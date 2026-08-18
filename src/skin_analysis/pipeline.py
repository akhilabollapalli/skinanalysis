"""Pipeline orchestration -- the only public entry point.

Order (architecture doc §3):

    capture QC  ->  landmarks + ROIs  ->  skin mask  ->  three image copies
                ->  active concern modules  ->  per-ROI severity  ->  aggregation
                ->  internal result  -> (only if calibrated) public payload

Two properties this module exists to guarantee:

* **Fails closed at QC.** A failed capture returns RETAKE and runs no concern logic.
  Disabled concerns report Severity.DISABLED without touching pixels.

* **Measurement and publication are separate (D2).** ``analyze_scan_internal`` always
  works. ``ScanResultInternal.to_public()`` refuses until calibration is complete. There
  is no code path that produces a user-facing severity from placeholder thresholds.

D10: this runs server-side, budget P50 < 1.5 s / P95 < 3.0 s after upload. The shared
intermediates below (one Lab conversion, one gradient pyramid) exist for that budget and
are PASSED IN to feature modules -- never cached in module globals, which would break the
purity contract in CLAUDE.md §4. Keep the QC and landmark stages free of server-only
assumptions so they can migrate on-device later without changing any concern API.

ORDERING NOTE. The architecture lists QC first, but three of its checks -- face size, pose,
ROI visibility -- are defined over the face layer's own output. So landmarks and the skin
mask are built BEFORE the gate closes, and only the concern modules are withheld. That
preserves the property CLAUDE.md actually states ("if capture QC fails, run no concern
logic"), and ``qc.precheck`` still rejects an obviously unusable frame before any of it is
paid for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .capture import qc
from .decision import decide as decision
from .face import landmarks as landmark_layer
from .face import rois as roi_layer
from .face import skin_mask as mask_layer
from .features import pigmentation, redness, texture, wrinkles
from .schemas import (
    ROI,
    CalibrationState,
    CaptureQC,
    Concern,
    FeatureContext,
    FeatureResultInternal,
    ImageCopy,
    PublicScanResult,
    RunMode,
    ScanResultInternal,
    Severity,
)
from .util import calibration, color, scale
from .util import config as cfg

#: Concern key in config -> the module implementing it. Only ACTIVE concerns appear here.
#: acne, fine_lines and pores are absent on purpose: they are feature-flagged off pending
#: commercially clear labels, and wiring them here "ready to switch on" is how a disabled
#: concern gets enabled by accident.
_ACTIVE_MODULES = {
    "redness": redness,
    "pigmentation": pigmentation,
    "texture": texture,
    "wrinkles": wrinkles,
}

#: Config key -> the Concern it reports as. ``pigmentation`` measures what the product
#: calls dark spots; the two names are kept distinct so neither layer has to know the
#: other's vocabulary.
_CONCERN_OF = {
    "redness": Concern.REDNESS,
    "pigmentation": Concern.DARK_SPOTS,
    "texture": Concern.TEXTURE,
    "wrinkles": Concern.WRINKLES,
    "fine_lines": Concern.FINE_LINES,
    "acne": Concern.ACNE,
    "pores": Concern.PORES,
}


@dataclass(frozen=True)
class ImageCopies:
    """The three preprocessing derivatives (D5).

    They deliberately do not share preprocessing:

    * ``color``   sRGB -> Lab/D65. No gray-world, no illuminant estimation, no CLAHE (D4).
                  Illumination is handled by each feature's within-image local baseline.
                  The term "color-calibrated" is retired: without a chart or RAW reference
                  it overclaims what this is.
    * ``texture`` Luminance with fixed normalization only. NO CLAHE. GLCM downstream of
                  adaptive gain would partly measure the gain, not the skin (D5).
    * ``ridge``   Optional validated enhancement for Gabor/Hessian. If enhancement is ever
                  adopted, its parameters join the frozen capture protocol.
    """

    color: np.ndarray
    texture: np.ndarray
    ridge: np.ndarray

    def get(self, which: ImageCopy) -> np.ndarray:
        return {
            ImageCopy.COLOR: self.color,
            ImageCopy.TEXTURE: self.texture,
            ImageCopy.RIDGE: self.ridge,
        }[which]


def analyze_scan_internal(
    image: np.ndarray,
    *,
    run_mode: RunMode = RunMode.PRODUCTION,
    device_metadata: dict | None = None,
) -> ScanResultInternal:
    """Run the full pipeline on one BGR image and return the internal result.

    Always available, including before calibration -- this is the entry point validation
    and threshold-fitting work uses.

    Args:
        image: BGR uint8, native resolution, as uploaded.
        run_mode: production refuses unverified ROI polygons (D15).
        device_metadata: recorded for a future capture-profile decision (D11). Internal.

    Returns:
        ScanResultInternal. Call ``.to_internal_payload()`` for validation logging, or
        ``.to_public()`` to attempt publication (which raises until calibrated, per D2).
    """
    profile = cfg.capture_profile()
    severity_cfg = cfg.load("severity_thresholds")
    roi_cfg = cfg.load("rois")

    # Cheap image-only rejection before landmarks and the mask are paid for (D10).
    early = qc.precheck(image, profile)
    if not early.passed:
        return _retake(early)

    detection = _detect(image, profile, run_mode)
    if detection is None:
        return _retake(qc.check(image, profile, face=None))

    landmark_points, n_faces, pose = detection
    anchor_px = scale.inter_ocular_distance(landmark_points)

    polygons = roi_layer.build(landmark_points, image.shape[:2], roi_cfg, run_mode=run_mode)
    mask, diagnostics = mask_layer.build_with_diagnostics(
        image, landmark_points, roi_cfg, roi_polygons=polygons
    )
    composed = roi_layer.compose(polygons, mask)
    visibility = roi_layer.measurable_fraction(polygons, composed)
    _exclude_unusable_rois(composed, visibility, mask, anchor_px, roi_cfg, profile)

    capture = qc.check(
        image,
        profile,
        face=qc.FaceObservation(
            n_faces=n_faces,
            face_box=_face_box(landmark_points, image.shape[:2]),
            pose_deg=pose,
            skin_mask=mask,
            roi_visibility=visibility,
            occlusion=_occlusion_fractions(diagnostics),
            device_metadata=dict(device_metadata or {}),
        ),
    )

    # The scale floor is a capture property, not a concern property, so it is enforced
    # once here rather than repeated in four feature modules.
    if not scale.anchor_is_sufficient(anchor_px, severity_cfg):
        capture = CaptureQC(
            passed=False,
            failures=[*capture.failures, qc.QCFailure.FACE_TOO_SMALL],
            metrics={**capture.metrics, "anchor_px": anchor_px},
            illumination_vector=capture.illumination_vector,
            device_metadata=capture.device_metadata,
            shadow_asymmetry_ok=capture.shadow_asymmetry_ok,
        )

    if not capture.passed:
        return _retake(capture)

    copies = make_image_copies(image, severity_cfg)
    context = FeatureContext(
        anchor_px=anchor_px,
        qc=capture.verdict(),
        capture_profile=str(cfg.load("capture_thresholds")["meta"]["active_profile"]),
        protocol_version=cfg.protocol_version(),
        run_mode=run_mode,
    )

    concerns = _run_concerns(copies, mask, composed, context)
    return ScanResultInternal(
        qc=capture,
        concerns=concerns,
        protocol_version=context.protocol_version,
        calibration_state=(
            CalibrationState.CALIBRATED
            if cfg.severity_calibrated()
            else CalibrationState.UNCALIBRATED
        ),
        module_versions=_module_versions(),
    )


def analyze_scan(image: np.ndarray) -> PublicScanResult:
    """Run the pipeline and return the user-facing payload.

    Raises:
        CalibrationRequiredError: while the active protocol is uncalibrated (D2).
    """
    return analyze_scan_internal(image).to_public()


def make_image_copies(image: np.ndarray, config: dict) -> ImageCopies:
    """Build the three preprocessing derivatives from one decoded image.

    The branches must NOT share preprocessing. Contrast enhancement helps structural
    detection but distorts the luminance and chroma relationships that redness and
    pigmentation depend on (architecture doc §12), and turns GLCM into a measurement of
    the enhancement itself (D5).

    Raises:
        ValueError: if any config flag would put an adaptive step in front of a
            measurement. The flags exist so that enabling one is a visible config change
            with a review attached, not an inline edit inside a feature module.
    """
    spec = config.get("image_copies", {}) or {}
    _reject_adaptive_preprocessing(spec)

    lab = color.bgr_to_lab(image)
    luminance = color.lab_to_luminance(lab)

    enhancement = str(spec.get("ridge", {}).get("enhancement", "none"))
    if enhancement != "none":
        raise ValueError(
            f"image_copies.ridge.enhancement is {enhancement!r}, which has no validated "
            "implementation. Adopting one refreezes calibration, because the ridge "
            "response distribution moves."
        )

    # The ridge branch is a COPY, not an alias. Sharing the array would let a future
    # in-place enhancement silently modify what the texture branch measures.
    return ImageCopies(color=lab, texture=luminance, ridge=luminance.copy())


# ------------------------------------------------------------------ internals


def _reject_adaptive_preprocessing(spec: dict) -> None:
    banned = {
        ("color", "gray_world"): "D4 forbids gray-world on the skin channels",
        ("color", "illuminant_estimation"): "D4 records the illuminant, never applies it",
        ("color", "clahe"): "CLAHE distorts the chroma relationships redness measures",
        ("texture", "clahe"): "a GLCM downstream of CLAHE partly measures CLAHE (D5)",
        ("ridge", "clahe"): "ridge CLAHE is an ablation candidate, not a default (D5)",
    }
    for (branch, flag), why in banned.items():
        if (spec.get(branch, {}) or {}).get(flag, False):
            raise ValueError(f"image_copies.{branch}.{flag} is true: {why}")


def _detect(
    image: np.ndarray,
    profile: dict,
    run_mode: RunMode,
) -> tuple[np.ndarray, int, tuple[float, float, float]] | None:
    """Landmarks, face count and pose, or None when no usable face was found."""
    faces = landmark_layer.detect_faces(image, profile)
    if not faces:
        return None
    points = faces[0]
    if points.shape[0] <= scale.RIGHT_IRIS_CENTER:
        # The 468-point surface mesh has no iris landmarks, so it cannot anchor scale (D1).
        return None
    del run_mode  # verification is enforced in roi_layer.build, not duplicated here
    return points, len(faces), landmark_layer.head_pose(points, image.shape[:2])


def _exclude_unusable_rois(
    composed: dict[str, np.ndarray],
    visibility: dict[str, float],
    mask: np.ndarray,
    anchor_px: float,
    roi_cfg: dict,
    profile: dict,
) -> None:
    """Zero any ROI too small or too occluded to trust, in place.

    Two independent reasons land here with the SAME treatment:

    * ``roi_layer.undersized`` -- an eroded polygon collapsed to a sliver (D7).
    * Below ``roi_visibility.min_visible_frac`` -- the polygon survived masking at all,
      but lost most of its area to hair, beard, shadow, or any other exclusion, so what
      remains is not a fair sample of that region.

    Zeroing here, rather than failing the whole capture, is what lets ONE bad ROI stay
    contained to itself: whichever other ROIs a concern needs are still available, and
    ``decision.decide`` (D7) reports a concern UNMEASURABLE only when EVERY ONE of its
    own primary ROIs was excluded. Before this existed, a hair-covered forehead alone
    discarded six otherwise-usable ROIs -- and every concern that needed them -- along
    with it. ``capture.qc._check_roi_visibility`` keeps a much looser capture-level floor
    for the case where nothing at all survives.
    """
    floor = float(profile["roi_visibility"]["min_visible_frac"])
    unusable = set(roi_layer.undersized(composed, anchor_px, roi_cfg))
    unusable |= {name for name, frac in visibility.items() if frac < floor}
    for name in unusable:
        visibility[name] = 0.0
        composed[name] = np.zeros_like(mask)


def _face_box(points: np.ndarray, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    xs, ys = points[:, 0], points[:, 1]
    x0 = int(max(0, np.floor(xs.min())))
    y0 = int(max(0, np.floor(ys.min())))
    x1 = int(min(shape[1], np.ceil(xs.max())))
    y1 = int(min(shape[0], np.ceil(ys.max())))
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


#: Mask stages QC gates on. ``specular`` is included because QC must NOT recompute it: the
#: mask measures it against the subject's own skin, and an absolute cutoff there was found
#: to be silently dead on brown skin (see qc._check_exposure).
_GATED_MASK_STAGES = ("hair", "beard", "glasses", "specular")


def _occlusion_fractions(diagnostics: dict) -> dict[str, float]:
    """Per-stage rejection fractions, named as the occlusion config expects them."""
    rejected = diagnostics.get("rejected_frac_of_face", {}) or {}
    return {stage: float(rejected.get(stage, 0.0)) for stage in _GATED_MASK_STAGES}


def _run_concerns(
    copies: ImageCopies,
    mask: np.ndarray,
    composed: dict[str, np.ndarray],
    context: FeatureContext,
) -> dict[Concern, FeatureResultInternal]:
    results: dict[Concern, FeatureResultInternal] = {}

    for key in cfg.configured_concerns():
        concern = _CONCERN_OF[key]

        if not cfg.feature_enabled(key):
            # DISABLED without touching pixels. A flagged-off concern must cost nothing and
            # must never be quietly enabled to fill the UI.
            results[concern] = FeatureResultInternal(
                concern=concern,
                severity=Severity.DISABLED,
                notes=[f"{key} is disabled pending commercially clear labels"],
            )
            continue

        module = _ACTIVE_MODULES.get(key)
        if module is None:
            raise KeyError(
                f"concern {key!r} is enabled in config but has no module in "
                "_ACTIVE_MODULES. Enabling a concern requires an implementation, not "
                "just a flag."
            )

        concern_cfg = cfg.concern_config(key)
        measurement = module.analyze(
            copies.get(module.IMAGE_COPY),
            mask,
            composed,
            concern_cfg,
            context=context,
        )
        results[concern] = _decide_or_leave_unmeasured(concern, measurement, concern_cfg)

    return results


def _decide_or_leave_unmeasured(
    concern: Concern,
    measurement: FeatureResultInternal,
    concern_cfg: dict,
) -> FeatureResultInternal:
    """Band the measurement if the cohort supports it; otherwise leave it UNMEASURABLE.

    Uncalibrated is not a finding (D2). Leaving the result UNMEASURABLE here is what lets
    ``analyze_scan_internal`` keep working for validation while ``to_public()`` still
    refuses -- and it is why there is no ``Severity.UNCALIBRATED`` to render.
    """
    measured = [r.roi for r in measurement.roi_results if r.raw]
    if not measured or not decision.decidable(concern, measured):
        return FeatureResultInternal(
            concern=measurement.concern,
            severity=Severity.UNMEASURABLE,
            regions=[],
            roi_results=measurement.roi_results,
            unmeasurable_regions=[r.roi for r in measurement.roi_results],
            raw=measurement.raw,
            notes=[
                *measurement.notes,
                "no calibrated population reference for the measured ROIs (D1 stage B)",
            ],
        )
    return decision.decide(measurement, concern_cfg)


def _retake(capture: CaptureQC) -> ScanResultInternal:
    """A failed capture. No concern logic runs -- not even for enabled concerns.

    Analyzing a bad capture yields a confident-looking wrong answer, which is worse for
    the user than being asked to retake.
    """
    return ScanResultInternal(
        qc=capture,
        concerns={},
        protocol_version=cfg.protocol_version(),
        calibration_state=(
            CalibrationState.CALIBRATED
            if cfg.severity_calibrated()
            else CalibrationState.UNCALIBRATED
        ),
        module_versions=_module_versions(),
    )


def _module_versions() -> dict[str, str]:
    """Recorded on every scan so a result can be traced to the code and assets that made it."""
    manifest = calibration.reference_manifest() or {}
    return {
        "landmark_bundle": landmark_layer.MODEL_VARIANT,
        "roi_schema": str(cfg.load("rois")["meta"].get("landmark_count", "")),
        "capture_schema": str(cfg.load("capture_thresholds")["meta"]["schema_version"]),
        "reference_set": str(manifest.get("version", "none")),
    }


__all__ = [
    "ImageCopies",
    "ROI",
    "analyze_scan",
    "analyze_scan_internal",
    "make_image_copies",
]
