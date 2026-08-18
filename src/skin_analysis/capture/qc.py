"""Capture quality gate.

Fails closed: if the capture does not pass, the pipeline returns RETAKE and no concern
logic runs at all. Analyzing a bad capture produces a confident-looking wrong answer,
which is worse for the user than being asked to retake.

This gate matters more here than in a typical pipeline. V1's redness and pigmentation
features are CIELAB proxies, so they partially measure the room's lighting. QC strictness
-- not feature sophistication -- is what makes those measurements trustworthy.

Thresholds: config/capture_thresholds.yaml.

TWO PHASES, because the checks need different inputs:

    precheck(image, config)              image alone -- blur, exposure, colour cast
    check(image, config, face=...)       everything, given a FaceObservation

``check`` with ``face=None`` fails with NO_FACE. That is not a special case to work
around; a frame with no detectable face has nothing to measure. Landmark detection loads a
model from disk, so it stays in the pipeline and out of this module -- which keeps every
function here pure and lets CI exercise the gate without the model bundle present.

D11: every scale-sensitive metric is computed on the CANONICAL FACE CROP, not the raw
upload. Normalising scale before measuring is what makes one `default` profile comparable
across devices, instead of maintaining a phone cohort nobody has validated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..schemas import CaptureQC, QCFailure
from ..util import color


@dataclass(frozen=True)
class FaceObservation:
    """What the face layer knows about a capture, handed to QC as plain data.

    Kept deliberately small and inert. QC must not reach back into the face layer: if it
    could, a QC threshold change could alter the mask, and the gate would be measuring
    something it had itself influenced.
    """

    #: Number of faces the detector returned. More than one is a rejection, not a crop.
    n_faces: int
    #: (x, y, w, h) of the face box in NATIVE image pixels.
    face_box: tuple[int, int, int, int]
    #: Yaw, pitch, roll in degrees.
    pose_deg: tuple[float, float, float]
    #: Analyzable-skin mask over the native frame.
    skin_mask: np.ndarray
    #: ROI name -> surviving fraction of its polygon, from ``rois.measurable_fraction``.
    roi_visibility: dict[str, float] = field(default_factory=dict)
    #: Per-stage rejection fractions from ``skin_mask.build_with_diagnostics``.
    #: Carries ``hair``, ``beard``, ``glasses`` AND ``specular`` -- see the note in
    #: ``_check_exposure`` for why specular coverage is not recomputed here.
    occlusion: dict[str, float] = field(default_factory=dict)
    device_metadata: dict[str, Any] = field(default_factory=dict)


def check(
    image: np.ndarray,
    config: dict,
    *,
    face: FaceObservation | None = None,
) -> CaptureQC:
    """Run every capture check and return a pass/fail with reasons.

    Checks: face presence/count/size, pose, blur, exposure and clipping, specular
    coverage, left/right shadow asymmetry, white balance, ROI visibility, occlusion,
    and a non-blocking filter/makeup warning.

    Args:
        image: BGR uint8, native resolution, as uploaded.
        config: a resolved capture profile from ``util.config.capture_profile``.
        face: what the face layer observed. ``None`` means no face was detected, which is
            a rejection.

    Returns:
        CaptureQC. ``metrics`` is internal only; the public payload carries just the
        boolean and the reason list.
    """
    failures: list[QCFailure] = []
    metrics: dict[str, float] = {}

    cast, illumination = color.gray_world_deviation(image)
    metrics["gray_world_deviation"] = cast
    if cast > float(config["white_balance"]["max_gray_world_deviation"]):
        # LOAD-BEARING under D4. Nothing downstream corrects a cast, so this check is the
        # only thing between a badly lit room and an a* measurement.
        failures.append(QCFailure.COLOR_CAST)

    if face is None:
        failures.append(QCFailure.NO_FACE)
        return CaptureQC(
            passed=False,
            failures=failures,
            metrics=metrics,
            illumination_vector=illumination,
        )

    failures.extend(_check_face_geometry(image, config, face, metrics))

    crop = canonical_crop(image, face.face_box, config)
    failures.extend(_check_blur(crop, config, metrics))

    skin = face.skin_mask
    # Recorded unconditionally. A metric present only on the failure path makes every
    # distribution over it a distribution over failures, which is how a validation summary
    # ends up describing the opposite of what it appears to.
    metrics["skin_px"] = float(np.count_nonzero(skin))
    shadow_asymmetry_ok = False
    if not skin.any():
        failures.append(QCFailure.OCCLUSION)
    else:
        failures.extend(
            _check_exposure(
                image, skin, config, metrics, float(face.occlusion.get("specular", 0.0))
            )
        )
        shadow_failures, shadow_asymmetry_ok = _check_shadow(image, skin, config, metrics)
        failures.extend(shadow_failures)

    failures.extend(_check_roi_visibility(config, face, metrics))
    failures.extend(_check_occlusion(config, face, metrics))

    # Filter/makeup is a WARNING, never a block (config: warn_on_suspected_filter). It
    # biases every classical colour proxy, so it is recorded for validation slicing, but
    # rejecting on a heuristic this weak would retake good captures.
    if config["occlusion"].get("warn_on_suspected_filter", False):
        metrics["suspected_filter"] = float(
            _suspected_filter(
                image, skin, float(config["occlusion"]["suspected_filter_detail_ratio"])
            )
        )

    ordered = sorted(dict.fromkeys(failures), key=lambda f: f.value)
    return CaptureQC(
        passed=not ordered,
        failures=ordered,
        metrics=metrics,
        illumination_vector=illumination,
        device_metadata=dict(face.device_metadata),
        shadow_asymmetry_ok=shadow_asymmetry_ok,
    )


def precheck(image: np.ndarray, config: dict) -> CaptureQC:
    """The subset of checks that need no face: blur over the whole frame, and colour cast.

    Exists for the D10 latency budget. A frame that is obviously out of focus or badly
    cast can be rejected before landmark detection and mask construction are paid for.
    A pass here is NOT a pass overall -- ``check`` still has to run.
    """
    failures: list[QCFailure] = []
    metrics: dict[str, float] = {}

    cast, illumination = color.gray_world_deviation(image)
    metrics["gray_world_deviation"] = cast
    if cast > float(config["white_balance"]["max_gray_world_deviation"]):
        failures.append(QCFailure.COLOR_CAST)

    failures.extend(_check_blur(image, config, metrics, prefix="precheck_"))

    return CaptureQC(
        passed=not failures,
        failures=sorted(dict.fromkeys(failures), key=lambda f: f.value),
        metrics=metrics,
        illumination_vector=illumination,
    )


# ------------------------------------------------------------------ canonical crop (D11)


def canonical_crop(
    image: np.ndarray,
    face_box: tuple[int, int, int, int],
    config: dict,
) -> np.ndarray:
    """Resample the face box to the profile's canonical width.

    Downsampling to reach the canonical width is legitimate. UPSAMPLING is not, and is
    refused: a blur score computed on interpolated pixels measures the interpolator. The
    ``face.min_face_px`` check is what guarantees there is something to downsample from.
    """
    import cv2

    x, y, w, h = face_box
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(image.shape[1], x + w), min(image.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return image

    crop = image[y0:y1, x0:x1]
    target = int(config["canonical"]["qc_face_width_px"])
    if crop.shape[1] <= target:
        return crop

    scale_factor = target / crop.shape[1]
    size = (target, max(1, int(round(crop.shape[0] * scale_factor))))
    resized: np.ndarray = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    return resized


# ------------------------------------------------------------------ individual checks


def _fraction(predicate: np.ndarray) -> float:
    """Share of True in a boolean array, 0.0 when empty."""
    if predicate.size == 0:
        return 0.0
    return float(np.count_nonzero(predicate) / predicate.size)


def _check_face_geometry(
    image: np.ndarray,
    config: dict,
    face: FaceObservation,
    metrics: dict[str, float],
) -> list[QCFailure]:
    spec = config["face"]
    failures: list[QCFailure] = []

    if face.n_faces > int(spec["max_faces"]):
        # Rejected, not cropped to the largest. Silently picking one face means the scan
        # may not be of the person who requested it.
        failures.append(QCFailure.MULTIPLE_FACES)

    _x, _y, box_w, box_h = face.face_box
    height_frac = box_h / image.shape[0] if image.shape[0] else 0.0
    metrics["face_height_frac"] = float(height_frac)
    metrics["face_px"] = float(max(box_w, box_h))

    # The native floor may not sit below the canonical width. Between the two, the face
    # clears the size gate but cannot be resampled UP to canonical -- upsampling would
    # invent pixels -- so every scale-sensitive metric would then be measured on a crop
    # narrower than the one its threshold was set on. D11 exists to stop exactly that.
    canonical_width = int(config["canonical"]["qc_face_width_px"])
    min_face_px = int(spec["min_face_px"])
    if min_face_px < canonical_width:
        raise ValueError(
            f"face.min_face_px ({min_face_px}) is below canonical.qc_face_width_px "
            f"({canonical_width}). Captures between the two would skip canonical "
            "normalisation silently, which defeats D11."
        )

    if height_frac < float(spec["min_face_height_frac"]):
        failures.append(QCFailure.FACE_TOO_SMALL)
    if box_w < min_face_px:
        # Width, not max(w, h): the canonical crop normalises on WIDTH, so width is what
        # has to clear the floor. A tall narrow box passing on its height would land back
        # in the sub-canonical case this check exists to prevent.
        failures.append(QCFailure.FACE_TOO_SMALL)

    pose = config["pose"]
    yaw, pitch, roll = face.pose_deg
    metrics.update({"yaw_deg": float(yaw), "pitch_deg": float(pitch), "roll_deg": float(roll)})
    if (
        abs(yaw) > float(pose["max_yaw_deg"])
        or abs(pitch) > float(pose["max_pitch_deg"])
        or abs(roll) > float(pose["max_roll_deg"])
    ):
        # Beyond these, ROI polygons stop corresponding to the same anatomy across scans,
        # which destroys repeatability -- the property this product is judged on.
        failures.append(QCFailure.EXTREME_POSE)

    return failures


def _check_blur(
    crop: np.ndarray,
    config: dict,
    metrics: dict[str, float],
    *,
    prefix: str = "",
) -> list[QCFailure]:
    """Focus check. ``prefix`` selects which support's thresholds apply.

    The full frame and the canonical face crop are DIFFERENT SUPPORTS and get different
    cutoffs. A frame is mostly background, which is smoother than skin, so reusing the
    crop's threshold on the frame rejects captures that are perfectly in focus. Applying
    one scale to two supports is the mistake that produced every skin-mask bug here.
    """
    import cv2

    spec = config["blur"]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = gray.astype(np.float32)

    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    tenengrad = float(np.asarray(gx * gx + gy * gy, dtype=np.float64).mean())

    metrics[f"{prefix}laplacian_var"] = laplacian_var
    metrics[f"{prefix}tenengrad"] = tenengrad

    min_laplacian = float(spec[f"{prefix}min_laplacian_var"])
    min_tenengrad = float(spec[f"{prefix}min_tenengrad"])
    if laplacian_var < min_laplacian or tenengrad < min_tenengrad:
        return [QCFailure.BLUR]
    return []


def _check_exposure(
    image: np.ndarray,
    skin: np.ndarray,
    config: dict,
    metrics: dict[str, float],
    specular_frac: float,
) -> list[QCFailure]:
    """Exposure over SKIN pixels, not the whole frame.

    A dark background would otherwise drag the mean down and reject a well-exposed face,
    and a blown background would do the reverse.

    ``specular_frac`` comes from the SKIN MASK, which measures it relative to the subject's
    own skin. It is not recomputed here, and that is the point.

    An earlier version of this function tested absolute values -- L* above 85 and chroma
    below 8, the signature of a blown highlight on light skin. Measured on 26 real captures
    of brown skin, the brightest 1% of skin pixels sat at L* 42-78 with chroma 18-43, so the
    gate matched NOTHING on any of them and specular coverage read identically zero while
    shine was plainly visible on foreheads and noses. A shiny patch on brown skin is bright
    RELATIVE TO THAT PERSON and still strongly pigmented; it never goes neutral.

    That is an absolute threshold applied to a tone-dependent quantity, which is the silent
    tone-dependent failure config/skin_mask.yaml exists to avoid, and it fails in the worst
    direction: it works on light skin and quietly stops working on dark skin. One source of
    truth, subject-relative, is the fix.
    """
    import cv2

    spec = config["exposure"]
    failures: list[QCFailure] = []

    luma = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)[skin]
    mean_luma = float(luma.mean())
    clipped_low = _fraction(luma <= float(spec["clip_low_level"]))
    clipped_high = _fraction(luma >= float(spec["clip_high_level"]))

    metrics.update(
        {
            "mean_luma": mean_luma,
            "clipped_low_frac": clipped_low,
            "clipped_high_frac": clipped_high,
        }
    )

    if mean_luma < float(spec["min_mean_luma"]) or clipped_low > float(
        spec["max_clipped_low_frac"]
    ):
        failures.append(QCFailure.UNDEREXPOSED)
    if mean_luma > float(spec["max_mean_luma"]) or clipped_high > float(
        spec["max_clipped_high_frac"]
    ):
        failures.append(QCFailure.OVEREXPOSED)

    metrics["specular_frac"] = specular_frac
    if specular_frac > float(spec["max_specular_frac"]):
        failures.append(QCFailure.OVEREXPOSED)

    return failures


def _check_shadow(
    image: np.ndarray,
    skin: np.ndarray,
    config: dict,
    metrics: dict[str, float],
) -> tuple[list[QCFailure], bool]:
    """Left/right luminance asymmetry and deep-shadow coverage over skin.

    Side lighting is the single most common cause of a false asymmetry finding, and V1
    corrects neither shadow nor cast (D4).

    Returns:
        (failures, shadow_asymmetry_ok). These are DELIBERATELY separate outcomes.
        Uneven left/right lighting invalidates a CROSS-SIDE comparison -- what redness
        calls asymmetry -- but not what either side reads on its own; each cheek's own
        colour is still real pixels. Rejecting the whole capture over an unreliable
        comparison discarded every other ROI along with it, so ``shadow_asymmetry_ok``
        alone drives ``QCVerdict.shadow_pass`` (suppressing just the comparison,
        features/redness.py ``asymmetry()``) and never enters ``failures``.

        Deep, pervasive shadow is different in kind: it is an exposure problem, and a
        pixel too dark to trust is too dark to trust on its own, not just relative to its
        mirror. That case stays in ``failures`` and blocks the capture.
    """
    import cv2

    spec = config["shadow"]
    failures: list[QCFailure] = []

    luma = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    columns = np.nonzero(skin.any(axis=0))[0]
    if columns.size == 0:
        return [QCFailure.OCCLUSION], False

    midline = int((columns.min() + columns.max()) / 2)
    left_half = skin.copy()
    left_half[:, midline:] = False
    right_half = skin.copy()
    right_half[:, :midline] = False

    overall = float(luma[skin].mean())
    if overall <= 1e-6 or not left_half.any() or not right_half.any():
        # No basis for a left/right comparison at all (e.g. skin entirely on one side).
        # Not blocking on its own -- see the docstring -- but nothing to compare either.
        return failures, False

    asymmetry = abs(float(luma[left_half].mean()) - float(luma[right_half].mean())) / overall
    metrics["lr_luma_asymmetry"] = asymmetry
    shadow_asymmetry_ok = asymmetry <= float(spec["max_lr_luma_asymmetry"])

    # Deep shadow RELATIVE TO THIS CAPTURE's own skin, so the test does not simply fire on
    # dark skin -- which is dark but evenly lit.
    deep_frac = _fraction(luma[skin] < float(spec["deep_shadow_luma_frac"]) * overall)
    metrics["deep_shadow_frac"] = deep_frac
    if deep_frac > float(spec["max_deep_shadow_frac"]):
        failures.append(QCFailure.SHADOW_ASYMMETRY)

    return failures, shadow_asymmetry_ok


def _check_roi_visibility(
    config: dict,
    face: FaceObservation,
    metrics: dict[str, float],
) -> list[QCFailure]:
    """Fail the capture only when NONE of the required ROIs is usable.

    This used to require EVERY required ROI to individually clear the floor, which meant
    one hair-covered forehead discarded the whole capture -- including left_cheek and
    right_cheek, which were often still perfectly visible. That duplicated a job the
    pipeline already does better: ``analyze_scan_internal`` zeroes any individual ROI
    below this same floor (see the comment there) so it is never measured, and D7 reports
    a concern UNMEASURABLE only when EVERY ONE of its own primary ROIs is gone -- a
    per-concern policy that already knows which ROIs each concern actually needs.

    What remains here is the genuine floor beneath that: if not even one of the three
    core regions survived, there is nothing left for any concern to report from, and
    asking for a clean retake is a better experience than running the full pipeline just
    to get UNMEASURABLE back from every concern.
    """
    spec = config["roi_visibility"]
    floor = float(spec["min_visible_frac"])
    required = [str(name) for name in spec.get("required", [])]

    visible = [face.roi_visibility.get(name, 0.0) for name in required]
    metrics["min_required_roi_visibility"] = float(min(visible)) if visible else 0.0

    if not visible or max(visible) < floor:
        return [QCFailure.INSUFFICIENT_ROI_VISIBILITY]
    return []


def _check_occlusion(
    config: dict,
    face: FaceObservation,
    metrics: dict[str, float],
) -> list[QCFailure]:
    """Beard and glasses still block the whole capture; hair no longer does.

    Hair coverage is a WHOLE-FACE fraction, and its actual effect on measurability is
    already better represented per ROI: hair over the forehead lowers only
    ``roi_visibility['forehead']``, which the pipeline already zeroes when it is too low
    (see the comment in ``analyze_scan_internal``) without discarding cheeks, nose, or
    chin along with it. Rejecting on the whole-face fraction on top of that penalized a
    hairstyle rather than a measurability problem -- a fringe or loose hair framing both
    sides of the face could cross 25% of total face area while the forehead itself was
    still mostly clear, or vice versa. The metric is still recorded for validation.

    Beard and glasses stay whole-face blocks: unlike hair, their real-world coverage
    pattern is closer to uniform across the ROI(s) they touch (chin; under-eye/nose
    bridge), so the per-ROI mechanism buys less here, and softening them is out of scope
    for this change.
    """
    spec = config["occlusion"]
    metrics["hair_frac"] = float(face.occlusion.get("hair", 0.0))

    limits = {
        "beard": float(spec["max_beard_frac"]),
        "glasses": float(spec["max_glasses_frac"]),
    }
    for stage, limit in limits.items():
        observed = float(face.occlusion.get(stage, 0.0))
        metrics[f"{stage}_frac"] = observed
        if observed > limit:
            return [QCFailure.OCCLUSION]
    return []


def _suspected_filter(image: np.ndarray, skin: np.ndarray, max_detail_ratio: float) -> bool:
    """Heuristic flag for beauty filters and heavy makeup. WARNING ONLY -- never blocks.

    A filter's signature is skin that is smoother than the rest of the frame is sharp: the
    smoothing is applied to the face, not to the background. Comparing the two is what
    makes this independent of overall image sharpness, which the blur check already covers.

    Deliberately not a rejection. It biases every classical colour proxy and so belongs in
    the validation slice, but the heuristic is far too weak to send a user back for a
    retake on its own.
    """
    import cv2

    if not skin.any() or skin.all():
        return False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    detail = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    face_detail = float(np.mean(detail[skin]))
    frame_detail = float(np.mean(detail[~skin]))
    if frame_detail <= 1e-6:
        return False
    return face_detail / frame_detail < max_detail_ratio
