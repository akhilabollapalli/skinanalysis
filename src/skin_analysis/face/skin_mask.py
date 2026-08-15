"""Skin mask construction.

V1 has NO learned face parser. CelebAMask-HQ is excluded by the licensing gate, which
excludes the BiSeNet weights trained on it, so the mask is built from landmark geometry
plus classical rejection of hair, beard, glasses, specular highlight and deep shadow.

This is the weakest link in the commercial-open V1 and the first place to look when a
feature produces false positives -- see .claude/skills/roi-debug. When a first-party or
commercially clear parser becomes available, replacing this module should be the highest-
value upgrade in the pipeline.

**The mask is subject-adaptive, not colour-absolute.** The textbook classical approach --
a fixed RGB/HSV/YCrCb "skin colour" box -- is precisely the technique that fails on dark
skin, and it fails silently: the mask shrinks, ROI coverage drops, and the concern quietly
reports UNMEASURABLE or scores a smaller region. Nothing looks broken from the outside.

So this module samples THIS face's own skin at landmark-stable points, builds a robust
CIELAB reference, and classifies every pixel relative to it. Every threshold is a multiple
of the subject's own MAD; there is no absolute colour constant anywhere in the module or
its config. The reasoning mirrors D1 stage A: ask "does this pixel differ from this
person's skin", never "does this pixel look like skin in general".

Fails closed throughout. When the reference cannot be trusted, or the surviving mask is
implausibly small, the answer is an empty mask -- which makes ROIs UNMEASURABLE -- rather
than a plausible-looking region that no longer corresponds to skin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..schemas import RunMode
from ..util import scale

MAD_SCALE = 1.4826


class ReferenceError(RuntimeError):
    """Raised when this subject's skin reference cannot be established.

    Not recoverable by relaxing a threshold: if the reference patches disagree, the mask
    has no idea what this person's skin looks like, and every rejection downstream would
    inherit that error.
    """


@dataclass(frozen=True)
class SkinReference:
    """Robust CIELAB statistics for one subject's normal skin.

    ``mad`` values are the subject's own scale. Every classification threshold in
    ``config/skin_mask.yaml`` is expressed as a multiple of these, which is what makes the
    mask tone-independent.
    """

    l_median: float
    l_mad: float
    a_median: float
    b_median: float
    chroma_mad: float
    n_pixels: int
    n_patches_used: int
    n_patches_rejected: int


# ------------------------------------------------------------------ helpers


def _to_lab(image: np.ndarray) -> np.ndarray:
    """BGR uint8 -> CIELAB float32, using the fixed D65 conversion (D4).

    No gray-world, no illuminant estimation. Illumination is handled by the fact that
    every threshold here is relative to the subject's own reference.
    """
    import cv2

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    # OpenCV packs 8-bit Lab as L in 0..255 and a/b offset by 128. Undo both so the values
    # are comparable to the literature and to the feature modules.
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1] -= 128.0
    lab[..., 2] -= 128.0
    return lab


def _mad(values: np.ndarray) -> float:
    """Median absolute deviation, scaled so it estimates a standard deviation."""
    if values.size == 0:
        return 0.0
    median = float(np.median(values))
    return float(MAD_SCALE * np.median(np.abs(values - median)))


def _disc(shape: tuple[int, int], centre: np.ndarray, radius_px: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    cy, cx = int(round(centre[1])), int(round(centre[0]))
    r = max(1, radius_px)
    y0, y1 = max(0, cy - r), min(shape[0], cy + r + 1)
    x0, x1 = max(0, cx - r), min(shape[1], cx + r + 1)
    if y0 >= y1 or x0 >= x1:
        return mask
    ys, xs = np.ogrid[y0:y1, x0:x1]
    mask[y0:y1, x0:x1] = (ys - cy) ** 2 + (xs - cx) ** 2 <= r * r
    return mask


def _kernel(size_px: int):  # type: ignore[no-untyped-def]
    import cv2

    size = max(1, size_px)
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _dilate(mask: np.ndarray, size_px: int) -> np.ndarray:
    import cv2

    if size_px < 1 or not mask.any():
        return mask
    dilated: np.ndarray = cv2.dilate(mask.astype(np.uint8), _kernel(2 * size_px + 1))
    return dilated.astype(bool)


def _local_texture_energy(lab: np.ndarray, window_px: int) -> np.ndarray:
    """Local standard deviation of L*, the high-frequency signature hair and beard share.

    Computed as sqrt(E[L^2] - E[L]^2) over a box window: cheap, separable, and it responds
    to strand structure rather than to overall darkness -- which is the whole point, since
    darkness alone would flag dark skin.
    """
    import cv2

    size = max(3, window_px | 1)
    lightness = lab[..., 0]
    mean: np.ndarray = cv2.blur(lightness, (size, size))
    mean_sq: np.ndarray = cv2.blur(lightness * lightness, (size, size))
    return np.asarray(np.sqrt(np.maximum(mean_sq - mean * mean, 0.0)))


def _polygon_mask(landmarks: np.ndarray, indices: list[int],
                  shape: tuple[int, int]) -> np.ndarray:
    import cv2

    mask = np.zeros(shape, dtype=np.uint8)
    points = landmarks[np.asarray(indices, dtype=int), :2]
    if points.shape[0] >= 3:
        cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
    return np.asarray(mask, dtype=np.uint8).astype(bool)


# ------------------------------------------------------------------ reference


def build_reference(
    lab: np.ndarray,
    roi_polygons: dict[str, np.ndarray],
    config: dict,
) -> SkinReference:
    """Establish this subject's skin statistics from the cheek and forehead ROI polygons.

    Sampled from ROIs rather than hand-picked landmark indices. Hand-picking was tried and
    failed on real faces: one index landed on hair and another on the chin silhouette,
    i.e. on background. The reference MAD blew up to 37 on a 0-100 scale, every +/-3.5 MAD
    gate then accepted everything, and the mask degenerated into the whole face oval with
    nothing rejected -- while still looking like a working mask.

    ROI polygons are eroded inward and already exclude eyes, brows and lips, so they cannot
    land on those features by construction.

    Patches that disagree with the others are discarded first. One that sits on a beard, a
    cast shadow or a specular hotspot would otherwise redefine "normal skin" for the whole
    face, and every rejection downstream would inherit the error.

    Raises:
        ReferenceError: when too few patches agree, too few pixels survive, or the surviving
            sample has no spread to measure against.
    """
    spec = config["reference"]

    patches: list[np.ndarray] = []
    for name in spec["patch_rois"]:
        polygon = roi_polygons.get(name)
        if polygon is not None and polygon.any():
            patches.append(polygon)

    if not patches:
        raise ReferenceError(
            f"none of the reference ROIs {spec['patch_rois']} produced a usable polygon"
        )

    medians = np.array([float(np.median(lab[..., 0][p])) for p in patches])
    consensus = float(np.median(medians))
    spread = _mad(medians)

    # With a tight spread every patch looks like an outlier, so only apply the filter when
    # there is real disagreement to act on.
    if spread > 1e-6:
        keep = np.abs(medians - consensus) <= spec["patch_outlier_mads"] * spread
    else:
        keep = np.ones(len(patches), dtype=bool)

    kept = [p for p, k in zip(patches, keep, strict=True) if k]
    if len(kept) < int(spec["min_valid_patches"]):
        raise ReferenceError(
            f"only {len(kept)} of {len(patches)} reference patches agree "
            f"(need {spec['min_valid_patches']}). The face is probably occluded, "
            "unevenly lit, or the landmarks are wrong."
        )

    combined = np.zeros(lab.shape[:2], dtype=bool)
    for patch in kept:
        combined |= patch

    if int(combined.sum()) < int(spec["min_reference_pixels"]):
        raise ReferenceError(
            f"reference covers {int(combined.sum())} px, below the floor of "
            f"{spec['min_reference_pixels']}. The capture is too small to characterise."
        )

    lightness = lab[..., 0][combined]
    a_values = lab[..., 1][combined]
    b_values = lab[..., 2][combined]

    a_median = float(np.median(a_values))
    b_median = float(np.median(b_values))
    chroma = np.hypot(a_values - a_median, b_values - b_median)

    skin_chroma = float(np.hypot(a_median, b_median))
    l_mad = _mad(lightness)
    if l_mad <= 0.0:
        # No spread means no scale, and every threshold here is a multiple of that scale.
        # Continuing would divide by zero or accept the entire image.
        raise ReferenceError(
            "reference skin has zero measurable spread; cannot establish a subject scale"
        )
    # Floored against the subject's ABSOLUTE skin chroma. Flooring against l_mad instead
    # collapses the mask on even skin: l_mad is estimated on the small reference ROIs but
    # the gate is applied across the whole face, where shading varies chroma far more.
    chroma_mad = max(
        _mad(chroma),
        float(config["classify"]["min_chroma_mad_frac_of_skin_chroma"]) * skin_chroma,
    )

    return SkinReference(
        l_median=float(np.median(lightness)),
        l_mad=l_mad,
        a_median=a_median,
        b_median=b_median,
        chroma_mad=chroma_mad,
        n_pixels=int(combined.sum()),
        n_patches_used=len(kept),
        n_patches_rejected=len(patches) - len(kept),
    )


# ------------------------------------------------------------------ rejection stages


def _specular(lab: np.ndarray, ref: SkinReference, anchor_px: float,
              config: dict) -> np.ndarray:
    """Clipped toward the illuminant, not toward the skin: bright AND washed out."""
    spec = config["specular"]
    if ref.l_mad <= 0:
        return np.zeros(lab.shape[:2], dtype=bool)
    lightness_z = (lab[..., 0] - ref.l_median) / ref.l_mad
    # ABSOLUTE chroma, compared against this subject's own skin chroma. Using distance
    # from the skin chroma instead is backwards: a blown highlight is neutral, so it sits
    # far from skin chroma, not near it.
    chroma_abs = np.hypot(lab[..., 1], lab[..., 2])
    skin_chroma = float(np.hypot(ref.a_median, ref.b_median))
    if skin_chroma <= 1e-6:
        return np.zeros(lab.shape[:2], dtype=bool)
    hit = (lightness_z > spec["min_lightness_mads"]) & (
        chroma_abs < spec["max_chroma_frac_of_skin"] * skin_chroma
    )
    return _dilate(hit, scale.to_px(spec["dilate_frac_of_iod"], anchor_px, minimum=0))


def _deep_shadow(lab: np.ndarray, ref: SkinReference, config: dict) -> np.ndarray:
    """Darkness RELATIVE TO THIS SUBJECT, which is what keeps it off dark skin."""
    if ref.l_mad <= 0:
        return np.zeros(lab.shape[:2], dtype=bool)
    lightness_z = (lab[..., 0] - ref.l_median) / ref.l_mad
    return np.asarray(lightness_z < float(config["shadow"]["max_lightness_mads"]))


def _hair_like(lab: np.ndarray, ref: SkinReference, anchor_px: float, spec: dict,
               support: np.ndarray, region: np.ndarray | None = None) -> np.ndarray:
    """Darker than reference AND high-frequency.

    Requiring BOTH is what keeps this off dark skin, which is dark but smooth. Either
    condition alone would classify an entire dark-skinned face as hair -- the exact
    silent failure this module exists to avoid.
    """
    if not spec.get("enabled", True) or ref.l_mad <= 0:
        return np.zeros(lab.shape[:2], dtype=bool)

    lightness_z = (lab[..., 0] - ref.l_median) / ref.l_mad
    window = scale.to_px(spec["texture_window_frac_of_iod"], anchor_px, odd=True, minimum=3)
    energy = _local_texture_energy(lab, window)

    # Normalise against the FACE, not the whole frame. A frame is mostly smooth
    # background, so whole-image statistics make ordinary skin texture look like an
    # outlier and the detector eats the face. Measured across 14 real captures this was
    # rejecting 18-51% of the face as "hair".
    sample = energy[support] if support.any() else energy.ravel()
    energy_scale = _mad(sample)
    if energy_scale <= 1e-6:
        return np.zeros(lab.shape[:2], dtype=bool)
    energy_z = (energy - float(np.median(sample))) / energy_scale

    hit = (lightness_z < spec["max_lightness_mads"]) & (
        energy_z > spec["min_texture_energy_mads"]
    )
    if region is not None:
        hit &= region
    return _dilate(hit, scale.to_px(spec["dilate_frac_of_iod"], anchor_px, minimum=0))


def _glasses_band(lab: np.ndarray, landmarks: np.ndarray, ref: SkinReference,
                  anchor_px: float, config: dict, support: np.ndarray) -> np.ndarray:
    """Detect frames, then drop the whole eye band rather than salvaging parts of it.

    Lenses refract and tint whatever is behind them, so skin visible through a lens is not
    measurable skin even where the frame does not cover it.
    """
    import cv2

    spec = config["glasses"]
    if not spec.get("enabled", True) or ref.l_mad <= 0:
        return np.zeros(lab.shape[:2], dtype=bool)

    lightness = lab[..., 0]
    gradient = np.abs(cv2.Sobel(lightness, cv2.CV_32F, 0, 1, ksize=3))
    sample = gradient[support] if support.any() else gradient.ravel()
    gradient_scale = _mad(sample)
    if gradient_scale <= 1e-6:
        return np.zeros(lab.shape[:2], dtype=bool)

    strong = (gradient - float(np.median(sample))) / gradient_scale
    edges = (strong > spec["min_edge_strength_mads"]).astype(np.uint8)

    min_length = scale.to_px(spec["min_edge_length_frac_of_iod"], anchor_px, minimum=3)
    # A frame is LONG and near-horizontal. Opening with a horizontal line keeps only runs
    # of that description, which is what separates a frame from brow or lash edges.
    horizontal = cv2.morphologyEx(
        edges, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (min_length, 1))
    )
    if not horizontal.any():
        return np.zeros(lab.shape[:2], dtype=bool)

    eye_indices = [33, 133, 263, 362, 168]
    valid = [i for i in eye_indices if i < landmarks.shape[0]]
    if not valid:
        return np.zeros(lab.shape[:2], dtype=bool)
    eye_y = landmarks[valid, 1]
    pad = scale.to_px(spec["band_pad_frac_of_iod"], anchor_px, minimum=1)

    band = np.zeros(lab.shape[:2], dtype=bool)
    top = max(0, int(eye_y.min()) - pad)
    bottom = min(lab.shape[0], int(eye_y.max()) + pad + 1)
    band[top:bottom, :] = True

    # Only drop the band if a frame-like edge was actually found inside it.
    if not (horizontal.astype(bool) & band).any():
        return np.zeros(lab.shape[:2], dtype=bool)
    return band


def _geometric_exclusions(landmarks: np.ndarray, shape: tuple[int, int], anchor_px: float,
                          roi_config: dict, config: dict) -> np.ndarray:
    """Eyes, brows and lips, dilated outward.

    Landmark rings sit ON the boundary of the feature, not outside it, so an undilated
    ring leaves a rim of lash, brow hair or lip vermilion inside the mask.
    """
    spec = config["geometric"]
    rings = roi_config.get("exclusion_rings", {})
    pads = {
        "left_eye": "eye_dilate_frac_of_iod",
        "right_eye": "eye_dilate_frac_of_iod",
        "left_eyebrow": "brow_dilate_frac_of_iod",
        "right_eyebrow": "brow_dilate_frac_of_iod",
        "lips": "lip_dilate_frac_of_iod",
    }

    excluded = np.zeros(shape, dtype=bool)
    for name, indices in rings.items():
        ring = _polygon_mask(landmarks, indices, shape)
        pad_key = pads.get(name, "eye_dilate_frac_of_iod")
        excluded |= _dilate(ring, scale.to_px(spec[pad_key], anchor_px, minimum=0))
    return excluded


def _cleanup(mask: np.ndarray, anchor_px: float, config: dict) -> np.ndarray:
    import cv2

    spec = config["morphology"]
    open_px = scale.to_px(spec["open_frac_of_iod"], anchor_px, minimum=0)
    close_px = scale.to_px(spec["close_frac_of_iod"], anchor_px, minimum=0)

    result: np.ndarray = mask.astype(np.uint8)
    if open_px >= 1:
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, _kernel(2 * open_px + 1))
    if close_px >= 1:
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, _kernel(2 * close_px + 1))

    if spec.get("keep_largest_component", True) and result.any():
        count, labels, stats, _ = cv2.connectedComponentsWithStats(result, connectivity=8)
        if count > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            result = (labels == largest).astype(np.uint8)

    return result.astype(bool)


# ------------------------------------------------------------------ public API


def build(
    image: np.ndarray,
    landmarks: np.ndarray,
    config: dict,
    *,
    roi_polygons: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Return a bool mask, True where pixels are analyzable skin.

    Excludes eyes, brows, lips, nostrils, hair, beard, glasses, specular highlights and
    deep shadow. When a region is ambiguous, it is excluded.

    Args:
        image: BGR uint8, native resolution.
        landmarks: (478, 3) pixel coordinates.
        config: the parsed ``config/rois.yaml``. Skin-mask parameters are read from
            ``config/skin_mask.yaml`` via the loader.

    Returns:
        Bool mask of ``image.shape[:2]``. Empty when the subject's skin reference could not
        be established or the surviving mask is implausibly small -- an empty mask makes
        ROIs UNMEASURABLE, which is the honest outcome, where a plausible-looking mask that
        no longer corresponds to skin would produce a confident wrong answer.
    """
    result, _ = build_with_diagnostics(image, landmarks, config, roi_polygons=roi_polygons)
    return result


def build_with_diagnostics(
    image: np.ndarray,
    landmarks: np.ndarray,
    roi_config: dict,
    *,
    roi_polygons: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Same as :func:`build`, plus per-stage rejection areas for validation and debug.

    The diagnostics are internal only. They are what ``scripts/debug_roi.py`` renders and
    what a validation run slices by skin tone -- if one stage starts rejecting far more
    area on one tone than another, that is the silent failure this module is designed to
    make visible.
    """
    from ..util import config as cfg

    mask_config = cfg.load("skin_mask")
    shape = image.shape[:2]
    empty = np.zeros(shape, dtype=bool)

    anchor_px = scale.inter_ocular_distance(landmarks)
    lab = _to_lab(image)

    if roi_polygons is None:
        # Standalone/debug use. DEVELOPMENT mode is correct here: the caller that owns the
        # D15 production gate is the pipeline, and it passes polygons in rather than
        # letting this module re-derive them.
        from . import rois as _rois

        roi_polygons = _rois.build(
            landmarks, shape, roi_config, run_mode=RunMode.DEVELOPMENT
        )

    try:
        reference = build_reference(lab, roi_polygons, mask_config)
    except ReferenceError as exc:
        return empty, {"failed": str(exc), "stage": "reference"}

    # Outer bound: the face oval. Everything outside is background by construction, which
    # is a far stronger signal than any colour test.
    oval = roi_config.get("face_oval") or _DEFAULT_FACE_OVAL
    face = _polygon_mask(landmarks, oval, shape)

    lightness_z = (lab[..., 0] - reference.l_median) / reference.l_mad
    chroma = np.hypot(lab[..., 1] - reference.a_median, lab[..., 2] - reference.b_median)
    chroma_z = chroma / reference.chroma_mad

    classify = mask_config["classify"]
    skin_like = (
        (np.abs(lightness_z) <= classify["max_lightness_mads"])
        & (chroma_z <= classify["max_chroma_mads"])
    )

    specular = _specular(lab, reference, anchor_px, mask_config)
    shadow = _deep_shadow(lab, reference, mask_config)
    hair = _hair_like(lab, reference, anchor_px, mask_config["hair"], face)

    # Beard only below the nose. Above it, the same signature is eyebrow or lash.
    below_nose = np.zeros(shape, dtype=bool)
    nose_y = int(landmarks[2, 1]) if landmarks.shape[0] > 2 else shape[0] // 2
    below_nose[max(0, nose_y):, :] = True
    beard = _hair_like(
        lab, reference, anchor_px, mask_config["beard"], face, region=below_nose
    )

    glasses = _glasses_band(lab, landmarks, reference, anchor_px, mask_config, face)
    geometric = _geometric_exclusions(landmarks, shape, anchor_px, roi_config, mask_config)

    rejected = specular | shadow | hair | beard | glasses | geometric
    mask = _cleanup(face & skin_like & ~rejected, anchor_px, mask_config)

    face_area = int(face.sum())
    coverage = float(mask.sum()) / face_area if face_area else 0.0

    diagnostics: dict[str, Any] = {
        "anchor_px": anchor_px,
        "reference": {
            "l_median": reference.l_median,
            "l_mad": reference.l_mad,
            "chroma_mad": reference.chroma_mad,
            "n_pixels": reference.n_pixels,
            "patches_used": reference.n_patches_used,
            "patches_rejected": reference.n_patches_rejected,
        },
        "rejected_frac_of_face": {
            "specular": float((specular & face).sum()) / face_area if face_area else 0.0,
            "deep_shadow": float((shadow & face).sum()) / face_area if face_area else 0.0,
            "hair": float((hair & face).sum()) / face_area if face_area else 0.0,
            "beard": float((beard & face).sum()) / face_area if face_area else 0.0,
            "glasses": float((glasses & face).sum()) / face_area if face_area else 0.0,
            "geometric": float((geometric & face).sum()) / face_area if face_area else 0.0,
        },
        "coverage_frac_of_face": coverage,
    }

    floor = float(mask_config["morphology"]["min_mask_frac_of_face"])
    if coverage < floor:
        diagnostics["failed"] = (
            f"mask covers {coverage:.1%} of the face box, below the {floor:.0%} floor. "
            "The classifier failed rather than the face being small."
        )
        diagnostics["stage"] = "coverage"
        return empty, diagnostics

    return mask, diagnostics


#: FaceMesh V2 face-oval ring, ordered. Used as the outer bound when rois.yaml does not
#: override it.
_DEFAULT_FACE_OVAL = [
    10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148,
    152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297,
    338,
]
