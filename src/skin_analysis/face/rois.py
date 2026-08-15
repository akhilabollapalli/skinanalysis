"""Anatomical ROI polygons built from landmarks.

Definitions live in config/rois.yaml and are deliberately conservative -- eroded inward
so boundary pixels stay out of measurements. Losing measurable area is cheap; a confident
false finding shown to a user is not.

An ROI is always a landmark polygon INTERSECTED with the skin mask. Landmarks alone are
never the mask, which is why :func:`build` and :func:`compose` are separate: a bad polygon
can hide behind a plausible skin mask, and stay hidden until a feature produces a finding
nobody can explain.

**LEFT and RIGHT are anatomical** -- the subject's own, following MediaPipe. The subject's
left cheek appears on the RIGHT of the image. Reversing this mirrors every paired ROI and
every asymmetry finding while passing any test that checks only magnitudes.

Three ROIs have no FaceMesh vertices at all (forehead, crow's feet, nasolabial) and are
constructed from anchors plus fractional offsets. The constructions are named in config;
this module implements them but holds none of their numbers.

Verification is enforced, not trusted: :func:`build` refuses to run in production while
``config/rois.yaml meta.verified`` is false (D15).
"""

from __future__ import annotations

import numpy as np

from ..schemas import RunMode, UnverifiedROIError
from ..util import scale


def assert_verified(config: dict, run_mode: RunMode) -> None:
    """Raise in production mode when ROI polygons have not been visually verified.

    Development mode is allowed through so the polygons can be defined and inspected in
    the first place -- that is the workflow that produces the verification.

    Raises:
        UnverifiedROIError: in production mode while ``meta.verified`` is false.
    """
    if run_mode is RunMode.PRODUCTION and not config.get("meta", {}).get("verified", False):
        raise UnverifiedROIError(
            "config/rois.yaml meta.verified is false. Verify every polygon with "
            "scripts/debug_roi.py across poses, hairstyles and facial hair before "
            "running in production."
        )


# ------------------------------------------------------------------ geometry helpers


def _pts(landmarks: np.ndarray, indices: list[int]) -> np.ndarray:
    return landmarks[np.asarray(indices, dtype=int), :2]


def _resolve_anchor(landmarks: np.ndarray, config: dict, name: str) -> np.ndarray:
    """Resolve a named anchor from config into an (M, 2) array of points."""
    value = config["anchors"][name]
    indices = value if isinstance(value, list) else [value]
    return _pts(landmarks, indices)


def _face_midline_x(landmarks: np.ndarray) -> float:
    """Horizontal centre of the face, used to decide which way 'lateral' points."""
    left = landmarks[scale.LEFT_IRIS_CENTER, 0]
    right = landmarks[scale.RIGHT_IRIS_CENTER, 0]
    return float((left + right) / 2.0)


def _rasterize(polygon: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Fill a polygon into a bool mask of ``shape`` = (height, width)."""
    import cv2

    mask = np.zeros(shape, dtype=np.uint8)
    if polygon.shape[0] >= 3:
        cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1)
    return mask.astype(bool)


def _erode(mask: np.ndarray, margin_px: int) -> np.ndarray:
    """Shrink a mask inward so boundary pixels stay out of the measurement."""
    import cv2

    if margin_px < 1 or not mask.any():
        return mask
    size = 2 * margin_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.erode(mask.astype(np.uint8), kernel).astype(bool)


def _convex_ring(points: np.ndarray) -> np.ndarray:
    """Convex hull of a point set, as an ordered ring.

    Used for derived regions. A hull cannot self-intersect, which a hand-ordered ring of
    constructed points easily can -- and a self-intersecting polygon rasterizes into
    something that still looks plausible in an overlay.
    """
    import cv2

    if points.shape[0] < 3:
        return points
    hull: np.ndarray = cv2.convexHull(np.round(points).astype(np.int32))
    return np.asarray(hull.reshape(-1, 2), dtype=np.float64)


# ------------------------------------------------------------------ constructions


def _construct_extrude_from_brow(
    landmarks: np.ndarray, spec: dict, config: dict, anchor_px: float
) -> np.ndarray:
    """Forehead: extrude upward from the brow line.

    FaceMesh's top edge follows the hairline, not the brow, and does so inconsistently
    across hairstyles. So the forehead is built from the brow upward by a bounded amount
    rather than from the mesh's top edge downward -- capping the height is what keeps hair
    out, and hair is this ROI's dominant failure mode.
    """
    brow = np.vstack([
        _resolve_anchor(landmarks, config, name) for name in spec["anchors"]
    ])
    order = np.argsort(brow[:, 0])
    brow = brow[order]

    clearance = spec["brow_clearance_frac_of_iod"] * anchor_px
    height = spec["extrude_frac_of_iod"] * anchor_px

    # Image y increases downward, so "up" is negative.
    lower = brow.copy()
    lower[:, 1] -= clearance
    upper = brow.copy()
    upper[:, 1] -= clearance + height

    return np.vstack([lower, upper[::-1]])


def _construct_lateral_patch(
    landmarks: np.ndarray, spec: dict, config: dict, anchor_px: float
) -> np.ndarray:
    """Crow's feet: a patch placed outboard of the outer eye corner.

    'Lateral' is resolved against the face midline rather than hardcoded per side, so the
    same construction serves both eyes and cannot be mirrored by a config typo.
    """
    corner = _resolve_anchor(landmarks, config, spec["anchors"][0])[0]
    midline = _face_midline_x(landmarks)
    direction = 1.0 if corner[0] >= midline else -1.0

    offset = spec["offset_frac_of_iod"] * anchor_px
    width = spec["width_frac_of_iod"] * anchor_px
    height = spec["height_frac_of_iod"] * anchor_px

    near_x = corner[0] + direction * offset
    far_x = near_x + direction * width
    top_y = corner[1] - height / 2.0
    bottom_y = corner[1] + height / 2.0

    return np.array([
        [near_x, top_y], [far_x, top_y], [far_x, bottom_y], [near_x, bottom_y],
    ])


def _construct_band_between(
    landmarks: np.ndarray, spec: dict, config: dict, anchor_px: float
) -> np.ndarray:
    """Nasolabial: a band along the segment between two anchors.

    The fold has no vertices, only its endpoints, so the band is built perpendicular to
    the alar-to-mouth-corner segment.
    """
    start = _resolve_anchor(landmarks, config, spec["anchors"][0])[0]
    end = _resolve_anchor(landmarks, config, spec["anchors"][1])[0]

    axis = end - start
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        return np.empty((0, 2))
    normal = np.array([-axis[1], axis[0]]) / length
    half = (spec["width_frac_of_iod"] * anchor_px) / 2.0

    return np.array([
        start + normal * half, end + normal * half,
        end - normal * half, start - normal * half,
    ])


_CONSTRUCTIONS = {
    "extrude_from_brow": _construct_extrude_from_brow,
    "lateral_patch": _construct_lateral_patch,
    "band_between": _construct_band_between,
}


# ------------------------------------------------------------------ public API


def build(
    landmarks: np.ndarray,
    image_shape: tuple[int, int],
    config: dict,
    *,
    run_mode: RunMode = RunMode.PRODUCTION,
) -> dict[str, np.ndarray]:
    """Return ROI name -> bool mask, eroded per config.

    These are the anatomical polygons ONLY. Intersection with the skin mask happens in
    :func:`compose`.

    Args:
        landmarks: (478, 3) pixel coordinates from ``face.landmarks.detect``.
        image_shape: (height, width) of the frame the landmarks are in.
        config: the parsed ``config/rois.yaml``.
        run_mode: production refuses unverified polygons (D15).

    Raises:
        UnverifiedROIError: in production while ``meta.verified`` is false.
    """
    assert_verified(config, run_mode)

    anchor_px = scale.inter_ocular_distance(landmarks)
    erosion = config["erosion"]
    per_roi = erosion.get("per_roi_margin_frac_of_iod", {}) or {}

    excluded = _exclusion_mask(landmarks, image_shape, config)

    built: dict[str, np.ndarray] = {}
    for name, spec in config["rois"].items():
        kind = spec.get("type", "polygon")
        if kind == "polygon":
            # Hulled, not used in listed order. A hand-ordered ring of mesh indices is
            # easy to get wrong, and a self-intersecting polygon does not fail loudly --
            # it rasterizes into a thin arc that still looks like a region in an overlay.
            # That is exactly how the first cheek definitions passed review.
            ring = _convex_ring(_pts(landmarks, spec["landmarks"]))
        elif kind == "derived":
            construction = _CONSTRUCTIONS.get(spec["construction"])
            if construction is None:
                raise KeyError(
                    f"{name}: unknown construction {spec['construction']!r}. "
                    f"Known: {sorted(_CONSTRUCTIONS)}"
                )
            ring = _convex_ring(construction(landmarks, spec, config, anchor_px))
        else:
            raise KeyError(f"{name}: unknown ROI type {kind!r}; expected polygon or derived")

        mask = _rasterize(ring, image_shape)
        # Excluded anatomy is subtracted BEFORE erosion, so the erosion margin also backs
        # the ROI away from the eye and lip boundaries rather than only from its own edge.
        mask &= ~excluded
        margin_frac = per_roi.get(name, erosion["margin_frac_of_iod"])
        built[name] = _erode(mask, scale.to_px(margin_frac, anchor_px, minimum=0))

    return built


def _exclusion_mask(
    landmarks: np.ndarray, image_shape: tuple[int, int], config: dict
) -> np.ndarray:
    """Union of the anatomy that is excluded from every ROI regardless of membership."""
    excluded = np.zeros(image_shape, dtype=bool)
    for ring in config.get("exclusion_rings", {}).values():
        excluded |= _rasterize(_pts(landmarks, ring), image_shape)
    return excluded


def undersized(
    roi_masks: dict[str, np.ndarray],
    anchor_px: float,
    config: dict,
) -> list[str]:
    """ROIs whose area falls below the configured floor.

    An eroded polygon that collapses to a sliver still produces numbers, and those numbers
    look like a measurement rather than like noise. Such ROIs must be reported
    UNMEASURABLE rather than scored (D7).

    The floor is a fraction of anchor^2 so it means the same thing at any capture scale.
    """
    floor_px = float(config.get("min_area_frac_of_iod2", 0.0)) * anchor_px**2
    return sorted(name for name, mask in roi_masks.items() if float(mask.sum()) < floor_px)


def compose(
    roi_polygons: dict[str, np.ndarray],
    skin_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Intersect each anatomical polygon with the skin mask (architecture doc §3.2).

        ROI_k = Polygon(p_i1, ..., p_in) INTERSECT SkinMask

    Returns:
        ROI name -> the final measurable region for that ROI.
    """
    return {name: mask & skin_mask for name, mask in roi_polygons.items()}


def measurable_fraction(
    roi_polygons: dict[str, np.ndarray],
    composed: dict[str, np.ndarray],
) -> dict[str, float]:
    """Per ROI: what fraction of the anatomical polygon survived skin masking.

    Drives the ``roi_visibility.min_visible_frac`` gate and the D7 decision about which
    ROIs a concern may report from. Logged internally; never shown to a user.
    """
    fractions: dict[str, float] = {}
    for name, polygon in roi_polygons.items():
        total = int(polygon.sum())
        fractions[name] = float(composed[name].sum()) / total if total else 0.0
    return fractions


def symmetry_ok(rois: dict[str, np.ndarray], config: dict) -> bool:
    """True when paired ROIs have comparable post-mask area.

    Unequal coverage manufactures asymmetry findings that look like real skin
    differences, so asymmetry must not be reported when this fails.
    """
    tolerance = float(config.get("symmetry_area_tolerance", 0.25))
    for left, right in config.get("symmetry_pairs", []):
        if left not in rois or right not in rois:
            return False
        a, b = float(rois[left].sum()), float(rois[right].sum())
        larger = max(a, b)
        if larger == 0.0:
            return False
        if abs(a - b) / larger > tolerance:
            return False
    return True
