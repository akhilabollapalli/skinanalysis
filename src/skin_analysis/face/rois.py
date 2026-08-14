"""Anatomical ROI polygons built from landmarks.

Definitions live in config/rois.yaml and are deliberately conservative -- eroded inward
so boundary pixels stay out of measurements. Losing measurable area is cheap; a confident
false finding shown to a user is not.

An ROI is always a landmark polygon INTERSECTED with the skin mask. Landmarks alone are
never the mask.

Verification is enforced, not trusted. ``meta.verified`` in config/rois.yaml flips to true
only after visual inspection with ``scripts/debug_roi.py`` across poses, hairstyles and
facial hair. Until then :func:`build` refuses to run in production mode: one forgotten
polygon silently moves every measurement taken inside it, and nothing downstream would
look wrong.
"""

from __future__ import annotations

import numpy as np

from ..schemas import RunMode, UnverifiedROIError


def assert_verified(config: dict, run_mode: RunMode) -> None:
    """Raise in production mode when ROI polygons have not been visually verified.

    Development mode is allowed through so the polygons can be defined and inspected in
    the first place -- that is the whole workflow that produces the verification.

    Raises:
        UnverifiedROIError: in production mode while ``meta.verified`` is false.
    """
    if run_mode is RunMode.PRODUCTION and not config.get("meta", {}).get("verified", False):
        raise UnverifiedROIError(
            "config/rois.yaml meta.verified is false. Verify every polygon with "
            "scripts/debug_roi.py across poses, hairstyles and facial hair before "
            "running in production."
        )


def build(
    landmarks: np.ndarray,
    image_shape: tuple[int, int],
    config: dict,
    *,
    run_mode: RunMode = RunMode.PRODUCTION,
) -> dict[str, np.ndarray]:
    """Return ROI name -> bool mask, eroded per config.

    These are the anatomical polygons ONLY. Intersection with the skin mask happens in
    :func:`compose`, kept separate so a bad polygon cannot be hidden behind a plausible
    skin mask.
    """
    raise NotImplementedError("face.rois.build is not implemented yet.")


def compose(
    roi_polygons: dict[str, np.ndarray],
    skin_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Intersect each anatomical polygon with the skin mask (architecture doc §3.2).

        ROI_k = Polygon(p_i1, ..., p_in) INTERSECT SkinMask

    Returns:
        ROI name -> the final measurable region for that ROI.
    """
    raise NotImplementedError("face.rois.compose is not implemented yet.")


def measurable_fraction(
    roi_polygons: dict[str, np.ndarray],
    composed: dict[str, np.ndarray],
) -> dict[str, float]:
    """Per ROI: what fraction of the anatomical polygon survived skin masking.

    Drives the ``roi_visibility.min_visible_frac`` gate and the D7 decision about which
    ROIs a concern may report from. Logged internally; never shown to a user.
    """
    raise NotImplementedError("face.rois.measurable_fraction is not implemented yet.")


def symmetry_ok(rois: dict[str, np.ndarray], config: dict) -> bool:
    """True when paired ROIs have comparable post-mask area.

    Unequal coverage manufactures asymmetry findings that look like real skin
    differences, so asymmetry must not be reported when this fails.
    """
    raise NotImplementedError("face.rois.symmetry_ok is not implemented yet.")
