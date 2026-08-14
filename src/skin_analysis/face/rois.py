"""Anatomical ROI polygons built from landmarks.

Definitions live in config/rois.yaml and are deliberately conservative -- eroded inward
so boundary pixels stay out of measurements. Losing measurable area is cheap; a confident
false finding shown to a user is not.

An ROI is always a landmark polygon INTERSECTED with the skin mask. Landmarks alone are
never the mask.
"""

from __future__ import annotations

import numpy as np


def build(
    landmarks: np.ndarray,
    image_shape: tuple[int, int],
    config: dict,
) -> dict[str, np.ndarray]:
    """Return ROI name -> bool mask, eroded per config."""
    raise NotImplementedError("face.rois.build is not implemented yet.")


def symmetry_ok(rois: dict[str, np.ndarray], config: dict) -> bool:
    """True when paired ROIs have comparable post-mask area.

    Unequal coverage manufactures asymmetry findings that look like real skin
    differences, so asymmetry must not be reported when this fails.
    """
    raise NotImplementedError("face.rois.symmetry_ok is not implemented yet.")
