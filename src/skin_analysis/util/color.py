"""Colour-space conversion, shared by the skin mask and the colour concerns.

One conversion, one place. The skin mask and the redness/pigmentation features must agree
exactly on what ``L*`` and ``a*`` mean, or the mask will reject pixels on one scale while a
feature measures them on another -- and nothing about that disagreement would look wrong in
an overlay.

D4 applies here and nowhere else is it easier to violate: this module performs a **fixed**
sRGB -> CIELAB/D65 conversion. No gray-world, no illuminant estimation, no adaptive gain,
no CLAHE. Skin is not a neutral calibration target, so forcing average skin toward gray
removes exactly the chromatic information redness and pigmentation exist to measure. The
illumination vector is recorded by capture QC as evidence; it never touches a pixel.

Pure functions: no I/O, no globals.
"""

from __future__ import annotations

import numpy as np

#: OpenCV packs 8-bit Lab with L in 0..255 and a/b offset by +128. Undoing both puts the
#: values on the literature scale (L* 0..100, a*/b* roughly -128..127), which is what every
#: threshold in config/skin_mask.yaml and config/severity_thresholds.yaml assumes.
_OPENCV_L_SCALE = 100.0 / 255.0
_OPENCV_AB_OFFSET = 128.0

#: L* is 0..100; the texture and ridge copies want a 0..255 luminance. This is a FIXED
#: linear rescale -- the whole point of D5's texture branch is that no adaptive step sits
#: upstream of GLCM, so the mapping may not depend on image content.
_L_TO_BYTE = 255.0 / 100.0


def bgr_to_lab(image: np.ndarray) -> np.ndarray:
    """BGR uint8 -> CIELAB float32 on the literature scale.

    Args:
        image: (H, W, 3) BGR uint8, as decoded by OpenCV.

    Returns:
        (H, W, 3) float32 with L* in 0..100 and a*/b* centred on zero.
    """
    import cv2

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) BGR image, got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"expected uint8 BGR image, got dtype {image.dtype}")

    lab: np.ndarray = np.asarray(
        cv2.cvtColor(image, cv2.COLOR_BGR2LAB), dtype=np.float32
    )
    lab[..., 0] *= _OPENCV_L_SCALE
    lab[..., 1] -= _OPENCV_AB_OFFSET
    lab[..., 2] -= _OPENCV_AB_OFFSET
    return lab


def lab_to_luminance(lab: np.ndarray) -> np.ndarray:
    """L* (0..100) -> a 0..255 float32 luminance plane.

    Fixed rescale only. Deriving luminance from L* rather than from a BGR->GRAY conversion
    keeps the texture branch perceptually consistent with the colour branch, so a patch
    that the mask considered mid-tone is mid-tone here too.
    """
    return np.asarray(lab[..., 0] * _L_TO_BYTE, dtype=np.float32)


def chroma(lab: np.ndarray) -> np.ndarray:
    """Absolute chroma, ``hypot(a*, b*)``.

    Absolute, not distance-from-skin-chroma. A blown specular highlight is *neutral*
    (a* ~ b* ~ 0) while skin sits at strongly positive a*/b*, so a highlight is maximally
    FAR from skin chroma. Measuring the distance instead of the magnitude inverts the test,
    which is a mistake this project has already made once.
    """
    return np.asarray(np.hypot(lab[..., 1], lab[..., 2]), dtype=np.float32)


def gray_world_deviation(image: np.ndarray) -> tuple[float, dict[str, float]]:
    """Colour-cast magnitude, for QC **evidence only** (D4).

    Returns the relative spread of the per-channel means about their average, plus the
    normalized illumination vector itself.

    This is the number ``white_balance.max_gray_world_deviation`` gates on. It is load
    bearing precisely because nothing downstream corrects a cast: V1 performs no white
    balance at all, so a capture that clears this check is the only guarantee a redness
    measurement is measuring skin rather than the room.

    Returns:
        ``(deviation, vector)`` where ``vector`` has ``r``/``g``/``b`` keys summing to 3.0.
        Applying ``vector`` to the pixels is forbidden (D4); record it and move on.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) BGR image, got shape {image.shape}")

    means = image.reshape(-1, 3).astype(np.float64).mean(axis=0)  # B, G, R
    overall = float(means.mean())
    if overall <= 1e-6:
        # A black frame has no measurable cast. Exposure QC rejects it; do not also
        # invent a colour-cast failure from a division by nothing.
        return 0.0, {"b": 1.0, "g": 1.0, "r": 1.0}

    vector = means / overall
    deviation = float(np.max(np.abs(vector - 1.0)))
    return deviation, {"b": float(vector[0]), "g": float(vector[1]), "r": float(vector[2])}
