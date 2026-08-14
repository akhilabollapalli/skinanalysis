"""Capture-scale resolution (D1).

Fixed pixel windows are retired. ``window_px: 129`` does not mean the same thing at two
resolutions, so a cohort statistic computed with one would not transfer between devices --
and transferable cohort statistics are exactly what the severity layer needs.

Every spatial parameter in ``config/severity_thresholds.yaml`` is therefore a fraction of
an anchor (inter-ocular distance) and is resolved to pixels here, at runtime, per capture.

Pure functions only: no I/O, no globals.
"""

from __future__ import annotations

import numpy as np

#: MediaPipe FaceMesh V2 iris centres. Used as the scale anchor because inter-ocular
#: distance is the most pose-stable facial measurement available from the landmark set.
#:
#: NAMING: MediaPipe's LEFT/RIGHT are ANATOMICAL -- the subject's own left and right. The
#: subject's left eye appears on the RIGHT side of the image. This project follows the same
#: convention everywhere (ROI.LEFT_CHEEK is the subject's left cheek), because a cosmetic
#: report says "your left cheek", not "the left of the photo". Getting this backwards
#: mirrors every paired ROI and every asymmetry finding without changing any test that
#: only checks magnitudes.
RIGHT_IRIS_CENTER = 468   #: subject's right eye; ring 469-472
LEFT_IRIS_CENTER = 473    #: subject's left eye;  ring 474-477


def inter_ocular_distance(landmarks: np.ndarray) -> float:
    """Anchor length in pixels: distance between iris centres.

    Args:
        landmarks: (N, 3) pixel-space landmarks. Requires the 478-point model -- the
            468-point surface mesh alone has no iris landmarks.

    Returns:
        Distance in pixels.
    """
    if landmarks.shape[0] <= RIGHT_IRIS_CENTER:
        raise ValueError(
            f"scale anchor needs the 478-point landmark model; got {landmarks.shape[0]} points"
        )
    left = landmarks[LEFT_IRIS_CENTER, :2]
    right = landmarks[RIGHT_IRIS_CENTER, :2]
    return float(np.linalg.norm(left - right))


def to_px(frac: float, anchor_px: float, *, odd: bool = False, minimum: int = 1) -> int:
    """Resolve a fraction-of-anchor parameter to pixels.

    Args:
        frac: the configured fraction, e.g. ``window_frac_of_iod``.
        anchor_px: inter-ocular distance for this capture.
        odd: force an odd result. Required for symmetric filter/median windows, where an
            even size has no centre pixel and biases the response by half a pixel.
        minimum: floor for the result.

    Returns:
        Pixel size, deterministic for a given (frac, anchor_px).
    """
    value = max(minimum, int(round(frac * anchor_px)))
    if odd and value % 2 == 0:
        value += 1
    return value


def anchor_is_sufficient(anchor_px: float, config: dict) -> bool:
    """False when the capture is too small to support local windows at all.

    Below the floor, a local baseline window would either exceed the ROI or contain too
    few pixels for a robust median/MAD. Either way the honest answer is UNMEASURABLE, not
    a plausible-looking number.
    """
    return anchor_px >= float(config.get("scale", {}).get("min_anchor_px", 0))
