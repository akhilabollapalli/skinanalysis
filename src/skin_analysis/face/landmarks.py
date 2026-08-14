"""MediaPipe Face Landmarker wrapper (468 surface + 10 iris points).

MediaPipe is Apache-2.0 and commercially usable -- one of the few face components that
clears this project's licensing gate. Record its version in the asset manifest and pin it;
landmark index semantics must not drift between releases or every calibrated ROI silently
changes meaning.
"""

from __future__ import annotations

import numpy as np


def detect(image: np.ndarray, config: dict) -> np.ndarray | None:
    """Return an (N, 3) array of landmark coordinates, or None if no usable face.

    Coordinates are in pixel units in the original image frame -- do not return
    normalized coordinates, since downstream ROI erosion is specified in pixels.
    """
    raise NotImplementedError("face.landmarks.detect is not implemented yet.")


def head_pose(landmarks: np.ndarray) -> tuple[float, float, float]:
    """Estimate (yaw, pitch, roll) in degrees for the pose gate."""
    raise NotImplementedError("face.landmarks.head_pose is not implemented yet.")
