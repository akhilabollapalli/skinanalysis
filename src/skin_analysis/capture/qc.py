"""Capture quality gate.

Fails closed: if the capture does not pass, the pipeline returns RETAKE and no concern
logic runs at all. Analyzing a bad capture produces a confident-looking wrong answer,
which is worse for the user than being asked to retake.

This gate matters more here than in a typical pipeline. V1's redness and pigmentation
features are CIELAB proxies, so they partially measure the room's lighting. QC strictness
-- not feature sophistication -- is what makes those measurements trustworthy.

Thresholds: config/capture_thresholds.yaml.
"""

from __future__ import annotations

import numpy as np

from ..schemas import CaptureQC


def check(image: np.ndarray, config: dict) -> CaptureQC:
    """Run every capture check and return a pass/fail with reasons.

    Checks: face presence/count/size, pose, blur, exposure and clipping, specular
    coverage, left/right shadow asymmetry, white balance, ROI visibility, occlusion,
    and a non-blocking filter/makeup warning.

    Returns:
        CaptureQC. ``metrics`` is internal only; the public payload carries just the
        boolean and the reason list.
    """
    raise NotImplementedError("capture.qc.check is not implemented yet.")
