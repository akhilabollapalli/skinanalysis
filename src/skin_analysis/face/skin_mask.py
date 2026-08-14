"""Skin mask construction.

V1 has NO learned face parser. CelebAMask-HQ is excluded by the licensing gate, which
excludes the BiSeNet weights trained on it, so the mask is built from landmark geometry
plus classical rejection of hair, beard, glasses, specular highlight and deep shadow.

This is the weakest link in the commercial-open V1 and the first place to look when a
feature produces false positives -- see .claude/skills/roi-debug. When a first-party or
commercially clear parser becomes available, replacing this module should be the highest
-value upgrade in the pipeline.
"""

from __future__ import annotations

import numpy as np


def build(image: np.ndarray, landmarks: np.ndarray, config: dict) -> np.ndarray:
    """Return a bool mask, True where pixels are analyzable skin.

    Excludes eyes, brows, lips, nostrils, hair, beard, glasses, specular highlights and
    deep shadow. When a region is ambiguous, exclude it.
    """
    raise NotImplementedError("face.skin_mask.build is not implemented yet.")
