"""Pipeline orchestration -- the only public entry point.

Order (architecture doc §3):

    capture QC  ->  landmarks + ROIs  ->  skin mask  ->  two image copies
                ->  active concern modules  ->  ordinal severity  ->  public payload

Fails closed at QC. Disabled concerns report Severity.DISABLED without touching pixels.
"""

from __future__ import annotations

import numpy as np

from .schemas import ScanResult
from .util import config as cfg


def analyze_scan(image: np.ndarray) -> ScanResult:
    """Run the full pipeline on one BGR image.

    Returns:
        ScanResult. Call ``.to_public()`` before anything leaves the backend -- that
        method is the enforcement point for the no-numbers contract.
    """
    raise NotImplementedError("pipeline.analyze_scan is not implemented yet.")


def make_image_copies(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (color_calibrated, structural).

    The two branches must NOT share preprocessing. Contrast enhancement helps structural
    detection but distorts the luminance and chroma relationships that redness and
    pigmentation depend on (architecture doc §12).
    """
    raise NotImplementedError("pipeline.make_image_copies is not implemented yet.")
