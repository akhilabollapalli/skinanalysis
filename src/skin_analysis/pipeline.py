"""Pipeline orchestration -- the only public entry point.

Order (architecture doc §3):

    capture QC  ->  landmarks + ROIs  ->  skin mask  ->  three image copies
                ->  active concern modules  ->  per-ROI severity  ->  aggregation
                ->  internal result  -> (only if calibrated) public payload

Two properties this module exists to guarantee:

* **Fails closed at QC.** A failed capture returns RETAKE and runs no concern logic.
  Disabled concerns report Severity.DISABLED without touching pixels.

* **Measurement and publication are separate (D2).** ``analyze_scan_internal`` always
  works. ``ScanResultInternal.to_public()`` refuses until calibration is complete. There
  is no code path that produces a user-facing severity from placeholder thresholds.

D10: this runs server-side, budget P50 < 1.5 s / P95 < 3.0 s after upload. The shared
intermediates below (one Lab conversion, one gradient pyramid) exist for that budget and
are PASSED IN to feature modules -- never cached in module globals, which would break the
purity contract in CLAUDE.md §4. Keep the QC and landmark stages free of server-only
assumptions so they can migrate on-device later without changing any concern API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schemas import ImageCopy, PublicScanResult, ScanResultInternal


@dataclass(frozen=True)
class ImageCopies:
    """The three preprocessing derivatives (D5).

    They deliberately do not share preprocessing:

    * ``color``   sRGB -> Lab/D65. No gray-world, no illuminant estimation, no CLAHE (D4).
                  Illumination is handled by each feature's within-image local baseline.
                  The term "color-calibrated" is retired: without a chart or RAW reference
                  it overclaims what this is.
    * ``texture`` Luminance with fixed normalization only. NO CLAHE. GLCM downstream of
                  adaptive gain would partly measure the gain, not the skin (D5).
    * ``ridge``   Optional validated enhancement for Gabor/Hessian. If enhancement is ever
                  adopted, its parameters join the frozen capture protocol.
    """

    color: np.ndarray
    texture: np.ndarray
    ridge: np.ndarray

    def get(self, which: ImageCopy) -> np.ndarray:
        return {
            ImageCopy.COLOR: self.color,
            ImageCopy.TEXTURE: self.texture,
            ImageCopy.RIDGE: self.ridge,
        }[which]


def analyze_scan_internal(image: np.ndarray) -> ScanResultInternal:
    """Run the full pipeline on one BGR image and return the internal result.

    Always available, including before calibration -- this is the entry point validation
    and threshold-fitting work uses.

    Returns:
        ScanResultInternal. Call ``.to_internal_payload()`` for validation logging, or
        ``.to_public()`` to attempt publication (which raises until calibrated, per D2).
    """
    raise NotImplementedError("pipeline.analyze_scan_internal is not implemented yet.")


def analyze_scan(image: np.ndarray) -> PublicScanResult:
    """Run the pipeline and return the user-facing payload.

    Raises:
        CalibrationRequiredError: while the active protocol is uncalibrated (D2).
    """
    return analyze_scan_internal(image).to_public()


def make_image_copies(image: np.ndarray, config: dict) -> ImageCopies:
    """Build the three preprocessing derivatives from one decoded image.

    The branches must NOT share preprocessing. Contrast enhancement helps structural
    detection but distorts the luminance and chroma relationships that redness and
    pigmentation depend on (architecture doc §12), and turns GLCM into a measurement of
    the enhancement itself (D5).
    """
    raise NotImplementedError("pipeline.make_image_copies is not implemented yet.")
