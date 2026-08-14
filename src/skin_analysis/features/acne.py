"""Acne — EXPERIMENTAL, disabled by default.

Lesion instance detection/segmentation on native-resolution crops.

Status: EXPERIMENTAL. Blocked on: commercial-open lesion-level labels

Design notes:
    Whole-face classification underperforms a trivial baseline for acne, so this must
    stay a localisation task. AcneSCU, ACNE04 and ACNE-DET are all excluded from
    production by the licensing gate.
"""

from __future__ import annotations

import numpy as np

from ..schemas import ROI, Concern, FeatureResult, Severity

CONCERN = Concern.ACNE


def analyze(
    image: np.ndarray,
    skin_mask: np.ndarray,
    rois: dict[str, np.ndarray],
    config: dict,
) -> FeatureResult:
    """Disabled in V1. Returns Severity.DISABLED without inspecting the image.

    The signature and schema are final so that enabling this concern later requires no
    change to the pipeline or the application contract.

    Blocked on: commercial-open lesion-level labels
    """
    if not config.get("enabled", False):
        return FeatureResult(
            concern=CONCERN,
            severity=Severity.DISABLED,
            regions=[],
            raw={},
            confidence_internal=0.0,
            notes=["blocked_on: commercial-open lesion-level labels"],
        )
    raise NotImplementedError(
        "acne was enabled in config but has no implementation. Enabling a concern "
        "requires commercially clear labels AND a passed acceptance gate."
    )
