"""Pores — EXPERIMENTAL, disabled by default.

Candidate morphology extraction on native-resolution ROI tiles.

Status: EXPERIMENTAL. Blocked on: commercial-open native-resolution pore masks

Design notes:
    Never resize the face before pore analysis. Report visible prominence, never a
    physical diameter, unless capture includes a known spatial scale.
"""

from __future__ import annotations

import numpy as np

from ..schemas import Concern, FeatureContext, FeatureResultInternal, ImageCopy, Severity

CONCERN = Concern.PORES
IMAGE_COPY = ImageCopy.TEXTURE


def analyze(
    image: np.ndarray,
    skin_mask: np.ndarray,
    rois: dict[str, np.ndarray],
    config: dict,
    *,
    context: FeatureContext,
) -> FeatureResultInternal:
    """Disabled in V1. Returns Severity.DISABLED without inspecting the image.

    The signature and schema are final so that enabling this concern later requires no
    change to the pipeline or the application contract.

    Blocked on: commercial-open native-resolution pore masks
    """
    if not config.get("enabled", False):
        return FeatureResultInternal(
            concern=CONCERN,
            severity=Severity.DISABLED,
            regions=[],
            raw={},
            confidence_internal=0.0,
            notes=["blocked_on: commercial-open native-resolution pore masks"],
        )
    raise NotImplementedError(
        "pores was enabled in config but has no implementation. Enabling a concern "
        "requires commercially clear labels AND a passed acceptance gate."
    )
