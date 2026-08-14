"""Fine Lines — EXPERIMENTAL, disabled by default.

Derived from the wrinkle line map: width, contrast, length, continuity.

Status: EXPERIMENTAL. Blocked on: validated high-resolution commercial labels

Design notes:
    One physical line must be detected once, then characterised -- never counted twice
    by a separate detector. In RGB this yields an appearance class, not a depth
    measurement.
"""

from __future__ import annotations

import numpy as np

from ..schemas import Concern, FeatureContext, FeatureResultInternal, ImageCopy, Severity

CONCERN = Concern.FINE_LINES
IMAGE_COPY = ImageCopy.RIDGE


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

    Blocked on: validated high-resolution commercial labels
    """
    if not config.get("enabled", False):
        return FeatureResultInternal(
            concern=CONCERN,
            severity=Severity.DISABLED,
            regions=[],
            raw={},
            confidence_internal=0.0,
            notes=["blocked_on: validated high-resolution commercial labels"],
        )
    raise NotImplementedError(
        "fine_lines was enabled in config but has no implementation. Enabling a concern "
        "requires commercially clear labels AND a passed acceptance gate."
    )
