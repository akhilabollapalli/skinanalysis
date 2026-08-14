"""Redness — ACTIVE (active).

CIELAB a* local excess on the color-calibrated copy.

Status: ACTIVE

Limitations and upgrade path:
    Without hemoglobin decomposition this partially measures illumination and skin
    tone, not only erythema. Upgrade path: learned melanin/hemoglobin decomposition
    once training data is commercially cleared.
"""

from __future__ import annotations

import numpy as np

from ..schemas import ROI, Concern, FeatureResult, Severity

CONCERN = Concern.REDNESS


def analyze(
    image: np.ndarray,
    skin_mask: np.ndarray,
    rois: dict[str, np.ndarray],
    config: dict,
) -> FeatureResult:
    """Analyze redness for one capture.

    Args:
        image: BGR uint8, native resolution. Use the color-calibrated copy; never a contrast-enhanced one.
        skin_mask: bool array, True where pixels are analyzable skin.
        rois: ROI name -> bool mask, already intersected with ``skin_mask``.
        config: the ``redness`` block of config/severity_thresholds.yaml.

    Returns:
        FeatureResult whose ``raw`` metrics stay internal; only ``severity`` and
        ``regions`` reach the user.

    Notes:
        Pure function: no I/O, no globals, no mutation of inputs. Determinism is a
        requirement, not a convenience -- repeatability is this product's core metric.
        Regions that are shadowed, clipped, occluded, or too low-resolution must return
        Severity.UNMEASURABLE rather than a plausible-looking number.
    """
    raise NotImplementedError(
        "redness.analyze is not implemented yet. See .claude/skills/add-feature-module."
    )
