"""Wrinkles — ACTIVE (active_baseline).

Multi-scale Gabor + Hessian ridge filtering on the structural copy.

Status: ACTIVE_BASELINE

Limitations and upgrade path:
    Ridge filters respond to hair, glasses frames and shadow edges as readily as to
    lines, so suppression is part of the algorithm. Upgrade path: high-resolution
    segmentation trained on first-party or commercially clear labels (FFHQ-Wrinkle is
    CC BY-NC-SA and therefore excluded from production).
"""

from __future__ import annotations

import numpy as np

from ..schemas import ROI, Concern, FeatureResult, Severity

CONCERN = Concern.WRINKLES


def analyze(
    image: np.ndarray,
    skin_mask: np.ndarray,
    rois: dict[str, np.ndarray],
    config: dict,
) -> FeatureResult:
    """Analyze wrinkles for one capture.

    Args:
        image: BGR uint8, native resolution. Use the structural copy.
        skin_mask: bool array, True where pixels are analyzable skin.
        rois: ROI name -> bool mask, already intersected with ``skin_mask``.
        config: the ``wrinkles`` block of config/severity_thresholds.yaml.

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
        "wrinkles.analyze is not implemented yet. See .claude/skills/add-feature-module."
    )
