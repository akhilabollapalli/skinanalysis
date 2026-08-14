"""Texture — ACTIVE (active).

GLCM, gradient and frequency descriptors on structural patches.

Status: ACTIVE

Limitations and upgrade path:
    Texture is a composite appearance property, not one physical variable. Values are
    comparable within a capture protocol only. Upgrade path: learned texture embedding.
"""

from __future__ import annotations

import numpy as np

from ..schemas import ROI, Concern, FeatureResult, Severity

CONCERN = Concern.TEXTURE


def analyze(
    image: np.ndarray,
    skin_mask: np.ndarray,
    rois: dict[str, np.ndarray],
    config: dict,
) -> FeatureResult:
    """Analyze texture for one capture.

    Args:
        image: BGR uint8, native resolution. Use the structural copy.
        skin_mask: bool array, True where pixels are analyzable skin.
        rois: ROI name -> bool mask, already intersected with ``skin_mask``.
        config: the ``texture`` block of config/severity_thresholds.yaml.

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
        "texture.analyze is not implemented yet. See .claude/skills/add-feature-module."
    )
