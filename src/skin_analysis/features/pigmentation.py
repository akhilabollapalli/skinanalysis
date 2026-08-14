"""Pigmentation — ACTIVE (active).

CIELAB L* local deficit with shadow rejection, on the standardized color-analysis copy.

Status: ACTIVE

Limitations and upgrade path:
    Shadow and pigment both darken pixels; V1 separates them by boundary gradient
    profile, which is imperfect. Upgrade path: learned melanin decomposition.
"""

from __future__ import annotations

import numpy as np

from ..schemas import Concern, FeatureContext, FeatureResultInternal, ImageCopy

CONCERN = Concern.DARK_SPOTS
IMAGE_COPY = ImageCopy.COLOR


def analyze(
    image: np.ndarray,
    skin_mask: np.ndarray,
    rois: dict[str, np.ndarray],
    config: dict,
    *,
    context: FeatureContext,
) -> FeatureResultInternal:
    """Analyze pigmentation for one capture.

    Args:
        image: BGR uint8, native resolution. Use the standardized color-analysis copy
            (sRGB -> Lab/D65, no gray-world, no CLAHE -- D4/D5).
        skin_mask: bool array, True where pixels are analyzable skin.
        rois: ROI name -> bool mask, already intersected with ``skin_mask``.
        config: the ``pigmentation`` block of config/severity_thresholds.yaml.
        context: per-capture facts -- scale anchor for resolving fraction-of-IOD
            parameters (D1), and the QC verdicts this concern must respect.

    Returns:
        FeatureResultInternal whose ``raw`` metrics stay internal; only ``severity`` and
        ``regions`` reach the user.

    Notes:
        Pure function: no I/O, no globals, no mutation of inputs. Determinism is a
        requirement, not a convenience -- repeatability is this product's core metric.
        Regions that are shadowed, clipped, occluded, or too low-resolution must return
        Severity.UNMEASURABLE rather than a plausible-looking number.

        Normalization (D1): this module computes z_local ONLY -- the within-image score
        that asks "does this area differ from surrounding skin?". It is not a severity
        signal, and a uniformly affected face has weak local contrast by construction.
        Population standardization is decision/standardize.py; calling its robust_z()
        from here is a bug.

        Spatial parameters are fractions of ``context.anchor_px``, never fixed pixel
        counts: a fixed window does not mean the same thing at two resolutions, so cohort
        statistics computed with one would not transfer between devices.
    """
    raise NotImplementedError(
        "pigmentation.analyze is not implemented yet. See .claude/skills/add-feature-module."
    )
