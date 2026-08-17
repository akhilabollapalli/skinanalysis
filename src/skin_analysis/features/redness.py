"""Redness — ACTIVE (active).

CIELAB a* local excess on the standardized color-analysis copy.

Status: ACTIVE

Limitations and upgrade path:
    Without hemoglobin decomposition this partially measures illumination and skin
    tone, not only erythema. Upgrade path: learned melanin/hemoglobin decomposition
    once training data is commercially cleared.

MEASUREMENT ONLY. ``analyze`` fills ``ROIResult.raw`` and leaves every severity at
UNMEASURABLE; population standardization and banding happen in ``decision/`` because they
need frozen cohort statistics from disk, and a feature module may not perform I/O
(CLAUDE.md §4). The consequence is deliberate: a result that never reaches the decision
layer stays UNMEASURABLE -- "could not assess", which is the fail-closed answer -- rather
than defaulting to NOT_DETECTED, which would be a claim the measurement never made.
"""

from __future__ import annotations

import numpy as np

from ..schemas import (
    ROI,
    Concern,
    FeatureContext,
    FeatureResultInternal,
    ImageCopy,
    ROIResult,
    Severity,
)
from . import _common

CONCERN = Concern.REDNESS
IMAGE_COPY = ImageCopy.COLOR

#: Index of a* in the CIELAB planes produced by ``util.color.bgr_to_lab``.
_A_STAR = 1


def analyze(
    image: np.ndarray,
    skin_mask: np.ndarray,
    rois: dict[str, np.ndarray],
    config: dict,
    *,
    context: FeatureContext,
) -> FeatureResultInternal:
    """Analyze redness for one capture.

    Args:
        image: the standardized color-analysis copy -- CIELAB float32 with L* in 0..100
            and a*/b* centred on zero, as produced by ``pipeline.make_image_copies``
            (sRGB -> Lab/D65, no gray-world, no CLAHE -- D4/D5).
        skin_mask: bool array, True where pixels are analyzable skin.
        rois: ROI name -> bool mask, already intersected with ``skin_mask``.
        config: the ``redness`` block of config/severity_thresholds.yaml.
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
    if image.shape[:2] != skin_mask.shape:
        raise ValueError(
            f"image {image.shape[:2]} and skin mask {skin_mask.shape} disagree on frame size"
        )

    primary = [str(name) for name in config["primary_rois"]]

    # A colour cast is uncorrected by design (D4), so an a* excess measured under one is a
    # property of the room. Refuse the concern rather than reporting the lighting as skin.
    if not context.qc.color_cast_pass:
        return _common.all_unmeasurable(
            CONCERN, primary, "capture failed the colour-cast check; a* is not skin here (D4)"
        )

    a_star = image[..., _A_STAR].astype(np.float64)
    eps = float(config["eps"])
    metric = config["metric"]
    local_cfg = config["local"]

    z_cut = float(metric["z_local_threshold"])
    percentile = float(metric["percentile_for_intensity"])
    min_component_px = float(metric["min_component_area_frac_of_iod2"]) * context.anchor_px**2

    usable, reasons = _common.measurable_rois(
        rois, primary, _common.min_support_px(config, context.anchor_px)
    )
    roi_results: list[ROIResult] = [
        _common.unmeasurable(ROI(name), why) for name, why in reasons.items()
    ]

    for name in usable:
        roi_mask = rois[name]
        score = _common.local_score(a_star, roi_mask, local_cfg, context.anchor_px, eps)

        # Redness is an EXCESS of a*: only the positive tail is erythema. The negative tail
        # is skin less red than its own neighbourhood, which is not a finding for this
        # concern and must not be folded in as magnitude.
        candidate = roi_mask & (score.z >= z_cut)
        affected, _components = _common.filter_components(candidate, min_component_px)

        roi_px = float(roi_mask.sum())
        positive = score.delta[roi_mask & (score.delta > 0.0)]

        roi_results.append(
            ROIResult(
                roi=ROI(name),
                severity=Severity.UNMEASURABLE,  # decided in decision/, not here
                raw={
                    "affected_area_ratio": float(affected.sum()) / roi_px,
                    "median_positive_delta_a": (
                        float(np.median(positive)) if positive.size else 0.0
                    ),
                    # Named p90 and configured at 90 -- see percentile_for_intensity.
                    "p90_delta_a": _common.signed_percentile(score.delta[roi_mask], percentile),
                },
            )
        )

    return _common.measurement_result(CONCERN, roi_results)


def asymmetry(
    roi_raw: dict[str, dict[str, float]],
    config: dict,
    *,
    context: FeatureContext,
) -> dict[str, float]:
    """Left/right redness difference, or an empty mapping when it may not be reported.

    Reported only when capture QC cleared the shadow-asymmetry check. Side lighting is the
    single most common cause of a false asymmetry finding, and V1 corrects neither shadow
    nor colour cast (D4) -- without that clearance the difference is the room, not the skin.

    Nasolabial never participates (D8): the fold's cast shadow and genuine perifacial
    erythema are not separable at V1 fidelity.

    Internal only. Asymmetry is a validation signal in V1 and has no public representation.
    """
    spec = config.get("asymmetry", {}) or {}
    if not spec.get("enabled", False):
        return {}
    if spec.get("require_qc_shadow_pass", True) and not context.qc.shadow_pass:
        return {}

    excluded = set(spec.get("excluded_rois", []))
    out: dict[str, float] = {}
    for left, right in _common.symmetry_pairs(roi_raw):
        if left in excluded or right in excluded:
            continue
        out[f"{left}_vs_{right}_delta_a"] = (
            roi_raw[left]["median_positive_delta_a"] - roi_raw[right]["median_positive_delta_a"]
        )
    return out
