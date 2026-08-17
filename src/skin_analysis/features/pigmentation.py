"""Dark spots / hyperpigmentation — ACTIVE.

CIELAB L* local deficit on the standardized color-analysis copy, with shadow rejection
and connected-component analysis.

Status: ACTIVE

Limitations and upgrade path:
    A local L* deficit does not distinguish melanin from a cast shadow, from a healing
    lesion, or from a mole. Shadow rejection below is a boundary-profile heuristic, not a
    physical model. Upgrade path: melanin/hemoglobin decomposition, plus lesion-level
    labels, once both are commercially cleared.

MEASUREMENT ONLY -- severities are assigned in ``decision/``. See ``features/redness.py``
for why, and for what happens when that step is skipped.
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

CONCERN = Concern.DARK_SPOTS
IMAGE_COPY = ImageCopy.COLOR

#: Index of L* in the CIELAB planes produced by ``util.color.bgr_to_lab``.
_L_STAR = 0


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
        image: the standardized color-analysis copy (CIELAB float32, L* 0..100).
        skin_mask: bool array, True where pixels are analyzable skin.
        rois: ROI name -> bool mask, already intersected with ``skin_mask``.
        config: the ``pigmentation`` block of config/severity_thresholds.yaml.
        context: per-capture facts (D1 scale anchor, QC verdicts).

    Returns:
        FeatureResultInternal with raw measurements per ROI. Nothing numeric here reaches
        a user.

    Notes:
        z_local ONLY (D1 stage A). The population comparison lives in
        ``decision/standardize.py``; calling its ``robust_z`` from here is a bug.

        Component area bounds are fractions of anchor^2, never pixels (D1). The upper
        bound matters as much as the lower one: a large dark region is shading or a beard
        edge, not a spot, and admitting it would let lighting drive the concern.
    """
    if image.shape[:2] != skin_mask.shape:
        raise ValueError(
            f"image {image.shape[:2]} and skin mask {skin_mask.shape} disagree on frame size"
        )

    primary = [str(name) for name in config["primary_rois"]]

    # Uncorrected shadow makes an L* deficit indistinguishable from pigment (D4). This is
    # the concern most exposed to that, because a shadow IS a local L* deficit.
    if not context.qc.shadow_pass:
        return _common.all_unmeasurable(
            CONCERN, primary, "capture failed the shadow check; an L* deficit here is lighting"
        )
    if not context.qc.exposure_pass:
        return _common.all_unmeasurable(
            CONCERN, primary, "capture failed the exposure check; clipped pixels carry no L*"
        )

    lightness = image[..., _L_STAR].astype(np.float64)
    eps = float(config["eps"])
    metric = config["metric"]
    local_cfg = config["local"]

    z_cut = float(metric["z_local_threshold"])
    min_px = float(metric["min_component_area_frac_of_iod2"]) * context.anchor_px**2
    max_px = float(metric["max_component_area_frac_of_iod2"]) * context.anchor_px**2
    reject_shadows = bool(metric.get("shadow_rejection", True))
    gradient_ratio = float(metric["shadow_gradient_ratio"])
    edge_width_px = max(
        1.0, float(metric["shadow_edge_width_frac_of_iod"]) * context.anchor_px
    )

    usable, reasons = _common.measurable_rois(
        rois, primary, _common.min_support_px(config, context.anchor_px)
    )
    roi_results: list[ROIResult] = [
        _common.unmeasurable(ROI(name), why) for name, why in reasons.items()
    ]

    for name in usable:
        roi_mask = rois[name]
        score = _common.local_score(lightness, roi_mask, local_cfg, context.anchor_px, eps)

        # DEFICIT: a spot is DARKER than its neighbourhood, so the negative tail of z is
        # the signal. delta_L is reported as a positive magnitude of darkening, which is
        # what `median_delta_L` and its strong-support threshold assume.
        candidate = roi_mask & (score.z <= -z_cut)
        kept, components = _common.filter_components(candidate, min_px, max_px)

        if reject_shadows and components:
            kept, components = _reject_shadow_edges(
                lightness, components, edge_width_px, gradient_ratio
            )

        roi_px = float(roi_mask.sum())
        darkening = -score.delta[kept] if kept.any() else np.empty(0)

        roi_results.append(
            ROIResult(
                roi=ROI(name),
                severity=Severity.UNMEASURABLE,  # decided in decision/, not here
                raw={
                    "spot_area_ratio": float(kept.sum()) / roi_px,
                    "median_delta_L": float(np.median(darkening)) if darkening.size else 0.0,
                    "spot_count": float(len(components)),
                },
            )
        )

    return _common.measurement_result(CONCERN, roi_results)


def _reject_shadow_edges(
    lightness: np.ndarray,
    components: list[np.ndarray],
    edge_width_px: float,
    min_ratio: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Drop candidates whose boundary ramps like a shadow instead of stepping like a spot.

    A pigmented spot has a comparatively sharp boundary: melanin concentration changes over
    a short distance. A cast shadow ramps, because it is a penumbra. So the discriminant is
    the boundary gradient measured against the component's own interior contrast --
    normalizing by contrast is what keeps this from simply preferring dark components.

    This is a heuristic on an appearance proxy, not a physical model of shading. It is the
    weakest step in this concern and the first place to look when pigmentation
    over-reports on side-lit captures.
    """
    import cv2

    kept_mask = np.zeros_like(lightness, dtype=bool)
    kept: list[np.ndarray] = []
    if not components:
        return kept_mask, kept

    gx = cv2.Sobel(lightness.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lightness.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.hypot(gx, gy)  # L* per pixel

    for component in components:
        boundary = _boundary_ring(component)
        if not boundary.any():
            continue
        interior = lightness[component]
        surround = lightness[boundary]
        contrast = float(np.median(surround) - np.median(interior))
        if contrast <= 0.0:
            # Not actually darker than its own boundary: not a spot by definition.
            continue

        # The reference is the gradient a STEP edge of this same contrast would produce
        # across the expected spot-edge width. Dividing by it makes the discriminant
        # dimensionless and scale-invariant: both the observed and reference gradients are
        # L* per pixel, and the width is a fraction of the anchor (D1).
        #
        # Normalizing by the component's own contrast is what stops this from simply
        # preferring dark components over pale ones.
        reference = contrast / edge_width_px
        observed = float(np.median(gradient[boundary]))
        if observed / reference < min_ratio:
            continue
        kept_mask |= component
        kept.append(component)

    return kept_mask, kept


def _boundary_ring(component: np.ndarray) -> np.ndarray:
    """The one-pixel ring immediately outside a component."""
    import cv2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(component.astype(np.uint8), kernel).astype(bool)
    return np.asarray(dilated & ~component)
