"""Helpers shared by the concern modules.

Everything here is pure: no I/O, no globals, no mutation of inputs. Feature modules are
required to be deterministic, and a shared helper that quietly cached state would break
that property for every concern at once.

**This module implements z_local and nothing else (D1 stage A).** The within-image score

    z_local(x) = (f(x) - median(f(N_x))) / (1.4826 * MAD(f(N_x)) + eps)

answers "does this area differ from surrounding skin?". It is NOT a severity signal: a
uniformly affected face has weak local contrast by construction, and reading z_local as
severity would score such a face as clear. Population standardization is a different
question, computed in ``decision/standardize.py`` against a frozen cohort. A feature module
calling ``standardize.robust_z`` is a bug.

Recurring failure to watch for here: **a scale estimated on one support and applied to a
wider one**. Every skin-mask bug in this project so far had that shape -- a MAD measured on
small reference patches, then used to gate the whole face, where shading varies far more.
z_local repeats the pattern by design, so each caller must state which support its baseline
came from, and it must be the same support the score is applied to.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..schemas import ROI, Concern, FeatureResultInternal, ROIResult, Severity
from ..util import scale

#: Scales a MAD so it estimates a standard deviation for normally distributed data. The
#: same constant is used by ``decision/standardize.py``, so a "1 MAD" step means the same
#: thing in both normalization stages and the two stay comparable while debugging.
MAD_SCALE = 1.4826


class LocalBaselineError(ValueError):
    """Raised when a local baseline cannot be estimated from the support given."""


@dataclass(frozen=True)
class LocalScore:
    """A within-image score plus the statistics it was derived from.

    ``support_px`` is recorded deliberately: it is the number of pixels the median and MAD
    were estimated on, and reviewing it is how the "scale from a narrow support applied to
    a wide one" failure gets caught.
    """

    z: np.ndarray
    delta: np.ndarray
    baseline: float
    mad: float
    support_px: int


def mad(values: np.ndarray) -> float:
    """Median absolute deviation, scaled so it estimates a standard deviation.

    Robust to what skin metrics actually produce: one specular blob or one missed hair
    strand moves a mean and a standard deviation far more than a median and a MAD.
    """
    if values.size == 0:
        return 0.0
    median = float(np.median(values))
    return float(MAD_SCALE * np.median(np.abs(values - median)))


def robust_mean(values: np.ndarray, trim_frac: float = 0.1) -> float:
    """Symmetrically trimmed mean. ``aggregate: robust_mean`` in config.

    Used across patches within an ROI, where a single patch landing on a stray hair or a
    mask edge should not move the ROI's value.
    """
    if values.size == 0:
        return 0.0
    if values.size < 3 or trim_frac <= 0.0:
        return float(np.mean(values))
    ordered = np.sort(values)
    k = int(np.floor(trim_frac * ordered.size))
    kept = ordered[k : ordered.size - k] if ordered.size - 2 * k > 0 else ordered
    return float(np.mean(kept))


def signed_percentile(values: np.ndarray, percentile: float) -> float:
    """Percentile of ``values``, or 0.0 when there is nothing to take it of."""
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, percentile))


def local_score(
    field: np.ndarray,
    roi_mask: np.ndarray,
    local_config: dict,
    anchor_px: float,
    eps: float,
) -> LocalScore:
    """Compute z_local for one ROI (D1 stage A).

    Args:
        field: (H, W) float scalar field, e.g. the a* plane.
        roi_mask: (H, W) bool. Already intersected with the skin mask, so every True pixel
            is analyzable skin.
        local_config: the concern's ``local`` block. ``use_roi_median`` selects the support.
        anchor_px: inter-ocular distance for this capture, for resolving window fractions.
        eps: division guard from config.

    Returns:
        LocalScore. ``z`` and ``delta`` are zero outside ``roi_mask`` -- they are undefined
        there, and zero is the value that cannot be mistaken for a finding.

    Raises:
        LocalBaselineError: when the ROI holds no analyzable pixels. Returning zeros in
            that case would read as "measured, nothing found" rather than "never looked".
    """
    values = field[roi_mask]
    if values.size == 0:
        raise LocalBaselineError("ROI contains no analyzable pixels")

    if not local_config.get("use_roi_median", True):
        # The sliding-window support of D1 stage A. No active concern selects it, and a
        # masked local median is not the same computation as an unmasked one -- doing it
        # approximately would put a silent bias in the only baseline the score has. Left
        # unimplemented on purpose rather than approximated.
        window_px = scale.to_px(
            float(local_config["window_frac_of_iod"]), anchor_px, odd=True, minimum=3
        )
        raise NotImplementedError(
            "local.use_roi_median: false selects a sliding-window baseline "
            f"({window_px}px), which is not implemented. Every active concern uses the "
            "ROI-median support; implement a masked local median before enabling this."
        )

    # SUPPORT: the ROI itself. The baseline is estimated on exactly the pixels the score is
    # then applied to, which is what keeps this free of the narrow-support failure.
    baseline = float(np.median(values))
    spread = mad(values)

    delta = np.zeros_like(field, dtype=np.float64)
    delta[roi_mask] = field[roi_mask] - baseline

    z = np.zeros_like(delta)
    z[roi_mask] = delta[roi_mask] / (spread + eps)

    return LocalScore(
        z=z, delta=delta, baseline=baseline, mad=spread, support_px=int(values.size)
    )


def filter_components(
    binary: np.ndarray,
    min_area_px: float,
    max_area_px: float | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Drop connected components outside the area bounds.

    Bounds arrive in pixels, resolved from ``*_frac_of_iod2`` by the caller (D1): a 40-pixel
    blob is a lesion at one capture scale and sensor noise at another, so a fixed pixel
    bound would mean different things on different devices.

    Returns:
        ``(kept_mask, component_masks)``. The component list is what lesion counts and
        per-component geometry are computed from.
    """
    import cv2

    if not binary.any():
        return np.zeros_like(binary, dtype=bool), []

    count, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    kept = np.zeros_like(binary, dtype=bool)
    components: list[np.ndarray] = []
    for label in range(1, count):
        component = labels == label
        area = float(component.sum())
        if area < min_area_px:
            continue
        if max_area_px is not None and area > max_area_px:
            continue
        kept |= component
        components.append(component)
    return kept, components


def unmeasurable(roi: ROI, reason: str) -> ROIResult:
    """An ROI that could not be assessed.

    Distinct from NOT_DETECTED, which is a real finding. Conflating them turns "we could
    not see your forehead" into "your forehead is clear", which is a claim the capture does
    not support (D7).
    """
    return ROIResult(roi=roi, severity=Severity.UNMEASURABLE, unmeasurable_reason=reason)


def measurement_result(
    concern: Concern,
    roi_results: list[ROIResult],
) -> FeatureResultInternal:
    """Wrap a measurement pass, with every severity still undecided.

    Severity stays UNMEASURABLE here on purpose. A feature module cannot decide a band: the
    decision needs frozen cohort statistics read from disk, and a feature module performs
    no I/O (CLAUDE.md §4). So the pipeline hands this to ``decision/`` for standardization
    and banding.

    The failure mode of skipping that step is therefore UNMEASURABLE -- "could not
    assess" -- and never NOT_DETECTED. A default of "clear" would be a claim the
    measurement never made, and it is the kind of claim nobody notices is wrong.
    """
    return FeatureResultInternal(
        concern=concern,
        severity=Severity.UNMEASURABLE,
        roi_results=roi_results,
        unmeasurable_regions=[
            r.roi for r in roi_results if r.severity is Severity.UNMEASURABLE and not r.raw
        ],
        notes=["measurement pass: severities are assigned by the decision layer"],
    )


def all_unmeasurable(
    concern: Concern,
    primary: list[str],
    reason: str,
) -> FeatureResultInternal:
    """Refuse a whole concern with one stated reason.

    Used when a capture-level fact makes the concern meaningless -- an uncorrected colour
    cast for a chroma measurement, for instance. Refusing the concern outright is better
    than measuring it and hoping the decision layer catches it later.
    """
    results = [unmeasurable(ROI(name), reason) for name in primary]
    return FeatureResultInternal(
        concern=concern,
        severity=Severity.UNMEASURABLE,
        roi_results=results,
        unmeasurable_regions=[r.roi for r in results],
        notes=[reason],
    )


def symmetry_pairs(available: dict[str, dict[str, float]]) -> list[tuple[str, str]]:
    """Left/right ROI pairs present in ``available``, by anatomical name.

    LEFT and RIGHT are the SUBJECT'S, following MediaPipe: the subject's left cheek appears
    on the right of the image. Deriving pairs from the names rather than hardcoding them
    means a new paired ROI is picked up without a second place to edit.
    """
    pairs: list[tuple[str, str]] = []
    for name in sorted(available):
        if not name.startswith("left_"):
            continue
        mirror = "right_" + name[len("left_") :]
        if mirror in available:
            pairs.append((name, mirror))
    return pairs


def measurable_rois(
    rois: dict[str, np.ndarray],
    primary: list[str],
    min_pixels: int,
) -> tuple[list[str], dict[str, str]]:
    """Split a concern's primary ROIs into measurable ones and reasons for the rest.

    Args:
        rois: ROI name -> composed mask (polygon INTERSECT skin mask).
        primary: the concern's ``primary_rois``.
        min_pixels: floor below which a robust median/MAD is not meaningful.

    Returns:
        ``(measurable_names, {name: reason})``.
    """
    usable: list[str] = []
    reasons: dict[str, str] = {}
    for name in primary:
        mask = rois.get(name)
        if mask is None:
            reasons[name] = "roi not produced for this capture"
            continue
        pixels = int(mask.sum())
        if pixels < min_pixels:
            reasons[name] = f"only {pixels} analyzable px, below the {min_pixels} px floor"
            continue
        usable.append(name)
    return usable, reasons


def min_support_px(config: dict, anchor_px: float) -> int:
    """Pixel floor for a robust per-ROI statistic, as a fraction of anchor^2 (D1).

    Raises:
        KeyError: if the concern's config has no floor. There is no code-side default on
            purpose: a threshold that lives in a .py file is not tunable or auditable
            without a code change (CLAUDE.md §4).
    """
    if "min_support_frac_of_iod2" not in config:
        raise KeyError(
            "concern config has no min_support_frac_of_iod2; add it under `defaults` in "
            "config/severity_thresholds.yaml rather than defaulting it in code"
        )
    return max(1, round(float(config["min_support_frac_of_iod2"]) * anchor_px**2))
