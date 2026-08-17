"""Measurement -> severity. The step that joins the feature layer to the ordinal scale.

A feature module measures and stops. It cannot decide a band, because deciding needs frozen
cohort statistics read from disk and a feature module performs no I/O (CLAUDE.md §4). This
module is where that boundary is crossed, in one place, so there is exactly one answer to
"where did this severity come from":

    feature.analyze()   raw per ROI, every severity UNMEASURABLE
        -> standardize  raw -> z_ref against the frozen cohort (D1 stage B)
        -> calibrator   z_ref -> that ROI's band
        -> severity     per-ROI bands -> one concern severity (D6, D7)

Skipping this step leaves the result UNMEASURABLE, which is "could not assess". That is the
fail-closed outcome and it is deliberate: a default of NOT_DETECTED would tell a user their
skin is clear on the strength of a decision nobody made.
"""

from __future__ import annotations

from ..schemas import ROI, Concern, FeatureResultInternal, ROIResult
from . import calibrator, severity, standardize


def decide(
    measurement: FeatureResultInternal,
    config: dict,
) -> FeatureResultInternal:
    """Standardize, band, and aggregate one concern's measurement pass.

    Args:
        measurement: the output of a feature module's ``analyze``. ROIs it could not
            measure carry ``unmeasurable_reason`` and an empty ``raw``.
        config: the concern's merged config block.

    Returns:
        FeatureResultInternal with real severities. Only ``severity`` and ``regions`` may
        ever reach a user.

    Raises:
        CalibrationRequiredError: if a measured ROI has no calibrated reference statistics.
            Not recoverable here -- the correct response is to run the calibration phase,
            never to fall back to a default band (D2).
    """
    decided: list[ROIResult] = []
    for result in measurement.roi_results:
        if not result.raw:
            # Nothing was measured. Preserve the reason so validation can tell "no finding"
            # apart from "never looked" (D7).
            decided.append(result)
            continue
        decided.append(_decide_roi(measurement.concern, result, config))

    aggregated = severity.aggregate(measurement.concern, decided, config)
    return FeatureResultInternal(
        concern=aggregated.concern,
        severity=aggregated.severity,
        regions=aggregated.regions,
        roi_results=aggregated.roi_results,
        unmeasurable_regions=aggregated.unmeasurable_regions,
        raw=measurement.raw,
        confidence_internal=aggregated.confidence_internal,
        notes=[*measurement.notes, *aggregated.notes],
    )


def _decide_roi(concern: Concern, result: ROIResult, config: dict) -> ROIResult:
    z_ref = standardize.standardize_roi(concern, result.roi, result.raw, config)
    return ROIResult(
        roi=result.roi,
        severity=calibrator.roi_severity(concern, result.roi, z_ref, config),
        raw=result.raw,
        z_ref=z_ref,
    )


def decidable(concern: Concern, rois: list[ROI]) -> bool:
    """Whether every one of ``rois`` has calibrated reference statistics for ``concern``.

    Lets the pipeline check before it measures, so an uncalibrated protocol produces one
    clear refusal rather than an exception raised halfway through a scan.
    """
    return all(standardize.load_reference(concern, roi) is not None for roi in rois)


__all__ = ["decidable", "decide"]
