"""Population reference standardization -- D1 stage B.

    z_ref = (m - Median_ref[ROI]) / (1.4826 * MAD_ref[ROI] + eps)

``ref`` means exactly one thing: **a frozen, commercially cleared calibration cohort
captured under the defined V1 acquisition protocol.**

It does NOT mean "people demographically like this user", and it does not define what
normal or healthy skin is. Its only job is to standardize raw measurements before the
ordinal calibrator sees them.

This is the ONLY module that may compute a population z-score. A feature module calling
:func:`robust_z` is a bug -- feature modules use the within-image ``z_local`` of D1
stage A, which answers a different question ("does this area differ from nearby skin?")
and is not a severity signal.
"""

from __future__ import annotations

from ..schemas import ROI, Concern
from ..util import calibration

#: Consistent with the MAD scaling used for z_local, so a MAD is a sigma estimate in both
#: stages and the two are at least dimensionally comparable during debugging.
MAD_SCALE = 1.4826


def robust_z(value: float, ref_median: float, ref_mad: float, eps: float) -> float:
    """Standardize one raw measurement against frozen cohort statistics.

    Robust to the outliers skin metrics produce: a single specular blob or one missed
    hair strand moves a mean and a standard deviation far more than a median and a MAD.
    """
    return (value - ref_median) / (MAD_SCALE * ref_mad + eps)


def standardize_roi(
    concern: Concern,
    roi: ROI,
    raw: dict[str, float],
    config: dict,
) -> dict[str, float]:
    """Standardize every raw measurement of one (concern, ROI) pair.

    Args:
        concern: the concern being standardized.
        roi: the anatomical region the measurements came from.
        raw: raw measurements as produced by the feature module. Keys must match the
            concern's ``raw_measurements`` list in config.
        config: the ``reference_stats`` block of config/severity_thresholds.yaml.

    Returns:
        Measurement name -> z_ref.

    Raises:
        CalibrationRequiredError: if this (concern, ROI) has no calibrated reference
            entry. Standardizing against absent statistics would silently produce zeros,
            which read as "perfectly average" rather than "unknown".
    """
    raise NotImplementedError("decision.standardize.standardize_roi is not implemented yet.")


def load_reference(concern: Concern, roi: ROI) -> dict[str, dict[str, float]] | None:
    """Frozen cohort statistics for one (concern, ROI), or None if not yet calibrated.

    Returns a mapping with ``median`` and ``mad`` sub-dicts keyed by measurement name.
    """
    stats = calibration.reference_stats(concern.value)
    if stats is None or not stats.get("calibrated", False):
        return None
    median = (stats.get("median", {}) or {}).get(roi.value)
    mad = (stats.get("mad", {}) or {}).get(roi.value)
    if not median or not mad:
        return None
    return {"median": median, "mad": mad}
