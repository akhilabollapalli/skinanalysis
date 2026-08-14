"""Ordinal decision -- standardized features to a severity band.

    Severity(z) = NOT_DETECTED if z < t0
                  MILD        if t0 <= z < t1
                  MODERATE    if t1 <= z < t2
                  HIGH        if z >= t2

The product exposes no numbers (CLAUDE.md Rule 3). This is the layer where an internal
measurement becomes the only concern-level value a user may see.

Thresholds must be learned or adjudicated from COMMERCIALLY USABLE annotations and
validated on a subject-level holdout. While ``meta.calibrated`` is false these are
placeholders and nothing may be published (D2).
"""

from __future__ import annotations

from ..schemas import ROI, Concern, Severity


def to_severity(z: float, thresholds: dict) -> Severity:
    """Map one standardized value onto the ordinal scale.

    Must be monotonic: severity may never decrease as the underlying measurement rises.
    That property is what makes the bands meaningful, and it is tested directly rather
    than assumed.
    """
    raise NotImplementedError("decision.calibrator.to_severity is not implemented yet.")


def roi_severity(
    concern: Concern,
    roi: ROI,
    z_ref: dict[str, float],
    config: dict,
) -> Severity:
    """Collapse one ROI's standardized measurements into that ROI's severity band.

    Args:
        concern: the concern being decided.
        roi: the region these measurements came from.
        z_ref: measurement name -> z_ref, from ``decision.standardize``.
        config: the concern's merged config block.

    Returns:
        The ROI-level severity. Concern-level aggregation across ROIs is a separate
        policy decision and lives in ``decision.severity`` (D6).
    """
    raise NotImplementedError("decision.calibrator.roi_severity is not implemented yet.")
