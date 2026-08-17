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

import numpy as np

from ..schemas import ROI, Concern, Severity


def to_severity(z: float, thresholds: dict) -> Severity:
    """Map one standardized value onto the ordinal scale.

    Must be monotonic: severity may never decrease as the underlying measurement rises.
    That property is what makes the bands meaningful, and it is tested directly rather
    than assumed.

    Raises:
        ValueError: if the cutoffs are not strictly increasing. Out-of-order cutoffs
            produce a non-monotonic band function -- a face could measure worse and be
            reported milder -- and nothing downstream would reveal it.
    """
    t0 = float(thresholds["t0"])
    t1 = float(thresholds["t1"])
    t2 = float(thresholds["t2"])
    if not t0 < t1 < t2:
        raise ValueError(
            f"severity cutoffs must be strictly increasing, got t0={t0}, t1={t1}, t2={t2}"
        )

    if z < t0:
        return Severity.NOT_DETECTED
    if z < t1:
        return Severity.MILD
    if z < t2:
        return Severity.MODERATE
    return Severity.HIGH


def combine(z_ref: dict[str, float], config: dict) -> float:
    """Collapse one ROI's several standardized measurements into a single value.

    Signs come from ``decision.direction`` in config. They are not cosmetic: GLCM
    homogeneity and energy FALL as texture gets rougher, so combining them unsigned would
    let a rough ROI partially cancel itself out and report as smooth.

    Args:
        z_ref: measurement name -> z_ref.
        config: the concern's merged config block.

    Raises:
        KeyError: if a measurement has no configured direction. Assuming +1 for an unknown
            measurement is how an inverted metric gets silently averaged in.
    """
    spec = config.get("decision", {}) or {}
    directions = spec.get("direction", {}) or {}
    method = str(spec.get("combine", "max"))

    signed: list[float] = []
    for name, value in sorted(z_ref.items()):
        if name not in directions:
            raise KeyError(
                f"measurement {name!r} has no entry in decision.direction. Defaulting to "
                "+1 would silently average in a metric that runs the other way."
            )
        signed.append(float(directions[name]) * value)

    if not signed:
        raise ValueError("no standardized measurements to combine")

    values = np.asarray(signed, dtype=np.float64)
    if method == "max":
        return float(values.max())
    if method == "median":
        return float(np.median(values))
    if method == "mean":
        return float(values.mean())
    raise ValueError(f"unknown decision.combine method {method!r}; expected max/median/mean")


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
    del concern, roi  # Policy is per concern via `config`; the identifiers are for tracing.
    return to_severity(combine(z_ref, config), config["thresholds"])
