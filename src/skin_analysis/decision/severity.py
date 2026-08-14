"""Raw metric -> ordinal severity.

The product exposes no numbers (CLAUDE.md Rule 3). This module is where internal
measurements become the only concern-level value a user may see:
not_detected / mild / moderate / high.

Thresholds must be learned or adjudicated from COMMERCIALLY USABLE annotations and
validated on a subject-level holdout. Until config meta.calibrated is true, severity is
not fit to display.
"""

from __future__ import annotations

import numpy as np

from ..schemas import Severity


def robust_z(value: float, ref_median: float, ref_mad: float, eps: float) -> float:
    """z = (m - median) / (MAD + eps). Robust to the outliers skin metrics produce."""
    return (value - ref_median) / (ref_mad + eps)


def to_severity(z: float, thresholds: dict) -> Severity:
    """Map a robust z-score onto the ordinal scale.

    Must be monotonic: severity may never decrease as the underlying measurement rises.
    """
    raise NotImplementedError("decision.severity.to_severity is not implemented yet.")


def assert_calibrated() -> None:
    """Raise if severity thresholds have not been calibrated.

    Guards against shipping placeholder cutoffs as if they were validated findings.
    """
    raise NotImplementedError("decision.severity.assert_calibrated is not implemented yet.")
