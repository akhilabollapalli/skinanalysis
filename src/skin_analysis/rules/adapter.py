"""Bridge from vision output to the existing recommendation engine.

The vision layer contains ZERO product-recommendation logic. Recommendations are derived
solely from concern + ordinal severity + affected regions. This boundary keeps the
measurement code honest: it cannot be tuned to produce a commercially convenient
recommendation.
"""

from __future__ import annotations

from ..schemas import Recommendation, ScanResult


def to_recommendations(result: ScanResult) -> list[Recommendation]:
    """Map a scan result to recommendations.

    Consumes only the public payload -- never raw metrics, z-scores or confidence.
    """
    raise NotImplementedError("rules.adapter.to_recommendations is not implemented yet.")
