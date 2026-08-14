"""The boundary between the vision package and the recommendation engine.

The vision layer contains ZERO product-recommendation logic. The engine is downstream and
separate, and this repository knows nothing about how it is implemented -- local rules, a
REST service, an existing backend, a database-driven ruleset. Only the contract is frozen
(D9); the destination is deliberately still open.

    RAW INTERNAL RESULT
            x   never crosses this line
            |
            v
    PublicScanResult          concern + severity + regions
            |
            v
    RecommendationEngine

The enforcement is structural, not conventional. ``recommend`` accepts
:class:`PublicScanResult` and there is no overload taking ``ScanResultInternal``, so there
is no code path that hands a raw metric to the recommendation layer.
"""

from __future__ import annotations

from ..schemas import PublicScanResult, RecommendationEngine, RecommendationResult


def to_recommendations(
    result: PublicScanResult,
    engine: RecommendationEngine,
) -> RecommendationResult:
    """Hand a sanitized scan result to a recommendation engine.

    Args:
        result: the public payload. Carries concern, ordinal severity, and regions --
            nothing else exists on the type.
        engine: any object satisfying the :class:`RecommendationEngine` protocol.

    Returns:
        Whatever the engine produced, unmodified. This adapter adds no product logic of
        its own; if it ever needs to, that logic belongs on the engine side of the line.
    """
    return engine.recommend(result)
