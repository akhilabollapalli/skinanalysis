"""Contract tests for CLAUDE.md Rule 3: the user never sees numbers.

If someone adds a numeric field to the public payload, this test is what catches it.
"""

from __future__ import annotations

from skin_analysis.schemas import (
    CaptureQC, Concern, FeatureResult, ROI, ScanResult, Severity,
)

FORBIDDEN_KEYS = {
    "raw", "confidence_internal", "confidence", "z", "z_score", "score",
    "probability", "area_fraction", "count", "density", "metrics", "notes",
}


def _sample() -> ScanResult:
    return ScanResult(
        qc=CaptureQC(passed=True, failures=[], metrics={"blur": 300.0}),
        concerns={
            Concern.REDNESS: FeatureResult(
                concern=Concern.REDNESS,
                severity=Severity.MODERATE,
                regions=[ROI.LEFT_CHEEK, ROI.RIGHT_CHEEK],
                raw={"area_fraction": 0.11, "z": 1.9},
                confidence_internal=0.91,
                notes=["internal note"],
            ),
            Concern.ACNE: FeatureResult(
                concern=Concern.ACNE, severity=Severity.DISABLED,
            ),
        },
    )


def test_public_payload_exposes_only_status_and_regions() -> None:
    public = _sample().to_public()["concerns"]["redness"]
    assert set(public.keys()) == {"status", "regions"}


def test_no_forbidden_key_anywhere_in_public_payload() -> None:
    def walk(node) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                assert key not in FORBIDDEN_KEYS, f"internal field {key!r} leaked to the user"
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(_sample().to_public())


def test_no_numeric_values_reach_the_user() -> None:
    """There is no 0-100 and no 0-5 score in this product."""
    def walk(node) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        else:
            assert not isinstance(node, (int, float)) or isinstance(node, bool), (
                f"numeric value {node!r} reached the public payload"
            )

    walk(_sample().to_public()["concerns"])


def test_disabled_concern_reports_disabled() -> None:
    assert _sample().to_public()["concerns"]["acne"]["status"] == "disabled"


def test_hydration_is_not_a_concern() -> None:
    """Hydration is deliberately out of scope and must stay out."""
    assert "hydration" not in {c.value for c in Concern}
