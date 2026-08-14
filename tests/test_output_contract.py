"""Contract tests for the two boundaries that protect the user.

* CLAUDE.md Rule 3 -- the user never sees numbers.
* D2 -- the user sees nothing at all until calibration is complete.

If someone adds a numeric field to the public payload, or finds a way to publish against
placeholder thresholds, these tests are what catch it.
"""

from __future__ import annotations

import pytest

from skin_analysis.rules.adapter import to_recommendations
from skin_analysis.schemas import (
    ROI,
    CalibrationRequiredError,
    CalibrationState,
    CaptureQC,
    Concern,
    FeatureResultInternal,
    PublicScanResult,
    RecommendationEngine,
    RecommendationResult,
    ROIResult,
    ScanResultInternal,
    Severity,
)

FORBIDDEN_KEYS = {
    "raw", "confidence_internal", "confidence", "z", "z_score", "z_ref", "score",
    "probability", "area_fraction", "count", "density", "metrics", "notes",
    "roi_results", "unmeasurable_regions", "illumination_vector",
}


def _sample() -> ScanResultInternal:
    return ScanResultInternal(
        qc=CaptureQC(
            passed=True,
            failures=[],
            metrics={"blur": 300.0},
            illumination_vector={"r_gain": 1.04, "b_gain": 0.97},
        ),
        concerns={
            Concern.REDNESS: FeatureResultInternal(
                concern=Concern.REDNESS,
                severity=Severity.MODERATE,
                regions=[ROI.LEFT_CHEEK, ROI.RIGHT_CHEEK],
                roi_results=[
                    ROIResult(
                        roi=ROI.LEFT_CHEEK,
                        severity=Severity.MODERATE,
                        raw={"affected_area_ratio": 0.11},
                        z_ref={"affected_area_ratio": 1.9},
                    )
                ],
                unmeasurable_regions=[ROI.FOREHEAD],
                raw={"affected_area_ratio": 0.11, "p90_delta_a": 5.2},
                confidence_internal=0.91,
                notes=["internal note"],
            ),
            Concern.ACNE: FeatureResultInternal(
                concern=Concern.ACNE, severity=Severity.DISABLED,
            ),
        },
    )


@pytest.fixture
def published(monkeypatch: pytest.MonkeyPatch):
    """A public payload, with the D2 gate stubbed out.

    Rule 3 and D2 are separate properties. This fixture isolates Rule 3 so it stays
    testable while the project is uncalibrated; D2 is tested on its own below, against
    the real gate.
    """
    from skin_analysis.util import calibration

    monkeypatch.setattr(calibration, "assert_public_ready", lambda **kwargs: None)
    return _sample().to_public()


# ------------------------------------------------------------------ D2: calibration gate


def test_uncalibrated_pipeline_cannot_create_public_payload() -> None:
    """The headline D2 invariant. Placeholder cutoffs must never reach a user."""
    with pytest.raises(CalibrationRequiredError):
        _sample().to_public()


def test_internal_payload_is_always_available() -> None:
    """Measurement is always possible; only publication is gated."""
    internal = _sample().to_internal_payload()
    assert internal["calibration_state"] == CalibrationState.UNCALIBRATED.value
    assert internal["concerns"]["redness"]["raw"]["p90_delta_a"] == 5.2


def test_severity_has_no_uncalibrated_member() -> None:
    """D2: no such value exists, so no UI can ever render one."""
    assert "uncalibrated" not in {s.value for s in Severity}


# ------------------------------------------------------------------ Rule 3: no numbers


def test_public_concern_has_exactly_three_fields(published) -> None:
    redness = published.by_concern(Concern.REDNESS)
    assert set(vars(redness)) == {"concern", "severity", "regions"}


def test_serialized_payload_exposes_only_status_and_regions(published) -> None:
    assert set(published.as_dict()["concerns"]["redness"].keys()) == {"status", "regions"}


def test_no_forbidden_key_anywhere_in_public_payload(published) -> None:
    def walk(node) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                assert key not in FORBIDDEN_KEYS, f"internal field {key!r} leaked to the user"
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(published.as_dict())


def test_no_numeric_values_reach_the_user(published) -> None:
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

    walk(published.concerns)


def test_unmeasurable_regions_stay_internal(published) -> None:
    """D7 keeps the list for validation; the user is not told which ROIs were skipped."""
    assert not hasattr(published.by_concern(Concern.REDNESS), "unmeasurable_regions")
    assert _sample().concerns[Concern.REDNESS].unmeasurable_regions == [ROI.FOREHEAD]


def test_disabled_concern_reports_disabled(published) -> None:
    assert published.by_concern(Concern.ACNE).severity is Severity.DISABLED


def test_hydration_is_not_a_concern() -> None:
    """Hydration is deliberately out of scope and must stay out."""
    assert "hydration" not in {c.value for c in Concern}


# ------------------------------------------------------------------ D9: sanitized DTO


def test_internal_result_cannot_leak_to_recommendation_adapter(published) -> None:
    """D9: the boundary is structural, not conventional.

    ``to_recommendations`` is annotated for PublicScanResult and there is no overload
    taking ScanResultInternal, so no code path hands a raw metric to the rules layer.
    """
    import inspect

    from skin_analysis.rules import adapter

    signature = inspect.signature(adapter.to_recommendations)
    assert signature.parameters["result"].annotation == "PublicScanResult"
    # The internal type is not even importable here: nothing in the module namespace can
    # be used to construct or accept one.
    assert not hasattr(adapter, "ScanResultInternal")
    assert not hasattr(adapter, "FeatureResultInternal")
    assert not hasattr(adapter, "ROIResult")


def test_engine_receives_only_concern_severity_regions(published) -> None:
    captured: list[PublicScanResult] = []

    class FakeEngine:
        def recommend(self, result: PublicScanResult) -> RecommendationResult:
            captured.append(result)
            return RecommendationResult(recommendations=())

    engine = FakeEngine()
    assert isinstance(engine, RecommendationEngine)
    to_recommendations(published, engine)

    for concern in captured[0].concerns:
        assert set(vars(concern)) == {"concern", "severity", "regions"}
