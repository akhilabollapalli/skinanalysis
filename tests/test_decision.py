"""The decision layer: standardization, banding, and the publication gate.

Two properties dominate here:

* **Monotonicity.** Severity may never fall as the underlying measurement rises. That is
  what makes an ordinal band mean anything, and it is tested directly rather than assumed.

* **No silent default.** Every path that lacks calibration must raise or report
  UNMEASURABLE. A default band is the one failure mode nobody would notice, because a
  defaulted result looks exactly like a measured one.
"""

from __future__ import annotations

import numpy as np
import pytest

from skin_analysis.decision import calibrator, decide, severity, standardize
from skin_analysis.schemas import (
    ROI,
    CalibrationRequiredError,
    Concern,
    FeatureResultInternal,
    ROIResult,
    Severity,
)
from skin_analysis.util import config as cfg

THRESHOLDS = {"t0": 0.5, "t1": 1.5, "t2": 2.5}


@pytest.fixture
def redness_config() -> dict:
    return cfg.concern_config("redness")


# ------------------------------------------------------------------ banding


@pytest.mark.parametrize(
    ("z", "expected"),
    [
        (-5.0, Severity.NOT_DETECTED),
        (0.0, Severity.NOT_DETECTED),
        (0.49, Severity.NOT_DETECTED),
        (0.5, Severity.MILD),
        (1.49, Severity.MILD),
        (1.5, Severity.MODERATE),
        (2.49, Severity.MODERATE),
        (2.5, Severity.HIGH),
        (99.0, Severity.HIGH),
    ],
)
def test_band_edges_are_left_closed(z: float, expected: Severity) -> None:
    """A cutoff belongs to the band it opens, so the mapping is total and unambiguous."""
    assert calibrator.to_severity(z, THRESHOLDS) is expected


def test_banding_is_monotonic() -> None:
    """The property the ordinal scale rests on: worse measurement, never a milder band."""
    ranks = [
        calibrator.to_severity(z, THRESHOLDS).rank
        for z in np.linspace(-6.0, 6.0, 400)
    ]
    assert ranks == sorted(ranks)


def test_out_of_order_cutoffs_are_rejected() -> None:
    """Non-monotonic cutoffs would let a face measure worse and be reported milder, and
    nothing downstream would reveal it."""
    with pytest.raises(ValueError, match="strictly increasing"):
        calibrator.to_severity(1.0, {"t0": 2.0, "t1": 1.0, "t2": 3.0})


def test_banding_returns_no_non_ordinal_state() -> None:
    """UNMEASURABLE and DISABLED are states, not findings; banding must never produce one."""
    for z in np.linspace(-10.0, 10.0, 50):
        assert calibrator.to_severity(float(z), THRESHOLDS) not in (
            Severity.UNMEASURABLE,
            Severity.DISABLED,
        )


# ------------------------------------------------------------------ combining measurements


def test_combine_applies_configured_signs(redness_config: dict) -> None:
    z_ref = {"affected_area_ratio": 1.0, "median_positive_delta_a": 2.0, "p90_delta_a": 0.5}
    assert calibrator.combine(z_ref, redness_config) == pytest.approx(2.0)


def test_texture_declares_its_inverted_measurements() -> None:
    """Homogeneity and energy FALL as roughness rises, so their sign must be negative."""
    directions = cfg.concern_config("texture")["decision"]["direction"]
    assert directions["glcm_homogeneity"] == -1
    assert directions["glcm_energy"] == -1
    assert directions["glcm_contrast"] == 1
    assert directions["glcm_entropy"] == 1


def test_combine_inverts_measurements_that_run_the_other_way() -> None:
    """An inverted metric must contribute a HIGHER combined value as its raw value falls.

    Tested against `combine: max` rather than texture's own `median`, which would mask the
    sign: moving two of seven measurements leaves a median of seven untouched. That is
    correct for a field property, so the sign logic has to be isolated from it.
    """
    config = {
        "decision": {"combine": "max", "direction": {"rises": 1, "falls": -1}},
    }
    assert calibrator.combine({"rises": 0.0, "falls": -3.0}, config) == pytest.approx(3.0)
    assert calibrator.combine({"rises": 0.0, "falls": 3.0}, config) == pytest.approx(0.0)


def test_texture_median_ignores_a_single_noisy_measurement() -> None:
    """The reason texture combines by median: one GLCM feature must not set the band."""
    config = cfg.concern_config("texture")
    quiet = dict.fromkeys(config["raw_measurements"], 0.0)
    one_spike = {**quiet, "glcm_contrast": 9.0}
    assert calibrator.combine(one_spike, config) == pytest.approx(
        calibrator.combine(quiet, config)
    )


def test_combine_refuses_an_undeclared_measurement(redness_config: dict) -> None:
    """Defaulting to +1 is how an inverted metric gets silently averaged in."""
    with pytest.raises(KeyError, match="direction"):
        calibrator.combine({"something_new": 1.0}, redness_config)


def test_texture_combines_by_median_not_max() -> None:
    """Roughness is a field property, so one noisy GLCM feature must not set the band."""
    assert cfg.concern_config("texture")["decision"]["combine"] == "median"


def test_unknown_combine_method_is_rejected(redness_config: dict) -> None:
    config = {**redness_config, "decision": {**redness_config["decision"], "combine": "sum"}}
    with pytest.raises(ValueError, match="combine"):
        calibrator.combine({"affected_area_ratio": 1.0}, config)


# ------------------------------------------------------------------ standardization (D1 B)


def test_standardize_refuses_without_a_reference(redness_config: dict) -> None:
    """Standardizing against absent statistics yields zeros, which read as 'perfectly
    average' rather than 'unknown'."""
    with pytest.raises(CalibrationRequiredError, match="no calibrated population"):
        standardize.standardize_roi(
            Concern.REDNESS, ROI.LEFT_CHEEK, {"affected_area_ratio": 0.2}, redness_config
        )


def test_standardize_uses_median_and_scaled_mad(
    monkeypatch: pytest.MonkeyPatch, redness_config: dict
) -> None:
    monkeypatch.setattr(
        standardize,
        "load_reference",
        lambda *_: {"median": {"affected_area_ratio": 0.10}, "mad": {"affected_area_ratio": 0.05}},
    )
    result = standardize.standardize_roi(
        Concern.REDNESS, ROI.LEFT_CHEEK, {"affected_area_ratio": 0.20}, redness_config
    )
    expected = (0.20 - 0.10) / (standardize.MAD_SCALE * 0.05 + float(redness_config["eps"]))
    assert result["affected_area_ratio"] == pytest.approx(expected)


def test_standardize_refuses_a_measurement_the_cohort_never_recorded(
    monkeypatch: pytest.MonkeyPatch, redness_config: dict
) -> None:
    """Skipping it silently would decide the concern on a subset nobody chose."""
    monkeypatch.setattr(
        standardize,
        "load_reference",
        lambda *_: {"median": {"affected_area_ratio": 0.1}, "mad": {"affected_area_ratio": 0.05}},
    )
    with pytest.raises(CalibrationRequiredError, match="p90_delta_a"):
        standardize.standardize_roi(
            Concern.REDNESS,
            ROI.LEFT_CHEEK,
            {"affected_area_ratio": 0.2, "p90_delta_a": 4.0},
            redness_config,
        )


def test_mad_scale_matches_the_local_stage() -> None:
    """One constant, so a '1 MAD' step means the same thing in both normalization stages."""
    from skin_analysis.features import _common

    assert standardize.MAD_SCALE == _common.MAD_SCALE


def test_shipped_reference_set_is_uncalibrated() -> None:
    """The repository must not ship a reference set that would let placeholders publish."""
    from skin_analysis.util import calibration

    manifest = calibration.reference_manifest()
    assert manifest is not None, "the placeholder reference set should exist"
    assert manifest["frozen"] is False
    for concern in ("redness", "pigmentation", "texture", "wrinkles"):
        stats = calibration.reference_stats(concern)
        assert stats is not None and stats["calibrated"] is False
        assert stats["n_subjects"] == 0


# ------------------------------------------------------------------ decide()


def _measurement(concern: Concern, per_roi: dict[str, dict[str, float]]) -> FeatureResultInternal:
    return FeatureResultInternal(
        concern=concern,
        severity=Severity.UNMEASURABLE,
        roi_results=[
            ROIResult(roi=ROI(name), severity=Severity.UNMEASURABLE, raw=raw)
            for name, raw in per_roi.items()
        ],
    )


def test_decide_preserves_unmeasurable_rois(redness_config: dict) -> None:
    """'No finding' and 'never looked' are different claims (D7)."""
    measurement = FeatureResultInternal(
        concern=Concern.REDNESS,
        severity=Severity.UNMEASURABLE,
        roi_results=[
            ROIResult(
                roi=ROI.CHIN,
                severity=Severity.UNMEASURABLE,
                unmeasurable_reason="occluded by beard",
            )
        ],
    )
    result = decide.decide(measurement, redness_config)
    assert result.severity is Severity.UNMEASURABLE
    assert result.roi_results[0].unmeasurable_reason == "occluded by beard"


def test_decide_bands_measured_rois(
    monkeypatch: pytest.MonkeyPatch, redness_config: dict
) -> None:
    monkeypatch.setattr(
        standardize,
        "load_reference",
        lambda *_: {
            "median": dict.fromkeys(redness_config["raw_measurements"], 0.0),
            "mad": dict.fromkeys(redness_config["raw_measurements"], 1.0),
        },
    )
    raw = {"affected_area_ratio": 6.0, "median_positive_delta_a": 0.0, "p90_delta_a": 0.0}
    measurement = _measurement(
        Concern.REDNESS, {"left_cheek": dict(raw), "right_cheek": dict(raw)}
    )
    result = decide.decide(measurement, redness_config)

    assert result.severity is Severity.HIGH, "two ROIs at max satisfies min_rois_at_max"
    assert set(result.regions) == {ROI.LEFT_CHEEK, ROI.RIGHT_CHEEK}
    assert all(r.z_ref for r in result.roi_results)


def test_decide_demotes_an_unsupported_single_roi(
    monkeypatch: pytest.MonkeyPatch, redness_config: dict
) -> None:
    """D6 max-with-support: one shadowed patch must not drive the whole result."""
    monkeypatch.setattr(
        standardize,
        "load_reference",
        lambda *_: {
            "median": dict.fromkeys(redness_config["raw_measurements"], 0.0),
            "mad": dict.fromkeys(redness_config["raw_measurements"], 1.0),
        },
    )
    measurement = _measurement(
        Concern.REDNESS,
        {
            # Well above t2, but the strong-support thresholds (area 0.18, p90 6.0) are not met.
            "left_cheek": {
                "affected_area_ratio": 6.0,
                "median_positive_delta_a": 0.0,
                "p90_delta_a": 0.0,
            },
            "right_cheek": {
                "affected_area_ratio": 0.0,
                "median_positive_delta_a": 0.0,
                "p90_delta_a": 0.0,
            },
        },
    )
    result = decide.decide(measurement, redness_config)
    assert result.severity is Severity.MODERATE, "HIGH demoted one band for lack of support"
    assert any("demoted" in note for note in result.notes)


def test_decidable_is_false_while_uncalibrated() -> None:
    """Lets the pipeline refuse once and clearly, instead of raising mid-scan."""
    assert not decide.decidable(Concern.REDNESS, [ROI.LEFT_CHEEK, ROI.FOREHEAD])


def test_decide_raises_rather_than_defaulting(redness_config: dict) -> None:
    """The one thing that must never happen: a band invented where calibration is missing."""
    measurement = _measurement(
        Concern.REDNESS,
        {
            "left_cheek": {
                "affected_area_ratio": 0.2,
                "median_positive_delta_a": 2.0,
                "p90_delta_a": 4.0,
            }
        },
    )
    with pytest.raises(CalibrationRequiredError):
        decide.decide(measurement, redness_config)


# ------------------------------------------------------------------ aggregation policy


def test_texture_has_no_single_roi_escape() -> None:
    """One rough ROI is a mask defect until proven otherwise (D6)."""
    assert cfg.concern_config("texture")["aggregation"]["strong_single_roi"] is None


def test_uncalibrated_strong_support_thresholds_block_the_escape() -> None:
    """A null threshold is unset, and unset must not widen the escape hatch."""
    assert not severity._meets_strong_support(
        {"lesion_burden": 999.0}, {"min_lesion_burden": None}
    )


def test_demote_leaves_non_ordinal_states_alone() -> None:
    """Demoting UNMEASURABLE would invent a finding out of an absence of one."""
    assert severity.demote(Severity.UNMEASURABLE) is Severity.UNMEASURABLE
    assert severity.demote(Severity.DISABLED) is Severity.DISABLED
    assert severity.demote(Severity.NOT_DETECTED) is Severity.NOT_DETECTED
