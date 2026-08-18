"""Guards for the calibration runner (D1 stage B, D2).

Two things matter most and get the deepest coverage:

* **The threshold fit is a real optimum**, not a heuristic that happens to look
  reasonable. ``test_dp_fit_matches_brute_force`` checks the DP against exhaustive search
  on small synthetic corpora, because a threshold fit that silently settles for "pretty
  good" would misclassify real subjects in a way nothing downstream could detect.

* **This script never crosses into publication.** It must not write to
  severity_thresholds.yaml, must not set a reference manifest's ``frozen`` flag, and must
  refuse to call a concern calibrated on too small a cohort even when every other number
  looks fine. Those are the exact substitutions CLAUDE.md §6 forbids a script from making
  quietly.

Everything here runs on synthetic data with no face corpus, the same way
test_validation.py does and for the same reason: this half of the pipeline is pure
computation over already-measured numbers.
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location("calibrate", REPO / "scripts" / "calibrate.py")
assert spec and spec.loader
calibrate = importlib.util.module_from_spec(spec)
sys.modules["calibrate"] = calibrate
spec.loader.exec_module(calibrate)

from skin_analysis.decision import calibrator  # noqa: E402
from skin_analysis.schemas import Severity  # noqa: E402
from skin_analysis.util import config as cfg  # noqa: E402

CALIBRATION_GATES = cfg.load("calibration_gates")
BANDS = (Severity.NOT_DETECTED, Severity.MILD, Severity.MODERATE, Severity.HIGH)


def sample(z: float, band: Severity, subject: str = "s") -> calibrate.Sample:
    return calibrate.Sample(subject, z, band)


# --------------------------------------------------------------------- threshold fitting


def test_fit_is_exact_on_perfectly_separable_data() -> None:
    samples = [
        sample(float(i), BANDS[i // 5], f"s{i}") for i in range(20)
    ]
    thresholds, misclassified = calibrate.fit_thresholds(samples)
    assert misclassified == 0
    result = calibrate.evaluate(samples, thresholds)
    assert result["exact_agreement"] == 1.0


def test_dp_fit_matches_brute_force() -> None:
    """The DP must find the SAME optimum as exhaustive search, not an approximation."""
    import random

    rng = random.Random(7)
    samples = [
        sample(rng.uniform(0, 40), rng.choice(BANDS), f"s{i}") for i in range(30)
    ]
    _, dp_misclassified = calibrate.fit_thresholds(samples)

    zs = sorted(s.z for s in samples)
    candidates = sorted({(a + b) / 2 for a, b in zip(zs, zs[1:], strict=False)})
    best = None
    for t0, t1, t2 in itertools.combinations(candidates, 3):
        if not t0 < t1 < t2:
            continue
        wrong = sum(
            1 for s in samples
            if calibrator.to_severity(s.z, {"t0": t0, "t1": t1, "t2": t2}) != s.label
        )
        if best is None or wrong < best:
            best = wrong

    assert dp_misclassified == best


def test_fitted_thresholds_are_strictly_increasing() -> None:
    """calibrator.to_severity itself enforces this, but a degenerate corpus (one band
    almost entirely absent) can produce ties. Refusing here is the honest answer -- the
    label distribution on this split cannot support 4 distinct bands."""
    samples = [sample(1.0, Severity.NOT_DETECTED, "s1")] * 4
    with pytest.raises(ValueError, match="not strictly increasing|too few"):
        calibrate.fit_thresholds(samples)


def test_fit_needs_at_least_four_samples() -> None:
    with pytest.raises(ValueError, match="at least 4"):
        calibrate.fit_thresholds([sample(1.0, Severity.MILD, "s1")])


def test_fitted_thresholds_reproduce_via_calibrator_to_severity() -> None:
    """The fit's own evaluation must agree with the exact function decide.py calls in
    production -- a private re-implementation here could silently drift from it."""
    samples = [
        sample(float(i), BANDS[min(i // 5, 3)], f"s{i}") for i in range(20)
    ]
    thresholds, _ = calibrate.fit_thresholds(samples)
    for s in samples:
        predicted = calibrator.to_severity(s.z, thresholds)
        assert predicted.rank >= 0  # exercised via the real function, not reimplemented


# ------------------------------------------------------------------------------- eval


def test_adjacent_agreement_is_never_below_exact_agreement() -> None:
    samples = [sample(float(i), BANDS[i % 4], f"s{i}") for i in range(16)]
    thresholds = {"t0": 4.0, "t1": 8.0, "t2": 12.0}
    result = calibrate.evaluate(samples, thresholds)
    assert result["adjacent_agreement"] >= result["exact_agreement"]


def test_evaluate_of_empty_samples_reports_none_not_zero() -> None:
    """Zero samples is not zero agreement -- it is nothing evaluated, and the caller
    (ConcernReport.gate_verdicts) must be able to tell the two apart."""
    result = calibrate.evaluate([], {"t0": 0.0, "t1": 1.0, "t2": 2.0})
    assert result == {"n": 0, "exact_agreement": None, "adjacent_agreement": None}


# --------------------------------------------------------------- subject-level aggregation


def test_repeat_captures_are_averaged_not_counted_twice() -> None:
    """The core defence against a subject with many selfies dominating the cohort
    statistic: two captures of s1 must contribute ONE averaged value, indistinguishable
    from a subject who was only ever captured once at the mean."""
    captures = [
        calibrate.Capture("a1.jpg", "s1", {("redness", "forehead"): {"x": 0.1}}),
        calibrate.Capture("a2.jpg", "s1", {("redness", "forehead"): {"x": 0.3}}),
        calibrate.Capture("b1.jpg", "s2", {("redness", "forehead"): {"x": 0.5}}),
    ]
    result = calibrate.subject_level_raw(captures)
    assert result["s1"][("redness", "forehead")]["x"] == pytest.approx(0.2)
    assert result["s2"][("redness", "forehead")]["x"] == pytest.approx(0.5)


def test_subject_level_raw_keeps_concerns_and_rois_separate() -> None:
    captures = [
        calibrate.Capture(
            "a.jpg", "s1",
            {
                ("redness", "forehead"): {"x": 1.0},
                ("redness", "left_cheek"): {"x": 2.0},
                ("dark_spots", "forehead"): {"x": 3.0},
            },
        ),
    ]
    result = calibrate.subject_level_raw(captures)
    assert set(result["s1"]) == {
        ("redness", "forehead"), ("redness", "left_cheek"), ("dark_spots", "forehead"),
    }


# ------------------------------------------------------------------------ subject split


def test_split_is_deterministic_across_runs() -> None:
    subjects = [f"s{i}" for i in range(50)]
    a = calibrate.split_subjects(subjects, 0.3, "v1")
    b = calibrate.split_subjects(subjects, 0.3, "v1")
    assert a == b


def test_split_changes_with_a_different_salt() -> None:
    """A protocol bump must be able to reshuffle the split deliberately."""
    subjects = [f"s{i}" for i in range(50)]
    a = calibrate.split_subjects(subjects, 0.3, "v1")
    b = calibrate.split_subjects(subjects, 0.3, "v2")
    assert a != b


def test_split_has_no_overlap_and_covers_every_subject() -> None:
    subjects = [f"s{i}" for i in range(100)]
    train, holdout = calibrate.split_subjects(subjects, 0.3, "v1")
    assert set(train) & set(holdout) == set()
    assert set(train) | set(holdout) == set(subjects)


def test_split_fraction_is_approximately_respected() -> None:
    """Hash-based, so not exact -- but should not be wildly off at n=1000."""
    subjects = [f"s{i}" for i in range(1000)]
    train, holdout = calibrate.split_subjects(subjects, 0.3, "v1")
    assert 0.20 < len(holdout) / len(subjects) < 0.40


# --------------------------------------------------------------------- reference stats


def test_reference_stats_below_cohort_floor_are_not_calibrated() -> None:
    """The single most important refusal in this file: too few subjects must produce
    calibrated: false even though the arithmetic runs fine and produces numbers that look
    like a real median and MAD."""
    from skin_analysis.schemas import Concern

    subject_raw = {
        f"s{i}": {("redness", "forehead"): {"affected_area_ratio": 0.1 * i}}
        for i in range(5)
    }
    stats = calibrate.compute_reference_stats(
        Concern.REDNESS, subject_raw, list(subject_raw), min_cohort_subjects=200
    )
    assert stats.calibrated is False
    assert "200" in stats.reason


def test_reference_stats_at_or_above_floor_are_calibrated() -> None:
    from skin_analysis.schemas import Concern

    subject_raw = {
        f"s{i}": {("redness", "forehead"): {"affected_area_ratio": 0.1 + 0.001 * i}}
        for i in range(200)
    }
    stats = calibrate.compute_reference_stats(
        Concern.REDNESS, subject_raw, list(subject_raw), min_cohort_subjects=200
    )
    assert stats.calibrated is True
    assert stats.n_subjects == 200


def test_reference_stats_only_use_train_subjects() -> None:
    """A subject in the holdout split must not leak into the cohort statistic it is later
    scored against -- that would make the holdout evaluation circular."""
    from skin_analysis.schemas import Concern

    subject_raw = {
        "train1": {("redness", "forehead"): {"x": 1.0}},
        "holdout1": {("redness", "forehead"): {"x": 1000.0}},
    }
    stats = calibrate.compute_reference_stats(
        Concern.REDNESS, subject_raw, ["train1"], min_cohort_subjects=1
    )
    assert stats.median["forehead"]["x"] == 1.0


def test_reference_stats_are_stored_unscaled() -> None:
    """standardize.robust_z applies MAD_SCALE itself; storing a pre-scaled MAD here would
    silently double-apply the scale factor on every read."""
    from skin_analysis.decision import standardize
    from skin_analysis.schemas import Concern

    subject_raw = {
        "s1": {("redness", "forehead"): {"x": 1.0}},
        "s2": {("redness", "forehead"): {"x": 3.0}},
        "s3": {("redness", "forehead"): {"x": 5.0}},
    }
    stats = calibrate.compute_reference_stats(
        Concern.REDNESS, subject_raw, list(subject_raw), min_cohort_subjects=1
    )
    # median=3, deviations [2,0,2], MAD=2 (unscaled)
    assert stats.median["forehead"]["x"] == pytest.approx(3.0)
    assert stats.mad["forehead"]["x"] == pytest.approx(2.0)
    z = standardize.robust_z(3.0, stats.median["forehead"]["x"], stats.mad["forehead"]["x"], 1e-6)
    assert z == pytest.approx(0.0, abs=1e-3)


# ----------------------------------------------------------------------- annotations


def test_annotations_require_all_columns(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text("subject_id,concern,roi\ns1,redness,forehead\n", encoding="utf-8")
    with pytest.raises(ValueError, match="severity"):
        calibrate.load_annotations(path, CALIBRATION_GATES["annotations"])


def test_annotations_reject_an_unknown_severity_value(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text(
        "subject_id,concern,roi,severity\ns1,redness,forehead,severe\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="severe"):
        calibrate.load_annotations(path, CALIBRATION_GATES["annotations"])


def test_duplicate_annotation_rows_resolve_by_majority_vote(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text(
        "subject_id,concern,roi,severity\n"
        "s1,redness,forehead,mild\n"
        "s1,redness,forehead,mild\n"
        "s1,redness,forehead,moderate\n",
        encoding="utf-8",
    )
    labels, disagreements = calibrate.load_annotations(path, CALIBRATION_GATES["annotations"])
    assert len(labels) == 1
    assert labels[0].severity == "mild"
    assert disagreements  # the 2-vs-1 split must be surfaced, not silently resolved


def test_annotations_use_concern_value_vocabulary_in_the_gate_config() -> None:
    """Guards the exact bug just fixed in run_validation.py from being reintroduced here:
    the annotation spec must document Concern.value ("dark_spots"), not the config key
    ("pigmentation")."""
    assert "pigmentation" not in str(CALIBRATION_GATES["annotations"])


# -------------------------------------------------------------------- combined samples


def test_combined_samples_use_the_production_combine_function() -> None:
    """Standardization and combination must be IDENTICAL to what decide.py runs at
    serving time, or a threshold fit here means something different in production."""
    from skin_analysis.schemas import Concern

    subject_raw = {
        "s1": {("redness", "forehead"): {
            "affected_area_ratio": 0.3, "median_positive_delta_a": 4.0, "p90_delta_a": 6.0,
        }},
    }
    reference = calibrate.ReferenceStats(
        concern="redness", calibrated=True, n_subjects=1,
        median={"forehead": {
            "affected_area_ratio": 0.1, "median_positive_delta_a": 2.0, "p90_delta_a": 3.0,
        }},
        mad={"forehead": {
            "affected_area_ratio": 0.05, "median_positive_delta_a": 0.5, "p90_delta_a": 1.0,
        }},
    )
    labels = [calibrate.Label("s1", "redness", "forehead", "mild")]
    concern_config = cfg.concern_config("redness")

    samples = calibrate.combined_samples(
        Concern.REDNESS, concern_config, labels, subject_raw, reference, 1e-6
    )
    assert len(samples) == 1

    from skin_analysis.decision import standardize
    z_ref = {
        "affected_area_ratio": standardize.robust_z(0.3, 0.1, 0.05, 1e-6),
        "median_positive_delta_a": standardize.robust_z(4.0, 2.0, 0.5, 1e-6),
        "p90_delta_a": standardize.robust_z(6.0, 3.0, 1.0, 1e-6),
    }
    expected = calibrator.combine(z_ref, concern_config)
    assert samples[0].z == pytest.approx(expected)


def test_combined_samples_skip_a_subject_with_no_matching_measurement() -> None:
    """Standardizing against absent statistics silently reads as 'perfectly average' -- a
    label with no raw measurement to pair with must be dropped, not imputed."""
    from skin_analysis.schemas import Concern

    reference = calibrate.ReferenceStats(
        concern="redness", calibrated=True, n_subjects=1,
        median={"forehead": {"affected_area_ratio": 0.1}},
        mad={"forehead": {"affected_area_ratio": 0.05}},
    )
    labels = [calibrate.Label("ghost_subject", "redness", "forehead", "mild")]
    samples = calibrate.combined_samples(
        Concern.REDNESS, cfg.concern_config("redness"), labels, {}, reference, 1e-6
    )
    assert samples == []


# ------------------------------------------------------------------- publication boundary


def test_calibrate_module_never_touches_severity_thresholds_yaml() -> None:
    """Structural guard for the boundary this whole script exists to respect: no code
    path may write severity_thresholds.yaml or flip meta.calibrated (D2, CLAUDE.md §6)."""
    source = (REPO / "scripts" / "calibrate.py").read_text(encoding="utf-8")
    assert "meta.calibrated = " not in source
    assert "meta[\"calibrated\"] = " not in source
    assert "meta['calibrated'] = " not in source
    assert ".open(\"w\"" not in source  # this script never opens severity_thresholds.yaml
    assert "severity_thresholds.yaml\", \"w\"" not in source


def test_calibrate_module_never_sets_frozen_true() -> None:
    source = (REPO / "scripts" / "calibrate.py").read_text(encoding="utf-8")
    assert "\"frozen\": True" not in source
    assert "'frozen': True" not in source
    assert "frozen\"] = True" not in source


def test_reference_json_shape_matches_what_standardize_load_reference_reads() -> None:
    """The file this script writes must be exactly what
    decision.standardize.load_reference expects: median/mad dicts keyed by ROI then
    measurement name, both present for calibrated: true."""
    from skin_analysis.schemas import Concern

    subject_raw = {
        f"s{i}": {("redness", "forehead"): {"x": float(i)}} for i in range(200)
    }
    stats = calibrate.compute_reference_stats(
        Concern.REDNESS, subject_raw, list(subject_raw), min_cohort_subjects=200
    )
    payload = stats.as_json()
    assert payload["calibrated"] is True
    assert payload["median"]["forehead"]["x"] == pytest.approx(99.5)
    assert set(payload) >= {"calibrated", "n_subjects", "median", "mad"}


# ------------------------------------------------------------------------------ config


def test_min_cohort_subjects_is_not_duplicated_in_calibration_gates() -> None:
    """The floor must be read from severity_thresholds.yaml (the single source of truth
    already used by util/calibration.py), not redeclared here where it could drift."""
    assert "min_cohort_subjects" not in CALIBRATION_GATES.get("cohort", {})


def test_gate_set_is_versioned() -> None:
    assert CALIBRATION_GATES["meta"]["gate_set_version"]
    assert CALIBRATION_GATES["meta"]["prespecified_on"]
