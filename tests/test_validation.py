"""Guards for the release-validation runner.

The runner's job is to be hard to fool. Every test here defends one way a validation
report could look green while proving nothing:

* counting UNMEASURABLE rescans as agreement,
* pooling a failing subgroup into a passing average,
* treating an unimplemented gate as satisfied,
* comparing two people because their filenames looked similar.

All statistics are computed from Records, never from images, so this runs in CI without a
face corpus -- which is the same reason repeatability ITSELF cannot run in CI (D13).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "run_validation", REPO / "scripts" / "run_validation.py"
)
assert spec and spec.loader
run_validation = importlib.util.module_from_spec(spec)
sys.modules["run_validation"] = run_validation
spec.loader.exec_module(run_validation)

from skin_analysis.util import config as cfg  # noqa: E402

GATES = cfg.load("validation_gates")
SESSION = GATES["corpus"]["session_column"]


def record(
    image: str,
    subject: str,
    *,
    session: str = "a",
    severity: str = "mild",
    concern: str = "redness",
    raw: dict[str, dict[str, float]] | None = None,
    qc_passed: bool = True,
    meta: dict[str, str] | None = None,
):
    """One synthetic analysed capture."""
    raw = {"forehead": {"affected_area_ratio": 0.10}} if raw is None else raw
    full_meta = {"subject_id": subject, SESSION: session}
    full_meta.update(meta or {})
    return run_validation.Record(
        image=image,
        meta=full_meta,
        qc_passed=qc_passed,
        qc_failures=() if qc_passed else ("blur",),
        severities={concern: severity},
        measurable_rois={concern: tuple(sorted(raw))},
        raw={concern: raw},
    )


def grouped(records, min_captures: int = 2):
    return run_validation.group_records(records, SESSION, min_captures)


# ------------------------------------------------------------------ ordinal agreement


def test_unmeasurable_rescans_are_not_counted_as_agreement() -> None:
    """The single most important test in this file.

    Until the calibration cohort exists every concern returns UNMEASURABLE (D2). If those
    counted as agreement, the headline repeatability gate would read 100% on a build that
    has never measured anything, and would keep reading 100% through every regression.
    """
    records = [
        record("a.jpg", "s1", severity="unmeasurable"),
        record("b.jpg", "s1", severity="unmeasurable"),
    ]
    agreed, comparable = run_validation.ordinal_agreement(grouped(records), "redness")
    assert comparable == 0
    assert agreed == 0


def test_disabled_concerns_are_not_counted_as_agreement() -> None:
    records = [
        record("a.jpg", "s1", severity="disabled", concern="acne"),
        record("b.jpg", "s1", severity="disabled", concern="acne"),
    ]
    _, comparable = run_validation.ordinal_agreement(grouped(records), "acne")
    assert comparable == 0


def test_mixed_measurable_and_unmeasurable_group_is_not_comparable() -> None:
    """A group that measured once and failed once has no agreement to report.

    It is an ROI-availability finding, counted by the availability gate, and calling it
    either agreement or disagreement would double-count it.
    """
    records = [
        record("a.jpg", "s1", severity="mild"),
        record("b.jpg", "s1", severity="unmeasurable"),
    ]
    _, comparable = run_validation.ordinal_agreement(grouped(records), "redness")
    assert comparable == 0


def test_agreement_counts_identical_bands_and_catches_a_flip() -> None:
    records = [
        record("a1.jpg", "s1", severity="mild"),
        record("a2.jpg", "s1", severity="mild"),
        record("b1.jpg", "s2", severity="mild"),
        record("b2.jpg", "s2", severity="high"),
    ]
    agreed, comparable = run_validation.ordinal_agreement(grouped(records), "redness")
    assert (agreed, comparable) == (1, 2)


def test_sessions_are_not_merged_across_days() -> None:
    """Same subject, two sessions, so two groups -- not one four-capture group.

    Cross-session skin change is real change. Merging the sessions would charge it to
    measurement noise and make the pipeline look unstable when the skin simply moved.
    """
    records = [
        record("a1.jpg", "s1", session="mon", severity="mild"),
        record("a2.jpg", "s1", session="mon", severity="mild"),
        record("b1.jpg", "s1", session="tue", severity="high"),
        record("b2.jpg", "s1", session="tue", severity="high"),
    ]
    groups = grouped(records)
    assert len(groups) == 2
    agreed, comparable = run_validation.ordinal_agreement(groups, "redness")
    assert (agreed, comparable) == (2, 2)


def test_single_capture_subjects_are_dropped() -> None:
    """One shot of a person cannot demonstrate anything about rescan stability."""
    assert grouped([record("only.jpg", "s1")]) == {}


def test_records_without_a_subject_are_dropped() -> None:
    """Anonymous captures would silently merge into one enormous fake subject."""
    records = [record("a.jpg", ""), record("b.jpg", "")]
    assert grouped(records) == {}


# ------------------------------------------------------------------------- raw CV


def test_cv_is_zero_for_identical_rescans() -> None:
    raw = {"forehead": {"affected_area_ratio": 0.2}}
    samples, undefined = run_validation.raw_metric_cvs(
        grouped([record("a.jpg", "s1", raw=raw), record("b.jpg", "s1", raw=raw)]), "redness"
    )
    assert undefined == 0
    assert [s.cv for s in samples] == [0.0]


def test_cv_grows_with_drift() -> None:
    samples, _ = run_validation.raw_metric_cvs(
        grouped(
            [
                record("a.jpg", "s1", raw={"forehead": {"affected_area_ratio": 0.10}}),
                record("b.jpg", "s1", raw={"forehead": {"affected_area_ratio": 0.30}}),
            ]
        ),
        "redness",
    )
    assert samples[0].cv > 0.5


def test_zero_mean_measurements_are_counted_not_silently_dropped() -> None:
    """A CV needs a non-zero mean. Many of these means the concern found nothing --
    a statement about the concern, which must not vanish from the report."""
    raw = {"forehead": {"line_density": 0.0}}
    samples, undefined = run_validation.raw_metric_cvs(
        grouped([record("a.jpg", "s1", raw=raw, concern="wrinkles"),
                 record("b.jpg", "s1", raw=raw, concern="wrinkles")]),
        "wrinkles",
    )
    assert samples == []
    assert undefined == 1


def test_cv_only_uses_measurements_present_in_every_capture() -> None:
    """A value present twice and missing once is an availability finding, not a sample.

    Averaging over whatever happened to be present would quietly shrink the denominator
    and make a flaky ROI look like a stable one.
    """
    samples, _ = run_validation.raw_metric_cvs(
        grouped(
            [
                record("a.jpg", "s1", raw={"forehead": {"x": 1.0}, "nose": {"x": 2.0}}),
                record("b.jpg", "s1", raw={"forehead": {"x": 1.0}}),
            ]
        ),
        "redness",
    )
    assert {s.roi for s in samples} == {"forehead"}


# --------------------------------------------------------------- ROI availability


def test_roi_availability_catches_a_region_that_comes_and_goes() -> None:
    records = [
        record("a.jpg", "s1", raw={"forehead": {"x": 1.0}, "nose": {"x": 1.0}}),
        record("b.jpg", "s1", raw={"forehead": {"x": 1.0}}),
    ]
    agreed, examined = run_validation.roi_availability_agreement(grouped(records), "redness")
    assert (agreed, examined) == (0, 1)


def test_groups_that_measured_nothing_are_not_examined_for_availability() -> None:
    """Perfect agreement on an empty set is not evidence of stability."""
    records = [record("a.jpg", "s1", raw={}), record("b.jpg", "s1", raw={})]
    agreed, examined = run_validation.roi_availability_agreement(grouped(records), "redness")
    assert (agreed, examined) == (0, 0)


# ---------------------------------------------------------------------- slicing


def test_blank_slice_values_are_dropped_not_bucketed() -> None:
    """An "unknown" bucket looks like a demographic group and is a data-entry gap."""
    records = [
        record("a.jpg", "s1", meta={"skin_tone": "mst-06"}),
        record("b.jpg", "s2", meta={"skin_tone": "  "}),
    ]
    assert set(run_validation.partition(records, "skin_tone")) == {"mst-06"}


def test_missing_required_slice_fails_the_fairness_gate() -> None:
    """A one-family corpus must not be labelled release evidence."""
    records = [record("a.jpg", "s1"), record("b.jpg", "s1")]
    results = run_validation.run_fairness(records, GATES)
    assert results
    assert all(g.status == "fail" for g in results)


def test_fairness_gate_passes_with_enough_distinct_values() -> None:
    meta_a = {"skin_tone": "mst-03", "device": "pixel-8", "lighting": "natural"}
    meta_b = {"skin_tone": "mst-09", "device": "iphone-15", "lighting": "warm"}
    records = [record("a.jpg", "s1", meta=meta_a), record("b.jpg", "s2", meta=meta_b)]
    assert all(g.status == "pass" for g in run_validation.run_fairness(records, GATES))


def test_slice_spread_needs_two_gateable_subgroups() -> None:
    """One subgroup has no spread. Reporting 0.0 would read as perfect equality."""
    slices = {"skin_tone": {"mst-03": {"rejection_rate": 0.1, "gated": True}}}
    spread, _ = run_validation._max_spread(slices, "rejection_rate")
    assert spread is None


def test_slice_spread_ignores_subgroups_below_the_gate_floor() -> None:
    slices = {
        "skin_tone": {
            "mst-03": {"rejection_rate": 0.10, "gated": True},
            "mst-09": {"rejection_rate": 0.90, "gated": False},
        }
    }
    spread, _ = run_validation._max_spread(slices, "rejection_rate")
    assert spread is None


# ------------------------------------------------------------------- gate mechanics


def test_capture_rejection_gate_reads_the_configured_bound() -> None:
    bound = GATES["capture"]["gates"]["rejection_rate"]["max_fraction"]
    records = [record(f"{i}.jpg", f"s{i}", qc_passed=False) for i in range(10)]
    level = run_validation.run_capture(records, GATES, [], 5)[0]
    assert level.status == "fail"
    assert level.value == 1.0
    assert level.bound == bound


def test_candidate_map_stability_is_reported_unevaluated_not_passed() -> None:
    """CLAUDE.md §5 names four repeatability gates. The fourth is not implemented, and an
    unimplemented gate must stay visible rather than dropping off the list."""
    assert GATES["repeatability"]["gates"]["candidate_map_stability"]["implemented"] is False
    records = [record("a.jpg", "s1"), record("b.jpg", "s1")]
    results = run_validation.run_repeatability(records, GATES, [], 5)
    gate = next(g for g in results if g.name == "candidate_map_stability")
    assert gate.status == "not_evaluated"
    assert not gate.evaluated


def test_unevaluated_is_neither_pass_nor_fail() -> None:
    gate = run_validation.GateResult("x", "not_evaluated", "")
    assert not gate.failed
    assert not gate.evaluated


def test_every_enabled_concern_has_a_prespecified_cv_bound() -> None:
    """An enabled concern with no bound is ungated, and ungated is indistinguishable from
    passing in a summary table."""
    bounds = GATES["repeatability"]["gates"]["raw_metric_cv"]["p95_max_cv"]
    enabled = [c for c in cfg.configured_concerns() if cfg.feature_enabled(c)]
    assert not [c for c in enabled if c not in bounds]


def test_gate_set_is_versioned() -> None:
    """A report that cannot name the gate set it was judged against proves nothing."""
    assert GATES["meta"]["gate_set_version"]
    assert GATES["meta"]["prespecified_on"]


# ------------------------------------------------------------------------ manifest


def test_manifest_requires_subject_id(tmp_path: Path) -> None:
    """Subject identity is never inferred from a filename: a wrong guess merges two people
    into one subject and every repeatability number then compares strangers."""
    path = tmp_path / "manifest.csv"
    path.write_text("image,device\na.jpg,pixel-8\n", encoding="utf-8")
    with pytest.raises(ValueError, match="subject_id"):
        run_validation.load_manifest(path, ["image", "subject_id"])


def test_manifest_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    path.write_text("image,subject_id\na.jpg, s1 \n", encoding="utf-8")
    manifest = run_validation.load_manifest(path, ["image", "subject_id"])
    assert manifest["a.jpg"]["subject_id"] == "s1"


# -------------------------------------------------------------------------- percentile


def test_percentile_matches_numpy() -> None:
    np = pytest.importorskip("numpy")
    values = [0.1, 0.4, 0.2, 0.9, 0.3, 0.7]
    for q in (5, 50, 95):
        assert run_validation.percentile(values, q) == pytest.approx(
            float(np.percentile(values, q))
        )


def test_percentile_of_one_value() -> None:
    assert run_validation.percentile([0.42], 95) == 0.42
