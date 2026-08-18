"""Guards for the blur-sensitivity measurement (data-driven blur.min_laplacian_var).

Everything here runs on synthetic DriftSample data, no face corpus needed -- the same
reason test_validation.py and test_calibrate.py stay pure. The property under test is
narrow but important: `recommend_floor` must pick the LOOSEST threshold that still keeps
drift within budget, not the strictest tested. Recommending stricter than the evidence
requires would reintroduce exactly the "guessed, not measured" cutoff this script exists
to replace.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "blur_sensitivity", REPO / "scripts" / "blur_sensitivity.py"
)
assert spec and spec.loader
blur_sensitivity = importlib.util.module_from_spec(spec)
sys.modules["blur_sensitivity"] = blur_sensitivity
spec.loader.exec_module(blur_sensitivity)


def row(threshold: float, n: int, tenengrad: float, p50: float, p95: float):
    return blur_sensitivity.ThresholdRow(
        threshold=threshold, n=n, min_tenengrad=tenengrad, p50_drift=p50, p95_drift=p95,
    )


def sample(laplacian_var: float, baseline: float, blurred: float, sigma: float = 1.0):
    return blur_sensitivity.DriftSample(
        image="x.jpg", concern="texture", roi="forehead", measurement="glcm_contrast",
        sigma=sigma, laplacian_var=laplacian_var, tenengrad=laplacian_var * 2,
        baseline=baseline, blurred=blurred,
    )


# --------------------------------------------------------------------------- drift math


def test_relative_drift_is_the_fractional_change_from_baseline() -> None:
    s = sample(laplacian_var=100.0, baseline=10.0, blurred=12.0)
    assert s.relative_drift == pytest.approx(0.2)


def test_relative_drift_at_zero_blur_change_is_zero() -> None:
    s = sample(laplacian_var=1000.0, baseline=5.0, blurred=5.0)
    assert s.relative_drift == 0.0


def test_relative_drift_near_zero_baseline_does_not_divide_by_zero() -> None:
    """A baseline of exactly 0 (e.g. no wrinkles detected at all) must not raise or
    produce inf -- the eps guard makes this a large-but-finite number instead."""
    s = sample(laplacian_var=100.0, baseline=0.0, blurred=0.3)
    assert s.relative_drift > 0
    assert s.relative_drift < float("inf")


# ------------------------------------------------------------------------- percentile


def test_percentile_matches_numpy() -> None:
    np = pytest.importorskip("numpy")
    values = [0.1, 0.4, 0.2, 0.9, 0.3, 0.7]
    for q in (5, 50, 95):
        assert blur_sensitivity.percentile(values, q) == pytest.approx(
            float(np.percentile(values, q))
        )


# --------------------------------------------------------------------- threshold sweep


def test_sweep_thresholds_skips_candidates_below_min_n() -> None:
    """A p95 over 2 points is noise; the sweep must not report one as if it meant
    something."""
    samples = [sample(100.0, 10.0, 10.5), sample(100.0, 10.0, 10.2)]
    rows = blur_sensitivity.sweep_thresholds(samples, min_n=5)
    assert rows == []


def test_sweep_thresholds_drift_grows_as_the_floor_loosens() -> None:
    """More blur (lower laplacian_var) should show up as MORE drift once included --
    the whole premise the recommendation logic depends on."""
    sharp = [sample(1000.0, 10.0, 10.05 + i * 0.001) for i in range(6)]
    blurry = [sample(50.0, 10.0, 15.0 + i * 0.1) for i in range(6)]
    rows = blur_sensitivity.sweep_thresholds(sharp + blurry, min_n=5)
    by_threshold = {row.threshold: row for row in rows}
    # At the strict floor (1000), only sharp samples qualify -- low drift.
    assert by_threshold[1000.0].p95_drift < 0.1
    # At the loose floor (50), both groups qualify -- drift rises because the blurry
    # group is now included.
    assert by_threshold[50.0].p95_drift > by_threshold[1000.0].p95_drift


# ---------------------------------------------------------------------- recommendation


def test_recommend_floor_picks_the_loosest_threshold_within_budget() -> None:
    """The central property: NOT the strictest (sharpest-only) threshold tested, and NOT
    a threshold whose drift exceeds the budget -- the loosest one that still clears it."""
    rows = [
        row(1000.0, 10, 2000.0, 0.01, 0.02),
        row(500.0, 10, 1000.0, 0.03, 0.08),
        row(200.0, 10, 400.0, 0.10, 0.20),
        row(50.0, 10, 100.0, 0.40, 0.90),
    ]
    # Budget of 0.15: 1000 and 500 clear it (0.02, 0.08 <= 0.15), 200 and 50 do not.
    assert blur_sensitivity.recommend_floor(rows, bound=0.15) == 500.0


def test_recommend_floor_is_none_when_nothing_tested_clears_the_bound() -> None:
    rows = [row(1000.0, 10, 2000.0, 0.5, 0.9)]
    assert blur_sensitivity.recommend_floor(rows, bound=0.15) is None


def test_recommend_floor_is_none_for_an_undeclared_bound() -> None:
    """A concern with no configured CV bound must not silently get a recommendation --
    that would be inventing a tolerance nobody prespecified."""
    rows = [row(1000.0, 10, 2000.0, 0.01, 0.02)]
    assert blur_sensitivity.recommend_floor(rows, bound=None) is None


def test_recommend_floor_of_empty_rows_is_none() -> None:
    assert blur_sensitivity.recommend_floor([], bound=0.15) is None


# -------------------------------------------------------------------------- config wiring


def test_recommendation_reuses_the_prespecified_repeatability_bound_not_a_new_one() -> None:
    """The whole point of this script: reuse validation_gates.yaml's raw_metric_cv bound
    rather than inventing a fresh tolerance. Guards against a future edit adding a
    parallel, undocumented bound here instead of reading the existing one."""
    from skin_analysis.util import config as cfg

    gates = cfg.load("validation_gates")
    bounds = gates["repeatability"]["gates"]["raw_metric_cv"]["p95_max_cv"]
    assert set(blur_sensitivity._SWEPT) <= set(bounds) | {"texture", "wrinkles"}
    assert "texture" in bounds
    assert "wrinkles" in bounds


def test_only_structurally_blur_sensitive_concerns_are_swept() -> None:
    """Redness/pigmentation are colour proxies over a whole-ROI statistic; sweeping them
    would cost real runtime for a relationship the module docstring already argues is
    weak. This test pins the deliberate scope, not an accidental omission."""
    assert set(blur_sensitivity._SWEPT) == {"texture", "wrinkles"}


# --------------------------------------------------------------- reference selection


def test_content_hash_is_identical_for_identical_bytes(tmp_path: Path) -> None:
    """The dedup mechanism itself. This project's real 50-photo corpus turned out to be
    29 unique files under "- Copy" / "(copy)" names; counting a duplicate twice is not a
    second, independent piece of evidence about where blur breaks a measurement."""
    a = tmp_path / "photo.jpg"
    b = tmp_path / "photo_-_Copy.jpg"
    a.write_bytes(b"same content, different filename")
    b.write_bytes(b"same content, different filename")
    assert blur_sensitivity._content_hash(a) == blur_sensitivity._content_hash(b)


def test_content_hash_differs_for_different_bytes(tmp_path: Path) -> None:
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    a.write_bytes(b"one photo")
    b.write_bytes(b"a different photo")
    assert blur_sensitivity._content_hash(a) != blur_sensitivity._content_hash(b)


def test_reference_floor_checks_both_blur_metrics_not_one() -> None:
    """Regression test for the actual bug found on the real corpus: an earlier version
    filtered candidate references on tenengrad alone. Every one of the first 5 unique
    photos in data/raw/real_orig passed that filter while sitting BELOW the current
    laplacian_var floor even at zero synthetic blur, so the sweep never bracketed the
    threshold it was meant to evaluate -- it silently measured a different, unrelated
    question. MIN_REFERENCE_METRIC_MULTIPLE must apply to laplacian_var too."""
    import inspect

    source = inspect.getsource(blur_sensitivity.build_reference)
    assert "laplacian_var" in source
    assert "MIN_REFERENCE_METRIC_MULTIPLE" in source
