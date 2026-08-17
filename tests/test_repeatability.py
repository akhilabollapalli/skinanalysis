"""Determinism tests, plus the config invariants that guard Rule 2 and D2.

D13 separates two properties that are easy to confuse:

    Determinism   same input array twice -> byte-identical output.   HERE, in CI.
    Repeatability distinct captures of one subject -> same ordinal severity.
                  Release validation only, against a local gitignored corpus:
                      python scripts/run_validation.py --suite repeatability --corpus PATH

The repo cannot hold real face images (CLAUDE.md §5), so true repeatability cannot be a CI
check. Naming them apart stops the cheap test from being mistaken for the meaningful one:
a user who rescans and sees "mild" become "high" stops trusting the product regardless of
how it scores against expert labels.
"""

from __future__ import annotations

import numpy as np
import pytest

from skin_analysis import pipeline
from skin_analysis.util import calibration
from skin_analysis.util import config as cfg

ACTIVE = ["redness", "pigmentation", "texture", "wrinkles"]
DISABLED = ["fine_lines", "acne", "pores"]


@pytest.mark.parametrize("concern", ACTIVE)
def test_active_features_are_enabled(concern: str) -> None:
    assert cfg.feature_enabled(concern), f"{concern} should be active in V1"


@pytest.mark.parametrize("concern", DISABLED)
def test_experimental_features_stay_disabled(concern: str) -> None:
    """Guards CLAUDE.md Rule 2: disabled concerns must not be switched on to fill the UI."""
    assert not cfg.feature_enabled(concern), (
        f"{concern} must stay disabled until commercially clear labels exist "
        "and it passes its acceptance gate"
    )
    assert cfg.load("severity_thresholds")[concern].get("blocked_on"), (
        f"{concern} must record what it is blocked on"
    )


def test_severity_not_marked_calibrated_prematurely() -> None:
    """Placeholder thresholds must not be presented as validated."""
    meta = cfg.load("severity_thresholds")["meta"]
    if meta.get("calibrated"):
        assert meta.get("calibration_source"), "calibrated=true requires a named source"
        assert meta.get("holdout") == "subject_level"


@pytest.mark.parametrize("concern", ACTIVE)
def test_active_concerns_are_not_publishable_before_reference_stats(concern: str) -> None:
    """D1 stage B: standardizing against absent statistics would silently produce zeros,
    which read as 'perfectly average' rather than 'unknown'."""
    assert not calibration.concern_public_ready(concern)
    assert calibration.missing_reference_rois(concern), (
        f"{concern} claims calibrated reference stats; if real, update meta and this test"
    )


@pytest.mark.parametrize("concern", ACTIVE + DISABLED)
def test_every_concern_declares_primary_rois(concern: str) -> None:
    """D7 needs to know which ROIs a concern requires before it can report partially."""
    assert cfg.concern_config(concern).get("primary_rois")


@pytest.mark.parametrize("concern", ACTIVE)
def test_no_fixed_pixel_windows_in_active_concerns(concern: str) -> None:
    """D1: a fixed window means different things at different resolutions, so cohort
    statistics computed with one would not transfer between devices."""
    block = cfg.concern_config(concern)

    def walk(node, path="") -> list[str]:
        found = []
        if isinstance(node, dict):
            for key, value in node.items():
                here = f"{path}.{key}" if path else key
                if key.endswith("_px") and not key.endswith("_frac_of_iod"):
                    found.append(here)
                found += walk(value, here)
        return found

    offenders = walk(block)
    assert not offenders, (
        f"{concern}: fixed pixel parameters must be fractions of the anchor: {offenders}"
    )


def test_pipeline_is_deterministic_for_identical_input() -> None:
    """The floor beneath repeatability, and the only half of it CI can check.

    Noise, so this exercises the fail-closed path rather than the measurement path. Both
    have to be deterministic: a scan that gives two different RETAKE reasons for the same
    bytes is as broken as one that gives two different severities.
    """
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (720, 720, 3), dtype=np.uint8)
    first = pipeline.analyze_scan_internal(image).to_internal_payload()
    second = pipeline.analyze_scan_internal(image).to_internal_payload()
    assert first == second
