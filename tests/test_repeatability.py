"""Repeatability tests -- the product's primary quality metric.

A user who rescans within five minutes and sees "mild" become "high" stops trusting the
product, regardless of how it scores against expert labels. These tests defend that
property directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from skin_analysis import pipeline
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


@pytest.mark.xfail(reason="pipeline not implemented yet", strict=True)
def test_same_input_gives_identical_result() -> None:
    """Determinism is the floor for repeatability."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (720, 720, 3), dtype=np.uint8)
    assert pipeline.analyze_scan(image).to_public() == pipeline.analyze_scan(image).to_public()
