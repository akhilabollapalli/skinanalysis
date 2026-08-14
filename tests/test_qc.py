"""Capture quality gate tests.

The gate must fail CLOSED. Every test here defends the property that a bad capture is
rejected rather than analyzed -- analyzing it yields a confident wrong answer.
"""

from __future__ import annotations

import numpy as np
import pytest

from skin_analysis.capture import qc
from skin_analysis.util import config as cfg


@pytest.fixture
def capture_config() -> dict:
    return cfg.load("capture_thresholds")


def test_config_loads(capture_config: dict) -> None:
    assert "blur" in capture_config
    assert "exposure" in capture_config
    assert capture_config["face"]["max_faces"] == 1


@pytest.mark.xfail(reason="qc.check not implemented yet", strict=True)
def test_blank_image_is_rejected(capture_config: dict) -> None:
    blank = np.zeros((720, 720, 3), dtype=np.uint8)
    result = qc.check(blank, capture_config)
    assert result.passed is False
    assert result.failures


@pytest.mark.xfail(reason="qc.check not implemented yet", strict=True)
def test_failure_reasons_are_public_but_metrics_are_not(capture_config: dict) -> None:
    blank = np.zeros((720, 720, 3), dtype=np.uint8)
    public = qc.check(blank, capture_config).to_public()
    assert set(public.keys()) == {"pass", "reasons"}
