"""Capture quality gate tests.

The gate must fail CLOSED. Every test here defends the property that a bad capture is
rejected rather than analyzed -- analyzing it yields a confident wrong answer.

D4 raised the stakes: V1 performs no white-balance correction at all, so the colour-cast
check here is the only thing between a badly lit room and a redness measurement.
"""

from __future__ import annotations

import numpy as np
import pytest

from skin_analysis.capture import qc
from skin_analysis.schemas import QCFailure
from skin_analysis.util import config as cfg


@pytest.fixture
def capture_config() -> dict:
    return cfg.capture_profile()


def test_default_profile_resolves(capture_config: dict) -> None:
    assert "blur" in capture_config
    assert "exposure" in capture_config
    assert capture_config["face"]["max_faces"] == 1


def test_canonical_crop_is_defined(capture_config: dict) -> None:
    """D11: normalise scale before measuring, instead of parameterising per device."""
    assert capture_config["canonical"]["qc_face_width_px"] > 0


def test_unknown_device_falls_back_to_default(capture_config: dict) -> None:
    """An unrecognised phone is the expected case in V1, not an error."""
    assert cfg.capture_profile("some-unreleased-phone") == capture_config


def test_overrides_start_empty() -> None:
    """D11: a device profile is added only with the validation run that justified it."""
    assert cfg.load("capture_thresholds").get("overrides") == {}


def test_colour_cast_is_a_rejectable_failure() -> None:
    """D4: nothing downstream corrects a cast, so QC must be able to reject one."""
    assert QCFailure.COLOR_CAST in set(QCFailure)


def test_white_balance_is_recorded_but_not_applied(capture_config: dict) -> None:
    """The illumination vector is evidence, not a correction (D4)."""
    assert capture_config["white_balance"]["record_illumination_vector"] is True


def test_device_metadata_is_logged_for_future_profiling() -> None:
    """D11: collect the evidence now so a later profile decision has something behind it."""
    recorded = cfg.load("capture_thresholds")["device_metadata"]["record"]
    assert {"make", "model", "image_width", "image_height"} <= set(recorded)


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
