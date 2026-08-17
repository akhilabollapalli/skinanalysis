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


def test_native_face_floor_is_at_least_the_canonical_width(capture_config: dict) -> None:
    """Otherwise canonical normalisation silently does not happen.

    A face between the two floors clears the size gate, but ``canonical_crop`` refuses to
    upsample it -- correctly, since measurements must trace back to real sensor pixels. Its
    blur is then measured on a narrower crop than the threshold was set on. Found on a real
    2316x3088 capture whose subject stood a bit further back: a 617px face box scored
    against a cutoff calibrated for 768px.
    """
    assert (
        capture_config["face"]["min_face_px"]
        >= capture_config["canonical"]["qc_face_width_px"]
    )


def test_canonical_crop_never_upsamples(capture_config: dict) -> None:
    """Upsampling would measure the interpolator, not the skin (CLAUDE.md §6)."""
    small = np.full((300, 240, 3), 128, dtype=np.uint8)
    crop = qc.canonical_crop(small, (0, 0, 240, 300), capture_config)
    assert crop.shape[1] <= 240


def test_canonical_crop_downsamples_to_target(capture_config: dict) -> None:
    target = capture_config["canonical"]["qc_face_width_px"]
    big = np.full((2000, 1600, 3), 128, dtype=np.uint8)
    crop = qc.canonical_crop(big, (0, 0, 1600, 2000), capture_config)
    assert crop.shape[1] == target


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


def test_blank_image_is_rejected(capture_config: dict) -> None:
    blank = np.zeros((720, 720, 3), dtype=np.uint8)
    result = qc.check(blank, capture_config)
    assert result.passed is False
    assert result.failures


def test_failure_reasons_are_public_but_metrics_are_not(capture_config: dict) -> None:
    blank = np.zeros((720, 720, 3), dtype=np.uint8)
    public = qc.check(blank, capture_config).to_public()
    assert set(public.keys()) == {"pass", "reasons"}


def test_no_face_is_a_rejection_not_a_default(capture_config: dict) -> None:
    """A frame with no detectable face has nothing to measure. It is not a pass."""
    frame = np.full((720, 720, 3), 128, dtype=np.uint8)
    result = qc.check(frame, capture_config, face=None)
    assert result.passed is False
    assert QCFailure.NO_FACE in result.failures


def test_precheck_does_not_stand_in_for_the_full_gate(capture_config: dict) -> None:
    """precheck exists for latency (D10). Passing it is not passing QC."""
    blank = np.zeros((720, 720, 3), dtype=np.uint8)
    assert qc.precheck(blank, capture_config).passed is False


def test_metrics_stay_out_of_the_public_payload(capture_config: dict) -> None:
    """Rule 3: internal numbers are logged, never rendered."""
    blank = np.zeros((720, 720, 3), dtype=np.uint8)
    result = qc.check(blank, capture_config)
    assert result.metrics, "QC must record metrics internally for validation"
    assert "metrics" not in result.to_public()
    assert "illumination_vector" not in result.to_public()


def test_illumination_vector_is_recorded(capture_config: dict) -> None:
    """D4: recorded as evidence, never applied to the pixels."""
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[..., 2] = 200  # a strong red cast
    result = qc.check(frame, capture_config, face=None)
    assert set(result.illumination_vector) == {"r", "g", "b"}
    assert QCFailure.COLOR_CAST in result.failures
