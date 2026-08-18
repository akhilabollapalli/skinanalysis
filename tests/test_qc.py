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


# ------------------------------------------------------------- partial-region tolerance
#
# Hair coverage and left/right shadow asymmetry used to fail the WHOLE capture -- no
# concern logic ran, for anyone, over a problem that usually affects one region (a hair-
# covered forehead) or one specific comparison (asymmetry needs both sides evenly lit;
# each side's own reading does not). These tests defend the fix: a local problem now stays
# local, and the capture-level safety net still catches a genuinely unusable photo.


def _flat_skin_image(width: int = 200, height: int = 200, value: int = 140) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def _passing_image(
    width: int = 800, height: int = 800, base: int = 130, noise: int = 25, seed: int = 0
) -> np.ndarray:
    """A synthetic capture that clears every OTHER gate: sized above min_face_px, gray
    (zero colour cast), correctly exposed, and textured enough to clear blur.

    Per-pixel independent noise, identical across channels so the image stays neutral
    grey (no cast) while giving the blur check real high-frequency content to measure --
    a flat image has zero laplacian variance and fails blur outright. Values verified
    directly against config/capture_thresholds.yaml before being used here: laplacian_var
    ~1200 (floor 120), tenengrad ~2900 (floor 60), mean_luma 130 (range 60-200), zero
    clipping, zero gray-world deviation.
    """
    rng = np.random.default_rng(seed)
    texture = rng.integers(-noise, noise + 1, size=(height, width))
    channel = np.clip(base + texture, 0, 255).astype(np.uint8)
    return np.stack([channel, channel, channel], axis=-1)


def test_hair_coverage_no_longer_blocks_the_whole_capture(capture_config: dict) -> None:
    """The flagship promise: a hair-covered forehead alone must not veto an otherwise
    good capture. Every OTHER check here clears -- this is a full ``passed is True``,
    not just an absent OCCLUSION reason, because that is the actual claim being made.

    max_hair_frac is exceeded, which USED to be an automatic OCCLUSION failure
    regardless of anything else in the photo.
    """
    hair_frac = capture_config["occlusion"]["max_hair_frac"] + 0.30
    image = _passing_image()
    skin = np.ones(image.shape[:2], dtype=bool)
    face = qc.FaceObservation(
        n_faces=1,
        face_box=(0, 0, image.shape[1], image.shape[0]),
        pose_deg=(0.0, 0.0, 0.0),
        skin_mask=skin,
        roi_visibility={"forehead": 0.1, "left_cheek": 0.95, "right_cheek": 0.95},
        occlusion={"hair": hair_frac, "beard": 0.0, "glasses": 0.0, "specular": 0.0},
    )
    result = qc.check(image, capture_config, face=face)
    assert result.passed is True
    assert result.failures == []
    assert result.metrics["hair_frac"] == pytest.approx(hair_frac)


def test_beard_coverage_still_blocks_the_capture(capture_config: dict) -> None:
    """Only hair was softened. Beard stays a whole-capture block (out of scope here)."""
    beard_frac = capture_config["occlusion"]["max_beard_frac"] + 0.30
    image = _flat_skin_image()
    skin = np.ones(image.shape[:2], dtype=bool)
    face = qc.FaceObservation(
        n_faces=1,
        face_box=(0, 0, image.shape[1], image.shape[0]),
        pose_deg=(0.0, 0.0, 0.0),
        skin_mask=skin,
        roi_visibility={"forehead": 0.95, "left_cheek": 0.95, "right_cheek": 0.95},
        occlusion={"hair": 0.0, "beard": beard_frac, "glasses": 0.0, "specular": 0.0},
    )
    result = qc.check(image, capture_config, face=face)
    assert QCFailure.OCCLUSION in result.failures


def test_glasses_coverage_still_blocks_the_capture(capture_config: dict) -> None:
    glasses_frac = capture_config["occlusion"]["max_glasses_frac"] + 0.30
    image = _flat_skin_image()
    skin = np.ones(image.shape[:2], dtype=bool)
    face = qc.FaceObservation(
        n_faces=1,
        face_box=(0, 0, image.shape[1], image.shape[0]),
        pose_deg=(0.0, 0.0, 0.0),
        skin_mask=skin,
        roi_visibility={"forehead": 0.95, "left_cheek": 0.95, "right_cheek": 0.95},
        occlusion={"hair": 0.0, "beard": 0.0, "glasses": glasses_frac, "specular": 0.0},
    )
    result = qc.check(image, capture_config, face=face)
    assert QCFailure.OCCLUSION in result.failures


def test_one_visible_required_roi_is_enough_to_proceed(capture_config: dict) -> None:
    """Only the WORST required ROI used to matter. Now the BEST one does: the capture
    proceeds if at least one core region survived, and D7 sorts out per-concern
    UNMEASURABLE from there."""
    image = _flat_skin_image()
    skin = np.ones(image.shape[:2], dtype=bool)
    face = qc.FaceObservation(
        n_faces=1,
        face_box=(0, 0, image.shape[1], image.shape[0]),
        pose_deg=(0.0, 0.0, 0.0),
        skin_mask=skin,
        roi_visibility={"forehead": 0.0, "left_cheek": 0.0, "right_cheek": 0.95},
        occlusion={"hair": 0.0, "beard": 0.0, "glasses": 0.0, "specular": 0.0},
    )
    result = qc.check(image, capture_config, face=face)
    assert QCFailure.INSUFFICIENT_ROI_VISIBILITY not in result.failures


def test_no_visible_required_roi_still_blocks(capture_config: dict) -> None:
    """The safety net that remains: if NOTHING core survived, retake is still correct --
    running the full pipeline just to get UNMEASURABLE back from every concern is worse
    than one clean retake message."""
    image = _flat_skin_image()
    skin = np.ones(image.shape[:2], dtype=bool)
    face = qc.FaceObservation(
        n_faces=1,
        face_box=(0, 0, image.shape[1], image.shape[0]),
        pose_deg=(0.0, 0.0, 0.0),
        skin_mask=skin,
        roi_visibility={"forehead": 0.0, "left_cheek": 0.0, "right_cheek": 0.0},
        occlusion={"hair": 0.0, "beard": 0.0, "glasses": 0.0, "specular": 0.0},
    )
    result = qc.check(image, capture_config, face=face)
    assert QCFailure.INSUFFICIENT_ROI_VISIBILITY in result.failures


def test_shadow_asymmetry_alone_does_not_block_the_capture(capture_config: dict) -> None:
    """The second flagship promise: uneven left/right lighting invalidates a CROSS-SIDE
    comparison, not what either side reads on its own. It must suppress the comparison
    (shadow_pass=False) without veto-ing the rest of the scan -- a full ``passed is True``.

    A moderate, gentle left/right gradient on top of real texture: enough to clear
    max_lr_luma_asymmetry on its own without also crushing enough pixels dark to trip the
    still-blocking deep-shadow check (verified below), which a harsher gradient would
    conflate with the thing actually being tested.
    """
    image = _passing_image(base=130, noise=15, seed=1).astype(np.int16)
    width = image.shape[1]
    shift = np.zeros_like(image)
    shift[:, : width // 2] -= 35
    shift[:, width // 2 :] += 35
    image = np.clip(image + shift, 0, 255).astype(np.uint8)

    skin = np.ones(image.shape[:2], dtype=bool)
    face = qc.FaceObservation(
        n_faces=1,
        face_box=(0, 0, image.shape[1], image.shape[0]),
        pose_deg=(0.0, 0.0, 0.0),
        skin_mask=skin,
        roi_visibility={"forehead": 0.95, "left_cheek": 0.95, "right_cheek": 0.95},
        occlusion={"hair": 0.0, "beard": 0.0, "glasses": 0.0, "specular": 0.0},
    )
    result = qc.check(image, capture_config, face=face)
    assert result.metrics["lr_luma_asymmetry"] > capture_config["shadow"]["max_lr_luma_asymmetry"]
    assert result.metrics["deep_shadow_frac"] <= capture_config["shadow"]["max_deep_shadow_frac"]
    assert result.passed is True
    assert result.failures == []
    assert result.shadow_asymmetry_ok is False
    assert result.verdict().shadow_pass is False


def test_symmetric_lighting_leaves_shadow_pass_true(capture_config: dict) -> None:
    image = _flat_skin_image()
    skin = np.ones(image.shape[:2], dtype=bool)
    face = qc.FaceObservation(
        n_faces=1,
        face_box=(0, 0, image.shape[1], image.shape[0]),
        pose_deg=(0.0, 0.0, 0.0),
        skin_mask=skin,
        roi_visibility={"forehead": 0.95, "left_cheek": 0.95, "right_cheek": 0.95},
        occlusion={"hair": 0.0, "beard": 0.0, "glasses": 0.0, "specular": 0.0},
    )
    result = qc.check(image, capture_config, face=face)
    assert result.shadow_asymmetry_ok is True
    assert result.verdict().shadow_pass is True


def test_pervasive_deep_shadow_still_blocks_the_capture(capture_config: dict) -> None:
    """Deep, pervasive shadow is an exposure problem -- a pixel too dark to trust is too
    dark to trust on its own, not just relative to its mirror. Unlike asymmetry, this
    stays a capture-level block (scope cut, disclosed: not made per-ROI in this change)."""
    height, width = 200, 200
    image = np.full((height, width, 3), 200, dtype=np.uint8)
    # Crush most of the frame near-black, leaving a thin bright sliver on the right so a
    # left/right comparison is still computable and does not itself trip first.
    image[:, : width - 10] = 5
    skin = np.ones((height, width), dtype=bool)
    face = qc.FaceObservation(
        n_faces=1,
        face_box=(0, 0, width, height),
        pose_deg=(0.0, 0.0, 0.0),
        skin_mask=skin,
        roi_visibility={"forehead": 0.95, "left_cheek": 0.95, "right_cheek": 0.95},
        occlusion={"hair": 0.0, "beard": 0.0, "glasses": 0.0, "specular": 0.0},
    )
    result = qc.check(image, capture_config, face=face)
    assert QCFailure.SHADOW_ASYMMETRY in result.failures
    assert result.metrics["deep_shadow_frac"] > capture_config["shadow"]["max_deep_shadow_frac"]


def test_capture_qc_verdict_reads_shadow_asymmetry_ok_not_the_failures_list() -> None:
    """Direct guard on the field CaptureQC.verdict() actually consults.

    QCFailure.SHADOW_ASYMMETRY can appear in ``failures`` (from deep shadow) even while
    ``shadow_asymmetry_ok`` is True (the comparison itself was fine) -- verdict() must not
    fall back to inferring shadow_pass from membership in ``failures``, or this exact
    combination would silently suppress a comparison that was never actually unreliable.
    """
    from skin_analysis.schemas import CaptureQC

    result = CaptureQC(
        passed=False,
        failures=[QCFailure.SHADOW_ASYMMETRY],
        shadow_asymmetry_ok=True,
    )
    assert result.verdict().shadow_pass is True
