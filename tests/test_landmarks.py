"""Face Landmarker wrapper tests (Stage B1).

No real face images: the repo may not contain them (CLAUDE.md §5). What CAN be tested
without one, and is tested here:

* the model-integrity gate, which is a licensing control as much as a correctness one;
* the no-face and multi-face paths, which decide whether a capture is analyzed at all;
* the pose solver's decomposition, on landmarks projected from a known rotation.

Detection quality on real faces is not a unit test. It is verified through
``scripts/debug_roi.py`` and the release validation corpus (D13).
"""

from __future__ import annotations

import numpy as np
import pytest

from skin_analysis.face import landmarks as lm
from skin_analysis.util import config as cfg

mediapipe = pytest.importorskip("mediapipe")


@pytest.fixture(scope="module")
def profile() -> dict:
    return cfg.capture_profile()


def _blank(h: int = 720, w: int = 720) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _projected(yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0,
               size: int = 800) -> np.ndarray:
    """Landmarks obtained by projecting the canonical rigid model under a known rotation.

    This exercises the part of :func:`head_pose` that actually breaks -- the Rodrigues to
    Euler decomposition, the axis conventions, and the pitch wrap -- rather than the model
    points themselves, which it shares by construction.
    """
    rx, ry, rz = np.radians([pitch, yaw, roll])
    r_x = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    r_y = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    r_z = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])

    points = (r_z @ r_y @ r_x @ lm._POSE_MODEL_POINTS.T).T + np.array([0.0, 0.0, 600.0])
    focal = float(size)
    projected = np.column_stack([
        points[:, 0] * focal / points[:, 2] + size / 2,
        points[:, 1] * focal / points[:, 2] + size / 2,
    ])

    out = np.zeros((lm.EXPECTED_LANDMARK_COUNT, 3))
    for i, index in enumerate(lm._POSE_LANDMARKS):
        out[index, :2] = projected[i]
    return out


# ------------------------------------------------------------------ model integrity


def test_pinned_model_matches_its_hash() -> None:
    """The shipped bundle must be the artifact that went through the licensing audit."""
    assert lm.verify_model() == lm.MODEL_SHA256


def test_missing_model_fails_closed(tmp_path) -> None:
    with pytest.raises(lm.ModelIntegrityError, match="not found"):
        lm.verify_model(tmp_path / "absent.task")


def test_wrong_model_hash_is_refused(tmp_path) -> None:
    """A silent bundle swap would move every ROI without changing a line of code."""
    impostor = tmp_path / "face_landmarker.task"
    impostor.write_bytes(b"not the audited bundle")
    with pytest.raises(lm.ModelIntegrityError, match="has not been through the licensing audit"):
        lm.verify_model(impostor)


def test_model_hash_matches_the_manifest() -> None:
    """Code and manifest must not drift apart; the manifest is the auditable record."""
    import csv
    from pathlib import Path

    manifest = Path(__file__).resolve().parents[1] / "LICENSES" / "asset_manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as fh:
        weights = [r for r in csv.DictReader(fh) if r["asset"].startswith("MediaPipe Face")
                   or r["asset"].startswith("MediaPipe Blaze")
                   or r["asset"].startswith("MediaPipe Blend")]
    assert weights, "MediaPipe weight rows missing from the manifest"
    for row in weights:
        assert row["sha256"] == lm.MODEL_SHA256, (
            f"{row['asset']}: manifest sha256 differs from landmarks.MODEL_SHA256"
        )


# ------------------------------------------------------------------ detection paths


def test_no_face_returns_none(profile: dict) -> None:
    assert lm.detect(_blank(), profile) is None


def test_no_face_returns_empty_list(profile: dict) -> None:
    assert lm.detect_faces(_blank(), profile) == []


def test_noise_is_not_a_face(profile: dict) -> None:
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (720, 720, 3), dtype=np.uint8)
    assert lm.detect_faces(noise, profile) == []


def test_non_bgr_input_is_rejected(profile: dict) -> None:
    with pytest.raises(ValueError, match="HxWx3"):
        lm.detect_faces(np.zeros((720, 720), dtype=np.uint8), profile)


def test_detector_looks_for_one_more_face_than_qc_allows(
    profile: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requesting exactly max_faces would make a two-person photo look like a one-person
    photo, and the second person would be silently cropped out instead of rejected."""
    requested: list[int] = []
    real = lm._landmarker

    def spy(num_faces: int):
        requested.append(num_faces)
        return real(num_faces)

    monkeypatch.setattr(lm, "_landmarker", spy)
    lm.detect_faces(_blank(), profile)
    assert requested == [profile["face"]["max_faces"] + 1]


def test_geometry_returns_none_without_a_face(profile: dict) -> None:
    assert lm.geometry(_blank(), profile) is None


# ------------------------------------------------------------------ head pose


def test_frontal_face_has_near_zero_pose() -> None:
    yaw, pitch, roll = lm.head_pose(_projected(), (800, 800))
    assert abs(yaw) < 0.5 and abs(pitch) < 0.5 and abs(roll) < 0.5


@pytest.mark.parametrize("yaw", [-25.0, -10.0, 10.0, 25.0])
def test_yaw_is_recovered(yaw: float) -> None:
    assert lm.head_pose(_projected(yaw=yaw), (800, 800))[0] == pytest.approx(yaw, abs=0.5)


@pytest.mark.parametrize("pitch", [-15.0, 15.0])
def test_pitch_is_recovered_without_wrap_artifacts(pitch: float) -> None:
    """solvePnP reports pitch near +/-180 for a forward-facing head; the wrap must not
    turn a compliant capture into an EXTREME_POSE rejection."""
    assert lm.head_pose(_projected(pitch=pitch), (800, 800))[1] == pytest.approx(pitch, abs=0.5)


@pytest.mark.parametrize("roll", [-10.0, 10.0])
def test_roll_is_recovered(roll: float) -> None:
    assert lm.head_pose(_projected(roll=roll), (800, 800))[2] == pytest.approx(roll, abs=0.5)


def test_combined_rotation_does_not_bleed_between_axes() -> None:
    yaw, pitch, roll = lm.head_pose(_projected(yaw=10.0, pitch=10.0), (800, 800))
    assert yaw == pytest.approx(10.0, abs=0.5)
    assert pitch == pytest.approx(10.0, abs=0.5)
    assert roll == pytest.approx(0.0, abs=0.5)


def test_pose_thresholds_are_expressible_from_config() -> None:
    """The gate this feeds is a coarse threshold, not a measurement."""
    pose = cfg.capture_profile()["pose"]
    assert pose["max_yaw_deg"] > 0 and pose["max_pitch_deg"] > 0 and pose["max_roll_deg"] > 0


def test_head_pose_requires_the_478_point_model() -> None:
    with pytest.raises(ValueError, match="478"):
        lm.head_pose(np.zeros((468, 3)), (800, 800))


# ------------------------------------------------------------------ determinism (D13)


def test_detection_is_deterministic(profile: dict) -> None:
    """The floor beneath repeatability, and the only half of it CI can check."""
    image = _blank()
    assert lm.detect_faces(image, profile) == lm.detect_faces(image, profile)


def test_pose_is_deterministic() -> None:
    points = _projected(yaw=12.0, pitch=-7.0, roll=3.0)
    assert lm.head_pose(points, (800, 800)) == lm.head_pose(points, (800, 800))
