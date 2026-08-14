"""Definition of done for Stage B.

These tests exist before the implementations do. Several xfail today; each one flipping to
pass is a Stage B milestone, and none may be deleted to make the suite green.

Ordering note (Stage B): polygons are verified with ``scripts/debug_roi.py`` BEFORE the
skin mask is relied on. Three things must be judged independently -- the anatomical
polygon, the semantic skin mask, and their intersection -- because a bad polygon can hide
behind a plausible skin mask and stay hidden until a feature produces a finding nobody can
explain.
"""

from __future__ import annotations

import numpy as np
import pytest

from skin_analysis.face import rois, skin_mask
from skin_analysis.schemas import ROI, RunMode, UnverifiedROIError
from skin_analysis.util import config as cfg
from skin_analysis.util import scale


@pytest.fixture
def roi_config() -> dict:
    return cfg.load("rois")


def _landmarks(anchor_px: float = 240.0, n: int = 478) -> np.ndarray:
    """Synthetic landmark array with a known inter-ocular distance."""
    pts = np.zeros((n, 3), dtype=np.float64)
    if n > scale.RIGHT_IRIS_CENTER:
        pts[scale.LEFT_IRIS_CENTER, :2] = (400.0 - anchor_px / 2, 300.0)
        pts[scale.RIGHT_IRIS_CENTER, :2] = (400.0 + anchor_px / 2, 300.0)
    return pts


# ------------------------------------------------------------------ ROI verification gate


def test_unverified_roi_rejected_in_production(roi_config: dict) -> None:
    """One forgotten polygon silently moves every measurement taken inside it."""
    assert roi_config["meta"]["verified"] is False
    with pytest.raises(UnverifiedROIError):
        rois.assert_verified(roi_config, RunMode.PRODUCTION)


def test_unverified_roi_allowed_in_development(roi_config: dict) -> None:
    """Development mode is how the polygons get defined and inspected in the first place."""
    rois.assert_verified(roi_config, RunMode.DEVELOPMENT)


def test_all_enabled_rois_are_verified(roi_config: dict) -> None:
    """Flips to a real assertion the day meta.verified goes true."""
    if not roi_config["meta"]["verified"]:
        pytest.skip("polygons not yet verified; test_unverified_roi_rejected covers this state")
    for name, body in roi_config["rois"].items():
        assert body.get("landmarks"), f"{name} is marked verified but has no landmark indices"


# ------------------------------------------------------------------ ROI definitions


def test_every_schema_roi_has_a_config_entry(roi_config: dict) -> None:
    assert {r.value for r in ROI} == set(roi_config["rois"])


@pytest.mark.xfail(reason="rois.yaml landmark indices not filled yet", strict=True)
def test_roi_polygon_indices_are_valid_landmarks(roi_config: dict) -> None:
    """Indices must exist in the 478-point model and form an actual polygon."""
    count = roi_config["meta"]["landmark_count"]
    for name, body in roi_config["rois"].items():
        indices = body.get("landmarks", [])
        assert len(indices) >= 3, f"{name}: a polygon needs at least 3 points"
        assert len(set(indices)) == len(indices), f"{name}: duplicate landmark index"
        assert all(0 <= i < count for i in indices), f"{name}: index outside the model"


@pytest.mark.xfail(reason="face.rois.build not implemented yet", strict=True)
def test_roi_has_minimum_area(roi_config: dict) -> None:
    """An eroded polygon that collapses to a sliver produces noise, not a measurement."""
    built = rois.build(_landmarks(), (720, 800), roi_config, run_mode=RunMode.DEVELOPMENT)
    for name, mask in built.items():
        assert mask.sum() > 0, f"{name}: empty after erosion"


@pytest.mark.xfail(reason="face.skin_mask.build not implemented yet", strict=True)
def test_skin_mask_does_not_include_background() -> None:
    """The mask is the weakest link in commercial-open V1; background leakage is the
    failure that most often looks like a real skin finding."""
    image = np.zeros((720, 800, 3), dtype=np.uint8)
    mask = skin_mask.build(image, _landmarks(), cfg.load("rois"))
    assert mask.dtype == np.bool_
    assert not mask[:10, :].any(), "top image border classified as skin"
    assert not mask[-10:, :].any(), "bottom image border classified as skin"


@pytest.mark.xfail(reason="face.rois.compose not implemented yet", strict=True)
def test_intersection_is_never_larger_than_the_polygon() -> None:
    polygons = {"forehead": np.ones((10, 10), dtype=bool)}
    mask = np.zeros((10, 10), dtype=bool)
    mask[:5] = True
    composed = rois.compose(polygons, mask)
    assert composed["forehead"].sum() <= polygons["forehead"].sum()


# ------------------------------------------------------------------ scale anchor (D1)


def test_anchor_px_positive() -> None:
    assert scale.inter_ocular_distance(_landmarks(240.0)) > 0


def test_anchor_scales_with_face_size() -> None:
    """A face twice as large must yield twice the anchor, or every fraction-of-anchor
    parameter means something different at different distances."""
    small = scale.inter_ocular_distance(_landmarks(120.0))
    large = scale.inter_ocular_distance(_landmarks(240.0))
    assert large == pytest.approx(2 * small)


def test_anchor_requires_the_478_point_model() -> None:
    """The 468-point surface mesh has no iris landmarks, so it cannot anchor scale."""
    with pytest.raises(ValueError):
        scale.inter_ocular_distance(_landmarks(n=468))


def test_window_resolution_tracks_the_anchor() -> None:
    frac = cfg.concern_config("redness")["local"]["window_frac_of_iod"]
    small = scale.to_px(frac, 120.0, odd=True)
    large = scale.to_px(frac, 240.0, odd=True)
    assert large > small
    assert small % 2 == 1 and large % 2 == 1, "symmetric windows need a centre pixel"


def test_anchor_floor_rejects_tiny_captures() -> None:
    config = cfg.load("severity_thresholds")
    floor = config["scale"]["min_anchor_px"]
    assert not scale.anchor_is_sufficient(floor - 1, config)
    assert scale.anchor_is_sufficient(floor, config)
