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

from skin_analysis import pipeline
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


# ROI polygon validity and minimum area moved to tests/test_roi_construction.py, which
# has a landmark fixture that places the anchors the derived constructions need. The
# assertions there are stronger: they are schema-aware (polygon vs derived) and they check
# handedness and anchor-proportional scaling as well.


def test_skin_mask_does_not_include_background() -> None:
    """The mask is the weakest link in commercial-open V1; background leakage is the
    failure that most often looks like a real skin finding."""
    image = np.zeros((720, 800, 3), dtype=np.uint8)
    mask = skin_mask.build(image, _landmarks(), cfg.load("rois"))
    assert mask.dtype == np.bool_
    assert not mask[:10, :].any(), "top image border classified as skin"
    assert not mask[-10:, :].any(), "bottom image border classified as skin"


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


# ---------------------------------------------------------- unusable-ROI exclusion (D7)


def _profile_with_visibility_floor(floor: float) -> dict:
    profile = cfg.capture_profile()
    return {**profile, "roi_visibility": {**profile["roi_visibility"], "min_visible_frac": floor}}


def test_low_visibility_roi_is_zeroed_but_others_survive() -> None:
    """The mechanism the hair/shadow partial-degradation fix actually relies on: a low-
    visibility ROI is excluded from measurement, ROIs above the floor are left untouched."""
    shape = (300, 300)  # comfortably above every ROI's min_area_frac_of_iod2 at anchor=240
    mask = np.ones(shape, dtype=bool)
    composed = {
        "forehead": np.zeros(shape, dtype=bool),  # already fully occluded
        "left_cheek": mask.copy(),
        "right_cheek": mask.copy(),
    }
    composed["forehead"][:5, :5] = True  # a sliver: low visibility, not literally empty
    visibility = {"forehead": 0.05, "left_cheek": 0.95, "right_cheek": 0.95}
    roi_cfg = cfg.load("rois")

    pipeline._exclude_unusable_rois(
        composed, visibility, mask, anchor_px=240.0, roi_cfg=roi_cfg,
        profile=_profile_with_visibility_floor(0.60),
    )

    assert not composed["forehead"].any()
    assert visibility["forehead"] == 0.0
    assert composed["left_cheek"].all()
    assert composed["right_cheek"].all()
    assert visibility["left_cheek"] == 0.95
    assert visibility["right_cheek"] == 0.95


def test_visibility_floor_is_read_from_the_profile_not_hardcoded() -> None:
    """CLAUDE.md §4: no magic numbers in code. Changing the configured floor must change
    which ROIs get excluded, proving the value actually came from config."""
    shape = (300, 300)  # comfortably above every ROI's min_area_frac_of_iod2 at anchor=240
    mask = np.ones(shape, dtype=bool)
    roi_cfg = cfg.load("rois")

    composed_strict = {"left_cheek": mask.copy()}
    visibility_strict = {"left_cheek": 0.50}
    pipeline._exclude_unusable_rois(
        composed_strict, visibility_strict, mask, 240.0, roi_cfg,
        _profile_with_visibility_floor(0.60),
    )
    assert not composed_strict["left_cheek"].any()

    composed_loose = {"left_cheek": mask.copy()}
    visibility_loose = {"left_cheek": 0.50}
    pipeline._exclude_unusable_rois(
        composed_loose, visibility_loose, mask, 240.0, roi_cfg,
        _profile_with_visibility_floor(0.40),
    )
    assert composed_loose["left_cheek"].all()


def test_undersized_by_area_is_still_excluded_independent_of_visibility_fraction() -> None:
    """The pre-existing D7 area floor and the visibility-fraction floor are two different
    reasons for the SAME treatment; fixing one path must not have quietly dropped the
    other. A tiny composed ROI can have a high visibility FRACTION (its polygon eroded to
    almost nothing, but skin-masking cost it none of that) while still being far too few
    pixels to trust."""
    shape = (300, 300)  # comfortably above every ROI's min_area_frac_of_iod2 at anchor=240
    mask = np.ones(shape, dtype=bool)
    composed = {"left_crows_feet": np.zeros(shape, dtype=bool)}
    composed["left_crows_feet"][:2, :2] = True  # 4px: far below any real area floor
    visibility = {"left_crows_feet": 1.0}  # the polygon itself was already this small
    roi_cfg = cfg.load("rois")

    pipeline._exclude_unusable_rois(
        composed, visibility, mask, anchor_px=240.0, roi_cfg=roi_cfg,
        profile=_profile_with_visibility_floor(0.0),  # visibility floor disabled
    )

    assert not composed["left_crows_feet"].any()
    assert visibility["left_crows_feet"] == 0.0
