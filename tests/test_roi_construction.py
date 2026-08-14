"""Derived-ROI construction tests (Stage B2).

Three ROIs have no FaceMesh vertices and are built from anchors plus fractional offsets:
forehead, crow's feet, nasolabial. Those constructions are pure geometry, so they are
fully testable here -- placement relative to their anchors, handedness, and scale
proportionality.

What is NOT tested here, and cannot be: whether the resulting polygons land on the right
anatomy of a real face. That is what ``scripts/debug_roi.py`` and B3 verification are for,
and it is why ``meta.verified`` gates production (D15).
"""

from __future__ import annotations

import numpy as np
import pytest

from skin_analysis.face import rois
from skin_analysis.schemas import ROI, RunMode
from skin_analysis.util import config as cfg
from skin_analysis.util import scale

IOD = 240.0
MIDLINE = 400.0
EYE_Y = 300.0


@pytest.fixture(scope="module")
def roi_config() -> dict:
    return cfg.load("rois")


@pytest.fixture
def landmarks(roi_config: dict) -> np.ndarray:
    """A synthetic 478-point array with anatomically plausible ANCHOR positions.

    Only the anchors the derived constructions use are placed meaningfully; the polygon
    ROIs get generic positions, because their correctness is a visual question rather than
    an arithmetic one.

    Anatomical handedness: the subject's LEFT is at HIGHER image x.
    """
    pts = np.zeros((478, 3), dtype=np.float64)
    pts[:, 0] = MIDLINE
    pts[:, 1] = EYE_Y

    pts[scale.LEFT_IRIS_CENTER, :2] = (MIDLINE + IOD / 2, EYE_Y)
    pts[scale.RIGHT_IRIS_CENTER, :2] = (MIDLINE - IOD / 2, EYE_Y)

    # Exclusion rings first. Left at the array default they would collapse to the face
    # centre, and the resulting lips polygon would be a large triangle swallowing the
    # nasolabial bands -- a fixture artifact that looks exactly like a real over-aggressive
    # exclusion. Placed BEFORE the named anchors because the two sets overlap: index 263 is
    # both `left_eye_outer` and a member of the left eye ring, so whichever runs last wins.
    def _ellipse(indices: list[int], cx: float, cy: float, rx: float, ry: float) -> None:
        for i, index in enumerate(indices):
            theta = 2 * np.pi * i / len(indices)
            pts[index, :2] = (cx + rx * np.cos(theta), cy + ry * np.sin(theta))

    rings = roi_config["exclusion_rings"]
    _ellipse(rings["left_eye"], MIDLINE + 0.55 * IOD, EYE_Y, 0.24 * IOD, 0.13 * IOD)
    _ellipse(rings["right_eye"], MIDLINE - 0.55 * IOD, EYE_Y, 0.24 * IOD, 0.13 * IOD)
    _ellipse(rings["lips"], MIDLINE, EYE_Y + 1.05 * IOD, 0.28 * IOD, 0.13 * IOD)

    anchors = roi_config["anchors"]
    brow_y = EYE_Y - 0.25 * IOD
    for i, index in enumerate(anchors["left_brow"]):      # medial -> lateral
        pts[index, :2] = (MIDLINE + 0.12 * IOD + i * 0.10 * IOD, brow_y)
    for i, index in enumerate(anchors["right_brow"]):
        pts[index, :2] = (MIDLINE - 0.12 * IOD - i * 0.10 * IOD, brow_y)

    pts[anchors["left_eye_outer"], :2] = (MIDLINE + 0.80 * IOD, EYE_Y)
    pts[anchors["right_eye_outer"], :2] = (MIDLINE - 0.80 * IOD, EYE_Y)

    pts[anchors["left_alar"], :2] = (MIDLINE + 0.18 * IOD, EYE_Y + 0.62 * IOD)
    pts[anchors["right_alar"], :2] = (MIDLINE - 0.18 * IOD, EYE_Y + 0.62 * IOD)
    pts[anchors["left_mouth_corner"], :2] = (MIDLINE + 0.30 * IOD, EYE_Y + 1.05 * IOD)
    pts[anchors["right_mouth_corner"], :2] = (MIDLINE - 0.30 * IOD, EYE_Y + 1.05 * IOD)

    return pts


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


# ------------------------------------------------------------------ schema


def test_every_schema_roi_has_a_config_entry(roi_config: dict) -> None:
    assert {r.value for r in ROI} == set(roi_config["rois"])


def test_every_roi_declares_a_known_type(roi_config: dict) -> None:
    for name, spec in roi_config["rois"].items():
        kind = spec.get("type")
        assert kind in {"polygon", "derived"}, f"{name}: bad type {kind!r}"
        if kind == "derived":
            assert spec["construction"] in rois._CONSTRUCTIONS, (
                f"{name}: unknown construction {spec['construction']!r}"
            )
        else:
            assert len(spec["landmarks"]) >= 3, f"{name}: a polygon needs 3+ points"


def test_polygon_indices_are_within_the_model(roi_config: dict) -> None:
    count = roi_config["meta"]["landmark_count"]
    for name, spec in roi_config["rois"].items():
        for index in spec.get("landmarks", []):
            assert 0 <= index < count, f"{name}: index {index} outside the 478-point model"


def test_polygon_indices_are_not_duplicated(roi_config: dict) -> None:
    for name, spec in roi_config["rois"].items():
        indices = spec.get("landmarks", [])
        assert len(set(indices)) == len(indices), f"{name}: duplicate landmark index"


def test_derived_offsets_are_anchor_fractions_not_pixels(roi_config: dict) -> None:
    """D1: a pixel offset means different things at two resolutions."""
    for name, spec in roi_config["rois"].items():
        offenders = [k for k in spec if k.endswith("_px")]
        assert not offenders, f"{name}: pixel-valued parameters {offenders}"


def test_erosion_is_anchored_to_iod_not_face_height(roi_config: dict) -> None:
    """Face height varies far more with pitch than inter-ocular distance does, so a
    face-height anchor would scale ROI erosion and feature windows differently under the
    same head tilt."""
    assert "margin_frac_of_iod" in roi_config["erosion"]
    assert "margin_frac_of_face_height" not in roi_config["erosion"]


def test_handedness_is_declared_anatomical(roi_config: dict) -> None:
    assert roi_config["meta"]["handedness"] == "anatomical"


# ------------------------------------------------------------------ forehead


def test_forehead_sits_above_the_brow(landmarks: np.ndarray, roi_config: dict) -> None:
    spec = roi_config["rois"]["forehead"]
    ring = rois._construct_extrude_from_brow(landmarks, spec, roi_config, IOD)
    brow_y = landmarks[roi_config["anchors"]["left_brow"][0], 1]
    assert ring[:, 1].max() < brow_y, "forehead polygon overlaps the brow line"


def test_forehead_clears_the_brow_by_the_configured_margin(
    landmarks: np.ndarray, roi_config: dict
) -> None:
    spec = roi_config["rois"]["forehead"]
    ring = rois._construct_extrude_from_brow(landmarks, spec, roi_config, IOD)
    brow_y = landmarks[roi_config["anchors"]["left_brow"][0], 1]
    clearance = brow_y - ring[:, 1].max()
    assert clearance == pytest.approx(spec["brow_clearance_frac_of_iod"] * IOD, rel=0.02)


def test_forehead_height_scales_with_the_anchor(
    landmarks: np.ndarray, roi_config: dict
) -> None:
    """A forehead fixed in pixels would swallow hair on a close capture."""
    spec = roi_config["rois"]["forehead"]
    small = rois._construct_extrude_from_brow(landmarks, spec, roi_config, IOD)
    large = rois._construct_extrude_from_brow(landmarks, spec, roi_config, 2 * IOD)
    span = lambda r: r[:, 1].max() - r[:, 1].min()  # noqa: E731
    assert span(large) == pytest.approx(2 * span(small), rel=0.02)


# ------------------------------------------------------------------ crow's feet


def test_crows_feet_are_lateral_to_their_eye_corner(
    landmarks: np.ndarray, roi_config: dict
) -> None:
    """'Lateral' is resolved against the face midline, so a config typo cannot mirror it."""
    left = rois._construct_lateral_patch(
        landmarks, roi_config["rois"]["left_crows_feet"], roi_config, IOD
    )
    right = rois._construct_lateral_patch(
        landmarks, roi_config["rois"]["right_crows_feet"], roi_config, IOD
    )
    # Subject's left is at higher image x, so its patch must sit further right.
    assert left[:, 0].min() > MIDLINE
    assert right[:, 0].max() < MIDLINE


def test_crows_feet_clear_the_lash_line(landmarks: np.ndarray, roi_config: dict) -> None:
    spec = roi_config["rois"]["left_crows_feet"]
    ring = rois._construct_lateral_patch(landmarks, spec, roi_config, IOD)
    corner_x = landmarks[roi_config["anchors"]["left_eye_outer"], 0]
    gap = ring[:, 0].min() - corner_x
    assert gap == pytest.approx(spec["offset_frac_of_iod"] * IOD, rel=0.02)


def test_crows_feet_pair_is_symmetric(landmarks: np.ndarray, roi_config: dict) -> None:
    """Unequal paired ROIs manufacture asymmetry findings that look like real differences."""
    left = rois._construct_lateral_patch(
        landmarks, roi_config["rois"]["left_crows_feet"], roi_config, IOD
    )
    right = rois._construct_lateral_patch(
        landmarks, roi_config["rois"]["right_crows_feet"], roi_config, IOD
    )
    width = lambda r: r[:, 0].max() - r[:, 0].min()  # noqa: E731
    assert width(left) == pytest.approx(width(right), rel=1e-6)
    assert abs(left[:, 0].mean() - MIDLINE) == pytest.approx(
        abs(right[:, 0].mean() - MIDLINE), rel=1e-6
    )


# ------------------------------------------------------------------ nasolabial


def test_nasolabial_band_spans_alar_to_mouth_corner(
    landmarks: np.ndarray, roi_config: dict
) -> None:
    spec = roi_config["rois"]["left_nasolabial"]
    ring = rois._construct_band_between(landmarks, spec, roi_config, IOD)
    alar = landmarks[roi_config["anchors"]["left_alar"], :2]
    mouth = landmarks[roi_config["anchors"]["left_mouth_corner"], :2]
    assert ring[:, 1].min() <= alar[1] <= ring[:, 1].max()
    assert ring[:, 1].min() <= mouth[1] <= ring[:, 1].max()


def test_nasolabial_band_width_matches_config(
    landmarks: np.ndarray, roi_config: dict
) -> None:
    spec = roi_config["rois"]["left_nasolabial"]
    ring = rois._construct_band_between(landmarks, spec, roi_config, IOD)
    # Opposite corners of the band are one width apart across the axis.
    width = float(np.linalg.norm(ring[0] - ring[3]))
    assert width == pytest.approx(spec["width_frac_of_iod"] * IOD, rel=0.02)


def test_degenerate_band_returns_no_polygon(
    landmarks: np.ndarray, roi_config: dict
) -> None:
    """Coincident anchors must not produce a normal vector from a zero-length axis."""
    spec = dict(roi_config["rois"]["left_nasolabial"])
    collapsed = landmarks.copy()
    collapsed[roi_config["anchors"]["left_mouth_corner"], :2] = collapsed[
        roi_config["anchors"]["left_alar"], :2
    ]
    assert rois._construct_band_between(collapsed, spec, roi_config, IOD).shape[0] == 0


# ------------------------------------------------------------------ build pipeline


def test_build_refuses_unverified_polygons_in_production(
    landmarks: np.ndarray, roi_config: dict
) -> None:
    from skin_analysis.schemas import UnverifiedROIError

    with pytest.raises(UnverifiedROIError):
        rois.build(landmarks, (720, 800), roi_config, run_mode=RunMode.PRODUCTION)


def test_build_produces_a_mask_per_configured_roi(
    landmarks: np.ndarray, roi_config: dict
) -> None:
    built = rois.build(landmarks, (720, 800), roi_config, run_mode=RunMode.DEVELOPMENT)
    assert set(built) == set(roi_config["rois"])
    for name, mask in built.items():
        assert mask.dtype == np.bool_, name
        assert mask.shape == (720, 800), name


def test_derived_rois_are_non_empty(landmarks: np.ndarray, roi_config: dict) -> None:
    """The polygon ROIs use generic synthetic positions, but the derived ones are built
    from real anchor geometry and must survive erosion."""
    built = rois.build(landmarks, (720, 800), roi_config, run_mode=RunMode.DEVELOPMENT)
    for name in ("forehead", "left_crows_feet", "right_crows_feet",
                 "left_nasolabial", "right_nasolabial"):
        assert built[name].sum() > 0, f"{name} collapsed to nothing after erosion"


def test_paired_derived_rois_have_equal_area(
    landmarks: np.ndarray, roi_config: dict
) -> None:
    built = rois.build(landmarks, (720, 800), roi_config, run_mode=RunMode.DEVELOPMENT)
    for left, right in [("left_crows_feet", "right_crows_feet"),
                        ("left_nasolabial", "right_nasolabial")]:
        a, b = built[left].sum(), built[right].sum()
        assert abs(int(a) - int(b)) <= max(2, 0.02 * max(a, b)), f"{left}/{right} differ"


def test_left_roi_is_on_the_image_right(landmarks: np.ndarray, roi_config: dict) -> None:
    """The handedness check that would otherwise only fail on a real face."""
    built = rois.build(landmarks, (720, 800), roi_config, run_mode=RunMode.DEVELOPMENT)
    left_x, _ = _centroid(built["left_crows_feet"])
    right_x, _ = _centroid(built["right_crows_feet"])
    assert left_x > MIDLINE > right_x


def test_build_is_deterministic(landmarks: np.ndarray, roi_config: dict) -> None:
    first = rois.build(landmarks, (720, 800), roi_config, run_mode=RunMode.DEVELOPMENT)
    second = rois.build(landmarks, (720, 800), roi_config, run_mode=RunMode.DEVELOPMENT)
    for name in first:
        assert np.array_equal(first[name], second[name]), name


# ------------------------------------------------------------------ compose


def test_compose_never_exceeds_the_polygon(
    landmarks: np.ndarray, roi_config: dict
) -> None:
    built = rois.build(landmarks, (720, 800), roi_config, run_mode=RunMode.DEVELOPMENT)
    skin = np.zeros((720, 800), dtype=bool)
    skin[:400] = True
    composed = rois.compose(built, skin)
    for name in built:
        assert composed[name].sum() <= built[name].sum(), name


def test_measurable_fraction_is_a_ratio(landmarks: np.ndarray, roi_config: dict) -> None:
    built = rois.build(landmarks, (720, 800), roi_config, run_mode=RunMode.DEVELOPMENT)
    skin = np.ones((720, 800), dtype=bool)
    fractions = rois.measurable_fraction(built, rois.compose(built, skin))
    for name, value in fractions.items():
        assert 0.0 <= value <= 1.0, f"{name}: {value}"
    assert fractions["forehead"] == pytest.approx(1.0)


def test_measurable_fraction_of_an_empty_polygon_is_zero() -> None:
    """Division by an empty polygon must not raise, and must not report full coverage."""
    empty = {"x": np.zeros((10, 10), dtype=bool)}
    assert rois.measurable_fraction(empty, empty)["x"] == 0.0


def test_undersized_rois_are_reported_not_scored(
    landmarks: np.ndarray, roi_config: dict
) -> None:
    """A sliver still produces numbers, and those numbers look like a measurement."""
    built = rois.build(landmarks, (720, 800), roi_config, run_mode=RunMode.DEVELOPMENT)
    assert rois.undersized(built, IOD, roi_config) != list(built), "everything undersized"

    sliver = {"forehead": np.zeros((720, 800), dtype=bool)}
    sliver["forehead"][0, :3] = True
    assert rois.undersized(sliver, IOD, roi_config) == ["forehead"]


def test_min_area_floor_scales_with_the_anchor(roi_config: dict) -> None:
    """A pixel-valued floor would call a close-up sliver a valid ROI."""
    mask = {"x": np.zeros((720, 800), dtype=bool)}
    mask["x"][:60, :60] = True          # 3600 px
    assert rois.undersized(mask, 240.0, roi_config) == []      # floor = 0.02*240^2 = 1152
    assert rois.undersized(mask, 600.0, roi_config) == ["x"]   # floor = 0.02*600^2 = 7200
