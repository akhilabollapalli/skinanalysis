"""Stage B6 -- the scale anchor actually works across face sizes.

This is the test D1 rests on. Every spatial parameter in this project is a fraction of
inter-ocular distance rather than a pixel count, and the entire justification is that a
measurement then means the same thing at two capture distances. If that is false, the
frozen cohort statistics of D1 stage B do not transfer between devices, and every severity
band inherits the error.

So the property under test is not "the code runs at two resolutions". It is:

    the same face, captured at two scales, produces the same measurement

Synthetic geometry only. No real face image is committed to this repository (CLAUDE.md §5),
and the arithmetic being checked here is scale proportionality, which synthetic landmarks
exercise exactly as well as real ones.
"""

from __future__ import annotations

import numpy as np
import pytest

from skin_analysis import pipeline
from skin_analysis.face import rois
from skin_analysis.features import pigmentation, redness, texture, wrinkles
from skin_analysis.schemas import FeatureContext, ImageCopy, QCVerdict, RunMode
from skin_analysis.util import config as cfg
from skin_analysis.util import scale

#: Two capture scales in a ratio a real product actually spans: arm's length versus a
#: closer selfie, or a 1x versus a 2x sensor crop.
SMALL_IOD = 130.0
LARGE_IOD = 260.0

MIDLINE_FRAC = 1.6
EYE_Y_FRAC = 1.2


@pytest.fixture(scope="module")
def roi_config() -> dict:
    return cfg.load("rois")


def _landmarks(iod: float, roi_config: dict) -> np.ndarray:
    """A synthetic face whose every landmark is placed as a multiple of ``iod``.

    Built this way on purpose: it is the ONLY construction under which the invariance claim
    is even meaningful. A fixture with any absolute offset in it would encode a face that
    changes shape with distance, and the test would then be measuring the fixture.
    """
    midline = MIDLINE_FRAC * iod
    eye_y = EYE_Y_FRAC * iod
    pts = np.zeros((478, 3), dtype=np.float64)
    pts[:, 0] = midline
    pts[:, 1] = eye_y

    pts[scale.LEFT_IRIS_CENTER, :2] = (midline + iod / 2, eye_y)
    pts[scale.RIGHT_IRIS_CENTER, :2] = (midline - iod / 2, eye_y)

    def ellipse(indices: list[int], cx: float, cy: float, rx: float, ry: float) -> None:
        for i, index in enumerate(indices):
            theta = 2 * np.pi * i / len(indices)
            pts[index, :2] = (cx + rx * np.cos(theta), cy + ry * np.sin(theta))

    rings = roi_config["exclusion_rings"]
    ellipse(rings["left_eye"], midline + 0.55 * iod, eye_y, 0.24 * iod, 0.13 * iod)
    ellipse(rings["right_eye"], midline - 0.55 * iod, eye_y, 0.24 * iod, 0.13 * iod)
    ellipse(rings["lips"], midline, eye_y + 1.05 * iod, 0.28 * iod, 0.13 * iod)

    anchors = roi_config["anchors"]
    brow_y = eye_y - 0.25 * iod
    for i, index in enumerate(anchors["left_brow"]):
        pts[index, :2] = (midline + 0.12 * iod + i * 0.10 * iod, brow_y)
    for i, index in enumerate(anchors["right_brow"]):
        pts[index, :2] = (midline - 0.12 * iod - i * 0.10 * iod, brow_y)

    pts[anchors["left_eye_outer"], :2] = (midline + 0.80 * iod, eye_y)
    pts[anchors["right_eye_outer"], :2] = (midline - 0.80 * iod, eye_y)
    pts[anchors["left_alar"], :2] = (midline + 0.18 * iod, eye_y + 0.62 * iod)
    pts[anchors["right_alar"], :2] = (midline - 0.18 * iod, eye_y + 0.62 * iod)
    pts[anchors["left_mouth_corner"], :2] = (midline + 0.30 * iod, eye_y + 1.05 * iod)
    pts[anchors["right_mouth_corner"], :2] = (midline - 0.30 * iod, eye_y + 1.05 * iod)
    return pts


def _frame_shape(iod: float) -> tuple[int, int]:
    return int(round(3.2 * iod)), int(round(3.2 * iod))


def _areas_in_anchor_units(iod: float, roi_config: dict) -> dict[str, float]:
    points = _landmarks(iod, roi_config)
    masks = rois.build(
        points, _frame_shape(iod), roi_config, run_mode=RunMode.DEVELOPMENT
    )
    anchor = scale.inter_ocular_distance(points)
    return {name: float(mask.sum()) / anchor**2 for name, mask in masks.items()}


# ------------------------------------------------------------------ the anchor itself


def test_anchor_is_proportional_to_face_size(roi_config: dict) -> None:
    small = scale.inter_ocular_distance(_landmarks(SMALL_IOD, roi_config))
    large = scale.inter_ocular_distance(_landmarks(LARGE_IOD, roi_config))
    assert large / small == pytest.approx(LARGE_IOD / SMALL_IOD, rel=1e-6)


def test_anchor_survives_a_resample(roi_config: dict) -> None:
    """Resizing the frame must scale the anchor by exactly the resize factor. If it did
    not, the canonical QC crop (D11) and the feature windows would disagree about scale."""
    points = _landmarks(SMALL_IOD, roi_config)
    doubled = points.copy()
    doubled[:, :2] *= 2.0
    assert scale.inter_ocular_distance(doubled) == pytest.approx(
        2.0 * scale.inter_ocular_distance(points)
    )


# ------------------------------------------------------------------ ROI geometry (B6)


def test_roi_area_in_anchor_units_is_scale_invariant(roi_config: dict) -> None:
    """The core B6 property. Areas are compared in anchor^2, not pixels: in pixels they
    differ by 4x between these two scales, and that difference is exactly what the anchor
    is supposed to divide out."""
    small = _areas_in_anchor_units(SMALL_IOD, roi_config)
    large = _areas_in_anchor_units(LARGE_IOD, roi_config)

    assert set(small) == set(large)
    for name in sorted(small):
        if small[name] == 0.0 and large[name] == 0.0:
            continue  # this ROI collapses under erosion at BOTH scales; consistent
        # 12% tolerance: erosion and rasterization are integer operations, so a thin band
        # loses proportionally more of itself to rounding at the smaller scale. Any real
        # scale dependence is far larger than this.
        assert large[name] == pytest.approx(small[name], rel=0.12), (
            f"{name}: {small[name]:.5f} anchor^2 at IOD {SMALL_IOD} vs "
            f"{large[name]:.5f} at IOD {LARGE_IOD} -- ROI geometry is scale dependent"
        )


def test_undersized_verdicts_agree_across_scales(roi_config: dict) -> None:
    """An ROI must not become measurable purely by moving the camera closer. If it did, the
    same subject would get different concerns reported at different distances -- which is a
    repeatability failure the user would experience directly."""
    small_masks = rois.build(
        _landmarks(SMALL_IOD, roi_config),
        _frame_shape(SMALL_IOD),
        roi_config,
        run_mode=RunMode.DEVELOPMENT,
    )
    large_masks = rois.build(
        _landmarks(LARGE_IOD, roi_config),
        _frame_shape(LARGE_IOD),
        roi_config,
        run_mode=RunMode.DEVELOPMENT,
    )
    assert rois.undersized(small_masks, SMALL_IOD, roi_config) == rois.undersized(
        large_masks, LARGE_IOD, roi_config
    )


def test_window_resolution_is_proportional_not_fixed() -> None:
    frac = cfg.concern_config("redness")["local"]["window_frac_of_iod"]
    small = scale.to_px(frac, SMALL_IOD, odd=True)
    large = scale.to_px(frac, LARGE_IOD, odd=True)
    assert large / small == pytest.approx(LARGE_IOD / SMALL_IOD, rel=0.05)


def test_no_config_carries_a_fixed_pixel_spatial_parameter() -> None:
    """Enumerated over every config, not just the concern blocks. A single `*_px` key is
    enough to make one measurement device dependent."""
    allowed = {
        # Genuinely resolution-space, by design and documented as such.
        "qc_face_width_px",  # D11 canonical crop target
        "min_face_px",  # native-pixel floor, checked BEFORE resampling
        "min_anchor_px",  # the floor on the anchor itself
        "min_reference_pixels",  # a count of pixels, not a distance
        "tile_px",  # pores: native-resolution tiling, concern disabled
        "stride_px",
        "min_px",
        "max_px",
        "min_lesion_px",
    }
    offenders: list[str] = []

    def walk(node: object, path: str = "") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                here = f"{path}.{key}" if path else str(key)
                if str(key).endswith("_px") and str(key) not in allowed:
                    offenders.append(here)
                walk(value, here)
        elif isinstance(node, list):
            for item in node:
                walk(item, path)

    for name in ("severity_thresholds", "capture_thresholds", "rois", "skin_mask"):
        walk(cfg.load(name), name)

    assert not offenders, f"fixed pixel parameters must be anchor fractions (D1): {offenders}"


# ------------------------------------------------------------------ feature measurements


#: Synthetic "skin" structure, defined entirely in ANCHOR UNITS: (wavelength, amplitude,
#: orientation). Wavelengths straddle the texture frequency band so at least one component
#: sits inside it at both scales, and all are far above the Nyquist limit of the smaller
#: capture, so neither scale is resolution-starved.
_GRATINGS = ((0.05, 4.0, 0.0), (0.09, 6.0, 1.05), (0.19, 8.0, 2.10))

#: Dark blobs for the pigmentation branch: (u, v, radius, depth) in anchor units.
_BLOBS = ((-0.30, -0.10, 0.030, 14.0), (0.25, 0.20, 0.022, 11.0), (0.05, -0.35, 0.026, 9.0))

#: Dark line valleys for the ridge branch: (v offset, half-width, depth) in anchor units.
#: A wrinkle is a DARK VALLEY, not a bright ridge, and the half-widths sit inside the
#: configured Gabor and Frangi scale range at both capture scales.
_LINES = ((-0.22, 0.008, 16.0), (0.00, 0.010, 20.0), (0.24, 0.007, 14.0))


def _anchor_field(side: int, iod: float) -> np.ndarray:
    """Render the same anchor-defined structure at an arbitrary pixel resolution.

    Rendered analytically at each scale rather than resampled from a common source. Both
    resampling approaches were tried and both measure the fixture instead of the code:

    * Nearest-neighbour upsampling duplicates pixels, so GLCM homogeneity at an offset of
      one pixel jumps for a reason no lens produces (reported a spurious 52% drift).
    * Area-downsampling WHITE NOISE by two different factors changes its amplitude --
      averaging four pixels quarters the variance, and local_variance duly moved by exactly
      4x. White noise is pure Nyquist-limited content, so it has no scale-invariant
      structure to preserve; it is not a stand-in for skin.

    Defining the structure in anchor units and evaluating it per scale is the actual
    physical analogue: the same face, more pixels spent on it.
    """
    coords = (np.arange(side, dtype=np.float64) - side / 2.0) / iod
    u, v = np.meshgrid(coords, coords, indexing="xy")

    field = np.zeros((side, side), dtype=np.float64)
    for wavelength, amplitude, theta in _GRATINGS:
        projection = u * np.cos(theta) + v * np.sin(theta)
        field += amplitude * np.sin(2.0 * np.pi * projection / wavelength)
    return field


def _blob_field(side: int, iod: float) -> np.ndarray:
    coords = (np.arange(side, dtype=np.float64) - side / 2.0) / iod
    u, v = np.meshgrid(coords, coords, indexing="xy")
    field = np.zeros((side, side), dtype=np.float64)
    for cu, cv, radius, depth in _BLOBS:
        field -= depth * np.exp(-(((u - cu) ** 2 + (v - cv) ** 2) / (2.0 * radius**2)))
    return field


def _line_field(side: int, iod: float) -> np.ndarray:
    """Long dark valleys running across the frame, defined in anchor units."""
    coords = (np.arange(side, dtype=np.float64) - side / 2.0) / iod
    _u, v = np.meshgrid(coords, coords, indexing="xy")
    field = np.zeros((side, side), dtype=np.float64)
    for offset, half_width, depth in _LINES:
        field -= depth * np.exp(-(((v - offset) ** 2) / (2.0 * half_width**2)))
    return field


def _scaled_pair(module: object) -> tuple[dict, dict]:
    """The same synthetic skin at two capture scales, measured with the matching anchor."""
    copy_kind = module.IMAGE_COPY  # type: ignore[attr-defined]

    def render(iod: float) -> np.ndarray:
        side = int(round(2.0 * iod))
        structure = _anchor_field(side, iod)
        blobs = _blob_field(side, iod)
        if copy_kind is ImageCopy.COLOR:
            image = np.zeros((side, side, 3), dtype=np.float32)
            image[..., 0] = 62.0 + 0.35 * structure + blobs
            image[..., 1] = 14.0 + 0.20 * structure
            image[..., 2] = 18.0 + 0.15 * structure
            return image
        if copy_kind is ImageCopy.RIDGE:
            return np.asarray(
                150.0 + 0.4 * structure + _line_field(side, iod), dtype=np.float32
            )
        return np.asarray(128.0 + structure + blobs, dtype=np.float32)

    small_img = render(SMALL_IOD)
    large_img = render(LARGE_IOD)

    config = cfg.concern_config(_key_for(module))
    results = []
    for image, iod in ((small_img, SMALL_IOD), (large_img, LARGE_IOD)):
        shape = image.shape[:2]
        mask = np.zeros(shape, dtype=bool)
        inset = int(round(0.08 * shape[0]))
        mask[inset : shape[0] - inset, inset : shape[1] - inset] = True
        roi_masks = {name: mask.copy() for name in config["primary_rois"]}
        context = FeatureContext(
            anchor_px=iod,
            qc=QCVerdict(True, True, True, True),
            run_mode=RunMode.DEVELOPMENT,
        )
        outcome = module.analyze(  # type: ignore[attr-defined]
            image, mask, roi_masks, config, context=context
        )
        results.append(
            {r.roi.value: r.raw for r in outcome.roi_results if r.raw}
        )
    return results[0], results[1]


def _key_for(module: object) -> str:
    for key, candidate in (
        ("redness", redness),
        ("pigmentation", pigmentation),
        ("texture", texture),
        ("wrinkles", wrinkles),
    ):
        if candidate is module:
            return key
    raise AssertionError(f"unknown module {module!r}")


#: Per-measurement scale-drift budget, as a relative tolerance between IOD 130 and 260.
#:
#: These are OBSERVED values with headroom, not aspirations. Each one is a claim about how
#: much of a measurement is skin and how much is capture distance, so a regression that
#: loosens one is a regression in what the cohort statistics mean -- raise a tolerance only
#: with the measurement that justified it.
#:
#: The GLCM family carries the largest residual and the reason is known: co-occurrence
#: offsets must be WHOLE pixels, so anchor fractions round to a coarser set at the smaller
#: scale (offsets {1,1,2} at IOD 130 against {1,2,4} at 260). That is irreducible for a
#: GLCM; `scale.min_anchor_px` is what bounds it.
DRIFT_BUDGET = {
    # Colour: ratios and Lab deltas, neither of which has a length in it.
    "affected_area_ratio": 0.10,
    "median_positive_delta_a": 0.05,
    "p90_delta_a": 0.05,
    "spot_area_ratio": 0.10,
    "median_delta_L": 0.05,
    "spot_count": 0.10,
    # Texture.
    "local_variance": 0.05,
    "hf_ratio": 0.10,
    "glcm_entropy": 0.15,
    "gradient_energy": 0.20,
    "glcm_homogeneity": 0.25,
    "glcm_energy": 0.25,
    "glcm_contrast": 0.30,
    # Ridge.
    "line_density": 0.20,
    "line_length": 0.15,
    "ridge_contrast": 0.15,
}


@pytest.mark.parametrize(
    "module", [redness, pigmentation, texture, wrinkles], ids=_key_for
)
def test_measurements_are_scale_stable(module: object) -> None:
    """The same synthetic skin at two capture distances must measure the same.

    This is the property that makes a frozen cohort statistic transferable between devices
    (D1 stage B). Without it every severity band inherits the capture distance.

    Four real bugs were found by this test and none of them were visible in any output:
    hf_ratio banded in cycles per PIXEL (60% drift), gradient_energy reported per pixel
    rather than per anchor, GLCM offsets fixed at [1,2,4] px, and Gabor/Frangi filter scales
    in pixels while their wavelength was already fractional -- which changed the kernel's
    SHAPE, not merely its size, between two captures of one face.
    """
    small, large = _scaled_pair(module)
    assert small and large, "the fixture produced no measurable ROI"

    for roi in sorted(set(small) & set(large)):
        for key, small_value in small[roi].items():
            assert key in DRIFT_BUDGET, f"{key} has no declared scale-drift budget"
            assert large[roi][key] == pytest.approx(
                small_value, rel=DRIFT_BUDGET[key], abs=1e-4
            ), (
                f"{roi}.{key}: {small_value:.4f} at IOD {SMALL_IOD} vs "
                f"{large[roi][key]:.4f} at IOD {LARGE_IOD} -- exceeds its "
                f"{DRIFT_BUDGET[key]:.0%} budget, so this measurement is partly reporting "
                "capture distance"
            )


def test_every_active_measurement_has_a_drift_budget() -> None:
    """A new raw measurement must declare how scale-stable it is before it can be shipped."""
    for key in ("redness", "pigmentation", "texture", "wrinkles"):
        for measurement in cfg.concern_config(key)["raw_measurements"]:
            assert measurement in DRIFT_BUDGET, (
                f"{key}.{measurement} has no scale-drift budget in this test"
            )


# ------------------------------------------------------------------ pipeline level


def test_image_copies_preserve_frame_size() -> None:
    """A copy that quietly resampled would break the mapping from ROI masks to pixels."""
    frame = np.random.default_rng(0).integers(0, 255, (96, 128, 3), dtype=np.uint8)
    copies = pipeline.make_image_copies(frame, cfg.load("severity_thresholds"))
    assert copies.color.shape[:2] == frame.shape[:2]
    assert copies.texture.shape == frame.shape[:2]
    assert copies.ridge.shape == frame.shape[:2]
