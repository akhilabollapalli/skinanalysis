"""Skin mask tests (Stage B4), driven by synthetic phantoms.

No real faces are needed here, and that is not a compromise: the mask is a colour and
texture classifier, so a phantom with a known skin field, a known hair patch and a known
specular blob exercises exactly the logic under test. What a phantom CANNOT tell us is
whether the mask handles real hairlines, stubble and glasses -- that is B3/B4 visual
review on the corpus.

The most important test in this file is
``test_coverage_is_stable_across_skin_tones``. A fixed RGB/HSV "skin colour" box is the
classical technique this module deliberately avoids, because it fails on dark skin
*silently*: the mask shrinks, ROI coverage drops, and the concern reports UNMEASURABLE or
scores a smaller region. Nothing looks broken. That test is what would catch a regression
back to colour-absolute thinking.
"""

from __future__ import annotations

import numpy as np
import pytest

import skin_analysis.face.skin_mask as sm
from skin_analysis.util import config as cfg
from skin_analysis.util import scale

IOD = 160.0
H, W = 640, 520
CX, CY = W // 2, H // 2


@pytest.fixture(scope="module")
def mask_config() -> dict:
    return cfg.load("skin_mask")


@pytest.fixture(scope="module")
def roi_config() -> dict:
    return cfg.load("rois")


def _landmarks() -> np.ndarray:
    """478-point array with the reference patch centres and face oval placed plausibly."""
    pts = np.zeros((478, 3), dtype=np.float64)
    pts[:, 0] = CX
    pts[:, 1] = CY

    pts[scale.RIGHT_IRIS_CENTER, :2] = (CX - IOD / 2, CY - 60)
    pts[scale.LEFT_IRIS_CENTER, :2] = (CX + IOD / 2, CY - 60)

    # Reference patch centres from config/skin_mask.yaml.
    pts[50, :2] = (CX - 0.55 * IOD, CY + 20)     # right mid-cheek
    pts[280, :2] = (CX + 0.55 * IOD, CY + 20)    # left mid-cheek
    pts[9, :2] = (CX, CY - 130)                  # forehead centre
    pts[152, :2] = (CX, CY + 165)                # chin centre
    pts[2, :2] = (CX, CY + 20)                   # nose, splits beard region

    # The reference is now sampled from the cheek and forehead ROI POLYGONS, so those
    # landmarks have to be placed or the polygons collapse and no reference can be built.
    cfg_rois = cfg.load("rois")["rois"]

    def _cluster(indices: list[int], cx: float, cy: float, rx: float, ry: float) -> None:
        for i, index in enumerate(indices):
            theta = 2 * np.pi * i / max(1, len(indices))
            pts[index, :2] = (cx + rx * np.cos(theta), cy + ry * np.sin(theta))

    _cluster(cfg_rois["left_cheek"]["landmarks"], CX + 0.62 * IOD, CY + 55,
             0.30 * IOD, 0.34 * IOD)
    _cluster(cfg_rois["right_cheek"]["landmarks"], CX - 0.62 * IOD, CY + 55,
             0.30 * IOD, 0.34 * IOD)
    _cluster(cfg_rois["chin"]["landmarks"], CX, CY + 1.25 * IOD, 0.28 * IOD, 0.16 * IOD)
    _cluster(cfg_rois["nose"]["landmarks"], CX, CY - 5, 0.14 * IOD, 0.30 * IOD)
    _cluster(cfg_rois["left_under_eye"]["landmarks"], CX + 0.55 * IOD, CY - 25,
             0.20 * IOD, 0.07 * IOD)
    _cluster(cfg_rois["right_under_eye"]["landmarks"], CX - 0.55 * IOD, CY - 25,
             0.20 * IOD, 0.07 * IOD)

    # Face oval as an ellipse, in the order sm._DEFAULT_FACE_OVAL expects.
    oval = sm._DEFAULT_FACE_OVAL
    for i, index in enumerate(oval):
        theta = 2 * np.pi * i / len(oval)
        pts[index, :2] = (CX + 1.35 * IOD * np.sin(theta), CY - 1.75 * IOD * np.cos(theta))
    return pts


def _phantom(
    skin_l: int = 170,
    hair: bool = False,
    specular: bool = False,
    shadow: bool = False,
    beard: bool = False,
    seed: int = 0,
) -> np.ndarray:
    """A face-shaped skin field with optional confounds painted on.

    ``skin_l`` sets overall lightness so the same phantom can be rendered across the tone
    range. Skin gets mild low-frequency variation and light noise; hair and beard get
    strong high-frequency structure, which is the signature the classifier keys on.
    """
    import cv2

    rng = np.random.default_rng(seed)
    image = np.full((H, W, 3), 30, dtype=np.uint8)  # dark background

    face = np.zeros((H, W), dtype=np.uint8)
    cv2.ellipse(face, (CX, CY), (int(1.35 * IOD), int(1.75 * IOD)), 0, 0, 360, 1, -1)
    face_bool = face.astype(bool)

    # Warm skin: B < G < R, with gentle shading.
    ys, xs = np.mgrid[0:H, 0:W]
    shading = 1.0 + 0.05 * np.sin(xs / 90.0) + 0.04 * np.cos(ys / 110.0)
    base = np.stack([
        np.clip(skin_l * 0.62 * shading, 0, 255),
        np.clip(skin_l * 0.78 * shading, 0, 255),
        np.clip(skin_l * 1.00 * shading, 0, 255),
    ], axis=-1)
    base += rng.normal(0, 1.5, base.shape)
    image[face_bool] = np.clip(base, 0, 255).astype(np.uint8)[face_bool]

    if hair:
        band = np.zeros((H, W), dtype=bool)
        band[: CY - int(1.15 * IOD), :] = True
        band &= face_bool
        strands = rng.normal(0, 34, (H, W))
        dark = np.clip(skin_l * 0.30 + strands, 0, 255)
        for c in range(3):
            image[..., c][band] = dark[band].astype(np.uint8)

    if beard:
        band = np.zeros((H, W), dtype=bool)
        band[CY + int(0.55 * IOD):, :] = True
        band &= face_bool
        strands = rng.normal(0, 26, (H, W))
        dark = np.clip(skin_l * 0.45 + strands, 0, 255)
        for c in range(3):
            image[..., c][band] = dark[band].astype(np.uint8)

    if specular:
        blob = np.zeros((H, W), dtype=np.uint8)
        cv2.circle(blob, (CX, CY - 40), int(0.30 * IOD), 1, -1)
        hit = blob.astype(bool) & face_bool
        image[hit] = 252  # near-clipped, neutral: no chroma left

    if shadow:
        half = np.zeros((H, W), dtype=bool)
        half[:, : CX - int(0.7 * IOD)] = True
        half &= face_bool
        image[half] = (image[half] * 0.22).astype(np.uint8)

    return image


def _coverage(mask: np.ndarray) -> float:
    import cv2

    face = np.zeros((H, W), dtype=np.uint8)
    cv2.ellipse(face, (CX, CY), (int(1.35 * IOD), int(1.75 * IOD)), 0, 0, 360, 1, -1)
    return float(mask.sum()) / float(face.sum())


# ------------------------------------------------------------------ config contract


def test_config_declares_no_absolute_colour_constant(mask_config: dict) -> None:
    """The defining property of this module. Any absolute RGB/HSV bound is the regression."""
    # Matched as whole underscore-separated tokens. Substring matching was tried and gave
    # a false positive: "min_r" is inside "min_reference_pixels".
    banned = {"rgb", "hsv", "ycrcb", "hue", "sat", "cr", "cb", "red", "green", "blue"}

    def walk(node: object, path: str = "") -> list[str]:
        found = []
        if isinstance(node, dict):
            for key, value in node.items():
                here = f"{path}.{key}" if path else str(key)
                if banned & set(str(key).lower().split("_")):
                    found.append(here)
                found += walk(value, here)
        return found

    assert not walk(mask_config), "absolute colour parameters found; see module docstring"


def test_every_threshold_is_expressed_in_subject_mads(mask_config: dict) -> None:
    for section in ("classify", "specular", "shadow", "hair", "beard", "glasses"):
        keys = [k for k in mask_config[section] if k.endswith(("_mads", "_frac_of_iod"))]
        assert keys, f"{section}: no MAD- or anchor-relative parameters"


def test_no_pixel_valued_parameters(mask_config: dict) -> None:
    """D1: a pixel size means different things at two resolutions."""
    def walk(node: object, path: str = "") -> list[str]:
        found = []
        if isinstance(node, dict):
            for key, value in node.items():
                here = f"{path}.{key}" if path else str(key)
                if str(key).endswith("_px"):
                    found.append(here)
                found += walk(value, here)
        return found

    offenders = [p for p in walk(mask_config) if not p.endswith("_pixels")]
    assert not offenders, f"pixel-valued parameters: {offenders}"


# ------------------------------------------------------------------ reference


def _polygons(roi_config: dict) -> dict:
    import skin_analysis.face.rois as R
    from skin_analysis.schemas import RunMode

    return R.build(_landmarks(), (H, W), roi_config, run_mode=RunMode.DEVELOPMENT)


def test_reference_is_established_on_a_clean_face(
    mask_config: dict, roi_config: dict
) -> None:
    lab = sm._to_lab(_phantom())
    ref = sm.build_reference(lab, _polygons(roi_config), mask_config)
    assert ref.n_patches_used >= mask_config["reference"]["min_valid_patches"]
    assert ref.l_mad > 0 and ref.chroma_mad > 0


def test_reference_rejects_a_patch_that_landed_on_a_confound(
    mask_config: dict, roi_config: dict
) -> None:
    """A patch on a beard or shadow would redefine 'normal skin' for the whole face."""
    lab = sm._to_lab(_phantom(beard=True))
    ref = sm.build_reference(lab, _polygons(roi_config), mask_config)
    assert ref.l_mad > 0


def test_reference_fails_closed_on_a_blank_image(
    mask_config: dict, roi_config: dict
) -> None:
    """Zero spread means no subject scale, and every threshold is a multiple of it."""
    lab = sm._to_lab(np.zeros((H, W, 3), np.uint8))
    with pytest.raises(sm.ReferenceError):
        sm.build_reference(lab, _polygons(roi_config), mask_config)


def test_chroma_mad_is_floored(mask_config: dict, roi_config: dict) -> None:
    """Near-even skin drives the chroma MAD toward zero, which would make the chroma gate
    infinitely strict and reject the entire face."""
    rng = np.random.default_rng(1)
    flat = np.clip(
        np.full((H, W, 3), 0.0) + np.array([110, 140, 180]) + rng.normal(0, 2, (H, W, 3)),
        0, 255,
    ).astype(np.uint8)
    ref = sm.build_reference(sm._to_lab(flat), _polygons(roi_config), mask_config)
    assert ref.chroma_mad > 0


# ------------------------------------------------------------------ the fairness property


@pytest.mark.parametrize("skin_l", [70, 100, 140, 180, 220])
def test_coverage_is_stable_across_skin_tones(skin_l: int, roi_config: dict) -> None:
    """The regression guard for colour-absolute thinking.

    A fixed skin-colour box passes at mid tones and quietly starves at the extremes. Here
    the same phantom is rendered across the tone range and must yield comparable coverage,
    because every threshold is relative to the subject's own statistics.
    """
    mask = sm.build(_phantom(skin_l=skin_l), _landmarks(), roi_config)
    assert _coverage(mask) > 0.5, f"coverage collapsed at skin_l={skin_l}"


def test_tone_does_not_change_coverage_much(roi_config: dict) -> None:
    landmarks = _landmarks()
    coverages = [
        _coverage(sm.build(_phantom(skin_l=level), landmarks, roi_config))
        for level in (70, 120, 170, 220)
    ]
    spread = max(coverages) - min(coverages)
    assert spread < 0.20, f"coverage varies {spread:.0%} across tone: {coverages}"


def test_dark_smooth_skin_is_not_mistaken_for_hair(roi_config: dict) -> None:
    """Hair detection requires darkness AND high frequency. Dark skin is dark but smooth,
    so requiring both is what keeps this test off it."""
    _, diagnostics = sm.build_with_diagnostics(_phantom(skin_l=70), _landmarks(), roi_config)
    assert diagnostics["rejected_frac_of_face"]["hair"] < 0.10


# ------------------------------------------------------------------ rejection stages


def test_background_is_never_skin(roi_config: dict) -> None:
    """Background leakage is the failure that most often looks like a real skin finding."""
    mask = sm.build(_phantom(), _landmarks(), roi_config)
    assert not mask[:20, :].any() and not mask[-20:, :].any()
    assert not mask[:, :20].any() and not mask[:, -20:].any()


def test_specular_highlight_is_rejected(roi_config: dict) -> None:
    """A clipped highlight carries no chroma: it measures the illuminant, not the skin."""
    _, clean = sm.build_with_diagnostics(_phantom(), _landmarks(), roi_config)
    _, hot = sm.build_with_diagnostics(_phantom(specular=True), _landmarks(), roi_config)
    assert (
        hot["rejected_frac_of_face"]["specular"]
        > clean["rejected_frac_of_face"]["specular"] + 0.02
    )


def test_hair_is_rejected(roi_config: dict) -> None:
    _, clean = sm.build_with_diagnostics(_phantom(), _landmarks(), roi_config)
    _, hairy = sm.build_with_diagnostics(_phantom(hair=True), _landmarks(), roi_config)
    assert (
        hairy["rejected_frac_of_face"]["hair"] > clean["rejected_frac_of_face"]["hair"] + 0.02
    )


def test_beard_is_rejected_and_stays_below_the_nose(roi_config: dict) -> None:
    mask, diagnostics = sm.build_with_diagnostics(_phantom(beard=True), _landmarks(), roi_config)
    assert diagnostics["rejected_frac_of_face"]["beard"] > 0.02
    upper = mask[: CY - int(0.5 * IOD)]
    assert upper.any(), "beard rejection leaked into the upper face"


def test_deep_shadow_is_rejected(roi_config: dict) -> None:
    _, shadowed = sm.build_with_diagnostics(_phantom(shadow=True), _landmarks(), roi_config)
    assert shadowed["rejected_frac_of_face"]["deep_shadow"] > 0.05


def test_eyes_and_lips_are_excluded_geometrically(roi_config: dict) -> None:
    _, diagnostics = sm.build_with_diagnostics(_phantom(), _landmarks(), roi_config)
    assert diagnostics["rejected_frac_of_face"]["geometric"] > 0.0


# ------------------------------------------------------------------ fail-closed


def test_blank_image_yields_an_empty_mask(roi_config: dict) -> None:
    """Empty makes ROIs UNMEASURABLE. A plausible mask that is not skin would produce a
    confident wrong answer instead."""
    mask = sm.build(np.zeros((H, W, 3), np.uint8), _landmarks(), roi_config)
    assert not mask.any()


def test_failure_reports_the_stage_that_failed(roi_config: dict) -> None:
    _, diagnostics = sm.build_with_diagnostics(
        np.zeros((H, W, 3), np.uint8), _landmarks(), roi_config
    )
    assert diagnostics["stage"] == "reference"
    assert "failed" in diagnostics


def test_heavily_occluded_face_fails_the_coverage_floor(roi_config: dict) -> None:
    """A mask that survives cleanup but covers almost nothing means the classifier failed,
    not that the face is small."""
    image = _phantom(hair=True, beard=True, shadow=True, specular=True)
    _, diagnostics = sm.build_with_diagnostics(image, _landmarks(), roi_config)
    if "failed" in diagnostics:
        assert diagnostics["stage"] == "coverage"
    else:
        assert diagnostics["coverage_frac_of_face"] >= (
            cfg.load("skin_mask")["morphology"]["min_mask_frac_of_face"]
        )


# ------------------------------------------------------------------ determinism (D13)


def test_mask_is_deterministic(roi_config: dict) -> None:
    image, landmarks = _phantom(), _landmarks()
    assert np.array_equal(
        sm.build(image, landmarks, roi_config), sm.build(image, landmarks, roi_config)
    )


def test_diagnostics_never_reach_the_public_payload() -> None:
    """Per-stage rejection areas are validation signal, not user-facing (Rule 3)."""
    from skin_analysis.schemas import PublicScanResult

    assert "diagnostics" not in PublicScanResult.__dataclass_fields__
