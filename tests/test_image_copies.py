"""The three image copies (D5).

These tests defend the property that the branches do NOT share preprocessing, and that no
adaptive step can be introduced in front of a measurement without a config change that
fails loudly.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from skin_analysis import pipeline
from skin_analysis.util import color
from skin_analysis.util import config as cfg


@pytest.fixture
def severity_config() -> dict:
    return copy.deepcopy(cfg.load("severity_thresholds"))


def _frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)


# ------------------------------------------------------------------ colour conversion


def test_lab_is_on_the_literature_scale() -> None:
    """L* in 0..100 and a*/b* centred on zero, or every threshold in config means nothing."""
    white = np.full((4, 4, 3), 255, dtype=np.uint8)
    lab = color.bgr_to_lab(white)
    assert lab[..., 0].max() == pytest.approx(100.0, abs=0.5)
    assert abs(float(lab[..., 1].mean())) < 2.0
    assert abs(float(lab[..., 2].mean())) < 2.0


def test_conversion_is_deterministic() -> None:
    frame = _frame()
    assert np.array_equal(color.bgr_to_lab(frame), color.bgr_to_lab(frame))


def test_conversion_rejects_the_wrong_dtype() -> None:
    """A float image would silently be reinterpreted, shifting every colour threshold."""
    with pytest.raises(ValueError):
        color.bgr_to_lab(np.zeros((8, 8, 3), dtype=np.float32))


def test_skin_mask_and_features_share_one_conversion() -> None:
    """If the mask and the colour concerns disagreed on what a* means, the mask would
    reject pixels on one scale while a feature measured them on another -- and nothing
    about that would look wrong in an overlay."""
    from skin_analysis.face import skin_mask

    frame = _frame(3)
    assert np.array_equal(skin_mask._to_lab(frame), color.bgr_to_lab(frame))


# ------------------------------------------------------------------ the three copies


def test_three_copies_have_the_documented_shapes(severity_config: dict) -> None:
    copies = pipeline.make_image_copies(_frame(), severity_config)
    assert copies.color.ndim == 3, "the colour copy is CIELAB, three planes"
    assert copies.texture.ndim == 2, "the texture copy is one luminance plane"
    assert copies.ridge.ndim == 2, "the ridge copy is one luminance plane"


def test_ridge_is_a_copy_not_an_alias(severity_config: dict) -> None:
    """A future in-place ridge enhancement must not modify what texture measures."""
    copies = pipeline.make_image_copies(_frame(), severity_config)
    copies.ridge[0, 0] = 123.0
    assert copies.texture[0, 0] != 123.0


def test_texture_luminance_uses_fixed_normalization(severity_config: dict) -> None:
    """Fixed, so it cannot depend on image content. Two different images must map the same
    L* to the same luminance, or a GLCM partly measures the normalization."""
    dark = np.full((32, 32, 3), 40, dtype=np.uint8)
    bright = np.full((32, 32, 3), 200, dtype=np.uint8)
    mixed = np.concatenate([dark, bright], axis=0)

    from_alone = pipeline.make_image_copies(dark, severity_config).texture[0, 0]
    from_mixed = pipeline.make_image_copies(mixed, severity_config).texture[0, 0]
    assert from_alone == pytest.approx(from_mixed)


@pytest.mark.parametrize(
    ("branch", "flag"),
    [
        ("color", "gray_world"),
        ("color", "illuminant_estimation"),
        ("color", "clahe"),
        ("texture", "clahe"),
        ("ridge", "clahe"),
    ],
)
def test_adaptive_preprocessing_is_refused(
    severity_config: dict, branch: str, flag: str
) -> None:
    """D4/D5. Turning one of these on must fail loudly rather than quietly change every
    measurement the cohort statistics were computed against."""
    severity_config["image_copies"][branch][flag] = True
    with pytest.raises(ValueError, match=flag):
        pipeline.make_image_copies(_frame(), severity_config)


def test_unvalidated_ridge_enhancement_is_refused(severity_config: dict) -> None:
    """Adopting an enhancement refreezes calibration, so it cannot arrive by config alone."""
    severity_config["image_copies"]["ridge"]["enhancement"] = "retinex"
    with pytest.raises(ValueError, match="retinex"):
        pipeline.make_image_copies(_frame(), severity_config)


def test_shipped_config_has_every_adaptive_flag_off() -> None:
    """The repository's own config must satisfy D4/D5, not merely be able to."""
    spec = cfg.load("severity_thresholds")["image_copies"]
    assert spec["color"] == {
        "gray_world": False,
        "illuminant_estimation": False,
        "clahe": False,
    }
    assert spec["texture"]["clahe"] is False
    assert spec["ridge"]["clahe"] is False
    assert spec["ridge"]["enhancement"] == "none"
