"""The four ACTIVE concern modules.

These defend the contracts every feature module shares, rather than the numeric output of
any one of them -- the numbers are uncalibrated placeholders and testing them would just
pin the placeholder in place.

The contracts under test:

* the entry point is ``analyze(image, skin_mask, rois, config, *, context=...)``
* ``context`` is keyword-only, so an old positional call site fails loudly
* the module is pure and deterministic
* it computes z_local only, never a population z (D1)
* it reports UNMEASURABLE rather than a plausible-looking number
* spatial parameters scale with the anchor (D1)
"""

from __future__ import annotations

import numpy as np
import pytest

from skin_analysis.decision import standardize
from skin_analysis.features import _common, pigmentation, redness, texture, wrinkles
from skin_analysis.schemas import Concern, FeatureContext, ImageCopy, QCVerdict, RunMode, Severity
from skin_analysis.util import config as cfg

#: (config key, module). Only the four ACTIVE concerns; the experimental three are
#: deliberately absent, matching pipeline._ACTIVE_MODULES.
ACTIVE = [
    ("redness", redness),
    ("pigmentation", pigmentation),
    ("texture", texture),
    ("wrinkles", wrinkles),
]

ANCHOR_PX = 200.0


def _context(anchor_px: float = ANCHOR_PX, **qc: bool) -> FeatureContext:
    verdict = {
        "passed": True,
        "shadow_pass": True,
        "color_cast_pass": True,
        "exposure_pass": True,
    }
    verdict.update(qc)
    return FeatureContext(
        anchor_px=anchor_px,
        qc=QCVerdict(**verdict),
        run_mode=RunMode.DEVELOPMENT,
    )


def _synthetic(module: object, size: int = 420, seed: int = 0) -> tuple:
    """A textured synthetic patch plus a generous ROI set.

    Synthetic on purpose: no real face image is ever committed to this repository
    (CLAUDE.md §5). These tests check plumbing and contracts, not skin truth.
    """
    rng = np.random.default_rng(seed)
    copy_kind = module.IMAGE_COPY  # type: ignore[attr-defined]

    if copy_kind is ImageCopy.COLOR:
        image = np.zeros((size, size, 3), dtype=np.float32)
        image[..., 0] = 60.0 + rng.normal(0.0, 4.0, (size, size))
        image[..., 1] = 14.0 + rng.normal(0.0, 3.0, (size, size))
        image[..., 2] = 18.0 + rng.normal(0.0, 3.0, (size, size))
    else:
        image = (120.0 + rng.normal(0.0, 12.0, (size, size))).astype(np.float32)

    mask = np.zeros((size, size), dtype=bool)
    mask[20 : size - 20, 20 : size - 20] = True

    config = cfg.concern_config(_key_for(module))
    rois = {name: mask.copy() for name in config["primary_rois"]}
    return image, mask, rois, config


def _key_for(module: object) -> str:
    for key, candidate in ACTIVE:
        if candidate is module:
            return key
    raise AssertionError(f"unknown module {module!r}")


# ------------------------------------------------------------------ shared contracts


@pytest.mark.parametrize(("key", "module"), ACTIVE)
def test_entry_point_signature(key: str, module: object) -> None:
    """One entry point per module, with the documented name."""
    assert callable(module.analyze)  # type: ignore[attr-defined]
    assert isinstance(module.CONCERN, Concern)  # type: ignore[attr-defined]
    assert isinstance(module.IMAGE_COPY, ImageCopy)  # type: ignore[attr-defined]
    del key


@pytest.mark.parametrize(("key", "module"), ACTIVE)
def test_context_is_keyword_only(key: str, module: object) -> None:
    """A stale positional call site must fail loudly rather than bind the wrong argument."""
    image, mask, rois, config = _synthetic(module)
    with pytest.raises(TypeError):
        module.analyze(image, mask, rois, config, _context())  # type: ignore[attr-defined]
    del key


@pytest.mark.parametrize(("key", "module"), ACTIVE)
def test_deterministic_for_identical_input(key: str, module: object) -> None:
    """Determinism is checked in CI on every commit (D13). Repeatability is not this test."""
    image, mask, rois, config = _synthetic(module)
    first = module.analyze(image, mask, rois, config, context=_context())  # type: ignore[attr-defined]
    second = module.analyze(image, mask, rois, config, context=_context())  # type: ignore[attr-defined]
    assert [r.raw for r in first.roi_results] == [r.raw for r in second.roi_results]
    del key


@pytest.mark.parametrize(("key", "module"), ACTIVE)
def test_does_not_mutate_its_inputs(key: str, module: object) -> None:
    """Pure function: the pipeline shares one image copy across concerns."""
    image, mask, rois, config = _synthetic(module)
    image_before, mask_before = image.copy(), mask.copy()
    module.analyze(image, mask, rois, config, context=_context())  # type: ignore[attr-defined]
    assert np.array_equal(image, image_before)
    assert np.array_equal(mask, mask_before)
    del key


@pytest.mark.parametrize(("key", "module"), ACTIVE)
def test_measurement_pass_leaves_severity_undecided(key: str, module: object) -> None:
    """A feature module may not decide a band: that needs cohort statistics from disk, and
    a feature module performs no I/O. Skipping the decision layer must therefore leave the
    result UNMEASURABLE -- never NOT_DETECTED, which would be a claim nobody made."""
    image, mask, rois, config = _synthetic(module)
    result = module.analyze(image, mask, rois, config, context=_context())  # type: ignore[attr-defined]
    assert result.severity is Severity.UNMEASURABLE
    assert all(r.severity is Severity.UNMEASURABLE for r in result.roi_results)
    del key


@pytest.mark.parametrize(("key", "module"), ACTIVE)
def test_empty_roi_is_unmeasurable_not_a_finding(key: str, module: object) -> None:
    """UNMEASURABLE and NOT_DETECTED are different claims (D7)."""
    image, mask, rois, config = _synthetic(module)
    blank = np.zeros_like(mask)
    empty_rois = {name: blank.copy() for name in rois}
    result = module.analyze(image, blank, empty_rois, config, context=_context())  # type: ignore[attr-defined]
    assert result.severity is Severity.UNMEASURABLE
    assert all(r.unmeasurable_reason for r in result.roi_results)
    assert not any(r.raw for r in result.roi_results)
    del key


@pytest.mark.parametrize(("key", "module"), ACTIVE)
def test_frame_size_disagreement_is_loud(key: str, module: object) -> None:
    image, mask, rois, config = _synthetic(module)
    with pytest.raises(ValueError, match="disagree on frame size"):
        module.analyze(  # type: ignore[attr-defined]
            image, np.zeros((10, 10), dtype=bool), rois, config, context=_context()
        )
    del key


@pytest.mark.parametrize(("key", "module"), ACTIVE)
def test_raw_measurement_keys_match_config(key: str, module: object) -> None:
    """The cohort and the feature module must agree about what is being measured, or
    standardization has nothing to standardize against."""
    image, mask, rois, config = _synthetic(module)
    result = module.analyze(image, mask, rois, config, context=_context())  # type: ignore[attr-defined]
    expected = set(config["raw_measurements"])
    for roi_result in result.roi_results:
        if roi_result.raw:
            assert set(roi_result.raw) == expected, key


@pytest.mark.parametrize(("key", "module"), ACTIVE)
def test_every_measurement_has_a_configured_direction(key: str, module: object) -> None:
    """An unsigned metric that runs the other way would cancel out a real finding.
    Texture's homogeneity and energy FALL as roughness rises."""
    config = cfg.concern_config(key)
    directions = config["decision"]["direction"]
    assert set(directions) == set(config["raw_measurements"])
    assert all(v in (1, -1) for v in directions.values())
    del module


# ------------------------------------------------------------------ D1: two normalizations


def test_features_do_not_call_the_population_standardizer() -> None:
    """z_local and z_ref answer different questions. A feature calling robust_z is a bug --
    a uniformly affected face has weak LOCAL contrast by construction, so z_local read as
    severity would score it clear.

    Checked over the parsed AST, not the source text. A substring search matches the module
    docstrings that explain the rule, so it would fail on modules that comply and could be
    "fixed" by deleting the explanation.
    """
    import ast
    import inspect

    for key, module in ACTIVE:
        tree = ast.parse(inspect.getsource(module))  # type: ignore[arg-type]
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr != "robust_z", f"{key} calls the population standardizer"
            if isinstance(node, ast.ImportFrom):
                assert "decision" not in (node.module or ""), (
                    f"{key} imports the decision layer; features measure and stop"
                )
            if isinstance(node, ast.Name):
                assert node.id != "robust_z", f"{key} calls the population standardizer"

    # The function does exist -- it just belongs to the decision layer.
    assert callable(standardize.robust_z)


def test_local_score_support_is_the_roi_it_is_applied_to() -> None:
    """Every skin-mask bug in this project was a scale estimated on one support and applied
    to a wider one. z_local repeats that shape by design, so the support is recorded."""
    field = np.arange(100, dtype=np.float64).reshape(10, 10)
    roi = np.zeros((10, 10), dtype=bool)
    roi[2:5, 2:5] = True
    score = _common.local_score(field, roi, {"use_roi_median": True}, ANCHOR_PX, 1e-6)
    assert score.support_px == int(roi.sum())
    assert score.baseline == pytest.approx(float(np.median(field[roi])))


def test_local_score_is_zero_outside_the_roi() -> None:
    """Undefined there, and zero is the value that cannot be mistaken for a finding."""
    field = np.random.default_rng(0).normal(size=(10, 10))
    roi = np.zeros((10, 10), dtype=bool)
    roi[3:6, 3:6] = True
    score = _common.local_score(field, roi, {"use_roi_median": True}, ANCHOR_PX, 1e-6)
    assert not score.z[~roi].any()


def test_local_score_refuses_an_empty_support() -> None:
    with pytest.raises(_common.LocalBaselineError):
        _common.local_score(
            np.zeros((8, 8)), np.zeros((8, 8), dtype=bool), {"use_roi_median": True},
            ANCHOR_PX, 1e-6,
        )


def test_sliding_window_baseline_is_unimplemented_not_approximated() -> None:
    """A masked local median is not the same computation as an unmasked one. Approximating
    it would put a silent bias in the only baseline the score has."""
    roi = np.ones((8, 8), dtype=bool)
    with pytest.raises(NotImplementedError, match="use_roi_median"):
        _common.local_score(
            np.zeros((8, 8)),
            roi,
            {"use_roi_median": False, "window_frac_of_iod": 0.5},
            ANCHOR_PX,
            1e-6,
        )


def test_no_active_concern_selects_the_unimplemented_baseline() -> None:
    for key, _module in ACTIVE:
        local = cfg.concern_config(key).get("local")
        if local is not None:
            assert local.get("use_roi_median") is True, key


# ------------------------------------------------------------------ QC gating


def test_redness_refuses_a_colour_cast() -> None:
    """D4: nothing downstream corrects a cast, so an a* excess measured under one is the
    room, not the skin."""
    image, mask, rois, config = _synthetic(redness)
    result = redness.analyze(
        image, mask, rois, config, context=_context(color_cast_pass=False)
    )
    assert result.severity is Severity.UNMEASURABLE
    assert any("colour-cast" in note for note in result.notes)


def test_pigmentation_refuses_a_failed_shadow_check() -> None:
    """A cast shadow IS a local L* deficit, so this is the concern most exposed to it."""
    image, mask, rois, config = _synthetic(pigmentation)
    result = pigmentation.analyze(
        image, mask, rois, config, context=_context(shadow_pass=False)
    )
    assert result.severity is Severity.UNMEASURABLE


def test_redness_asymmetry_needs_the_shadow_check() -> None:
    """Side lighting is the commonest cause of a false asymmetry finding."""
    config = cfg.concern_config("redness")
    raw = {
        "left_cheek": {"median_positive_delta_a": 5.0},
        "right_cheek": {"median_positive_delta_a": 1.0},
    }
    assert redness.asymmetry(raw, config, context=_context()) != {}
    assert redness.asymmetry(raw, config, context=_context(shadow_pass=False)) == {}


def test_texture_refuses_clahe_in_config() -> None:
    """D5: a GLCM downstream of adaptive gain partly measures the gain."""
    image, mask, rois, config = _synthetic(texture)
    config = {**config, "clahe": True}
    with pytest.raises(ValueError, match="CLAHE"):
        texture.analyze(image, mask, rois, config, context=_context())


# ------------------------------------------------------------------ D1: anchor scaling


@pytest.mark.parametrize(("key", "module"), ACTIVE)
def test_spatial_parameters_track_the_anchor(key: str, module: object) -> None:
    """A fixed pixel window means different things at two resolutions, so cohort statistics
    computed with one would not transfer between devices. Halving the anchor must change
    what the module measures on the same pixels."""
    image, mask, rois, config = _synthetic(module)
    big = module.analyze(image, mask, rois, config, context=_context(ANCHOR_PX))  # type: ignore[attr-defined]
    small = module.analyze(  # type: ignore[attr-defined]
        image, mask, rois, config, context=_context(ANCHOR_PX / 2)
    )
    assert [r.raw for r in big.roi_results] != [r.raw for r in small.roi_results], key
