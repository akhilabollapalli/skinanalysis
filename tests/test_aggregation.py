"""Aggregation policy tests -- D6 (max-with-support) and D7 (partial ROI reporting).

This is the layer that decides what a scan actually claims, so it is tested directly
rather than only through the pipeline.
"""

from __future__ import annotations

import pytest

from skin_analysis.decision import severity as agg
from skin_analysis.schemas import ROI, Concern, ROIResult, Severity
from skin_analysis.util import config as cfg


def _roi(roi: ROI, sev: Severity, **raw: float) -> ROIResult:
    return ROIResult(roi=roi, severity=sev, raw=raw)


def _unmeasurable(roi: ROI) -> ROIResult:
    return ROIResult(roi=roi, severity=Severity.UNMEASURABLE, unmeasurable_reason="occlusion")


# ------------------------------------------------------------------ ordinal scale


def test_ordinal_ranks_are_monotonic() -> None:
    scale = [Severity.NOT_DETECTED, Severity.MILD, Severity.MODERATE, Severity.HIGH]
    assert [s.rank for s in scale] == sorted(s.rank for s in scale)


def test_non_ordinal_states_sit_outside_the_scale() -> None:
    """Comparing UNMEASURABLE against a finding is meaningless, not 'lowest'."""
    assert Severity.UNMEASURABLE.rank == -1
    assert Severity.DISABLED.rank == -1


def test_demote_floors_at_not_detected() -> None:
    assert agg.demote(Severity.MILD) is Severity.NOT_DETECTED
    assert agg.demote(Severity.NOT_DETECTED) is Severity.NOT_DETECTED


def test_demote_passes_non_ordinal_through() -> None:
    """Demoting UNMEASURABLE would invent a finding out of an absence of one."""
    assert agg.demote(Severity.UNMEASURABLE) is Severity.UNMEASURABLE


# ------------------------------------------------------------------ D6: max-with-support


def test_two_rois_at_max_hold_the_level() -> None:
    config = cfg.concern_config("redness")
    result = agg.aggregate(
        Concern.REDNESS,
        [
            _roi(ROI.LEFT_CHEEK, Severity.MODERATE),
            _roi(ROI.RIGHT_CHEEK, Severity.MODERATE),
            _roi(ROI.FOREHEAD, Severity.MILD),
        ],
        config,
    )
    assert result.severity is Severity.MODERATE


def test_single_roi_without_strong_support_is_demoted() -> None:
    """One hot ROI is more often a shadow or specular edge than a finding."""
    config = cfg.concern_config("redness")
    result = agg.aggregate(
        Concern.REDNESS,
        [
            _roi(ROI.LEFT_CHEEK, Severity.MODERATE, affected_area_ratio=0.05, p90_delta_a=2.0),
            _roi(ROI.RIGHT_CHEEK, Severity.MILD),
        ],
        config,
    )
    assert result.severity is Severity.MILD


def test_single_roi_with_strong_support_holds_the_level() -> None:
    """D6 escape hatch: strong single-ROI evidence substitutes for a second ROI."""
    config = cfg.concern_config("redness")
    result = agg.aggregate(
        Concern.REDNESS,
        [
            _roi(ROI.LEFT_CHEEK, Severity.MODERATE, affected_area_ratio=0.30, p90_delta_a=9.0),
            _roi(ROI.RIGHT_CHEEK, Severity.MILD),
        ],
        config,
    )
    assert result.severity is Severity.MODERATE


def test_strong_support_requires_every_threshold() -> None:
    """`any` semantics would make the escape wider than the rule it bypasses."""
    config = cfg.concern_config("redness")
    result = agg.aggregate(
        Concern.REDNESS,
        [
            _roi(ROI.LEFT_CHEEK, Severity.HIGH, affected_area_ratio=0.30, p90_delta_a=1.0),
            _roi(ROI.RIGHT_CHEEK, Severity.MILD),
        ],
        config,
    )
    assert result.severity is Severity.MODERATE  # demoted from HIGH


def test_texture_has_no_single_roi_escape() -> None:
    """Roughness is a field property; one rough ROI is a mask defect until proven otherwise."""
    config = cfg.concern_config("texture")
    assert config["aggregation"]["strong_single_roi"] is None
    result = agg.aggregate(
        Concern.TEXTURE,
        [
            _roi(ROI.FOREHEAD, Severity.HIGH, glcm_contrast=99.0),
            _roi(ROI.LEFT_CHEEK, Severity.MILD),
        ],
        config,
    )
    assert result.severity is Severity.MODERATE


def test_pigmentation_escape_uses_lesion_evidence_not_roi_count() -> None:
    """Spots are focal and genuinely can be one-sided (D6)."""
    config = cfg.concern_config("pigmentation")
    result = agg.aggregate(
        Concern.DARK_SPOTS,
        [
            _roi(
                ROI.LEFT_CHEEK, Severity.MODERATE,
                spot_count=9, spot_area_ratio=0.07, median_delta_L=5.5,
            ),
            _roi(ROI.RIGHT_CHEEK, Severity.NOT_DETECTED),
        ],
        config,
    )
    assert result.severity is Severity.MODERATE


def test_uncalibrated_strong_support_thresholds_block_the_escape() -> None:
    """A null threshold is unset, not satisfied. Acne must not sneak through its own hatch."""
    config = cfg.concern_config("acne")
    result = agg.aggregate(
        Concern.ACNE,
        [
            _roi(ROI.LEFT_CHEEK, Severity.HIGH, lesion_burden=999.0, affected_area_ratio=0.9),
            _roi(ROI.CHIN, Severity.MILD),
        ],
        config,
    )
    assert result.severity is Severity.MODERATE


# ------------------------------------------------------------------ D7: partial reporting


def test_reports_from_measurable_rois_when_some_fail() -> None:
    config = cfg.concern_config("redness")
    result = agg.aggregate(
        Concern.REDNESS,
        [
            _unmeasurable(ROI.FOREHEAD),
            _roi(ROI.LEFT_CHEEK, Severity.MODERATE),
            _roi(ROI.RIGHT_CHEEK, Severity.MODERATE),
        ],
        config,
    )
    assert result.severity is Severity.MODERATE
    assert result.regions == [ROI.LEFT_CHEEK, ROI.RIGHT_CHEEK]
    assert result.unmeasurable_regions == [ROI.FOREHEAD]


def test_all_rois_unmeasurable_gives_unmeasurable_not_not_detected() -> None:
    """'Could not assess' and 'nothing found' are different claims."""
    config = cfg.concern_config("redness")
    result = agg.aggregate(
        Concern.REDNESS,
        [_unmeasurable(ROI.FOREHEAD), _unmeasurable(ROI.LEFT_CHEEK)],
        config,
    )
    assert result.severity is Severity.UNMEASURABLE
    assert result.regions == []


def test_regions_are_independent_of_the_aggregated_level() -> None:
    """D6: regions list every ROI at or above min_reportable, whatever the concern level."""
    config = cfg.concern_config("redness")
    result = agg.aggregate(
        Concern.REDNESS,
        [
            _roi(ROI.LEFT_CHEEK, Severity.MODERATE, affected_area_ratio=0.01, p90_delta_a=0.1),
            _roi(ROI.RIGHT_CHEEK, Severity.MILD),
            _roi(ROI.FOREHEAD, Severity.NOT_DETECTED),
        ],
        config,
    )
    assert result.severity is Severity.MILD  # demoted
    assert result.regions == [ROI.LEFT_CHEEK, ROI.RIGHT_CHEEK]  # still both


# ------------------------------------------------------------------ D8: nasolabial


@pytest.mark.parametrize("concern", ["redness", "pigmentation", "texture"])
def test_nasolabial_excluded_from_colour_and_texture_concerns(concern: str) -> None:
    primary = cfg.concern_config(concern)["primary_rois"]
    assert "left_nasolabial" not in primary
    assert "right_nasolabial" not in primary


def test_nasolabial_is_available_to_wrinkles() -> None:
    """The fold IS a morphology target, so the structural detector may use it (D8)."""
    primary = cfg.concern_config("wrinkles")["primary_rois"]
    assert "left_nasolabial" in primary and "right_nasolabial" in primary


def test_redness_asymmetry_excludes_nasolabial() -> None:
    excluded = cfg.concern_config("redness")["asymmetry"]["excluded_rois"]
    assert set(excluded) == {"left_nasolabial", "right_nasolabial"}
