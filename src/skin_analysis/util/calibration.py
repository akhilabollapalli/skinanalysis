"""Calibration registry -- the gate between measurement and publication (D2).

The pipeline can always measure. It may only publish once ordinal thresholds have been
calibrated against commercially usable annotations on a subject-level holdout, AND every
concern being published has population reference statistics for its primary ROIs (D1
stage B), AND those statistics came from a cohort large enough to mean anything.

This module is deliberately the single place that answers "may this go to a user?".
``ScanResultInternal.to_public()`` calls it and there is no other route to a public
payload.

Reference artifacts are treated as source, not as derived output: they are versioned,
diffable, and **immutable once frozen**. A production reference is never edited in place;
a new protocol gets ``data/reference/v2/``, so an old scan or release stays reproducible.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import config as cfg

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, typing only
    from ..schemas import Concern

REPO_ROOT = Path(__file__).resolve().parents[3]


def _reference_root() -> Path:
    ref = cfg.load("severity_thresholds").get("reference_stats", {}) or {}
    return REPO_ROOT / str(ref.get("root", "data/reference/v1"))


def reference_manifest() -> dict[str, Any] | None:
    """The reference set's manifest, or None if the set does not exist."""
    path = _reference_root() / "manifest.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        loaded: dict[str, Any] = json.load(fh)
    return loaded


def reference_stats(concern: str) -> dict[str, Any] | None:
    """Frozen population statistics for one concern, or None if absent.

    An ASSET, not a tuning knob: produced by a calibration run and carrying the manifest
    hash of the cohort it came from. Hand-editing one silently rewrites what "moderate"
    means for every user.
    """
    path = _reference_root() / f"{concern}.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        loaded: dict[str, Any] = json.load(fh)
    return loaded


def min_cohort_subjects() -> int:
    """Floor on cohort size for a reference set to count as calibrated.

    Guards against statistics quietly generated from two or three people, which would
    produce a median and MAD that look entirely normal and mean nothing.
    """
    manifest = reference_manifest() or {}
    configured = (cfg.load("severity_thresholds").get("reference_stats", {}) or {}).get(
        "min_cohort_subjects"
    )
    return int(manifest.get("min_cohort_subjects", configured or 0))


def missing_reference_rois(concern: str) -> list[str]:
    """Primary ROIs of ``concern`` lacking usable calibrated statistics.

    An ROI counts as missing when the concern's reference file is absent, uncalibrated,
    drawn from too small a cohort, or simply has no entry for that ROI.
    """
    block = cfg.load("severity_thresholds").get(concern, {}) or {}
    primary = block.get("primary_rois", [])

    stats = reference_stats(concern)
    if stats is None or not stats.get("calibrated", False):
        return list(primary)
    if int(stats.get("n_subjects", 0)) < min_cohort_subjects():
        return list(primary)

    medians = stats.get("median", {}) or {}
    mads = stats.get("mad", {}) or {}
    return [roi for roi in primary if not medians.get(roi) or not mads.get(roi)]


def concern_public_ready(concern: str) -> bool:
    """True when ``concern`` is enabled, calibrated globally, and has every reference ROI."""
    if not cfg.feature_enabled(concern):
        return False
    if not cfg.severity_calibrated():
        return False
    return not missing_reference_rois(concern)


def assert_public_ready(
    protocol_version: str,
    concerns: Iterable[Concern] | None = None,
) -> None:
    """Raise unless the active protocol may produce user-facing output.

    Args:
        protocol_version: the protocol the scan was measured under. Must match the
            protocol the thresholds were calibrated for -- publishing a v1 scan against
            v2 thresholds is a silent recalibration.
        concerns: concerns about to be published. ``None`` checks every enabled concern.

    Raises:
        CalibrationRequiredError: with the specific reason. The correct response is to run
            the calibration phase, never to catch this and substitute a default severity.
    """
    from ..schemas import CalibrationRequiredError  # local import: schemas imports us

    meta = cfg.load("severity_thresholds").get("meta", {}) or {}

    if not meta.get("calibrated", False):
        raise CalibrationRequiredError(
            "severity_thresholds.yaml meta.calibrated is false. The pipeline may measure "
            "but may not publish: placeholder cutoffs must never reach a user (D2). "
            "Use analyze_scan_internal() for validation work."
        )

    expected = meta.get("protocol_version")
    if expected and expected != protocol_version:
        raise CalibrationRequiredError(
            f"scan protocol {protocol_version!r} was calibrated against {expected!r}. "
            "Publishing across protocol versions is a silent recalibration."
        )

    if not meta.get("calibration_source"):
        raise CalibrationRequiredError(
            "meta.calibrated is true but meta.calibration_source is unset. A calibration "
            "with no named source cannot be audited or reproduced."
        )

    manifest = reference_manifest()
    if manifest is None:
        raise CalibrationRequiredError(
            f"no reference manifest at {_reference_root()}. Population standardization "
            "(D1 stage B) has nothing to standardize against."
        )
    if not manifest.get("frozen", False):
        raise CalibrationRequiredError(
            "reference set is not frozen. A reference set that can still change under a "
            "released build makes its scans unreproducible."
        )

    names = (
        [c.value for c in concerns]
        if concerns is not None
        else [c for c in cfg.configured_concerns() if cfg.feature_enabled(c)]
    )
    for name in names:
        stats = reference_stats(name)
        if stats and int(stats.get("n_subjects", 0)) < min_cohort_subjects():
            raise CalibrationRequiredError(
                f"{name}: reference cohort has {stats.get('n_subjects', 0)} subjects, "
                f"below the floor of {min_cohort_subjects()}. A median and MAD from a "
                "handful of people look normal and mean nothing."
            )
        missing = missing_reference_rois(name)
        if missing:
            raise CalibrationRequiredError(
                f"{name}: no calibrated population reference statistics for ROIs {missing}. "
                "Severity standardization (D1 stage B) has nothing to standardize against."
            )
