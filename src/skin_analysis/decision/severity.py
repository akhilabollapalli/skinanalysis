"""Concern-level aggregation policy -- D6 and D7.

Per-ROI severities collapse to one concern severity by MAX-WITH-SUPPORT: take the highest
ROI severity, but require evidence before letting it stand.

    if n_rois_at_max >= min_rois_at_max:   final = max_severity
    elif strong_single_roi_support:        final = max_severity
    else:                                  final = demote(max_severity)

Two independent support routes, because a rigid "must appear in two ROIs" would suppress
genuinely localized pigmentation -- spots really can be one-sided -- while a bare maximum
would let one shadowed patch drive the whole result.

The support rule is CONCERN-SPECIFIC, not universal:

    redness       diffuse, usually bilateral -> escape needs red-area fraction + delta-a*
    pigmentation  focal and legitimately one-sided -> escape needs spot count + contrast
    wrinkles      forehead-only is normal anatomy -> escape needs line density + length
    texture       a field property; one rough ROI is a mask defect -> no escape at all

D7: a concern reports from whichever ROIs are measurable, and returns UNMEASURABLE only
when none of its primary ROIs is usable. Unmeasurable ROIs are preserved internally so
validation can tell "no finding" apart from "never looked".

Pure functions: policy only, no I/O, no image data.
"""

from __future__ import annotations

from ..schemas import ROI, Concern, FeatureResultInternal, ROIResult, Severity

#: Severities that carry no ordinal information and must not enter a maximum.
_NON_ORDINAL = (Severity.UNMEASURABLE, Severity.DISABLED)


def _min_reportable(config: dict) -> Severity:
    return Severity(str(config.get("min_reportable", "mild")))


def _meets_strong_support(raw: dict[str, float], rule: dict | None) -> bool:
    """True when one ROI's raw measurements clear every strong-support threshold.

    ``rule`` of None means the concern has no single-ROI escape (texture, D6). Every
    named threshold must be met -- ``any`` semantics would make the escape hatch wider
    than the rule it bypasses. A threshold configured as null is treated as unset and
    blocks the escape, so an uncalibrated concern cannot sneak through it.
    """
    if not rule:
        return False
    thresholds = {k: v for k, v in rule.items() if k.startswith("min_")}
    if not thresholds:
        return False
    for key, floor in thresholds.items():
        if floor is None:
            return False
        measurement = key[len("min_") :]
        value = raw.get(measurement)
        if value is None or value < floor:
            return False
    return True


def demote(severity: Severity, bands: int = 1) -> Severity:
    """Lower a severity by ``bands`` steps, floored at NOT_DETECTED.

    Non-ordinal states pass through unchanged: demoting UNMEASURABLE would invent a
    finding out of an absence of one.
    """
    if severity in _NON_ORDINAL:
        return severity
    return Severity.from_rank(severity.rank - bands)


def aggregate(
    concern: Concern,
    roi_results: list[ROIResult],
    config: dict,
) -> FeatureResultInternal:
    """Apply the concern's aggregation policy to its per-ROI results.

    Args:
        concern: the concern being aggregated.
        roi_results: one entry per primary ROI of this concern, including unmeasurable ones.
        config: the concern's merged config block (``defaults`` + concern overrides).

    Returns:
        FeatureResultInternal. ``severity`` and ``regions`` are the only fields that may
        later reach a user; everything else is retained for validation.
    """
    unmeasurable = [r.roi for r in roi_results if r.severity == Severity.UNMEASURABLE]
    measurable = [r for r in roi_results if r.severity not in _NON_ORDINAL]

    # D7: no usable primary ROI means the honest answer is "could not assess", not
    # "nothing found". Those are different claims and the product must not conflate them.
    if not measurable:
        return FeatureResultInternal(
            concern=concern,
            severity=Severity.UNMEASURABLE,
            regions=[],
            roi_results=roi_results,
            unmeasurable_regions=unmeasurable,
            notes=["no measurable primary ROI"],
        )

    policy = config.get("aggregation", {}) or {}
    min_rois_at_max = int(policy.get("min_rois_at_max", 2))
    demote_bands = int(policy.get("demote_bands", 1))
    strong_rule = policy.get("strong_single_roi")

    max_severity = max((r.severity for r in measurable), key=lambda s: s.rank)
    at_max = [r for r in measurable if r.severity == max_severity]

    notes: list[str] = []
    final: Severity
    if max_severity == Severity.NOT_DETECTED:
        final = max_severity
    elif len(at_max) >= min_rois_at_max:
        final = max_severity
        notes.append(f"support: {len(at_max)} ROIs at {max_severity.value}")
    elif any(_meets_strong_support(r.raw, strong_rule) for r in at_max):
        final = max_severity
        notes.append("support: single-ROI strong evidence")
    else:
        final = demote(max_severity, demote_bands)
        notes.append(
            f"demoted {max_severity.value} -> {final.value}: "
            f"{len(at_max)} ROI(s) at max, no strong single-ROI support"
        )

    # D6: the region list is independent of which ROI set produced the concern level.
    floor = _min_reportable(config)
    regions: list[ROI] = [r.roi for r in measurable if r.severity.rank >= floor.rank]

    return FeatureResultInternal(
        concern=concern,
        severity=final,
        regions=regions,
        roi_results=roi_results,
        unmeasurable_regions=unmeasurable,
        notes=notes,
    )
