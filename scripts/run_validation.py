"""Release validation: repeatability, capture rejection, and fairness slicing.

    python scripts/run_validation.py --suite repeatability --corpus <local path>
    python scripts/run_validation.py --suite slices --corpus <local path>
    python scripts/run_validation.py --suite all --corpus <local path> --out report.json

Gates come from ``config/validation_gates.yaml`` and are PRESPECIFIED. This script only
evaluates them; it never picks one. A threshold chosen after seeing the results is a
description of those results, not a gate.

WHY THIS IS NOT A CI CHECK (D13). Determinism -- same array twice, byte-identical output --
is in CI, in tests/test_repeatability.py. Repeatability is a different property: DISTINCT
captures of one subject in one session must land on the same ordinal severity. It needs
real faces, and real faces are never committed to this repo (CLAUDE.md §5), so it runs here
against a local gitignored corpus.

NEVER REPORT A POOLED NUMBER. Every gate is evaluated per slice as well as overall, and a
subgroup too small to gate is printed rather than folded into its neighbour. A model that
works on average and fails on one skin tone is a failed model, and pooling is precisely the
operation that hides that.

FAILS CLOSED. The default exit status is non-zero if any gate failed OR if any gate could
not be evaluated. An unevaluated gate is not a passed gate. Use --allow-unevaluated for
day-to-day development runs, never for a release decision.

Everything this script prints is internal (Rule 3). The report is a validation artifact;
nothing in it may reach an application payload.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from skin_analysis import pipeline  # noqa: E402
from skin_analysis.schemas import RunMode, Severity  # noqa: E402
from skin_analysis.util import config as cfg  # noqa: E402

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


def _enabled_concern_values() -> list[str]:
    """Enabled concerns, named the way a Record actually names them.

    ``cfg.configured_concerns()`` returns severity_thresholds.yaml's CONFIG keys
    (``redness``, ``pigmentation``, ``texture``, ``wrinkles``, ...). A Record's
    ``severities`` / ``raw`` / ``measurable_rois`` are keyed by ``Concern.value`` instead,
    because they are flattened straight from ``ScanResultInternal.to_internal_payload()``,
    which iterates ``dict[Concern, FeatureResultInternal]``.

    Those vocabularies agree for redness/texture/wrinkles and silently disagree for
    pigmentation, whose ``Concern.value`` is ``dark_spots`` (CLAUDE.md: "the product calls
    dark spots"; the two names are kept distinct on purpose so neither layer has to know
    the other's vocabulary -- see pipeline._CONCERN_OF). Looking a Record up by the config
    key ``"pigmentation"`` finds nothing, ever, and the gate reports NOT_EVALUATED forever
    without raising -- exactly the shape of silent failure this project keeps finding
    elsewhere (a value estimated on one vocabulary and applied under another).

    ``pipeline._CONCERN_OF`` is the single source of truth for this mapping and is reused
    here rather than duplicated, the same way qc_report.py reaches into
    ``pipeline._GATED_MASK_STAGES``.
    """
    return [
        concern.value
        for key, concern in pipeline._CONCERN_OF.items()
        if cfg.feature_enabled(key)
    ]


#: Severities that sit on the ordinal scale and can therefore be compared to each other.
#: UNMEASURABLE and DISABLED are states, not findings, and comparing them to a band is
#: meaningless -- see the ordinal_agreement comment in config/validation_gates.yaml.
_COMPARABLE = frozenset(s.value for s in (
    Severity.NOT_DETECTED,
    Severity.MILD,
    Severity.MODERATE,
    Severity.HIGH,
))

SUITES = ("repeatability", "capture", "slices", "all")


# --------------------------------------------------------------------------- records


@dataclass(frozen=True)
class Record:
    """One analysed capture, flattened to exactly what the gates need.

    Deliberately a plain data holder with no image and no pipeline object: every statistic
    below is computed from Records alone, so the whole analysis half of this script is
    testable without a face corpus.
    """

    image: str
    meta: dict[str, str]
    qc_passed: bool
    qc_failures: tuple[str, ...]
    #: concern -> severity value.
    severities: dict[str, str]
    #: concern -> the ROIs that produced raw measurements, sorted.
    measurable_rois: dict[str, tuple[str, ...]]
    #: concern -> roi -> measurement -> value.
    raw: dict[str, dict[str, dict[str, float]]]

    @property
    def subject(self) -> str:
        return self.meta.get("subject_id", "")

    def group(self, session_column: str) -> tuple[str, str]:
        """Subject + session. Absent a session column every capture is one session.

        That fallback is not free: it mixes cross-session skin change into what is
        supposed to be same-session measurement noise, so the runner warns when it applies.
        """
        return self.subject, self.meta.get(session_column, "") or "_"


def record_from_payload(image: str, meta: dict[str, str], payload: dict[str, Any]) -> Record:
    """Flatten one ``ScanResultInternal.to_internal_payload()`` into a Record."""
    capture = payload.get("capture_quality", {}) or {}
    severities: dict[str, str] = {}
    measurable: dict[str, tuple[str, ...]] = {}
    raw: dict[str, dict[str, dict[str, float]]] = {}

    for concern, block in (payload.get("concerns", {}) or {}).items():
        severities[concern] = str(block.get("severity", ""))
        per_roi = {
            str(rr["roi"]): {k: float(v) for k, v in (rr.get("raw") or {}).items()}
            for rr in block.get("roi_results", [])
            if rr.get("raw")
        }
        raw[concern] = per_roi
        measurable[concern] = tuple(sorted(per_roi))

    return Record(
        image=image,
        meta=dict(meta),
        qc_passed=bool(capture.get("pass", False)),
        qc_failures=tuple(str(r) for r in capture.get("reasons", []) or []),
        severities=severities,
        measurable_rois=measurable,
        raw=raw,
    )


# --------------------------------------------------------------------------- gates


@dataclass
class GateResult:
    """One gate's outcome. ``status`` is pass / fail / not_evaluated.

    not_evaluated is a distinct status on purpose. Collapsing it into "pass" is how an
    unimplemented check becomes an implicitly satisfied one, and collapsing it into "fail"
    would make a development run indistinguishable from a regression.
    """

    name: str
    status: str
    detail: str
    value: float | None = None
    bound: float | None = None
    slices: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.status == "fail"

    @property
    def evaluated(self) -> bool:
        return self.status in ("pass", "fail")


def _verdict(value: float, bound: float, *, higher_is_better: bool) -> str:
    return "pass" if (value >= bound if higher_is_better else value <= bound) else "fail"


# ------------------------------------------------------------ repeatability statistics


def group_records(
    records: list[Record], session_column: str, min_captures: int
) -> dict[tuple[str, str], list[Record]]:
    """Subject-session groups holding enough captures to say anything about stability."""
    groups: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for record in records:
        if record.subject:
            groups[record.group(session_column)].append(record)
    return {key: rs for key, rs in groups.items() if len(rs) >= min_captures}


def ordinal_agreement(
    groups: dict[tuple[str, str], list[Record]], concern: str
) -> tuple[int, int]:
    """(groups that agreed, groups that were comparable at all) for one concern.

    A group is comparable only when every capture in it produced an ordinal band. On an
    uncalibrated build nothing is comparable, so the denominator is zero and the gate
    reports not_evaluated rather than a vacuous 100%.
    """
    comparable = 0
    agreed = 0
    for members in groups.values():
        values = [r.severities.get(concern, "") for r in members]
        if not all(v in _COMPARABLE for v in values):
            continue
        comparable += 1
        if len(set(values)) == 1:
            agreed += 1
    return agreed, comparable


@dataclass(frozen=True)
class CVSample:
    """CV of one measurement, in one ROI, across one subject-session."""

    group: tuple[str, str]
    roi: str
    measurement: str
    cv: float


def raw_metric_cvs(
    groups: dict[tuple[str, str], list[Record]], concern: str
) -> tuple[list[CVSample], int]:
    """Per (group, ROI, measurement) coefficients of variation, plus the undefined count.

    A measurement is only compared where every capture in the group produced it: a value
    present in two scans and absent in a third is an ROI-availability finding, counted by
    ``roi_availability_agreement``, and folding it in here would silently shrink the
    sample instead of reporting it.
    """
    samples: list[CVSample] = []
    undefined = 0

    for key, members in groups.items():
        rois = set.intersection(*(set(r.raw.get(concern, {})) for r in members))
        for roi in sorted(rois):
            names = set.intersection(*(set(r.raw[concern][roi]) for r in members))
            for measurement in sorted(names):
                values = [r.raw[concern][roi][measurement] for r in members]
                mean = statistics.fmean(values)
                if mean == 0.0:
                    # No defined CV. Counted rather than dropped: many of these means the
                    # concern is mostly finding nothing, which is a statement about the
                    # concern, not about stability.
                    undefined += 1
                    continue
                cv = statistics.stdev(values) / abs(mean)
                samples.append(CVSample(key, roi, measurement, cv))

    return samples, undefined


def roi_availability_agreement(
    groups: dict[tuple[str, str], list[Record]], concern: str
) -> tuple[int, int]:
    """(groups where the measurable ROI set was identical, groups examined)."""
    agreed = 0
    examined = 0
    for members in groups.values():
        sets = {r.measurable_rois.get(concern, ()) for r in members}
        # A group where no capture measured anything says nothing about availability
        # agreement; it is a capture or calibration finding reported elsewhere.
        if sets == {()}:
            continue
        examined += 1
        if len(sets) == 1:
            agreed += 1
    return agreed, examined


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile. Local so this module needs no numpy import."""
    if not values:
        raise ValueError("percentile of an empty sample")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (q / 100.0)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


# ------------------------------------------------------------------- suite: repeatability


def run_repeatability(
    records: list[Record], gates: dict[str, Any], slice_columns: list[str], min_subgroup_n: int
) -> list[GateResult]:
    spec = gates["repeatability"]
    session_column = gates["corpus"]["session_column"]
    groups = group_records(records, session_column, int(spec["min_captures_per_group"]))
    concerns = _enabled_concern_values()
    results: list[GateResult] = []

    if not groups:
        return [
            GateResult(
                name="repeatability",
                status="not_evaluated",
                detail=(
                    "no subject-session has "
                    f"{spec['min_captures_per_group']}+ captures. Repeatability needs "
                    "rescans of the same subject; a corpus of one shot per person cannot "
                    "measure it at all."
                ),
            )
        ]

    # --- ordinal agreement -------------------------------------------------------------
    bound = float(spec["gates"]["ordinal_agreement"]["min_fraction"])
    for concern in concerns:
        agreed, comparable = ordinal_agreement(groups, concern)
        if comparable == 0:
            results.append(
                GateResult(
                    name=f"ordinal_agreement[{concern}]",
                    status="not_evaluated",
                    detail=(
                        "no group produced an ordinal band in every capture. On an "
                        "uncalibrated build every concern returns UNMEASURABLE (D2), so "
                        "there is nothing to agree about -- this is expected until the "
                        "calibration cohort exists, and it is NOT a pass."
                    ),
                    bound=bound,
                )
            )
            continue
        value = agreed / comparable
        results.append(
            GateResult(
                name=f"ordinal_agreement[{concern}]",
                status=_verdict(value, bound, higher_is_better=True),
                detail=f"{agreed}/{comparable} subject-sessions returned one severity",
                value=value,
                bound=bound,
                slices=_slice_ordinal_agreement(
                    records, concern, gates, slice_columns, min_subgroup_n
                ),
            )
        )

    # --- raw metric CV -----------------------------------------------------------------
    cv_bounds = spec["gates"]["raw_metric_cv"]["p95_max_cv"]
    for concern in concerns:
        if concern not in cv_bounds:
            results.append(
                GateResult(
                    name=f"raw_metric_cv[{concern}]",
                    status="not_evaluated",
                    detail=(
                        "no prespecified CV bound. An enabled concern with no bound is "
                        "ungated; add one to config/validation_gates.yaml BEFORE running."
                    ),
                )
            )
            continue
        samples, undefined = raw_metric_cvs(groups, concern)
        limit = float(cv_bounds[concern])
        if not samples:
            results.append(
                GateResult(
                    name=f"raw_metric_cv[{concern}]",
                    status="not_evaluated",
                    detail=(
                        f"no measurement was produced in every capture of any group "
                        f"({undefined} had a zero mean and no defined CV)"
                    ),
                    bound=limit,
                )
            )
            continue
        value = percentile([s.cv for s in samples], 95)
        worst = max(samples, key=lambda s: s.cv)
        results.append(
            GateResult(
                name=f"raw_metric_cv[{concern}]",
                status=_verdict(value, limit, higher_is_better=False),
                detail=(
                    f"p95 over {len(samples)} (group, ROI, measurement) tuples; "
                    f"median {percentile([s.cv for s in samples], 50):.3f}; "
                    f"worst {worst.cv:.3f} at {worst.roi}/{worst.measurement}; "
                    f"{undefined} undefined (zero mean)"
                ),
                value=value,
                bound=limit,
            )
        )

    # --- ROI availability agreement ----------------------------------------------------
    bound = float(spec["gates"]["roi_availability_agreement"]["min_fraction"])
    for concern in concerns:
        agreed, examined = roi_availability_agreement(groups, concern)
        if examined == 0:
            results.append(
                GateResult(
                    name=f"roi_availability[{concern}]",
                    status="not_evaluated",
                    detail="no group measured any ROI in any capture",
                    bound=bound,
                )
            )
            continue
        value = agreed / examined
        results.append(
            GateResult(
                name=f"roi_availability[{concern}]",
                status=_verdict(value, bound, higher_is_better=True),
                detail=f"{agreed}/{examined} subject-sessions kept the same measurable ROIs",
                value=value,
                bound=bound,
            )
        )

    # --- candidate map stability -------------------------------------------------------
    stability = spec["gates"]["candidate_map_stability"]
    if not stability.get("implemented", False):
        results.append(
            GateResult(
                name="candidate_map_stability",
                status="not_evaluated",
                detail=(
                    f"not implemented: blocked on {stability['blocked_on']}. Unregistered "
                    "IoU across distinct captures measures head motion, and the area "
                    "ratios are not a stand-in -- two disjoint maps of equal area score "
                    "identically and share no pixel."
                ),
            )
        )

    return results


def _slice_ordinal_agreement(
    records: list[Record],
    concern: str,
    gates: dict[str, Any],
    slice_columns: list[str],
    min_subgroup_n: int,
) -> dict[str, dict[str, Any]]:
    """Ordinal agreement recomputed inside each slice value.

    The spread across these is gated separately from the pooled figure, because a build
    can hold 0.95 overall while sitting at 0.70 on one skin tone -- and that build must
    not ship.
    """
    spec = gates["repeatability"]
    session_column = gates["corpus"]["session_column"]
    out: dict[str, dict[str, Any]] = {}

    for column in slice_columns:
        per_value: dict[str, Any] = {}
        for value, subset in partition(records, column).items():
            groups = group_records(subset, session_column, int(spec["min_captures_per_group"]))
            agreed, comparable = ordinal_agreement(groups, concern)
            per_value[value] = {
                "n_images": len(subset),
                "n_subjects": len({r.subject for r in subset}),
                "n_comparable_groups": comparable,
                "agreement": (agreed / comparable) if comparable else None,
                # Reported, not gated. A subgroup too small to gate is a finding about the
                # corpus, and printing it is the only thing that keeps it visible.
                "gated": comparable >= min_subgroup_n,
            }
        if per_value:
            out[column] = per_value
    return out


# ------------------------------------------------------------------------ suite: capture


def run_capture(
    records: list[Record], gates: dict[str, Any], slice_columns: list[str], min_subgroup_n: int
) -> list[GateResult]:
    spec = gates["capture"]["gates"]["rejection_rate"]
    if not records:
        return [GateResult("capture_rejection_rate", "not_evaluated", "no captures")]

    rejected = sum(1 for r in records if not r.qc_passed)
    value = rejected / len(records)
    bound = float(spec["max_fraction"])

    slices: dict[str, dict[str, Any]] = {}
    for column in slice_columns:
        per_value = {}
        for name, subset in partition(records, column).items():
            per_value[name] = {
                "n_images": len(subset),
                "n_subjects": len({r.subject for r in subset}),
                "rejection_rate": sum(1 for r in subset if not r.qc_passed) / len(subset),
                "gated": len(subset) >= min_subgroup_n,
            }
        if per_value:
            slices[column] = per_value

    level = GateResult(
        name="capture_rejection_rate",
        status=_verdict(value, bound, higher_is_better=False),
        detail=f"{rejected}/{len(records)} captures rejected",
        value=value,
        bound=bound,
        slices=slices,
    )

    spread_bound = float(spec["max_slice_spread"])
    spread, where = _max_spread(slices, "rejection_rate")
    if spread is None:
        spread_gate = GateResult(
            name="capture_rejection_slice_spread",
            status="not_evaluated",
            detail=(
                "fewer than two gateable subgroups in any slice. Rejection concentrated "
                "in one subgroup is a fairness problem, and a corpus that cannot be "
                "sliced cannot reveal one."
            ),
            bound=spread_bound,
        )
    else:
        spread_gate = GateResult(
            name="capture_rejection_slice_spread",
            status=_verdict(spread, spread_bound, higher_is_better=False),
            detail=f"widest gap within {where}",
            value=spread,
            bound=spread_bound,
        )

    return [level, spread_gate]


def _max_spread(
    slices: dict[str, dict[str, Any]], key: str
) -> tuple[float | None, str]:
    """Largest max-min gap of ``key`` across the gateable values of any one slice."""
    worst: float | None = None
    where = ""
    for column, values in slices.items():
        numbers = [
            v[key] for v in values.values() if v.get("gated") and v.get(key) is not None
        ]
        if len(numbers) < 2:
            continue
        spread = max(numbers) - min(numbers)
        if worst is None or spread > worst:
            worst, where = spread, column
    return worst, where


# ------------------------------------------------------------------------- suite: slices


def partition(records: list[Record], column: str) -> dict[str, list[Record]]:
    """Records grouped by one manifest column. Blank values are dropped, not bucketed.

    Bucketing them as "unknown" would create a slice that looks like a demographic group
    and is really a data-entry gap.
    """
    out: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        value = record.meta.get(column, "").strip()
        if value:
            out[value].append(record)
    return dict(sorted(out.items()))


def slice_report(records: list[Record], slice_columns: list[str], min_subgroup_n: int) -> str:
    """Per-slice severity distribution and rejection rate. Never a pooled headline."""
    lines: list[str] = []
    concerns = _enabled_concern_values()

    for column in slice_columns:
        buckets = partition(records, column)
        lines.append("")
        if not buckets:
            lines.append(
                f"{column}: ABSENT from the manifest. This run is not evidence about "
                f"{column}."
            )
            continue
        lines.append(f"{column}:")
        for value, subset in buckets.items():
            subjects = len({r.subject for r in subset})
            rejected = sum(1 for r in subset if not r.qc_passed)
            flag = "" if len(subset) >= min_subgroup_n else "   [n below gate floor]"
            lines.append(
                f"  {value:<20} n={len(subset):<4} subjects={subjects:<3} "
                f"rejected={rejected}/{len(subset)}{flag}"
            )
            for concern in concerns:
                tally: dict[str, int] = defaultdict(int)
                for record in subset:
                    # A rejected capture runs no concern logic at all, so its payload has
                    # no entry for this concern. Named rather than left blank: "no finding"
                    # and "never looked" must stay distinguishable in a validation report
                    # exactly as they do in the pipeline (D7).
                    tally[record.severities.get(concern, "not_analysed")] += 1
                spread = "  ".join(f"{k}={v}" for k, v in sorted(tally.items()))
                lines.append(f"      {concern:<16} {spread}")
    return "\n".join(lines)


def run_fairness(records: list[Record], gates: dict[str, Any]) -> list[GateResult]:
    """Is this corpus even capable of supporting a fairness claim?

    Checked as a gate rather than as a warning because the failure it guards against is
    labelling a one-family, one-lighting run as release evidence. This project has already
    shipped a tone-dependent bug that a corpus like that could not have caught.
    """
    spec = gates["fairness"]
    minimum = int(spec["min_values_per_required_slice"])
    results: list[GateResult] = []
    for column in spec["required_slices"]:
        values = partition(records, column)
        results.append(
            GateResult(
                name=f"slice_coverage[{column}]",
                status="pass" if len(values) >= minimum else "fail",
                detail=(
                    f"{len(values)} distinct value(s): {sorted(values) or 'none'}. "
                    f"Needs {minimum}."
                ),
                value=float(len(values)),
                bound=float(minimum),
            )
        )
    return results


# ---------------------------------------------------------------------------- corpus io


def load_manifest(path: Path, required: list[str]) -> dict[str, dict[str, str]]:
    """Read the corpus sidecar CSV, keyed by image filename.

    Raises:
        ValueError: if a required column is missing. Guessing subject identity from a
            filename was considered and rejected -- a wrong guess silently merges two
            people into one "subject", and every repeatability number downstream would be
            comparing strangers.
    """
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = [c for c in required if c not in columns]
        if missing:
            raise ValueError(f"{path}: manifest is missing required column(s) {missing}")
        return {row["image"]: {k: (v or "").strip() for k, v in row.items()} for row in reader}


def content_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324 - dedup only, not security


def collect(
    corpus: Path,
    manifest: dict[str, dict[str, str]],
    *,
    deduplicate: bool,
    run_mode: RunMode,
    limit: int | None,
) -> tuple[list[Record], list[str], int]:
    """Analyse every manifest-listed image. Returns (records, warnings, duplicates dropped)."""
    import cv2

    files = sorted(p for p in corpus.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    warnings: list[str] = []
    seen: dict[str, str] = {}
    duplicates = 0
    records: list[Record] = []

    for path in files:
        if limit is not None and len(records) >= limit:
            break
        meta = manifest.get(path.name)
        if meta is None:
            warnings.append(f"not in manifest, skipped: {path.name}")
            continue
        if deduplicate:
            digest = content_hash(path)
            if digest in seen:
                duplicates += 1
                warnings.append(f"duplicate of {seen[digest]}, skipped: {path.name}")
                continue
            seen[digest] = path.name

        image = cv2.imread(str(path))
        if image is None:
            warnings.append(f"unreadable, skipped: {path.name}")
            continue

        payload = pipeline.analyze_scan_internal(
            image, run_mode=run_mode
        ).to_internal_payload()
        records.append(record_from_payload(path.name, meta, payload))
        print(f"  [{len(records)}] {path.name}: {'PASS' if records[-1].qc_passed else 'RETAKE'}")

    listed = set(manifest) - {r.image for r in records}
    for name in sorted(listed):
        warnings.append(f"listed in manifest but not analysed: {name}")

    return records, warnings, duplicates


# ------------------------------------------------------------------------------- report


def print_gates(results: list[GateResult]) -> None:
    print()
    print(f"{'gate':<40}{'status':<16}{'value':<12}{'bound':<12}detail")
    print("-" * 118)
    for gate in results:
        value = f"{gate.value:.3f}" if gate.value is not None else "-"
        bound = f"{gate.bound:.3f}" if gate.bound is not None else "-"
        print(f"{gate.name:<40}{gate.status.upper():<16}{value:<12}{bound:<12}{gate.detail}")

    for gate in results:
        if not gate.slices:
            continue
        print()
        print(f"{gate.name} by slice:")
        for column, values in gate.slices.items():
            for name, stats in values.items():
                flag = "" if stats.get("gated") else "   [n below gate floor, not gated]"
                shown = {k: v for k, v in stats.items() if k != "gated"}
                print(f"  {column}={name:<16} {shown}{flag}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, default="all")
    parser.add_argument("--corpus", type=Path, required=True, help="directory of captures")
    parser.add_argument("--manifest", type=Path, help="defaults to <corpus>/manifest.csv")
    parser.add_argument("--out", type=Path, help="write the JSON report here")
    parser.add_argument("--limit", type=int, help="stop after N images")
    parser.add_argument(
        "--run-mode",
        choices=[m.value for m in RunMode],
        default=RunMode.DEVELOPMENT.value,
        help=(
            "development is the default because config/rois.yaml meta.verified is still "
            "false (D15). A development run is NOT release evidence."
        ),
    )
    parser.add_argument(
        "--allow-unevaluated",
        action="store_true",
        help="exit 0 despite gates that could not be evaluated. Development only.",
    )
    args = parser.parse_args()

    if not args.corpus.is_dir():
        print(f"not a directory: {args.corpus}", file=sys.stderr)
        return 2

    gates = cfg.load("validation_gates")
    corpus_spec = gates["corpus"]
    manifest_path = args.manifest or args.corpus / corpus_spec["manifest_filename"]
    if not manifest_path.exists():
        print(
            f"no manifest at {manifest_path}.\n\n"
            "Real face images are never committed (CLAUDE.md §5), so the corpus lives "
            "outside the repo and this CSV is what makes it interpretable. Minimum:\n\n"
            "    image,subject_id,session_id,skin_tone,device,lighting,age_band,"
            "makeup_facial_hair\n"
            "    IMG_0001.jpg,s01,a,mst-06,pixel-8,natural,30-39,none\n",
            file=sys.stderr,
        )
        return 2

    try:
        manifest = load_manifest(manifest_path, list(corpus_spec["required_columns"]))
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    run_mode = RunMode(args.run_mode)
    roi_verified = bool(cfg.load("rois")["meta"].get("verified", False))
    calibrated = cfg.severity_calibrated()

    print(f"corpus:     {args.corpus}")
    print(f"manifest:   {manifest_path} ({len(manifest)} rows)")
    print(f"gate set:   {gates['meta']['gate_set_version']} "
          f"(prespecified {gates['meta']['prespecified_on']})")
    print(f"run mode:   {run_mode.value}")
    print()

    records, warnings, duplicates = collect(
        args.corpus,
        manifest,
        deduplicate=bool(corpus_spec["deduplicate_by_content_hash"]),
        run_mode=run_mode,
        limit=args.limit,
    )
    if not records:
        print("no analysable captures", file=sys.stderr)
        return 2

    slice_columns = list(corpus_spec["slice_columns"])
    min_subgroup_n = int(corpus_spec["min_subgroup_n"])
    session_column = corpus_spec["session_column"]

    results: list[GateResult] = []
    if args.suite in ("repeatability", "all"):
        results += run_repeatability(records, gates, slice_columns, min_subgroup_n)
    if args.suite in ("capture", "all"):
        results += run_capture(records, gates, slice_columns, min_subgroup_n)
    if args.suite in ("slices", "all"):
        results += run_fairness(records, gates)

    print_gates(results)

    if args.suite in ("slices", "all"):
        print()
        print("=" * 78)
        print("SLICED REPORT -- there is deliberately no pooled headline here")
        print("=" * 78)
        print(slice_report(records, slice_columns, min_subgroup_n))

    # Conditions that invalidate the run as release evidence, printed together so a
    # green gate table is never mistaken for a release sign-off on its own.
    caveats: list[str] = []
    if run_mode is not RunMode.PRODUCTION:
        caveats.append("ran in development mode")
    if not roi_verified:
        caveats.append("config/rois.yaml meta.verified is false (D15)")
    if not calibrated:
        caveats.append(
            "severity thresholds are uncalibrated (D2), so every ordinal gate is vacuous"
        )
    if not any(session_column in r.meta for r in records):
        caveats.append(
            f"manifest has no {session_column!r} column, so every capture of a subject "
            "was treated as one session -- cross-session skin change is mixed into what "
            "is reported as measurement noise"
        )

    if warnings:
        print()
        print(f"corpus warnings ({len(warnings)}, {duplicates} duplicate files dropped):")
        for line in warnings[:40]:
            print(f"  {line}")
        if len(warnings) > 40:
            print(f"  ... {len(warnings) - 40} more")

    failed = [g for g in results if g.failed]
    unevaluated = [g for g in results if not g.evaluated]

    print()
    print(f"gates: {len(results)}   passed: {len(results) - len(failed) - len(unevaluated)}   "
          f"failed: {len(failed)}   not evaluated: {len(unevaluated)}")

    if caveats:
        print()
        print("THIS RUN IS NOT RELEASE EVIDENCE:")
        for line in caveats:
            print(f"  - {line}")

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "gate_set_version": gates["meta"]["gate_set_version"],
                    "protocol_version": cfg.protocol_version(),
                    "run_mode": run_mode.value,
                    "n_captures": len(records),
                    "n_subjects": len({r.subject for r in records}),
                    "duplicates_dropped": duplicates,
                    "release_evidence": not caveats,
                    "caveats": caveats,
                    "gates": [
                        {
                            "name": g.name,
                            "status": g.status,
                            "value": g.value,
                            "bound": g.bound,
                            "detail": g.detail,
                            "slices": g.slices,
                        }
                        for g in results
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")

    if failed:
        return 1
    if unevaluated and not args.allow_unevaluated:
        # An unevaluated gate is not a passed gate. --allow-unevaluated exists for
        # development runs and must not be used to reach a release decision.
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
