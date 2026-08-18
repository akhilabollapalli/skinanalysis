"""Fit population reference statistics and ordinal thresholds (D1 stage B).

    python scripts/calibrate.py --corpus <local path> --annotations labels.csv \
        --out-root data/reference/v1 --report calibration_report.json

Every module in this codebase that mentions "a calibration run" -- decision/standardize.py,
util/calibration.py, docs/DECISIONS.md D1/D2 -- has, until now, described this script
without it existing. This is that script.

------------------------------------------------------------------------------------------
WHAT THIS SCRIPT DOES NOT DO

It does not flip severity_thresholds.yaml `meta.calibrated`, and it does not flip a
reference manifest's `frozen` flag. Those two fields are what let ScanResultInternal.to_
public() hand a severity to a real user (util/calibration.py assert_public_ready). Setting
them is a human sign-off, not a computation, and no code path in this project may
substitute a calibrated result for a decision nobody made (CLAUDE.md §6).

This script MEASURES AND FITS. It writes proposed reference statistics under
data/reference/v1/, with each file's own `calibrated` flag set only when the evidence
clears the prespecified floor in config/calibration_gates.yaml, and it prints a report.
Whether that report is convincing enough to ship is for a person to decide, by editing
severity_thresholds.yaml and the reference manifest.json themselves, having read it.
------------------------------------------------------------------------------------------

METHOD, IN ONE PARAGRAPH. Subjects are split by subject_id into train/holdout (D2:
"subject_level" holdout -- never split by image, or the same face's repeat captures end up
on both sides). Reference median/MAD are computed from TRAIN subjects only, one value per
subject per (concern, ROI, measurement) -- averaged across that subject's own captures
first, so a subject photographed forty times does not outweigh one photographed twice.
Every annotated (subject, concern, ROI) is then standardized against those TRAIN
statistics using the exact production functions (decision.standardize.robust_z,
decision.calibrator.combine), so calibration-time math can never drift from serving-time
math. Ordinal thresholds are fit on the TRAIN side by exact dynamic programming: sorted by
combined z, split into 4 contiguous bins labelled NOT_DETECTED..HIGH in order to minimize
mislabelled subjects. This assumes the direction convention in each concern's
`decision.direction` is correct -- monotonic bins are only meaningful if higher z already
means worse. Holdout subjects, never touched during fitting, are scored against those cuts
to produce the numbers in config/calibration_gates.yaml.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from skin_analysis import pipeline  # noqa: E402
from skin_analysis.decision import calibrator, standardize  # noqa: E402
from skin_analysis.schemas import Concern, RunMode, Severity  # noqa: E402
from skin_analysis.util import config as cfg  # noqa: E402

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")

#: Ordinal order the DP fit assigns to the 4 contiguous z-sorted bins, low to high. Must
#: match decision.calibrator.to_severity's own band order, or a fitted threshold would mean
#: something different at fit time than it does in production.
_BANDS = (Severity.NOT_DETECTED, Severity.MILD, Severity.MODERATE, Severity.HIGH)
_BAND_RANK = {s.value: i for i, s in enumerate(_BANDS)}


def _enabled_concerns() -> list[Concern]:
    """Enabled concerns as Concern objects -- see run_validation._enabled_concern_values
    for why config keys (``pigmentation``) and Concern.value (``dark_spots``) must not be
    used interchangeably. Reusing pipeline._CONCERN_OF here for the same reason."""
    return [c for k, c in pipeline._CONCERN_OF.items() if cfg.feature_enabled(k)]


# --------------------------------------------------------------------------- corpus pass


@dataclass(frozen=True)
class Capture:
    """One analysed image's raw measurements, before any subject-level averaging."""

    image: str
    subject_id: str
    #: (concern value, roi value) -> measurement name -> value.
    raw: dict[tuple[str, str], dict[str, float]]


def capture_from_payload(image: str, subject_id: str, payload: dict[str, Any]) -> Capture:
    raw: dict[tuple[str, str], dict[str, float]] = {}
    for concern, block in (payload.get("concerns", {}) or {}).items():
        for rr in block.get("roi_results", []):
            values = rr.get("raw") or {}
            if values:
                raw[(concern, str(rr["roi"]))] = {k: float(v) for k, v in values.items()}
    return Capture(image=image, subject_id=subject_id, raw=raw)


def content_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324 - dedup only


def collect_captures(
    corpus: Path,
    manifest: dict[str, dict[str, str]],
    *,
    run_mode: RunMode,
    limit: int | None,
) -> tuple[list[Capture], list[str]]:
    """Analyse every manifest-listed image. Mirrors run_validation.collect(), separately:
    calibrate.py needs raw measurements only, never QC pass/fail or severities, and keeping
    the two scripts independent means neither can be broken by refactoring the other."""
    import cv2

    files = sorted(p for p in corpus.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    warnings: list[str] = []
    seen: dict[str, str] = {}
    captures: list[Capture] = []

    for path in files:
        if limit is not None and len(captures) >= limit:
            break
        meta = manifest.get(path.name)
        if meta is None:
            warnings.append(f"not in manifest, skipped: {path.name}")
            continue
        digest = content_hash(path)
        if digest in seen:
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
        captures.append(capture_from_payload(path.name, meta["subject_id"], payload))
        print(f"  [{len(captures)}] {path.name}")

    return captures, warnings


# --------------------------------------------------------------- subject-level aggregation


def subject_level_raw(
    captures: list[Capture],
) -> dict[str, dict[tuple[str, str], dict[str, float]]]:
    """Mean raw measurements per subject, per (concern, ROI), across that subject's own
    captures.

    This is the step that keeps reference statistics describing SUBJECT-TO-SUBJECT
    variation rather than repeat-capture noise. Without it, a subject photographed forty
    times contributes forty correlated samples to the cohort median while a subject
    photographed twice contributes two -- the "cohort" statistic would really describe
    whoever brought the most selfies, not the population.
    """
    by_subject: dict[str, dict[tuple[str, str], dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for capture in captures:
        for key, values in capture.raw.items():
            for name, value in values.items():
                by_subject[capture.subject_id][key][name].append(value)

    return {
        subject: {
            key: {name: statistics.fmean(values) for name, values in measurements.items()}
            for key, measurements in per_key.items()
        }
        for subject, per_key in by_subject.items()
    }


def split_subjects(
    subject_ids: list[str], holdout_fraction: float, salt: str
) -> tuple[list[str], list[str]]:
    """Deterministic subject-level train/holdout split (D2: holdout is subject_level).

    Hash-based rather than random.shuffle: re-running this script on the same corpus with
    the same salt always produces the same split, which matters because a reference set
    that silently reshuffles its holdout on every run cannot be audited or reproduced
    (docs/DECISIONS.md: "produced by a calibration run... immutable once frozen" -- the
    split itself needs the same property while calibration is still in progress).
    """

    def bucket(subject: str) -> float:
        digest = hashlib.sha256(f"{salt}:{subject}".encode()).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF

    holdout = sorted(s for s in subject_ids if bucket(s) < holdout_fraction)
    train = sorted(s for s in subject_ids if bucket(s) >= holdout_fraction)
    return train, holdout


# ------------------------------------------------------------------------ reference stats


@dataclass(frozen=True)
class ReferenceStats:
    """One concern's proposed data/reference/v1/{concern}.json content."""

    concern: str
    calibrated: bool
    n_subjects: int
    median: dict[str, dict[str, float]]
    mad: dict[str, dict[str, float]]
    reason: str = ""

    def as_json(self) -> dict[str, Any]:
        return {
            "calibrated": self.calibrated,
            "n_subjects": self.n_subjects,
            "median": self.median,
            "mad": self.mad,
            "reason": self.reason,
        }


def compute_reference_stats(
    concern: Concern,
    subject_raw: dict[str, dict[tuple[str, str], dict[str, float]]],
    train_subjects: list[str],
    min_cohort_subjects: int,
) -> ReferenceStats:
    """Median/MAD per (ROI, measurement) from TRAIN subjects' averaged raw measurements.

    Values are stored UNSCALED (raw median, raw MAD), matching what
    decision.standardize.robust_z expects to divide by (it applies MAD_SCALE itself) --
    storing a pre-scaled MAD here would silently double-apply the scale factor everywhere
    this reference file is read.
    """
    per_roi: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    contributing: set[str] = set()

    for subject in train_subjects:
        for (c, roi), measurements in subject_raw.get(subject, {}).items():
            if c != concern.value:
                continue
            contributing.add(subject)
            for name, value in measurements.items():
                per_roi[roi][name].append(value)

    median: dict[str, dict[str, float]] = {}
    mad: dict[str, dict[str, float]] = {}
    for roi, by_measurement in per_roi.items():
        median[roi] = {}
        mad[roi] = {}
        for name, values in by_measurement.items():
            arr = np.asarray(values, dtype=np.float64)
            med = float(np.median(arr))
            median[roi][name] = med
            mad[roi][name] = float(np.median(np.abs(arr - med)))

    n_subjects = len(contributing)
    if n_subjects < min_cohort_subjects:
        return ReferenceStats(
            concern=concern.value,
            calibrated=False,
            n_subjects=n_subjects,
            median=median,
            mad=mad,
            reason=(
                f"{n_subjects} contributing TRAIN subjects, below the floor of "
                f"{min_cohort_subjects} (severity_thresholds.yaml "
                "reference_stats.min_cohort_subjects). A median and MAD from this few "
                "subjects look entirely normal and mean nothing."
            ),
        )
    return ReferenceStats(
        concern=concern.value, calibrated=True, n_subjects=n_subjects, median=median, mad=mad
    )


# --------------------------------------------------------------------------- annotations


@dataclass(frozen=True)
class Label:
    subject_id: str
    concern: str
    roi: str
    severity: str


def load_annotations(path: Path, spec: dict[str, Any]) -> tuple[list[Label], list[str]]:
    """Read the annotation CSV. One row per (subject, concern, ROI): the annotator's
    judgement of the SUBJECT in that region, not of one photo (config/calibration_gates.yaml
    `annotations`). Duplicate keys are resolved by majority vote and reported, not averaged
    -- averaging two different raters' ordinal labels produces a value that is not itself an
    ordinal label.

    Raises:
        ValueError: a required column is missing, or a severity value is not one of the
            four the pipeline can ever produce. Silently dropping an unrecognised value
            would fit thresholds against a quietly smaller, self-selected label set.
    """
    required = list(spec["required_columns"])
    allowed = set(spec["severity_values"])
    grouped: dict[tuple[str, str, str], list[str]] = defaultdict(list)

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = [c for c in required if c not in columns]
        if missing:
            raise ValueError(f"{path}: annotations missing required column(s) {missing}")
        for row in reader:
            severity = row["severity"].strip()
            if severity not in allowed:
                raise ValueError(
                    f"{path}: severity {severity!r} is not one of {sorted(allowed)}"
                )
            key = (row["subject_id"].strip(), row["concern"].strip(), row["roi"].strip())
            grouped[key].append(severity)

    labels: list[Label] = []
    disagreements: list[str] = []
    for (subject, concern, roi), values in grouped.items():
        tally = Counter(values)
        winner, count = tally.most_common(1)[0]
        if len(tally) > 1:
            disagreements.append(
                f"{subject}/{concern}/{roi}: {dict(tally)} -> took {winner} ({count}/{len(values)})"
            )
        labels.append(Label(subject, concern, roi, winner))

    return labels, disagreements


# --------------------------------------------------------------------- threshold fitting


@dataclass(frozen=True)
class Sample:
    subject_id: str
    z: float
    label: Severity


def combined_samples(
    concern: Concern,
    concern_config: dict,
    labels: list[Label],
    subject_raw: dict[str, dict[tuple[str, str], dict[str, float]]],
    reference: ReferenceStats,
    eps: float,
) -> list[Sample]:
    """One combined z_ref per labelled (subject, ROI), using the SAME production functions
    (decision.standardize.robust_z, decision.calibrator.combine) that decide.py uses at
    serving time -- so a threshold fit here means exactly what it will mean in production.

    A label whose subject has no raw measurement for that ROI, or whose ROI has no
    reference statistics yet (too few TRAIN subjects touched it), is skipped rather than
    imputed: standardizing against absent statistics silently produces zeros, which read as
    "perfectly average" (decision.standardize docstring) and would corrupt the fit exactly
    the way it would corrupt a real scan.
    """
    samples: list[Sample] = []
    for label in labels:
        if label.concern != concern.value:
            continue
        measurements = subject_raw.get(label.subject_id, {}).get((concern.value, label.roi))
        roi_median = reference.median.get(label.roi)
        roi_mad = reference.mad.get(label.roi)
        if not measurements or not roi_median or not roi_mad:
            continue
        if not set(measurements) <= set(roi_median):
            continue
        z_ref = {
            name: standardize.robust_z(value, roi_median[name], roi_mad[name], eps)
            for name, value in measurements.items()
        }
        combined = calibrator.combine(z_ref, concern_config)
        samples.append(Sample(label.subject_id, combined, Severity(label.severity)))
    return samples


def fit_thresholds(samples: list[Sample]) -> tuple[dict[str, float], int]:
    """Optimal monotonic 4-band threshold triple via exact DP.

    Sorted by z, the samples are split into 4 CONTIGUOUS ranges assigned the fixed labels
    NOT_DETECTED, MILD, MODERATE, HIGH in that order -- not re-labelled by majority, since
    the whole point of a threshold classifier is that severity is monotonic in z (D1: "does
    this area differ from surrounding skin" -> combined and standardized -> assumed to run
    one direction, matching decision.direction's signs). The DP finds the 3 cut points that
    minimize the number of misclassified TRAIN samples under that constraint; a brute-force
    or a mean-based split would either be too slow at real cohort sizes or ignore that
    ordinal structure.

    Returns:
        (thresholds dict with t0/t1/t2, misclassified count on the samples given). t0/t1/t2
        are the midpoints between the z value below and above each cut, so a novel z lands
        unambiguously in a bin at serving time.

    Raises:
        ValueError: fewer than 4 samples. Four contiguous bins need at least one sample
            each, or the "optimum" is an artifact of an empty bin.
    """
    if len(samples) < len(_BANDS):
        raise ValueError(
            f"need at least {len(_BANDS)} labelled samples to fit 4 bands, got {len(samples)}"
        )

    ordered = sorted(samples, key=lambda s: s.z)
    n = len(ordered)
    ranks = np.array([_BAND_RANK[s.label.value] for s in ordered])

    # cum[k, j] = count of label k in ordered[:j]. O(1) misclassification lookups: the
    # cost of assigning label `k` to the contiguous range [i, j) is (j - i) minus how many
    # of those points already carry label k.
    cum = np.zeros((len(_BANDS), n + 1), dtype=np.int64)
    for k in range(len(_BANDS)):
        cum[k, 1:] = np.cumsum(ranks == k)

    def cost(i: int, j: int, k: int) -> int:
        return int((j - i) - (cum[k, j] - cum[k, i]))

    NEG = 10**9
    # dp[k][j] = min misclassifications covering ordered[:j] with the first k+1 bands.
    dp = [[NEG] * (n + 1) for _ in range(len(_BANDS))]
    back = [[0] * (n + 1) for _ in range(len(_BANDS))]
    for j in range(n + 1):
        dp[0][j] = cost(0, j, 0)
    for k in range(1, len(_BANDS)):
        for j in range(n + 1):
            best, best_i = NEG, 0
            for i in range(j + 1):
                candidate = dp[k - 1][i] + cost(i, j, k)
                if candidate < best:
                    best, best_i = candidate, i
            dp[k][j], back[k][j] = best, best_i

    cuts: list[int] = []
    j = n
    for k in range(len(_BANDS) - 1, 0, -1):
        i = back[k][j]
        cuts.append(i)
        j = i
    cuts.reverse()  # cuts[0] = start of MILD, cuts[1] = start of MODERATE, cuts[2] = start of HIGH

    def midpoint(index: int) -> float:
        if index <= 0:
            return ordered[0].z - 1.0
        if index >= n:
            return ordered[-1].z + 1.0
        return (ordered[index - 1].z + ordered[index].z) / 2.0

    thresholds = {
        "t0": midpoint(cuts[0]),
        "t1": midpoint(cuts[1]),
        "t2": midpoint(cuts[2]),
    }
    # A degenerate corpus (every label identical, or z ties straddling a cut) can produce
    # a non-increasing triple. Nudging apart would hide that; refusing is the honest answer.
    if not thresholds["t0"] < thresholds["t1"] < thresholds["t2"]:
        raise ValueError(
            f"fitted thresholds are not strictly increasing: {thresholds}. The label "
            "distribution on this TRAIN split cannot support 4 distinct bands -- likely "
            "too few subjects, or one band almost entirely absent."
        )
    return thresholds, dp[len(_BANDS) - 1][n]


def evaluate(samples: list[Sample], thresholds: dict[str, float]) -> dict[str, Any]:
    """Exact and adjacent-band agreement between fitted thresholds and annotated labels."""
    if not samples:
        return {"n": 0, "exact_agreement": None, "adjacent_agreement": None}
    exact = 0
    adjacent = 0
    for sample in samples:
        predicted = calibrator.to_severity(sample.z, thresholds)
        gap = abs(predicted.rank - sample.label.rank)
        exact += gap == 0
        adjacent += gap <= 1
    return {
        "n": len(samples),
        "exact_agreement": exact / len(samples),
        "adjacent_agreement": adjacent / len(samples),
    }


# ------------------------------------------------------------------------------- reporting


@dataclass
class ConcernReport:
    concern: str
    reference: ReferenceStats
    thresholds: dict[str, float] | None
    train_eval: dict[str, Any]
    holdout_eval: dict[str, Any]
    slice_spread: dict[str, dict[str, Any]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def gate_verdicts(self, gates: dict[str, Any]) -> list[tuple[str, str, str]]:
        """(gate name, verdict, detail) against config/calibration_gates.yaml."""
        spec = gates["thresholds"]["gates"]
        out: list[tuple[str, str, str]] = []

        if self.holdout_eval.get("n", 0) == 0:
            out.append(("holdout_evaluated", "not_evaluated", "no holdout samples"))
            return out

        exact = self.holdout_eval["exact_agreement"]
        bound = float(spec["exact_agreement"]["min_fraction"])
        out.append((
            "exact_agreement",
            "pass" if exact >= bound else "fail",
            f"{exact:.3f} (n={self.holdout_eval['n']}) vs floor {bound:.3f}",
        ))

        adjacent = self.holdout_eval["adjacent_agreement"]
        bound = float(spec["adjacent_agreement"]["min_fraction"])
        out.append((
            "adjacent_agreement",
            "pass" if adjacent >= bound else "fail",
            f"{adjacent:.3f} vs floor {bound:.3f}",
        ))

        if self.slice_spread:
            spread_bound = float(spec["max_slice_spread"]["exact_agreement"])
            for column, per_value in self.slice_spread.items():
                gateable = [
                    v["exact_agreement"] for v in per_value.values()
                    if v.get("gated") and v.get("exact_agreement") is not None
                ]
                if len(gateable) < 2:
                    out.append((
                        f"slice_spread[{column}]", "not_evaluated",
                        "fewer than two gateable slice values",
                    ))
                    continue
                spread = max(gateable) - min(gateable)
                out.append((
                    f"slice_spread[{column}]",
                    "pass" if spread <= spread_bound else "fail",
                    f"{spread:.3f} vs floor {spread_bound:.3f}",
                ))

        return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="defaults to <corpus>/manifest.csv")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "data/reference/v1")
    parser.add_argument("--report", type=Path, help="write the JSON report here")
    parser.add_argument("--limit", type=int, help="stop after N images")
    parser.add_argument(
        "--holdout-salt",
        default="v1",
        help="changes the train/holdout split deterministically; bump only when starting a "
        "new protocol version, never to search for a favourable split",
    )
    parser.add_argument(
        "--run-mode",
        choices=[m.value for m in RunMode],
        default=RunMode.DEVELOPMENT.value,
        help="development is the default because config/rois.yaml meta.verified is false",
    )
    args = parser.parse_args()

    if not args.corpus.is_dir():
        print(f"not a directory: {args.corpus}", file=sys.stderr)
        return 2

    validation_gates = cfg.load("validation_gates")
    calibration_gates = cfg.load("calibration_gates")
    severity_cfg = cfg.load("severity_thresholds")
    min_cohort_subjects = int(
        (severity_cfg.get("reference_stats", {}) or {}).get("min_cohort_subjects", 200)
    )
    eps = float((severity_cfg.get("reference_stats", {}) or {}).get("eps", 1e-6))

    manifest_path = args.manifest or args.corpus / validation_gates["corpus"]["manifest_filename"]
    if not manifest_path.exists():
        print(f"no manifest at {manifest_path}", file=sys.stderr)
        return 2
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "subject_id" not in (reader.fieldnames or []):
            print(f"{manifest_path}: manifest has no subject_id column", file=sys.stderr)
            return 2
        manifest = {row["image"]: row for row in reader}

    if not args.annotations.exists():
        print(f"no annotations at {args.annotations}", file=sys.stderr)
        return 2
    try:
        labels, disagreements = load_annotations(args.annotations, calibration_gates["annotations"])
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    run_mode = RunMode(args.run_mode)
    print(f"corpus:      {args.corpus}")
    print(f"annotations: {args.annotations} ({len(labels)} labelled subject-regions)")
    print(f"run mode:    {run_mode.value}")
    if disagreements:
        print(f"\nannotator disagreements ({len(disagreements)}), resolved by majority vote:")
        for line in disagreements[:20]:
            print(f"  {line}")
    print()

    captures, warnings = collect_captures(
        args.corpus, manifest, run_mode=run_mode, limit=args.limit
    )
    if not captures:
        print("no analysable captures", file=sys.stderr)
        return 2
    subject_raw = subject_level_raw(captures)

    slice_columns = list(validation_gates["corpus"]["slice_columns"])
    min_subgroup_n = int(validation_gates["corpus"]["min_subgroup_n"])
    holdout_fraction = float(calibration_gates["cohort"]["holdout_fraction"])
    train_subjects, holdout_subjects = split_subjects(
        sorted(subject_raw), holdout_fraction, args.holdout_salt
    )
    print(
        f"subjects: {len(subject_raw)} total, {len(train_subjects)} train, "
        f"{len(holdout_subjects)} holdout (salt={args.holdout_salt!r})"
    )

    reports: list[ConcernReport] = []
    for concern in _enabled_concerns():
        concern_config = cfg.concern_config(_config_key(concern))
        reference = compute_reference_stats(
            concern, subject_raw, train_subjects, min_cohort_subjects
        )
        notes = [reference.reason] if reference.reason else []

        train_samples = combined_samples(
            concern, concern_config, labels, subject_raw, reference, eps
        )
        # Only TRAIN subjects may inform the fit.
        train_samples = [s for s in train_samples if s.subject_id in set(train_subjects)]
        holdout_samples = combined_samples(
            concern, concern_config, labels, subject_raw, reference, eps
        )
        holdout_samples = [s for s in holdout_samples if s.subject_id in set(holdout_subjects)]

        thresholds: dict[str, float] | None = None
        train_eval: dict[str, Any] = {"n": len(train_samples)}
        holdout_eval: dict[str, Any] = {"n": len(holdout_samples)}
        try:
            thresholds, misclassified = fit_thresholds(train_samples)
            train_eval = evaluate(train_samples, thresholds)
            train_eval["misclassified"] = misclassified
            holdout_eval = evaluate(holdout_samples, thresholds)
        except ValueError as error:
            notes.append(f"threshold fit skipped: {error}")

        slice_spread: dict[str, dict[str, Any]] = {}
        if thresholds is not None:
            slice_spread = _slice_holdout(
                concern, thresholds, calibration_gates, labels, subject_raw, reference,
                eps, manifest, holdout_subjects, slice_columns, min_subgroup_n,
            )

        reports.append(
            ConcernReport(
                concern=concern.value,
                reference=reference,
                thresholds=thresholds,
                train_eval=train_eval,
                holdout_eval=holdout_eval,
                slice_spread=slice_spread,
                notes=notes,
            )
        )

    _print_report(reports, calibration_gates)

    args.out_root.mkdir(parents=True, exist_ok=True)
    for report in reports:
        path = args.out_root / f"{report.concern}.json"
        path.write_text(json.dumps(report.reference.as_json(), indent=2), encoding="utf-8")
        print(f"wrote {path}  (calibrated={report.reference.calibrated})")

    print(
        "\nThis script did NOT edit severity_thresholds.yaml or a reference manifest.json. "
        "Review the report above, then a person sets meta.calibrated and manifest.json "
        "frozen by hand once satisfied (D2, CLAUDE.md §6)."
    )

    if warnings:
        print(f"\ncorpus warnings ({len(warnings)}):")
        for line in warnings[:20]:
            print(f"  {line}")

    if args.report:
        args.report.write_text(
            json.dumps(
                {
                    "gate_set_version": calibration_gates["meta"]["gate_set_version"],
                    "protocol_version": cfg.protocol_version(),
                    "run_mode": run_mode.value,
                    "n_subjects_total": len(subject_raw),
                    "n_subjects_train": len(train_subjects),
                    "n_subjects_holdout": len(holdout_subjects),
                    "annotator_disagreements": disagreements,
                    "concerns": [
                        {
                            "concern": r.concern,
                            "calibrated": r.reference.calibrated,
                            "n_reference_subjects": r.reference.n_subjects,
                            "thresholds": r.thresholds,
                            "train_eval": r.train_eval,
                            "holdout_eval": r.holdout_eval,
                            "slice_spread": r.slice_spread,
                            "gates": [
                                {"name": n, "status": s, "detail": d}
                                for n, s, d in r.gate_verdicts(calibration_gates)
                            ],
                            "notes": r.notes,
                        }
                        for r in reports
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.report}")

    return 0


def _config_key(concern: Concern) -> str:
    """Inverse of pipeline._CONCERN_OF: Concern -> the config key it is defined under."""
    for key, value in pipeline._CONCERN_OF.items():
        if value is concern:
            return key
    raise KeyError(concern)


def _slice_holdout(
    concern: Concern,
    thresholds: dict[str, float],
    calibration_gates: dict[str, Any],
    labels: list[Label],
    subject_raw: dict[str, dict[tuple[str, str], dict[str, float]]],
    reference: ReferenceStats,
    eps: float,
    manifest: dict[str, dict[str, str]],
    holdout_subjects: list[str],
    slice_columns: list[str],
    min_subgroup_n: int,
) -> dict[str, dict[str, Any]]:
    """Holdout exact-agreement recomputed inside each slice value.

    A fitted threshold that scores well pooled and badly on one skin tone must not ship
    (same principle as run_validation.py's ordinal-agreement slicing), and slicing only
    the pooled headline would hide exactly that failure.
    """
    concern_config = cfg.concern_config(_config_key(concern))
    subject_slice: dict[str, dict[str, str]] = defaultdict(dict)
    for row in manifest.values():
        subject = row.get("subject_id", "")
        for column in slice_columns:
            value = (row.get(column) or "").strip()
            if subject and value:
                subject_slice[column][subject] = value

    out: dict[str, dict[str, Any]] = {}
    holdout_set = set(holdout_subjects)
    for column, by_subject in subject_slice.items():
        per_value: dict[str, list[Sample]] = defaultdict(list)
        all_samples = combined_samples(concern, concern_config, labels, subject_raw, reference, eps)
        for sample in all_samples:
            if sample.subject_id not in holdout_set:
                continue
            slice_value = by_subject.get(sample.subject_id)
            if slice_value:
                per_value[slice_value].append(sample)
        if not per_value:
            continue
        out[column] = {
            value: {**evaluate(samples, thresholds), "gated": len(samples) >= min_subgroup_n}
            for value, samples in per_value.items()
        }
    return out


def _print_report(reports: list[ConcernReport], calibration_gates: dict[str, Any]) -> None:
    print()
    print(f"{'concern':<14}{'calibrated':<12}{'ref_subj':<10}{'train_n':<9}{'holdout_n':<10}"
          f"{'exact':<8}{'adjacent':<9}thresholds")
    print("-" * 100)
    for r in reports:
        exact = r.holdout_eval.get("exact_agreement")
        adjacent = r.holdout_eval.get("adjacent_agreement")
        print(
            f"{r.concern:<14}{str(r.reference.calibrated):<12}{r.reference.n_subjects:<10}"
            f"{r.train_eval.get('n', 0):<9}{r.holdout_eval.get('n', 0):<10}"
            f"{'-' if exact is None else f'{exact:.3f}':<8}"
            f"{'-' if adjacent is None else f'{adjacent:.3f}':<9}"
            f"{r.thresholds or '-'}"
        )

    print()
    for r in reports:
        verdicts = r.gate_verdicts(calibration_gates)
        if not verdicts:
            continue
        print(f"{r.concern} gates:")
        for name, status, detail in verdicts:
            print(f"  {name:<24}{status.upper():<16}{detail}")
        for note in r.notes:
            print(f"  note: {note}")


if __name__ == "__main__":
    raise SystemExit(main())
