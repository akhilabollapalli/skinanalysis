"""Stage B7 -- collect capture-QC metric distributions over a local corpus.

This script MEASURES. It does not tune. B8 tunes, using what this produces.

    python scripts/qc_report.py --corpus data/raw/sfhq --out qc_metrics.csv

Why the two are separate steps: every threshold in ``config/capture_thresholds.yaml`` is
currently a placeholder chosen by reasoning rather than by evidence, and the honest way to
replace one is to look at where real captures actually fall. Adjusting a threshold until a
particular corpus passes is not tuning, it is overfitting to whatever that corpus is.

    ------------------------------------------------------------------------------
    WHICH CORPORA MAY SET A THRESHOLD

    SFHQ is GENERATIVE (StyleGAN). CLAUDE.md permits it for infrastructure and QC
    stress tests ONLY, never as skin-concern truth -- and blur, specular coverage and
    texture statistics are precisely where a generative image differs most from a
    camera. A StyleGAN face has no lens, no sensor noise, and no real defocus.

    So: run this against SFHQ to check the gate WORKS and to find checks that are dead
    or that fire on everything. Do NOT move `blur`, `exposure` or `specular` thresholds
    to fit it. Those need captures from real devices under the V1 protocol.
    ------------------------------------------------------------------------------

Every value written here is internal (Rule 3). The CSV is a validation artifact; nothing in
it may reach an application payload.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from skin_analysis.capture import qc  # noqa: E402
from skin_analysis.face import landmarks as landmark_layer  # noqa: E402
from skin_analysis.face import rois as roi_layer  # noqa: E402
from skin_analysis.face import skin_mask as mask_layer  # noqa: E402
from skin_analysis.schemas import RunMode  # noqa: E402
from skin_analysis.util import config as cfg  # noqa: E402
from skin_analysis.util import scale  # noqa: E402

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")

#: Percentiles reported per metric. p05/p95 matter more than the mean here: a QC threshold
#: is a decision about a tail, and a mean says nothing about where a tail sits.
PERCENTILES = (5, 25, 50, 75, 95)


def observe(image: np.ndarray, profile: dict, roi_config: dict) -> dict[str, Any]:
    """Run the face layer and the QC gate over one image, returning every metric.

    DEVELOPMENT run mode: ``config/rois.yaml meta.verified`` is still false, and this
    script is part of the workflow that produces that verification (D15).
    """
    faces = landmark_layer.detect_faces(image, profile)
    if not faces:
        capture = qc.check(image, profile, face=None)
        return {"n_faces": 0, "passed": False, "failures": _reasons(capture), **capture.metrics}

    points = faces[0]
    if points.shape[0] <= scale.RIGHT_IRIS_CENTER:
        capture = qc.check(image, profile, face=None)
        return {"n_faces": len(faces), "passed": False, "failures": "no_iris_landmarks"}

    anchor_px = scale.inter_ocular_distance(points)
    polygons = roi_layer.build(
        points, image.shape[:2], roi_config, run_mode=RunMode.DEVELOPMENT
    )
    mask, diagnostics = mask_layer.build_with_diagnostics(
        image, points, roi_config, roi_polygons=polygons
    )
    composed = roi_layer.compose(polygons, mask)
    visibility = roi_layer.measurable_fraction(polygons, composed)
    undersized = set(roi_layer.undersized(composed, anchor_px, roi_config))

    capture = qc.check(
        image,
        profile,
        face=qc.FaceObservation(
            n_faces=len(faces),
            face_box=_face_box(points, image.shape[:2]),
            pose_deg=landmark_layer.head_pose(points, image.shape[:2]),
            skin_mask=mask,
            roi_visibility=visibility,
            occlusion={
                stage: float((diagnostics.get("rejected_frac_of_face", {}) or {}).get(stage, 0.0))
                for stage in ("hair", "beard", "glasses")
            },
        ),
    )

    row: dict[str, Any] = {
        "n_faces": len(faces),
        "passed": capture.passed,
        "failures": _reasons(capture),
        "anchor_px": anchor_px,
        "mask_coverage_frac_of_face": diagnostics.get("coverage_frac_of_face", 0.0),
        "mask_failed_stage": diagnostics.get("stage", ""),
        **capture.metrics,
        **{f"reject_{k}": v for k, v in (diagnostics.get("rejected_frac_of_face") or {}).items()},
        **{f"visible_{k}": v for k, v in visibility.items()},
        **{f"undersized_{k}": int(k in undersized) for k in visibility},
    }
    return row


def _reasons(capture: Any) -> str:
    return "|".join(f.value for f in capture.failures)


def _face_box(points: np.ndarray, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    xs, ys = points[:, 0], points[:, 1]
    x0 = max(0, int(np.floor(xs.min())))
    y0 = max(0, int(np.floor(ys.min())))
    x1 = min(shape[1], int(np.ceil(xs.max())))
    y1 = min(shape[0], int(np.ceil(ys.max())))
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def summarize(rows: list[dict[str, Any]], profile: dict) -> str:
    """Per-metric percentiles alongside the configured threshold, and the failure tally.

    The threshold column is the point: a check whose cutoff sits outside the observed
    range is either dead (never fires) or saturated (always fires), and both are invisible
    in a pass/fail count alone.
    """
    lines: list[str] = []
    numeric = sorted(
        {k for row in rows for k, v in row.items() if isinstance(v, (int, float, bool))}
    )
    thresholds = _threshold_index(profile)

    header = f"{'metric':<34}" + "".join(f"p{p:<7}" for p in PERCENTILES) + "  threshold"
    lines.append(header)
    lines.append("-" * len(header))
    for name in numeric:
        values = np.array(
            [float(row[name]) for row in rows if isinstance(row.get(name), (int, float, bool))],
            dtype=np.float64,
        )
        if values.size == 0:
            continue
        cells = "".join(f"{np.percentile(values, p):<8.3f}" for p in PERCENTILES)
        limit = thresholds.get(name)
        lines.append(f"{name:<34}{cells}  {limit if limit is not None else ''}")

    lines.append("")
    lines.append(f"captures: {len(rows)}   passed: {sum(1 for r in rows if r['passed'])}")

    tally: dict[str, int] = {}
    for row in rows:
        for reason in str(row.get("failures", "")).split("|"):
            if reason:
                tally[reason] = tally.get(reason, 0) + 1
    lines.append("failure tally:")
    for reason, count in sorted(tally.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {reason:<32} {count}")

    dead = [name for name in _all_threshold_names(profile) if name not in tally]
    if dead:
        lines.append("")
        lines.append("checks that never fired on this corpus (dead, or the corpus is clean):")
        for name in dead:
            lines.append(f"  {name}")
    return "\n".join(lines)


def _threshold_index(profile: dict) -> dict[str, Any]:
    """Map a metric name to the configured limit it is compared against."""
    exposure = profile["exposure"]
    return {
        "laplacian_var": profile["blur"]["min_laplacian_var"],
        "tenengrad": profile["blur"]["min_tenengrad"],
        "precheck_laplacian_var": profile["blur"]["precheck_min_laplacian_var"],
        "precheck_tenengrad": profile["blur"]["precheck_min_tenengrad"],
        "mean_luma": f"{exposure['min_mean_luma']}..{exposure['max_mean_luma']}",
        "clipped_low_frac": profile["exposure"]["max_clipped_low_frac"],
        "clipped_high_frac": profile["exposure"]["max_clipped_high_frac"],
        "specular_frac": profile["exposure"]["max_specular_frac"],
        "lr_luma_asymmetry": profile["shadow"]["max_lr_luma_asymmetry"],
        "deep_shadow_frac": profile["shadow"]["max_deep_shadow_frac"],
        "gray_world_deviation": profile["white_balance"]["max_gray_world_deviation"],
        "face_height_frac": profile["face"]["min_face_height_frac"],
        "face_px": profile["face"]["min_face_px"],
        "yaw_deg": profile["pose"]["max_yaw_deg"],
        "pitch_deg": profile["pose"]["max_pitch_deg"],
        "roll_deg": profile["pose"]["max_roll_deg"],
        "min_required_roi_visibility": profile["roi_visibility"]["min_visible_frac"],
        "hair_frac": profile["occlusion"]["max_hair_frac"],
        "beard_frac": profile["occlusion"]["max_beard_frac"],
        "glasses_frac": profile["occlusion"]["max_glasses_frac"],
    }


def _all_threshold_names(profile: dict) -> list[str]:
    del profile
    from skin_analysis.schemas import QCFailure

    return [f.value for f in QCFailure]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path, required=True, help="directory of capture images"
    )
    parser.add_argument("--out", type=Path, help="write the per-image CSV here")
    parser.add_argument("--limit", type=int, help="stop after N images")
    args = parser.parse_args()

    if not args.corpus.is_dir():
        print(f"not a directory: {args.corpus}", file=sys.stderr)
        return 2

    import cv2

    profile = cfg.capture_profile()
    roi_config = cfg.load("rois")

    files = sorted(
        path
        for path in args.corpus.rglob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES
    )[: args.limit]
    if not files:
        print(f"no images under {args.corpus}", file=sys.stderr)
        return 2

    rows: list[dict[str, Any]] = []
    for index, path in enumerate(files, start=1):
        image = cv2.imread(str(path))
        if image is None:
            print(f"  unreadable, skipped: {path.name}", file=sys.stderr)
            continue
        row = observe(image, profile, roi_config)
        row["image"] = path.name
        rows.append(row)
        print(f"[{index}/{len(files)}] {path.name}: {row['failures'] or 'PASS'}")

    print()
    print(summarize(rows, profile))

    if args.out:
        fields = sorted({key for row in rows for key in row})
        fields.remove("image")
        with args.out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image", *fields])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
