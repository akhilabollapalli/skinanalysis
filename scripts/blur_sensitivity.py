"""Measure where blur actually breaks a measurement, instead of guessing a cutoff.

    python scripts/blur_sensitivity.py --corpus data/raw/real_orig --out blur_report.json

THE QUESTION THIS ANSWERS. capture_thresholds.yaml's blur floor
(``blur.min_laplacian_var`` / ``min_tenengrad``) has always been a retake-tolerance
judgement call: the measured sharpness distribution on real captures is a smooth
continuum with no natural split, so any cutoff on it alone is "how many retakes are we
willing to cause", not "where does the measurement stop being trustworthy". This script
answers the second question directly: it takes photos sharp enough to trust, blurs them by
KNOWN, CONTROLLED amounts, and watches when the ACTUAL texture/wrinkle measurements start
drifting away from their true (sharp) value. Wherever that drift first exceeds the SAME
tolerance the project already accepts for same-session repeatability
(config/validation_gates.yaml raw_metric_cv), that is a measured, defensible floor -- not
a picked one.

METHOD. Landmarks, the skin mask and ROI polygons are computed ONCE per reference photo,
from its sharpest (native) version, and held FIXED across the blur sweep. This isolates
the question on purpose: does blurring the pixels change what texture/wrinkles measure,
independent of whether extreme blur would also break face detection (a real, separate
failure mode this script does not test). A synthetic Gaussian blur sweep is used rather
than the corpus's own natural blur variation because it gives a KNOWN ground truth (sigma
= 0 is the true value) and a dense, controlled ladder -- the corpus's real blurry photos
have no sharp twin to compare against, so drift could not be measured from them alone.

ONLY texture and wrinkles are swept. Redness and pigmentation are CIELAB colour proxies
over a whole ROI's mean/percentile -- a slight defocus barely moves an average, and
qc_report.py's own instructions already forbid tuning colour thresholds from anything but
real captures under the V1 protocol. Texture and wrinkles are structural: GLCM contrast and
line detection are exactly the measurements a soft image should distort first.

Every number this script writes is internal (Rule 3), same as qc_report.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from skin_analysis import pipeline  # noqa: E402
from skin_analysis.capture import qc  # noqa: E402
from skin_analysis.face import landmarks as landmark_layer  # noqa: E402
from skin_analysis.face import rois as roi_layer  # noqa: E402
from skin_analysis.face import skin_mask as mask_layer  # noqa: E402
from skin_analysis.features import texture, wrinkles  # noqa: E402
from skin_analysis.schemas import FeatureContext, QCVerdict, RunMode  # noqa: E402
from skin_analysis.util import config as cfg  # noqa: E402
from skin_analysis.util import scale  # noqa: E402

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")

#: Synthetic Gaussian blur, in pixels of standard deviation, applied to the NATIVE image
#: before canonical cropping -- realistic in scale for a phone photo (a few pixels of blur
#: at native resolution is a lot of blur once downsampled to the canonical crop width).
SIGMA_SWEEP = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)

#: Concern -> feature module. Only the structurally blur-sensitive concerns (see module
#: docstring for why redness/pigmentation are excluded).
_SWEPT = {"texture": texture, "wrinkles": wrinkles}

#: Minimum native sharpness (tenengrad on the canonical crop) for a photo to serve as a
#: ground-truth reference. Below this the "sharp" baseline is itself dubious, and drift
#: measured against a dubious baseline is not evidence of anything.
MIN_REFERENCE_TENENGRAD_MULTIPLE = 3.0


@dataclass(frozen=True)
class DriftSample:
    image: str
    concern: str
    roi: str
    measurement: str
    sigma: float
    laplacian_var: float
    tenengrad: float
    baseline: float
    blurred: float

    @property
    def relative_drift(self) -> float:
        denom = abs(self.baseline) if abs(self.baseline) > 1e-9 else 1e-9
        return abs(self.blurred - self.baseline) / denom


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    import cv2

    if sigma <= 0.0:
        return image
    # ksize=0 lets OpenCV derive an odd kernel size from sigma; truncation past ~3 sigma is
    # OpenCV's own default and is not a parameter this experiment needs to control.
    return np.asarray(cv2.GaussianBlur(image, (0, 0), sigmaX=sigma), dtype=image.dtype)


def _blur_metrics(
    image: np.ndarray, face_box: tuple[int, int, int, int], profile: dict
) -> dict[str, float]:
    """The exact laplacian_var/tenengrad the capture gate itself computes, on the
    canonical crop. Reuses qc's private blur check directly (the same pattern
    qc_report.py already uses for pipeline._GATED_MASK_STAGES) rather than re-deriving the
    formula, so a future change to the gate's blur math cannot silently desync from what
    this script reports as "the same number"."""
    crop = qc.canonical_crop(image, face_box, profile)
    metrics: dict[str, float] = {}
    qc._check_blur(crop, profile, metrics)
    return metrics


def _face_box(points: np.ndarray, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    xs, ys = points[:, 0], points[:, 1]
    x0 = max(0, int(np.floor(xs.min())))
    y0 = max(0, int(np.floor(ys.min())))
    x1 = min(shape[1], int(np.ceil(xs.max())))
    y1 = min(shape[0], int(np.ceil(ys.max())))
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


@dataclass(frozen=True)
class Reference:
    """One sharp photo's fixed geometry, reused across the whole blur sweep."""

    image: str
    native_tenengrad: float
    mask: np.ndarray
    composed: dict[str, np.ndarray]
    anchor_px: float
    face_box: tuple[int, int, int, int]


def build_reference(
    path: Path, profile: dict, roi_cfg: dict, severity_cfg: dict
) -> Reference | None:
    """Detect the face and fix its geometry, or None if this photo cannot anchor one.

    DEVELOPMENT run mode: this is instrumentation, the same workflow debug_roi.py and
    qc_report.py already use ahead of config/rois.yaml meta.verified (D15).
    """
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        return None

    faces = landmark_layer.detect_faces(image, profile)
    if not faces:
        return None
    points = faces[0]
    if points.shape[0] <= scale.RIGHT_IRIS_CENTER:
        return None

    anchor_px = scale.inter_ocular_distance(points)
    if not scale.anchor_is_sufficient(anchor_px, severity_cfg):
        return None

    face_box = _face_box(points, image.shape[:2])
    blur = _blur_metrics(image, face_box, profile)
    floor = MIN_REFERENCE_TENENGRAD_MULTIPLE * float(profile["blur"]["min_tenengrad"])
    if blur["tenengrad"] < floor:
        return None  # not sharp enough to trust as a ground-truth baseline

    polygons = roi_layer.build(points, image.shape[:2], roi_cfg, run_mode=RunMode.DEVELOPMENT)
    mask = mask_layer.build(image, points, roi_cfg, roi_polygons=polygons)
    composed = roi_layer.compose(polygons, mask)
    return Reference(
        image=path.name, native_tenengrad=blur["tenengrad"], mask=mask,
        composed=composed, anchor_px=anchor_px, face_box=face_box,
    )


def measure(
    image: np.ndarray, reference: Reference, severity_cfg: dict
) -> dict[str, dict[str, dict[str, float]]]:
    """Run the swept feature modules with GEOMETRY HELD FIXED, varying only the pixels.

    Returns: concern -> roi -> measurement -> value.
    """
    context = FeatureContext(
        anchor_px=reference.anchor_px,
        qc=QCVerdict(passed=True, shadow_pass=True, color_cast_pass=True, exposure_pass=True),
        run_mode=RunMode.DEVELOPMENT,
    )
    copies = pipeline.make_image_copies(image, severity_cfg)

    out: dict[str, dict[str, dict[str, float]]] = {}
    for key, module in _SWEPT.items():
        concern_cfg = cfg.concern_config(key)
        result = module.analyze(
            copies.get(module.IMAGE_COPY), reference.mask, reference.composed, concern_cfg,
            context=context,
        )
        out[key] = {
            str(rr.roi.value): dict(rr.raw) for rr in result.roi_results if rr.raw
        }
    return out


def sweep_one(path: Path, profile: dict, roi_cfg: dict, severity_cfg: dict) -> list[DriftSample]:
    import cv2

    reference = build_reference(path, profile, roi_cfg, severity_cfg)
    if reference is None:
        return []

    native = cv2.imread(str(path))
    if native is None:
        return []  # succeeded once in build_reference; a second read failing is theoretical
    baseline = measure(native, reference, severity_cfg)
    if not any(baseline.values()):
        return []  # nothing measurable on this face at all; not a useful reference

    samples: list[DriftSample] = []
    for sigma in SIGMA_SWEEP:
        blurred_image = _gaussian_blur(native, sigma)
        blur_metrics = _blur_metrics(blurred_image, reference.face_box, profile)
        measured = measure(blurred_image, reference, severity_cfg)
        for concern, by_roi in baseline.items():
            for roi, measurements in by_roi.items():
                blurred_roi = measured.get(concern, {}).get(roi, {})
                for name, base_value in measurements.items():
                    if name not in blurred_roi:
                        continue
                    samples.append(
                        DriftSample(
                            image=path.name, concern=concern, roi=roi, measurement=name,
                            sigma=sigma,
                            laplacian_var=blur_metrics["laplacian_var"],
                            tenengrad=blur_metrics["tenengrad"],
                            baseline=base_value, blurred=blurred_roi[name],
                        )
                    )
    return samples


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile of an empty sample")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (q / 100.0)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


@dataclass(frozen=True)
class ThresholdRow:
    """Drift stats over every sample whose laplacian_var is at or above ``threshold``."""

    threshold: float
    n: int
    min_tenengrad: float
    p50_drift: float
    p95_drift: float


def sweep_thresholds(samples: list[DriftSample], min_n: int = 5) -> list[ThresholdRow]:
    """One row per candidate laplacian_var floor, sweeping every value actually observed.

    A candidate with fewer than ``min_n`` samples at-or-above it is skipped: a p95 computed
    over a handful of points is noise, not evidence, and would make the recommendation
    below depend on which few points happened to be sharpest.
    """
    thresholds = sorted({round(s.laplacian_var, 1) for s in samples})
    rows: list[ThresholdRow] = []
    for threshold in thresholds:
        at_or_above = [s for s in samples if s.laplacian_var >= threshold]
        if len(at_or_above) < min_n:
            continue
        drifts = [s.relative_drift for s in at_or_above]
        rows.append(
            ThresholdRow(
                threshold=threshold,
                n=len(at_or_above),
                min_tenengrad=min(s.tenengrad for s in at_or_above),
                p50_drift=percentile(drifts, 50),
                p95_drift=percentile(drifts, 95),
            )
        )
    return rows


def recommend_floor(rows: list[ThresholdRow], bound: float | None) -> float | None:
    """The LOOSEST candidate whose p95 drift still clears ``bound``, or None.

    A LOWER threshold value is the looser requirement -- it admits blurrier images -- so
    "loosest passing" is the MINIMUM of the candidates that clear the bound, not the
    maximum. As threshold falls, its "at or above" sample set only grows (more, blurrier
    samples are admitted), so drift should rise; the recommendation is the point just
    before it crosses what repeatability already tolerates, not the most conservative
    (highest, sharpest-only) point tested, which would recommend far stricter than the
    evidence requires.
    """
    if bound is None:
        return None
    candidates = [row.threshold for row in rows if row.p95_drift <= bound]
    return min(candidates) if candidates else None


def summarize(samples: list[DriftSample], cv_bounds: dict[str, float]) -> str:
    """Drift by blur bucket, and the recommended floor: the loosest laplacian_var value
    whose p95 relative drift still clears each concern's ALREADY-PRESPECIFIED
    repeatability CV bound -- reusing that bound rather than inventing a new one, since it
    already says how much drift this project accepts as "the same result on a rescan"."""
    lines: list[str] = []

    by_concern: dict[str, list[DriftSample]] = defaultdict(list)
    for s in samples:
        by_concern[s.concern].append(s)

    recommendation: dict[str, float] = {}
    for concern, concern_samples in by_concern.items():
        bound = cv_bounds.get(concern)
        lines.append(f"\n{concern} (n={len(concern_samples)} sweep points, "
                     f"CV floor {bound if bound is not None else 'undeclared'}):")
        lines.append(f"{'laplacian_var >=':<20}{'tenengrad >=':<16}{'n':<8}"
                      f"{'p50 drift':<12}{'p95 drift':<12}")

        rows = sweep_thresholds(concern_samples)
        for row in rows:
            lines.append(f"{row.threshold:<20.1f}{row.min_tenengrad:<16.1f}{row.n:<8}"
                         f"{row.p50_drift:<12.4f}{row.p95_drift:<12.4f}")

        best = recommend_floor(rows, bound)
        if best is not None:
            recommendation[concern] = best
            lines.append(f"  -> recommended floor for {concern}: laplacian_var >= {best:.1f} "
                         f"(loosest level tested whose p95 drift stays within the {bound} "
                         "repeatability CV bound)")
        else:
            lines.append(f"  -> no level tested kept {concern} within its CV bound; "
                         "even the sharpest bucket exceeded it, or the bound is undeclared")

    if recommendation:
        overall = max(recommendation.values())
        lines.append(
            f"\ncombined recommendation (strictest across swept concerns): "
            f"blur.min_laplacian_var >= {overall:.1f}\n"
            f"current config/capture_thresholds.yaml value: "
            f"{cfg.capture_profile()['blur']['min_laplacian_var']}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="write per-sample CSV here")
    parser.add_argument("--report", type=Path, help="write the JSON summary here")
    parser.add_argument("--limit", type=int, help="stop after N reference photos")
    args = parser.parse_args()

    if not args.corpus.is_dir():
        print(f"not a directory: {args.corpus}", file=sys.stderr)
        return 2

    profile = cfg.capture_profile()
    roi_cfg = cfg.load("rois")
    severity_cfg = cfg.load("severity_thresholds")
    validation_gates = cfg.load("validation_gates")
    cv_bounds = validation_gates["repeatability"]["gates"]["raw_metric_cv"]["p95_max_cv"]

    files = sorted(p for p in args.corpus.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    samples: list[DriftSample] = []
    used = 0
    for path in files:
        if args.limit is not None and used >= args.limit:
            break
        found = sweep_one(path, profile, roi_cfg, severity_cfg)
        if found:
            used += 1
            samples.extend(found)
            print(f"  [{used}] {path.name}: {len({s.concern for s in found})} concern(s), "
                  f"{len(SIGMA_SWEEP)} blur levels")
        else:
            print(f"  skip  {path.name}: not usable as a sharp reference")

    if not samples:
        print(
            "\nno usable reference photos: none of these were sharp enough (>= "
            f"{MIN_REFERENCE_TENENGRAD_MULTIPLE}x the current tenengrad floor) AND had a "
            "detectable, sufficiently large face with at least one measurable ROI.",
            file=sys.stderr,
        )
        return 2

    print(f"\n{used} reference photo(s), {len(samples)} (image, blur level, ROI, "
          "measurement) samples")
    print(summarize(samples, cv_bounds))

    if args.out:
        with args.out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["image", "concern", "roi", "measurement", "sigma",
                            "laplacian_var", "tenengrad", "baseline", "blurred",
                            "relative_drift"],
            )
            writer.writeheader()
            for s in samples:
                writer.writerow({
                    "image": s.image, "concern": s.concern, "roi": s.roi,
                    "measurement": s.measurement, "sigma": s.sigma,
                    "laplacian_var": s.laplacian_var, "tenengrad": s.tenengrad,
                    "baseline": s.baseline, "blurred": s.blurred,
                    "relative_drift": s.relative_drift,
                })
        print(f"\nwrote {len(samples)} rows to {args.out}")

    if args.report:
        payload: dict[str, Any] = {
            "n_reference_photos": used,
            "n_samples": len(samples),
            "sigma_sweep": list(SIGMA_SWEEP),
            "cv_bounds_used": cv_bounds,
        }
        args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
