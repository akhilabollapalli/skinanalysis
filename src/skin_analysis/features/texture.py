"""Visual texture / roughness — ACTIVE.

GLCM + gradient energy + local variance + high-frequency ratio, on the texture copy.

Status: ACTIVE

Limitations and upgrade path:
    This measures APPEARANCE of roughness in a 2D projection. It is not a topographic
    measurement and must never be presented as one -- there is no depth information in a
    single RGB selfie. Upgrade path: none within V1's constraints; a real roughness
    measurement needs structured light or a profilometer.

D5 is load-bearing here. The texture copy carries luminance with FIXED normalization only.
CLAHE is local adaptive gain, so a GLCM computed downstream of it partly measures CLAHE's
response to the neighbourhood rather than the skin. The GLCM quantization is likewise fixed
over the full range, not per-patch -- rescaling each patch to its own min/max would be
adaptive gain reintroduced by the back door.

MEASUREMENT ONLY -- severities are assigned in ``decision/``. See ``features/redness.py``.
"""

from __future__ import annotations

import numpy as np

from ..schemas import (
    ROI,
    Concern,
    FeatureContext,
    FeatureResultInternal,
    ImageCopy,
    ROIResult,
    Severity,
)
from ..util import scale
from . import _common

CONCERN = Concern.TEXTURE
IMAGE_COPY = ImageCopy.TEXTURE


def analyze(
    image: np.ndarray,
    skin_mask: np.ndarray,
    rois: dict[str, np.ndarray],
    config: dict,
    *,
    context: FeatureContext,
) -> FeatureResultInternal:
    """Analyze visual texture for one capture.

    Args:
        image: the texture copy -- (H, W) float32 luminance, 0..255, fixed normalization.
        skin_mask: bool array, True where pixels are analyzable skin.
        rois: ROI name -> bool mask, already intersected with ``skin_mask``.
        config: the ``texture`` block of config/severity_thresholds.yaml.
        context: per-capture facts (D1 scale anchor, QC verdicts).

    Returns:
        FeatureResultInternal with per-ROI raw measurements.

    Notes:
        Patches are square, anchor-proportional, and sampled deterministically. Both
        properties are requirements rather than conveniences: a GLCM computed on a
        different pixel count is a different statistic, and a randomly sampled patch set
        would break determinism, which CI checks on every commit (D13).

        A patch is used only if ``min_valid_frac`` of it is analyzable skin. Partial
        patches are the main route by which a mask edge is measured as roughness.
    """
    if image.ndim != 2:
        raise ValueError(f"texture copy must be a single luminance plane, got {image.shape}")
    if image.shape != skin_mask.shape:
        raise ValueError(
            f"image {image.shape} and skin mask {skin_mask.shape} disagree on frame size"
        )
    if config.get("clahe", False):
        raise ValueError(
            "texture.clahe is true. CLAHE is local adaptive gain, so the GLCM downstream "
            "would partly measure the gain rather than the skin (D5)."
        )

    primary = [str(name) for name in config["primary_rois"]]

    if not context.qc.exposure_pass:
        return _common.all_unmeasurable(
            CONCERN, primary, "capture failed the exposure check; clipped pixels carry no texture"
        )

    patch_cfg = config["patch"]
    size_px = scale.to_px(
        float(patch_cfg["size_frac_of_iod"]), context.anchor_px, odd=True, minimum=9
    )
    stride_px = scale.to_px(
        float(patch_cfg["stride_frac_of_iod"]), context.anchor_px, minimum=1
    )
    min_valid = float(patch_cfg["min_valid_frac"])
    max_patches = int(patch_cfg["max_patches_per_roi"])

    usable, reasons = _common.measurable_rois(
        rois, primary, _common.min_support_px(config, context.anchor_px)
    )
    roi_results: list[ROIResult] = [
        _common.unmeasurable(ROI(name), why) for name, why in reasons.items()
    ]

    for name in usable:
        patches = _sample_patches(rois[name], size_px, stride_px, min_valid, max_patches)
        if not patches:
            roi_results.append(
                _common.unmeasurable(
                    ROI(name),
                    f"no {size_px}px patch is at least {min_valid:.0%} analyzable skin",
                )
            )
            continue

        per_patch = [
            _patch_measurements(image[sl], config, context.anchor_px) for sl in patches
        ]
        raw = {
            key: _common.robust_mean(np.array([m[key] for m in per_patch], dtype=np.float64))
            for key in per_patch[0]
        }
        roi_results.append(
            ROIResult(
                roi=ROI(name),
                severity=Severity.UNMEASURABLE,  # decided in decision/, not here
                raw=raw,
            )
        )

    return _common.measurement_result(CONCERN, roi_results)


# ------------------------------------------------------------------ patch sampling


def _sample_patches(
    roi_mask: np.ndarray,
    size_px: int,
    stride_px: int,
    min_valid_frac: float,
    max_patches: int,
) -> list[tuple[slice, slice]]:
    """Deterministic raster scan over the ROI, keeping only sufficiently covered patches.

    When more patches qualify than the latency budget allows (D10), the kept set is an
    evenly spaced subsample of the raster order -- NOT a random sample and NOT the first N.
    Random would break determinism; the first N would bias every ROI toward its top-left.
    """
    ys, xs = np.nonzero(roi_mask)
    if ys.size == 0:
        return []

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    needed = min_valid_frac * size_px * size_px

    candidates: list[tuple[slice, slice]] = []
    for top in range(y0, max(y0 + 1, y1 - size_px + 2), stride_px):
        for left in range(x0, max(x0 + 1, x1 - size_px + 2), stride_px):
            window = (slice(top, top + size_px), slice(left, left + size_px))
            block = roi_mask[window]
            if block.shape != (size_px, size_px):
                continue
            if float(block.sum()) < needed:
                continue
            candidates.append(window)

    if len(candidates) <= max_patches:
        return candidates
    picks = np.linspace(0, len(candidates) - 1, num=max_patches).round().astype(int)
    unique: list[int] = list(dict.fromkeys(int(i) for i in picks))
    return [candidates[i] for i in unique]


# ------------------------------------------------------------------ per-patch metrics


def _patch_measurements(
    patch: np.ndarray, config: dict, anchor_px: float
) -> dict[str, float]:
    """Every raw texture measurement for one patch."""
    glcm = _glcm_features(patch, config, anchor_px)
    return {
        **glcm,
        "gradient_energy": _gradient_energy(patch, anchor_px),
        "local_variance": float(np.var(patch, dtype=np.float64)),
        "hf_ratio": _hf_ratio(patch, config["frequency"]),
    }


def _glcm_features(patch: np.ndarray, config: dict, anchor_px: float) -> dict[str, float]:
    """Grey-level co-occurrence statistics, averaged over distances and angles.

    Quantization is FIXED over the configured range rather than over the patch's own
    min/max. Per-patch rescaling would make a flat patch and a high-contrast patch produce
    the same matrix, which is the same objection that keeps CLAHE out of this branch (D5).

    Co-occurrence offsets are anchor fractions resolved to integer pixels (D1), so the same
    offset spans the same physical skin at any capture distance. Duplicate offsets after
    rounding are collapsed: passing the same distance twice would weight it twice in the
    average without saying so.
    """
    from skimage.feature import graycomatrix, graycoprops

    spec = config["glcm"]
    levels = int(spec["levels"])
    lo, hi = (float(v) for v in config["quantization"]["range"])

    scaled = (patch - lo) * (levels / max(hi - lo, 1e-6))
    quantized = np.clip(scaled, 0, levels - 1).astype(np.uint8)

    distances = sorted(
        {scale.to_px(float(frac), anchor_px, minimum=1) for frac in spec["distances_frac_of_iod"]}
    )

    matrix = graycomatrix(
        quantized,
        distances=distances,
        angles=[np.deg2rad(float(a)) for a in spec["angles_deg"]],
        levels=levels,
        symmetric=True,
        normed=True,
    )

    features: dict[str, float] = {}
    for name in spec["features"]:
        if name == "entropy":
            # graycoprops has no entropy. Computed per (distance, angle) plane and then
            # averaged, matching how graycoprops reduces its own outputs.
            planes = matrix.astype(np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                logs = np.where(planes > 0, np.log2(planes, where=planes > 0), 0.0)
            features["glcm_entropy"] = float(np.mean(-np.sum(planes * logs, axis=(0, 1))))
        else:
            features[f"glcm_{name}"] = float(np.mean(graycoprops(matrix, name)))
    return features


def _gradient_energy(patch: np.ndarray, anchor_px: float) -> float:
    """Mean Sobel gradient magnitude. Responds to edge density, not to overall brightness.

    Reported in luminance per ANCHOR, not per pixel. A Sobel gradient is intrinsically
    per-pixel, so a closer capture spreads the same physical luminance step over more pixels
    and reads a smaller gradient -- the same unit error that was in the pigmentation shadow
    test. Multiplying by the anchor converts the axis to physical facial scale.
    """
    import cv2

    source = patch.astype(np.float32)
    gx = cv2.Sobel(source, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(source, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.hypot(gx, gy)) * anchor_px)


def _hf_ratio(patch: np.ndarray, spec: dict) -> float:
    """Share of spectral energy in the configured high-frequency band.

    Band edges are in CYCLES PER PATCH. That unit is the whole point: because the patch is a
    fixed fraction of the anchor, it covers the same physical skin at every capture
    distance, so a structure of a given physical size sits at the same cycles-per-patch
    regardless of how many pixels the sensor spent on it.

    Cycles per PIXEL was the first implementation and it was scale dependent -- a closer
    capture spends more pixels on the same skin, so every feature slid down that axis and
    hf_ratio moved 60% between two scales of the same synthetic skin. The frequency axis has
    to be normalized by the patch, not by the sampling rate.
    """
    lo = float(spec["hf_band_lo_cycles_per_patch"])
    hi = float(spec["hf_band_hi_cycles_per_patch"])

    windowed = patch.astype(np.float64) - float(np.mean(patch))
    # Hann window in both axes: without it the patch edges act as a step discontinuity and
    # leak broadband energy straight into the band being measured.
    window = np.hanning(patch.shape[0])[:, None] * np.hanning(patch.shape[1])[None, :]
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(windowed * window))) ** 2

    # fftfreq gives cycles per sample; multiplying by the axis length converts to cycles
    # across the patch, which is the scale-invariant axis.
    rows = np.fft.fftshift(np.fft.fftfreq(patch.shape[0])) * patch.shape[0]
    cols = np.fft.fftshift(np.fft.fftfreq(patch.shape[1])) * patch.shape[1]
    radius = np.hypot(rows[:, None], cols[None, :])

    total = float(spectrum.sum())
    if total <= 0.0:
        return 0.0
    band = (radius >= lo) & (radius <= hi)
    return float(spectrum[band].sum()) / total
