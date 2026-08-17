"""Wrinkles — ACTIVE BASELINE.

Multi-scale Gabor + Hessian ridge filtering + morphology, on the ridge copy.

Status: ACTIVE BASELINE

Limitations and upgrade path:
    A ridge filter responds to hair, glasses frames, shadow edges and mask boundaries as
    readily as it responds to a line. The suppression below is therefore part of the
    algorithm, not an optional extra. Upgrade path: a supervised line segmenter once
    commercially clear wrinkle annotations exist -- FFHQ-Wrinkle is CC BY-NC-SA and is
    EXCLUDED, so this cannot simply be trained today.

The ridge response is computed ONCE over the union of this concern's ROIs rather than per
ROI. Twenty-four Gabor convolutions plus a Frangi filter per region would not fit the D10
latency budget, and computing it once also guarantees that two overlapping ROIs cannot
disagree about the same pixel.

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

CONCERN = Concern.WRINKLES
IMAGE_COPY = ImageCopy.RIDGE


def analyze(
    image: np.ndarray,
    skin_mask: np.ndarray,
    rois: dict[str, np.ndarray],
    config: dict,
    *,
    context: FeatureContext,
) -> FeatureResultInternal:
    """Analyze wrinkles for one capture.

    Args:
        image: the ridge copy -- (H, W) float32 luminance, 0..255.
        skin_mask: bool array, True where pixels are analyzable skin.
        rois: ROI name -> bool mask, already intersected with ``skin_mask``.
        config: the ``wrinkles`` block of config/severity_thresholds.yaml.
        context: per-capture facts (D1 scale anchor, QC verdicts).

    Returns:
        FeatureResultInternal with per-ROI raw measurements.

    Notes:
        This is the only concern that may use the nasolabial ROIs, because there the fold
        IS the morphology being measured (D8). Every other concern excludes them.

        Responses hugging an ROI boundary are discarded (``roi_edge_margin_frac_of_iod``).
        A mask edge is a perfect ridge, and without that margin the strongest "wrinkle" in
        a region is frequently its own outline.
    """
    if image.ndim != 2:
        raise ValueError(f"ridge copy must be a single luminance plane, got {image.shape}")
    if image.shape != skin_mask.shape:
        raise ValueError(
            f"image {image.shape} and skin mask {skin_mask.shape} disagree on frame size"
        )

    primary = [str(name) for name in config["primary_rois"]]

    if not context.qc.exposure_pass:
        return _common.all_unmeasurable(
            CONCERN, primary, "capture failed the exposure check; ridge contrast is not skin here"
        )

    usable, reasons = _common.measurable_rois(
        rois, primary, _common.min_support_px(config, context.anchor_px)
    )
    roi_results: list[ROIResult] = [
        _common.unmeasurable(ROI(name), why) for name, why in reasons.items()
    ]
    if not usable:
        return _common.measurement_result(CONCERN, roi_results)

    support = np.zeros_like(skin_mask, dtype=bool)
    for name in usable:
        support |= rois[name]

    response = ridge_response(image, support, config, context.anchor_px)

    percentile_cut = float(config["ridge_percentile_cut"])
    morph = config["morphology"]
    edge_margin_px = scale.to_px(
        float(config["suppression"]["roi_edge_margin_frac_of_iod"]),
        context.anchor_px,
        minimum=0,
    )
    min_length_px = float(morph["min_line_length_frac_of_iod"]) * context.anchor_px
    max_width_px = float(morph["max_line_width_frac_of_iod"]) * context.anchor_px
    close_px = scale.to_px(float(morph["close_kernel_frac_of_iod"]), context.anchor_px, minimum=0)

    for name in usable:
        # An ROI's own outline is a perfect ridge. Back away from it before scoring, and do
        # it per ROI rather than on the union, or interior boundaries between adjacent ROIs
        # would survive.
        interior = _erode(rois[name], edge_margin_px)
        if not interior.any():
            roi_results.append(
                _common.unmeasurable(
                    ROI(name), f"ROI vanishes under the {edge_margin_px}px edge margin"
                )
            )
            continue

        # Percentile within THIS ROI, so the cut adapts to how much ridge structure the
        # region actually has rather than to a global response level -- the same
        # within-image logic as z_local, on a distribution where a MAD is meaningless.
        cut = float(np.percentile(response[interior], percentile_cut))
        lines = _close(interior & (response >= cut), close_px)
        lines, segments = _filter_by_geometry(lines, min_length_px, max_width_px)

        roi_px = float(interior.sum())
        lengths = np.array([s["length_px"] for s in segments], dtype=np.float64)

        roi_results.append(
            ROIResult(
                roi=ROI(name),
                severity=Severity.UNMEASURABLE,  # decided in decision/, not here
                raw={
                    "line_density": float(lines.sum()) / roi_px,
                    # MEDIAN segment length, in anchor units so it transfers across
                    # devices. Not the maximum: two lines that touch merge into one
                    # component, and a maximum would then report a single value several
                    # times the length of any real line on the face.
                    "line_length": (
                        float(np.median(lengths) / context.anchor_px) if lengths.size else 0.0
                    ),
                    "ridge_contrast": (
                        float(np.median(response[lines])) if lines.any() else 0.0
                    ),
                },
            )
        )

    return _common.measurement_result(CONCERN, roi_results)


# ------------------------------------------------------------------ ridge response


def ridge_response(
    image: np.ndarray,
    support: np.ndarray,
    config: dict,
    anchor_px: float,
) -> np.ndarray:
    """Combined multi-scale Gabor and Hessian ridge response, zero outside ``support``.

    Gabor supplies orientation selectivity, which is what separates a line from a blob;
    Frangi supplies scale-selective ridge geometry, which is what separates a line from an
    edge. Taking the pointwise minimum of the two normalized responses requires BOTH to
    agree, which is the cheapest available defence against hair -- hair is oriented and
    ridge-like, so either filter alone lights it up.
    """
    from skimage.filters import frangi

    hessian_cfg = config["hessian"]
    if str(hessian_cfg.get("ridge_response", "frangi")) != "frangi":
        raise ValueError(
            f"unsupported hessian.ridge_response {hessian_cfg['ridge_response']!r}; "
            "only 'frangi' is implemented"
        )

    out = np.zeros(image.shape, dtype=np.float64)
    if not support.any():
        return out

    # Restricted to the support's bounding box, padded by the widest kernel. 24 Gabor
    # convolutions plus a Frangi filter over a full 1024x1024 frame took 7 s per face,
    # against a P50 budget of 1.5 s for the WHOLE pipeline (D10). The padding is what keeps
    # this identical to the full-frame result: without it, every kernel would see a
    # zero border and manufacture a ridge along the crop edge.
    window = _support_window(support, _kernel_reach(config, anchor_px))
    plane = image[window].astype(np.float32)
    local_support = support[window]

    gabor = _gabor_bank(plane, config["gabor"], anchor_px)
    # black_ridges: a wrinkle is a DARK valley in luminance, not a bright ridge. Getting
    # this backwards produces a filter that measures the highlights beside each line.
    ridges = frangi(
        plane / 255.0,
        sigmas=_scales_px(hessian_cfg["scales_frac_of_iod"], anchor_px),
        black_ridges=True,
    ).astype(np.float64)

    # Pointwise minimum: BOTH filters must agree. Gabor supplies orientation selectivity
    # and Frangi supplies ridge geometry, and hair satisfies either one alone.
    combined = np.minimum(
        _unit_scale(gabor, local_support), _unit_scale(ridges, local_support)
    )
    combined[~local_support] = 0.0

    out[window] = combined
    return out


def _scales_px(fracs: list, anchor_px: float) -> list[float]:
    """Filter scales in pixels, resolved from anchor fractions (D1).

    Floored at one pixel: a kernel narrower than a pixel is not a smaller filter, it is a
    filter the sampling grid cannot express, and scipy would silently return near-zero.
    """
    return [max(1.0, float(frac) * anchor_px) for frac in fracs]


def _kernel_reach(config: dict, anchor_px: float) -> int:
    """Padding needed so a cropped convolution matches the full-frame result."""
    sigmas = _scales_px(config["gabor"]["sigmas_frac_of_iod"], anchor_px)
    scales = _scales_px(config["hessian"]["scales_frac_of_iod"], anchor_px)
    lambdas = [float(v) * anchor_px for v in config["gabor"]["lambda_frac_of_iod"]]
    return int(max([3.0 * max(sigmas), 4.0 * max(scales), max(lambdas)])) + 2


def _support_window(support: np.ndarray, pad_px: int) -> tuple[slice, slice]:
    """Bounding box of ``support``, grown by ``pad_px`` and clipped to the frame."""
    ys, xs = np.nonzero(support)
    return (
        slice(max(0, int(ys.min()) - pad_px), min(support.shape[0], int(ys.max()) + pad_px + 1)),
        slice(max(0, int(xs.min()) - pad_px), min(support.shape[1], int(xs.max()) + pad_px + 1)),
    )


def _gabor_bank(plane: np.ndarray, spec: dict, anchor_px: float) -> np.ndarray:
    """Maximum even-symmetric Gabor magnitude over orientations and scales.

    Wavelengths are fractions of the anchor (D1). A fixed wavelength in pixels would tune
    the bank to one capture distance, so the same face measured from further away would
    return different line densities -- which is exactly the repeatability failure this
    product cares most about.
    """
    import cv2

    sigmas = _scales_px(spec["sigmas_frac_of_iod"], anchor_px)
    lambdas = [float(v) * anchor_px for v in spec["lambda_frac_of_iod"]]
    if len(sigmas) != len(lambdas):
        raise ValueError(
            f"gabor.sigmas has {len(sigmas)} entries but lambda_frac_of_iod has "
            f"{len(lambdas)}; each scale needs its own wavelength"
        )
    orientations = int(spec["orientations"])

    best = np.zeros(plane.shape, dtype=np.float64)
    for sigma, wavelength in zip(sigmas, lambdas, strict=True):
        ksize = 2 * round(3.0 * sigma) + 1
        for k in range(orientations):
            theta = np.pi * k / orientations
            kernel = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, max(wavelength, 2.0), 0.5, 0.0, ktype=cv2.CV_32F
            )
            # Zero-mean the kernel so the response measures local structure rather than
            # local brightness. Without it, a bright forehead scores as a wrinkled one.
            kernel -= float(kernel.mean())
            best = np.maximum(best, np.abs(cv2.filter2D(plane, cv2.CV_32F, kernel)))
    return best


def _unit_scale(values: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Rescale to 0..1 using percentiles WITHIN the support, so the two filters combine.

    Percentiles rather than min/max: one specular sliver would otherwise set the top of the
    range and flatten every real line to near zero. Scoped to the support because the
    background is most of the frame and would otherwise define "normal".
    """
    inside = values[support]
    if inside.size == 0:
        return np.zeros_like(values, dtype=np.float64)
    lo, hi = np.percentile(inside, [1.0, 99.0])
    if hi - lo <= 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return np.asarray(np.clip((values - lo) / (hi - lo), 0.0, 1.0), dtype=np.float64)


# ------------------------------------------------------------------ morphology


def _erode(mask: np.ndarray, margin_px: int) -> np.ndarray:
    import cv2

    if margin_px < 1 or not mask.any():
        return mask
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * margin_px + 1, 2 * margin_px + 1)
    )
    return np.asarray(cv2.erode(mask.astype(np.uint8), kernel), dtype=np.uint8).astype(bool)


def _close(mask: np.ndarray, size_px: int) -> np.ndarray:
    """Bridge the gaps a thresholded line breaks into, without joining separate lines."""
    import cv2

    if size_px < 1 or not mask.any():
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * size_px + 1, 2 * size_px + 1))
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return np.asarray(closed, dtype=np.uint8).astype(bool)


def _filter_by_geometry(
    mask: np.ndarray,
    min_length_px: float,
    max_width_px: float,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Keep components that are long AND thin. Both conditions, or neither means anything.

    Length alone admits the boundary of a shadow; thinness alone admits speckle. A wrinkle
    is specifically an elongated, narrow structure, and the width bound is what excludes
    the broad dark bands that hair and glasses temples produce.

    Length comes from the SKELETON and width from the DISTANCE TRANSFORM, both of which are
    independent of curvature. A minimum-area rectangle was tried first and was wrong: a
    wrinkle curves, and the min-area rect around a curved thin line is wide, so the width
    bound rejected every genuine line while keeping nothing. That failure is silent -- it
    reports a line density of exactly zero, which reads as "no wrinkles" rather than as a
    broken filter.
    """
    import cv2
    from skimage.morphology import skeletonize

    if not mask.any():
        return np.zeros_like(mask, dtype=bool), []

    # Local half-thickness at every pixel; doubling it gives the line's own width.
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    skeleton = skeletonize(mask)

    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    kept = np.zeros_like(mask, dtype=bool)
    segments: list[dict[str, float]] = []

    for label in range(1, count):
        component = labels == label
        # Skeleton pixel count is the arc length in pixels, following the curve instead of
        # measuring the chord across it.
        length = float(np.count_nonzero(skeleton & component))
        if length < 1.0:
            continue
        width = 2.0 * float(np.mean(distance[skeleton & component]))

        if length < min_length_px or width > max_width_px:
            continue
        kept |= component
        segments.append(
            {"length_px": length, "width_px": width, "area_px": float(component.sum())}
        )

    return kept, segments
