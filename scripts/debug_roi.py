"""Render landmarks, ROI polygons and the skin mask over an image.

Most false positives in this pipeline trace to the mask, not the feature math -- V1 has
no learned face parser, so boundaries are the weak point. Run this before tuning any
feature threshold. See .claude/skills/roi-debug for the review checklist.

**Run this immediately after defining the polygons in config/rois.yaml, before leaning on
the skin mask at all.** Three things have to be verified independently:

    1. Is the anatomical polygon correct?
    2. Is the semantic skin mask correct?
    3. Is their intersection correct?

Checking only the intersection is a trap: a bad polygon can be hidden behind a plausible
skin mask, and it will stay hidden until a feature produces a finding nobody can explain.
Hence the separate overlay modes.

``config/rois.yaml meta.verified`` may be set to true only after this has been reviewed
across poses, hairstyles and facial hair. Until then the pipeline refuses to run in
production mode (D15).

Usage:
    python scripts/debug_roi.py --image face.jpg --mode all --out overlay.png
    python scripts/debug_roi.py --image-dir data/raw/sfhq --mode polygons --out-dir out/
    python scripts/debug_roi.py --image face.jpg --roi forehead --show-anchor
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import skin_analysis.face.landmarks as lm  # noqa: E402
import skin_analysis.face.rois as roi_mod  # noqa: E402
import skin_analysis.util.config as cfg  # noqa: E402
import skin_analysis.util.scale as scale  # noqa: E402
from skin_analysis.schemas import RunMode  # noqa: E402

#: Overlay modes, deliberately separable so each stage can be judged on its own.
MODES = (
    "landmarks",     # raw landmark points, with the pose-solver anchors highlighted
    "polygons",      # ROI polygons from config/rois.yaml, after erosion
    "skin_mask",     # the semantic skin mask alone
    "intersection",  # polygon AND skin_mask
    "measurable",    # the final measurable region handed to features, with coverage
    "all",           # every mode above, tiled
)

#: Distinct hues per ROI so adjacent regions never share a colour. BGR.
_ROI_COLORS = {
    "forehead": (0, 215, 255),
    "left_cheek": (0, 255, 0),
    "right_cheek": (0, 200, 120),
    "nose": (255, 128, 0),
    "left_crows_feet": (255, 0, 255),
    "right_crows_feet": (200, 0, 200),
    "left_under_eye": (255, 255, 0),
    "right_under_eye": (200, 200, 0),
    "left_nasolabial": (0, 0, 255),
    "right_nasolabial": (60, 60, 255),
    "chin": (180, 105, 255),
}
_FALLBACK_COLOR = (200, 200, 200)


def _tint(canvas: np.ndarray, mask: np.ndarray, color: tuple[int, int, int],
          alpha: float = 0.45) -> None:
    """Blend a colour into ``canvas`` wherever ``mask`` is set, in place."""
    if not mask.any():
        return
    patch = canvas[mask].astype(np.float32)
    canvas[mask] = (patch * (1 - alpha) + np.array(color, np.float32) * alpha).astype(np.uint8)


def _outline(canvas: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> None:
    import cv2

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(canvas, contours, -1, color, 1)


def _label(canvas: np.ndarray, text: str, origin: tuple[int, int],
           color: tuple[int, int, int] = (255, 255, 255)) -> None:
    import cv2

    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def _panel_landmarks(image: np.ndarray, geometry: lm.FaceGeometry, anchor_px: float,
                     show_anchor: bool) -> np.ndarray:
    import cv2

    canvas = image.copy()
    points = geometry.landmarks
    for x, y, _ in points:
        cv2.circle(canvas, (int(round(x)), int(round(y))), 1, (0, 255, 0), -1)

    # Highlight the pose anchors: if these sit wrong, every pose number is wrong.
    for index in lm._POSE_LANDMARKS:
        x, y = points[index, :2]
        cv2.circle(canvas, (int(round(x)), int(round(y))), 4, (0, 0, 255), -1)

    if show_anchor:
        left = points[scale.LEFT_IRIS_CENTER, :2]
        right = points[scale.RIGHT_IRIS_CENTER, :2]
        cv2.line(canvas, tuple(np.round(left).astype(int)),
                 tuple(np.round(right).astype(int)), (255, 255, 0), 2)
        _label(canvas, f"IOD anchor = {anchor_px:.0f}px",
               (int(min(left[0], right[0])), int(left[1]) - 10), (255, 255, 0))

    _label(canvas, "LANDMARKS", (10, 24))
    _label(
        canvas,
        f"yaw {geometry.yaw_deg:+.1f}  pitch {geometry.pitch_deg:+.1f} "
        f" roll {geometry.roll_deg:+.1f}",
        (10, 46),
    )
    # Handedness is the error that survives every magnitude-only check, so name the sides
    # on the image itself rather than trusting the reviewer to remember the convention.
    _label(canvas, "subject RIGHT", (10, image.shape[0] - 14), (120, 220, 255))
    _label(canvas, "subject LEFT", (image.shape[1] - 130, image.shape[0] - 14), (120, 220, 255))
    return canvas


def _panel_masks(image: np.ndarray, masks: dict[str, np.ndarray], title: str,
                 only: str | None = None) -> np.ndarray:
    canvas = image.copy()
    for name, mask in masks.items():
        if only and name != only:
            continue
        color = _ROI_COLORS.get(name, _FALLBACK_COLOR)
        _tint(canvas, mask, color)
        _outline(canvas, mask, color)
    _label(canvas, title, (10, 24))
    return canvas


def _panel_skin(image: np.ndarray, skin: np.ndarray | None) -> np.ndarray:
    canvas = image.copy()
    if skin is None:
        _label(canvas, "SKIN MASK - not implemented (B4)", (10, 24), (0, 165, 255))
        return canvas
    _tint(canvas, skin, (255, 255, 255), alpha=0.35)
    _label(canvas, "SKIN MASK", (10, 24))
    return canvas


def _panel_measurable(image: np.ndarray, polygons: dict[str, np.ndarray],
                      composed: dict[str, np.ndarray], anchor_px: float,
                      roi_config: dict) -> np.ndarray:
    canvas = _panel_masks(image, composed, "MEASURABLE")
    fractions = roi_mod.measurable_fraction(polygons, composed)
    undersized = set(roi_mod.undersized(composed, anchor_px, roi_config))

    for row, (name, frac) in enumerate(sorted(fractions.items())):
        flag = "  UNDERSIZED" if name in undersized else ""
        color = (0, 0, 255) if name in undersized else _ROI_COLORS.get(name, _FALLBACK_COLOR)
        _label(canvas, f"{name:<18} {frac * 100:5.1f}%{flag}", (10, 48 + row * 18), color)
    return canvas


def _tile(panels: list[np.ndarray], columns: int = 3) -> np.ndarray:
    height, width = panels[0].shape[:2]
    rows = (len(panels) + columns - 1) // columns
    sheet = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for i, panel in enumerate(panels):
        r, c = divmod(i, columns)
        sheet[r * height:(r + 1) * height, c * width:(c + 1) * width] = panel
    return sheet


def render(image: np.ndarray, mode: str, roi: str | None, show_anchor: bool) -> np.ndarray:
    """Build the requested overlay. Returns a BGR image."""
    profile = cfg.capture_profile()
    roi_config = cfg.load("rois")

    geometry = lm.geometry(image, profile)
    if geometry is None:
        canvas = image.copy()
        _label(canvas, "NO SINGLE FACE DETECTED", (10, 24), (0, 0, 255))
        return canvas

    anchor_px = scale.inter_ocular_distance(geometry.landmarks)

    # DEVELOPMENT mode on purpose: this script is the workflow that produces the
    # verification that meta.verified records (D15).
    polygons = roi_mod.build(
        geometry.landmarks, image.shape[:2], roi_config, run_mode=RunMode.DEVELOPMENT
    )

    skin: np.ndarray | None = None
    try:
        import skin_analysis.face.skin_mask as skin_mask

        skin = skin_mask.build(image, geometry.landmarks, roi_config)
    except NotImplementedError:
        skin = None  # B4 not landed yet; polygons are still fully reviewable without it

    composed = roi_mod.compose(polygons, skin) if skin is not None else polygons

    panels = {
        "landmarks": lambda: _panel_landmarks(image, geometry, anchor_px, show_anchor),
        "polygons": lambda: _panel_masks(image, polygons, "ROI POLYGONS (eroded)", roi),
        "skin_mask": lambda: _panel_skin(image, skin),
        "intersection": lambda: _panel_masks(image, composed, "ROI AND SKIN", roi),
        "measurable": lambda: _panel_measurable(
            image, polygons, composed, anchor_px, roi_config
        ),
    }

    if mode == "all":
        return _tile([panels[name]() for name in MODES if name != "all"])
    return panels[mode]()


def _process(path: Path, out: Path, mode: str, roi: str | None, show_anchor: bool) -> bool:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"  SKIP  {path.name}: could not decode", file=sys.stderr)
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), render(image, mode, roi, show_anchor))
    print(f"  ok    {path.name} -> {out}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="single image to overlay")
    source.add_argument("--image-dir", help="directory of images; renders each one")
    parser.add_argument("--mode", default="all", choices=MODES)
    parser.add_argument("--roi", default=None, help="restrict the overlay to one ROI by name")
    parser.add_argument(
        "--show-anchor",
        action="store_true",
        help="draw the inter-ocular anchor used to resolve every fraction-of-IOD size (D1)",
    )
    parser.add_argument("--out", default="debug_overlay.png", help="output file (--image)")
    parser.add_argument("--out-dir", default="debug_overlays", help="output dir (--image-dir)")
    parser.add_argument("--limit", type=int, default=0, help="cap images processed")
    args = parser.parse_args()

    if args.roi and args.roi not in cfg.load("rois")["rois"]:
        parser.error(f"unknown ROI {args.roi!r}; known: {sorted(cfg.load('rois')['rois'])}")

    if args.image:
        _process(Path(args.image), Path(args.out), args.mode, args.roi, args.show_anchor)
        return

    directory = Path(args.image_dir)
    files = sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if args.limit:
        files = files[: args.limit]
    if not files:
        parser.error(f"no images found in {directory}")

    out_dir = Path(args.out_dir)
    rendered = sum(
        _process(path, out_dir / f"{path.stem}_{args.mode}.png", args.mode, args.roi,
                 args.show_anchor)
        for path in files
    )
    print(f"\n{rendered}/{len(files)} rendered into {out_dir}")
    print(
        "\nReview checklist before setting rois.yaml meta.verified: true\n"
        "  - forehead stops short of the hairline on LOOSE hair, not just tied back\n"
        "  - chin excludes beard and stubble\n"
        "  - crow's feet sit outboard of the lash line, not on the eye\n"
        "  - nasolabial follows the fold, and is present ONLY for wrinkles (D8)\n"
        "  - paired ROIs are mirror images in size and placement\n"
        "  - subject LEFT is on the image RIGHT (labelled in the landmarks panel)\n"
        "  - no ROI reaches background, hair, glasses frames, or lips"
    )


if __name__ == "__main__":
    main()
