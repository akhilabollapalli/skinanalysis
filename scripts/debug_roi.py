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
Hence the separate overlay modes below.

``config/rois.yaml meta.verified`` may be set to true only after this has been reviewed
across poses, hairstyles and facial hair. Until then the pipeline refuses to run in
production mode.

Usage:
    python scripts/debug_roi.py --image <path> --mode all --out debug_overlay.png
"""

from __future__ import annotations

import argparse

#: Overlay modes, deliberately separable so each stage can be judged on its own.
MODES = (
    "landmarks",     # raw landmark points and indices
    "polygons",      # ROI polygons from config/rois.yaml, pre-erosion and post-erosion
    "skin_mask",     # the semantic skin mask alone
    "intersection",  # polygon AND skin_mask
    "measurable",    # the final measurable region actually handed to features
    "all",           # every mode above, tiled
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--mode", default="all", choices=MODES)
    parser.add_argument("--roi", default=None, help="restrict to one ROI by name")
    parser.add_argument(
        "--show-anchor",
        action="store_true",
        help="draw the inter-ocular anchor and the resolved window sizes (D1)",
    )
    parser.add_argument("--out", default="debug_overlay.png")
    parser.parse_args()
    raise NotImplementedError("scripts/debug_roi.py is not implemented yet.")


if __name__ == "__main__":
    main()
