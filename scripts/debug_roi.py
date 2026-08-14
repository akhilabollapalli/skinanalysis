"""Render landmarks, ROI polygons and the skin mask over an image.

Most false positives in this pipeline trace to the mask, not the feature math -- V1 has
no learned face parser, so boundaries are the weak point. Run this before tuning any
feature threshold. See .claude/skills/roi-debug for the review checklist.

Usage:
    python scripts/debug_roi.py --image <path> --out debug_overlay.png
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="debug_overlay.png")
    parser.parse_args()
    raise NotImplementedError("scripts/debug_roi.py is not implemented yet.")


if __name__ == "__main__":
    main()
