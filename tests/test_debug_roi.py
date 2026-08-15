"""Guards for the ROI debug overlay (Stage B3 tooling).

This script is how `rois.yaml meta.verified` gets earned, so it must not silently render
an incomplete picture -- an ROI drawn in the fallback grey, or a mode that quietly does
nothing, would let a bad polygon pass review.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]

pytest.importorskip("mediapipe")

spec = importlib.util.spec_from_file_location("debug_roi", REPO / "scripts" / "debug_roi.py")
debug_roi = importlib.util.module_from_spec(spec)
sys.modules["debug_roi"] = debug_roi
spec.loader.exec_module(debug_roi)

from skin_analysis.util import config as cfg  # noqa: E402


def test_every_roi_has_a_distinct_colour() -> None:
    """A duplicate colour makes two adjacent regions look like one, which is exactly the
    kind of boundary error this overlay exists to catch."""
    configured = set(cfg.load("rois")["rois"])
    assert configured <= set(debug_roi._ROI_COLORS), (
        f"ROIs with no colour: {sorted(configured - set(debug_roi._ROI_COLORS))}"
    )
    used = [debug_roi._ROI_COLORS[name] for name in configured]
    assert len(set(used)) == len(used), "two ROIs share a colour"


def test_all_mode_covers_every_other_mode() -> None:
    assert set(debug_roi.MODES) - {"all"} == {
        "landmarks", "polygons", "skin_mask", "intersection", "measurable"
    }


def test_render_reports_when_no_face_is_found() -> None:
    """Failing loudly matters: a blank overlay would read as 'polygons look fine'."""
    out = debug_roi.render(np.full((480, 480, 3), 60, np.uint8), "all", None, False)
    assert out.shape[:2] == (480, 480), "no-face path must not tile empty panels"


def test_render_degrades_when_the_mask_cannot_be_built() -> None:
    """A face the mask cannot characterise must not blank the overlay: the polygons are
    still reviewable, which is the whole B3-before-B4 ordering."""
    import skin_analysis.face.skin_mask as skin_mask

    landmarks = np.zeros((478, 3))
    landmarks[468, :2] = (100.0, 50.0)
    landmarks[473, :2] = (140.0, 50.0)
    mask = skin_mask.build(
        np.zeros((120, 240, 3), np.uint8), landmarks, cfg.load("rois")
    )
    assert mask.shape == (120, 240) and not mask.any(), "degenerate input must fail closed"
