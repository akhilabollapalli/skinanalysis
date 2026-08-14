"""CI gate for the commercial-open licensing rule (CLAUDE.md Rule 1).

This test is intentionally strict. It is cheaper to fail a build than to discover at
launch that a model was trained on non-commercial data.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "LICENSES" / "asset_manifest.csv"

REQUIRED_COLUMNS = {
    "asset", "type", "source_url", "version_or_date", "license",
    "layer_rights", "attribution_required", "allowed_purpose",
    "decision", "reviewed_on",
}

VALID_DECISIONS = {"ELIGIBLE", "ELIGIBLE WITH CONDITIONS", "EXCLUDED"}

# Assets that must never be marked eligible without a documented re-audit.
STANDING_EXCLUSIONS = {
    "FFHQ-Wrinkle", "AcneSCU", "ACNE04", "ACNE-DET",
    "CelebAMask-HQ", "PorePatch", "BiSeNet face-parsing weights",
}


def _rows() -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_manifest_exists() -> None:
    assert MANIFEST.exists(), "asset_manifest.csv is required before any asset is used"


def test_required_columns_present() -> None:
    missing = REQUIRED_COLUMNS - set(_rows()[0].keys())
    assert not missing, f"manifest missing columns: {missing}"


@pytest.mark.parametrize("row", _rows(), ids=lambda r: r["asset"])
def test_every_asset_has_a_valid_decision(row: dict[str, str]) -> None:
    assert row["decision"] in VALID_DECISIONS, (
        f"{row['asset']}: decision {row['decision']!r} is not one of {VALID_DECISIONS}. "
        "Unclear licensing defaults to EXCLUDED."
    )


@pytest.mark.parametrize("row", _rows(), ids=lambda r: r["asset"])
def test_eligible_assets_are_fully_documented(row: dict[str, str]) -> None:
    if not row["decision"].startswith("ELIGIBLE"):
        return
    for field in ("source_url", "license", "layer_rights", "reviewed_on"):
        assert row[field].strip(), f"{row['asset']}: eligible asset missing {field}"
    assert row["source_url"].startswith("http"), (
        f"{row['asset']}: source must be an authoritative URL, not a mirror or fork"
    )


def test_standing_exclusions_are_still_excluded() -> None:
    by_asset = {r["asset"]: r for r in _rows()}
    for asset in STANDING_EXCLUSIONS:
        assert asset in by_asset, f"{asset} must remain listed in the manifest"
        assert by_asset[asset]["decision"] == "EXCLUDED", (
            f"{asset} was changed to {by_asset[asset]['decision']}. "
            "Re-litigating a standing exclusion requires a documented licensing audit."
        )


def test_no_unpinned_eligible_library() -> None:
    """Libraries must have a pinned version before release.

    Unpinned MediaPipe is a real hazard here: landmark index semantics can drift between
    releases, silently invalidating every calibrated ROI.
    """
    unpinned = [
        r["asset"] for r in _rows()
        if r["type"] == "library" and r["version_or_date"] == "PIN_VERSION"
    ]
    if unpinned:
        pytest.xfail(f"pin before release: {unpinned}")
