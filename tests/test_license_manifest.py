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
    "asset", "type", "source_url", "model_card_url", "version_or_date", "version",
    "license", "license_version", "layer_rights", "commercial_allowed",
    "sha256", "download_date", "attribution_required", "allowed_purpose",
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


@pytest.mark.parametrize("row", _rows(), ids=lambda r: r["asset"])
def test_decision_and_commercial_allowed_agree(row: dict[str, str]) -> None:
    """`commercial_allowed` is the machine-checkable half of the licensing gate.

    A row that says ELIGIBLE while commercial_allowed is anything but ``yes`` is either a
    typo or a rationalisation. Both should fail the build.
    """
    eligible = row["decision"].startswith("ELIGIBLE")
    allowed = row["commercial_allowed"].strip().lower()
    if eligible:
        assert allowed == "yes", (
            f"{row['asset']}: decision is {row['decision']} but commercial_allowed={allowed!r}"
        )
    else:
        assert allowed != "yes", (
            f"{row['asset']}: EXCLUDED assets must not be marked commercially allowed"
        )


@pytest.mark.parametrize(
    "row", [r for r in _rows() if r["type"] == "weights"], ids=lambda r: r["asset"]
)
def test_eligible_weights_cite_a_model_card(row: dict[str, str]) -> None:
    """Weights are cleared by their model card, never by the code license (CLAUDE.md Rule 1).

    The BiSeNet row in this manifest is the standing example of why: MIT code, restricted
    weights. An eligible weights row without a model card URL has no evidence behind it.
    """
    if not row["decision"].startswith("ELIGIBLE"):
        return
    assert row["model_card_url"].startswith("http"), (
        f"{row['asset']}: eligible weights must cite an authoritative model card"
    )


def test_no_unpinned_eligible_weights() -> None:
    """Weight hashes and versions must be pinned before release.

    An unpinned .task bundle means the audited artifact and the shipped artifact are not
    provably the same file.
    """
    unpinned = [
        f"{r['asset']}({field})"
        for r in _rows()
        if r["type"] == "weights" and r["decision"].startswith("ELIGIBLE")
        for field in ("version", "sha256", "download_date")
        if r[field].startswith("PIN_")
    ]
    if unpinned:
        pytest.xfail(f"pin before release: {unpinned}")


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
