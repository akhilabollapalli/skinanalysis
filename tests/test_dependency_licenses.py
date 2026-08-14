"""The installed environment must match the audited manifest (CLAUDE.md Rule 1).

`test_license_manifest.py` checks what the manifest *says*. This file checks what is
actually importable, because a transitive dependency can enter the graph without anyone
adding a row -- which is exactly how an unaudited component reaches production.

Also guards two properties of the OpenCV build that are correctness issues, not just
licensing ones.
"""

from __future__ import annotations

import csv
import importlib.metadata as md
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "LICENSES" / "asset_manifest.csv"

#: Packages that ship with CPython or with pip itself, and tooling that never reaches a
#: release artifact. Everything else in the import graph needs a manifest row.
_STDLIB_OR_TOOLING = {
    "pip", "setuptools", "wheel", "pytest", "ruff", "mypy", "iniconfig", "pluggy",
    "mypy-extensions", "typing-extensions", "typing_extensions", "colorama", "exceptiongroup",
    "tomli", "pathspec", "attrs", "six", "python-dateutil", "pygments",
}


def _manifest_assets() -> set[str]:
    with MANIFEST.open(newline="", encoding="utf-8") as fh:
        return {row["asset"].strip().lower() for row in csv.DictReader(fh)}


def _runtime_requirements() -> set[str]:
    """Transitive closure of the runtime dependencies declared in requirements.txt."""
    roots = []
    for line in (REPO / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            roots.append(line.split(">=")[0].split("==")[0].strip())

    seen: set[str] = set()
    queue = list(roots)
    while queue:
        name = queue.pop().lower()
        if name in seen:
            continue
        seen.add(name)
        try:
            requires = md.requires(name) or []
        except md.PackageNotFoundError:
            continue
        for req in requires:
            # Skip optional extras: they are not installed by a plain `pip install -r`.
            if "extra ==" in req:
                continue
            dep = req.split(";")[0].split("[")[0]
            for sep in (">=", "<=", "==", "!=", "~=", ">", "<", " "):
                dep = dep.split(sep)[0]
            dep = dep.strip()
            if dep and dep.lower() not in seen:
                queue.append(dep)
    return seen


def test_every_runtime_dependency_has_a_manifest_row() -> None:
    """A dependency with no row is an unaudited component in the release artifact."""
    assets = _manifest_assets()
    missing = sorted(
        name for name in _runtime_requirements()
        if name not in assets and name not in _STDLIB_OR_TOOLING
    )
    assert not missing, (
        f"runtime dependencies absent from asset_manifest.csv: {missing}. "
        "Rule 1: every external asset needs a row before it is used."
    )


def test_exactly_one_opencv_distribution_is_installed() -> None:
    """Two cv2 distributions install into the same directory and clobber each other.

    The resolved version then depends on install order, so two machines running identical
    code can compute different results -- a repeatability bug wearing a packaging costume.
    MediaPipe requires the contrib build, so that is the one this project standardises on.
    """
    installed = sorted(
        d.metadata["Name"]
        for d in md.distributions()
        if d.metadata["Name"] and "opencv" in d.metadata["Name"].lower()
    )
    assert installed == ["opencv-contrib-python"], (
        f"expected only opencv-contrib-python, found {installed}. "
        "Uninstall all opencv distributions and reinstall opencv-contrib-python."
    )


def test_opencv_build_has_nonfree_disabled() -> None:
    """Non-free algorithms are patent-encumbered and are not commercially clear."""
    cv2 = pytest.importorskip("cv2")
    line = next(
        ln for ln in cv2.getBuildInformation().splitlines() if "Non-free algorithms" in ln
    )
    assert line.split(":")[1].strip().upper() == "NO", f"non-free OpenCV build: {line.strip()}"


def test_no_model_weights_are_tracked_by_git() -> None:
    """Weights are never vendored, even when their licence is clear (CLAUDE.md Rule 1).

    This test exists because the .gitignore pattern that was supposed to stop it had a
    trailing inline comment. .gitignore has no inline comments, so the pattern became
    ``*.task            # ...`` and matched nothing, and a 3.6 MB bundle was committed.
    Checking the ignore rules would have missed it; checking what git actually tracks does
    not.
    """
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=False
    )
    if tracked.returncode != 0:
        pytest.skip("not a git checkout")

    suffixes = (".task", ".tflite", ".pt", ".pth", ".onnx", ".pb", ".h5", ".safetensors")
    offenders = [
        line for line in tracked.stdout.splitlines()
        if line.strip().lower().endswith(suffixes)
    ]
    assert not offenders, (
        f"model weights tracked in git: {offenders}. Weights are downloaded and verified "
        "by hash at runtime, never committed."
    )


def test_gitignore_has_no_inline_comments() -> None:
    """An inline comment silently becomes part of the pattern, disabling the rule.

    The failure is invisible: the line still looks like an ignore rule.
    """
    offenders = []
    for number, raw in enumerate((REPO / ".gitignore").read_text(encoding="utf-8").splitlines(), 1):
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if " #" in line or "\t#" in line:
            offenders.append(f"{number}: {line}")
    assert not offenders, (
        f".gitignore lines with inline comments (these do not work): {offenders}"
    )


def test_mediapipe_is_importable_and_apache_licensed() -> None:
    """D3 cleared the Face Landmarker; this checks the installed artifact agrees."""
    pytest.importorskip("mediapipe")
    metadata = md.metadata("mediapipe")
    classifiers = " ".join(metadata.get_all("Classifier") or [])
    declared = str(metadata["License"] or "") if "License" in metadata else ""
    assert "Apache" in declared or "Apache" in classifiers
