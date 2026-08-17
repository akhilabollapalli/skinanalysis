"""Config loading. Every threshold in this project lives in config/*.yaml, never in code."""

from __future__ import annotations

import copy
from functools import cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"

#: Top-level keys in severity_thresholds.yaml that are not concerns.
_NON_CONCERN_KEYS = frozenset(
    {"meta", "defaults", "reference_stats", "scale", "image_copies"}
)


@cache
def load(name: str) -> dict[str, Any]:
    """Load a YAML config by stem, e.g. load("severity_thresholds")."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge; ``override`` wins at every leaf."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def capture_profile(name: str | None = None) -> dict[str, Any]:
    """Return the resolved capture-QC profile (D11).

    Profiles inherit from ``default``; an entry under ``overrides`` layers on top of it.
    ``overrides`` is empty by design -- a device profile is added only with the validation
    run that demonstrated the bias it corrects. Until then every device uses ``default``
    and its metadata is logged so a future profile decision has evidence.
    """
    raw = load("capture_thresholds")
    profiles = raw.get("capture_profiles", {})
    default = profiles.get("default")
    if default is None:
        raise KeyError("capture_thresholds.yaml has no capture_profiles.default")

    requested = name or raw.get("meta", {}).get("active_profile", "default")
    if requested == "default":
        return copy.deepcopy(default)

    override = raw.get("overrides", {}).get(requested)
    if override is None:
        # Unknown device is not an error: it is the expected case in V1.
        return copy.deepcopy(default)
    return _deep_merge(default, override)


def concern_config(concern: str) -> dict[str, Any]:
    """Return a concern's block with ``defaults`` merged underneath it."""
    raw = load("severity_thresholds")
    block = raw.get(concern)
    if block is None:
        raise KeyError(f"severity_thresholds.yaml has no block for concern {concern!r}")
    return _deep_merge(raw.get("defaults", {}), block)


def configured_concerns() -> list[str]:
    """Concern keys present in severity_thresholds.yaml, enabled or not."""
    return [k for k in load("severity_thresholds") if k not in _NON_CONCERN_KEYS]


def feature_enabled(concern: str) -> bool:
    """A concern runs only when explicitly enabled in config.

    Defaults to False: acne, fine_lines and pores stay off until commercially clear
    labels exist and the concern passes its acceptance gate.
    """
    return bool(load("severity_thresholds").get(concern, {}).get("enabled", False))


def severity_calibrated() -> bool:
    """True once ordinal thresholds have been calibrated on usable annotations.

    This is only half the publication gate. See ``util.calibration.assert_public_ready``,
    which additionally requires population reference statistics per ROI (D1, D2).
    """
    return bool(load("severity_thresholds").get("meta", {}).get("calibrated", False))


def protocol_version() -> str:
    """The acquisition/calibration protocol the configs describe."""
    return str(load("severity_thresholds").get("meta", {}).get("protocol_version", "v1"))
