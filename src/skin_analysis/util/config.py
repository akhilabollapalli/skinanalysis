"""Config loading. Every threshold in this project lives in config/*.yaml, never in code."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


@lru_cache(maxsize=None)
def load(name: str) -> dict[str, Any]:
    """Load a YAML config by stem, e.g. load("severity_thresholds")."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def feature_enabled(concern: str) -> bool:
    """A concern runs only when explicitly enabled in config.

    Defaults to False: acne, fine_lines and pores stay off until commercially clear
    labels exist and the concern passes its acceptance gate.
    """
    return bool(load("severity_thresholds").get(concern, {}).get("enabled", False))


def severity_calibrated() -> bool:
    """True once ordinal thresholds have been calibrated on usable annotations."""
    return bool(load("severity_thresholds").get("meta", {}).get("calibrated", False))
