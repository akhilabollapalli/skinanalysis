"""Run sliced validation and print per-slice results.

Never report a pooled number: a model that works on average and fails on one skin tone
is a failed model. See .claude/skills/repeatability-test.

Usage:
    python scripts/run_validation.py --slice skin_tone --slice device --slice lighting
"""

from __future__ import annotations

import argparse

SLICES = ["skin_tone", "device", "lighting", "age", "makeup_facial_hair", "longitudinal"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", action="append", choices=SLICES, default=[])
    parser.parse_args()
    raise NotImplementedError("scripts/run_validation.py is not implemented yet.")


if __name__ == "__main__":
    main()
