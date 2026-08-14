# Skin Analysis — RGB Selfie Cosmetic Concern Detection

Analyzes seven cosmetic skin concerns from a consumer RGB selfie: **pores, wrinkles, fine
lines, acne, dark spots, texture, and redness**.

Built to be **owned IP**. Every production asset — dataset, annotation, library, and weight —
must carry an authoritative license that already permits commercial use. That constraint shapes
the entire architecture, so read `CLAUDE.md` before writing code.

---

## Status: Phase 0 → 1 (scaffold)

The skeleton, contracts, configs, and licensing gate are in place. Feature implementations are
stubs raising `NotImplementedError`.

| Concern | V1 status | Method |
|---|---|---|
| Redness | **Active** | CIELAB a* local excess + ROI baseline + asymmetry |
| Dark spots | **Active** | CIELAB L* local deficit + shadow rejection |
| Texture | **Active** | GLCM + gradients + frequency features |
| Wrinkles | **Active baseline** | Multi-scale Gabor + Hessian ridge + morphology |
| Fine lines | Disabled | Needs validated high-resolution commercial labels |
| Acne | Disabled | Needs commercial-open lesion-level labels |
| Pores | Disabled | Needs commercial-open native-resolution pore masks |

Hydration is **out of scope** — RGB-selfie hydration estimation is not reliable enough to
present as a measurement.

### Why three concerns ship disabled

The strongest public datasets for acne, pores, and fine lines (AcneSCU, ACNE04, PorePatch,
FFHQ-Wrinkle) are all non-commercial. Rather than train on data the product cannot legally
use, those concerns keep their full interface and stay behind feature flags until commercially
clear labels exist. Turning one on requires cleared data **and** a passed acceptance gate.

---

## Architecture

```
capture QC  →  landmarks + ROIs  →  skin mask  →  two image copies
            →  active concern modules  →  ordinal severity  →  public payload
```

Two image copies, deliberately not sharing preprocessing: a **color-calibrated** copy for
redness and pigmentation, and a **structural** copy for wrinkles and texture. Contrast
enhancement helps structure detection but destroys the luminance relationships that color
features depend on.

The pipeline **fails closed** — a failed capture returns RETAKE and runs no concern logic.

### Output contract

Users see ordinal severity only:

```json
{"redness": {"status": "moderate", "regions": ["left_cheek", "right_cheek"]}}
```

No 0–100 score, no 0–5 score, no percentages, counts, or confidence. Raw metrics stay internal
for validation. `ScanResult.to_public()` is the enforcement point, and
`tests/test_output_contract.py` is the guard.

---

## Layout

```
.claude/              Project agents and skills
  agents/             licensing-auditor, cv-feature-implementer,
                      validation-engineer, annotation-lead
  skills/             license-check, add-feature-module,
                      repeatability-test, roi-debug
config/               All thresholds. No magic numbers in code.
LICENSES/             asset_manifest.csv (CI-enforced), THIRD_PARTY_NOTICES.md
src/skin_analysis/    capture · face · features · decision · rules · pipeline · schemas
tests/                incl. test_license_manifest.py and test_output_contract.py
scripts/              debug_roi.py, run_validation.py
docs/                 Architecture document
data/                 Gitignored. Never commit face images.
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .
pytest
```

---

## Known limitations of V1

Stated plainly because they should drive the roadmap, not be discovered later:

- **No learned face parser.** CelebAMask-HQ is excluded, which excludes the BiSeNet weights
  trained on it. The skin mask comes from landmark polygons plus classical rejection — weaker
  at hair, beard, and lip boundaries. This is the highest-value upgrade once a clear parser
  exists.
- **CIELAB proxies partially measure lighting.** Without hemoglobin/melanin decomposition,
  redness and pigmentation carry some of the room in them. Strict capture QC — not feature
  sophistication — is what makes them trustworthy.
- **Severity thresholds are uncalibrated placeholders.** `config/severity_thresholds.yaml` has
  `meta.calibrated: false`. Do not ship user-facing severity until calibration lands.
- **Ridge filters respond to hair and glasses** as readily as to wrinkles; suppression is part
  of the algorithm.

---

## Ground rules

1. **Licensing gate** — nothing enters production without an authoritative commercial license,
   checked separately for images, annotations, code, weights, and source media.
2. **Disabled stays disabled** — until cleared labels and a passed gate.
3. **No numbers to the user** — ordinal severity and regions only.

Full detail in `CLAUDE.md`. This product estimates **cosmetic appearance**; it does not
diagnose.
