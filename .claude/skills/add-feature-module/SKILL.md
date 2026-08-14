---
name: add-feature-module
description: Scaffold or implement a concern-feature module (redness, pigmentation, texture, wrinkles, fine_lines, acne, pores) following the project's module contract, config-driven thresholds, and ordinal output rules. Use when building or rewriting any feature in src/skin_analysis/features/.
---

# Add / Implement a Feature Module

Build a concern module that conforms to the pipeline contract so it can be enabled, disabled, or
later swapped for a learned implementation without touching app-facing code.

## The contract

```python
def analyze(
    image: np.ndarray,           # BGR uint8, native resolution
    skin_mask: np.ndarray,       # bool, True = analyzable skin
    rois: dict[str, np.ndarray], # ROI name -> bool mask
    config: dict,                # from config/severity_thresholds.yaml
) -> FeatureResult
```

Return a `FeatureResult` (see `src/skin_analysis/schemas.py`) carrying:

- `concern` — the concern name
- `status` — `not_detected | mild | moderate | high | unmeasurable | disabled`
- `regions` — list of affected ROI names
- `raw` — dict of internal metrics (**never** surfaced to the user)
- `confidence_internal` — float, internal only

## Steps

### 1. Pick the right image copy

| Concern type | Copy | Why |
|---|---|---|
| redness, pigmentation | **color-calibrated** original | contrast enhancement destroys the luminance/chroma relationships being measured |
| wrinkles, texture, pores | **structural** (high-pass / local contrast) | relief and thin structure need frequency emphasis |
| acne | **native-resolution** crop | small-lesion localization needs real pixels |

### 2. Compute against a local baseline, never a global constant

Absolute CIELAB values conflate lighting and skin tone with the concern. Always normalize
against a local or ROI-level robust baseline:

```
z = (m - median_local) / (MAD_local + eps)
```

### 3. Exclude aggressively

Lips, nostrils, eye margins, brows, hair, beard, glasses, specular highlights, and deep shadow.
Under-exclusion is the single largest source of false positives in this pipeline.

### 4. Put every threshold in config

No magic numbers in `.py` files. Add each knob to `config/severity_thresholds.yaml` with a
comment giving its units and re-tuning guidance.

### 5. Mark unmeasurable rather than guessing

If the ROI is shadowed, blown out, occluded, or lacks pixel density for the concern's scale,
return `unmeasurable`. A confident-looking number from bad pixels is worse than no answer.

### 6. Keep it pure

No I/O, no network, no globals, no input mutation. Deterministic given the same inputs —
repeatability is this product's core quality metric.

### 7. Document the ceiling

In the module docstring, state plainly what this classical implementation cannot do and what
would replace it once commercially clear labels exist.

### 8. Add tests

- repeatability: same input twice → identical ordinal severity
- exclusion: synthetic hair/shadow/lip pixels do not create findings
- unmeasurable: degraded input returns `unmeasurable`, not a severity

## Feature flags

`acne`, `fine_lines`, and `pores` are **disabled by default** and must stay that way until
commercially clear labels exist and the concern passes its acceptance gate. Implement their
interface and schema now; do not enable them to fill the UI.

## Before finishing

```bash
pytest tests/ -v
```

Confirm: no new undeclared dependency, no numeric value in the user-facing payload, thresholds
all live in config.
