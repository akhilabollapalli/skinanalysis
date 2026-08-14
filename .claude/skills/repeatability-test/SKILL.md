---
name: repeatability-test
description: Design or run repeatability and robustness validation for the skin pipeline — same-subject rescans, cross-device, lighting, and skin-tone slicing. Use before enabling a concern, before a release, or when a measurement seems unstable.
---

# Repeatability & Robustness Testing

Repeatability is this product's primary quality metric. A user who rescans within five minutes
and sees "mild" become "high" stops trusting the product, regardless of how it scores against
expert labels.

## What to measure

### Same-session repeatability

Capture the same subject 3–5 times in immediate succession under unchanged conditions.

- **Ordinal stability**: what fraction of rescans return the identical severity? This is the
  headline number.
- **Raw drift**: coefficient of variation of the internal metric.
- **Boundary proximity**: flag subjects sitting near a threshold — instability there is expected
  and is a calibration problem, not a CV bug. Distinguish the two.

### Cross-session repeatability

Same subject, different days, standardized capture. Skin genuinely changes, so separate real
change from measurement noise before declaring a regression.

### Cross-device

Same subject and lighting, different phone models. Report per-device severity distribution and
capture-rejection rate. Systematic per-device offsets mean the color pipeline needs calibration,
not that the feature is broken.

### Lighting

natural / white / warm / dim / backlit. Expect redness and pigmentation to be the most
lighting-sensitive — they are CIELAB proxies in V1, so they partially measure the room. Quantify
that rather than assuming it away.

## Required slicing

Never report a pooled number. Always slice by:

skin tone (with CIs and subgroup n) · device · lighting · age · makeup/facial hair

A subgroup with small n is itself a finding — report it rather than smoothing it over.

## Pass criteria

Gates must be **prespecified**. A threshold chosen after seeing the results is not a gate.

| Check | Typical gate |
|---|---|
| Same-session ordinal agreement | high, and equal across skin-tone slices |
| Raw metric CV | within prespecified bound per concern |
| Cross-device severity shift | no systematic offset beyond bound |
| Capture rejection rate | acceptable and not concentrated in one subgroup |

## Running

```bash
pytest tests/test_repeatability.py -v
python scripts/run_validation.py --slice skin_tone --slice device --slice lighting
```

## Interpreting failure

- Unstable **raw** metric → CV problem (exclusion, baseline, or noise sensitivity)
- Stable raw but unstable **ordinal** → threshold calibration problem
- Fails on one subgroup only → fairness problem; do not ship, and do not fix by loosening the gate
- Fails only in poor lighting → tighten capture QC rather than weakening the feature

When a concern cannot pass, recommend disabling it. This project already ships with three
concerns disabled by design — that is the intended behavior, not a failure.
