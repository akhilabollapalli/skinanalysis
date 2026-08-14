# Population reference statistics

D1 stage B. These files hold the frozen cohort statistics that
`decision/standardize.py` standardizes raw measurements against:

    z_ref = (m - median) / (1.4826 * mad + eps)

## What `ref` means

**A frozen, commercially cleared calibration cohort captured under the defined V1
acquisition protocol.**

It does **not** mean "people demographically like this user", and it does **not** define
what normal or healthy skin is. Its only job is to standardize a raw measurement before
the ordinal calibrator sees it. Subgroup-specific references are permitted only when a
validation result justifies one — never as a shortcut for demographic stereotyping
(architecture doc §13).

## Why this directory is not gitignored

The rest of `data/` is excluded because it holds face images. These files hold aggregate
median/MAD per (concern, ROI) — no images, nothing identifiable — and they determine every
user-facing severity the product produces. A calibration that cannot be diffed, reviewed,
and reverted is not auditable. Image patterns in `.gitignore` still apply inside this tree.

## Layout

    data/reference/v1/<concern>/<roi>.json

## Schema

```json
{
  "protocol_version": "v1",
  "concern": "redness",
  "roi": "left_cheek",
  "n_subjects": 0,
  "median": null,
  "mad": null,
  "quantiles": {},
  "capture_profile": "selfie-v1",
  "source_manifest_hash": null,
  "calibrated": false
}
```

`median` and `mad` are keyed by measurement name, matching the concern's
`raw_measurements` list in `config/severity_thresholds.yaml`.

## Rules

- **Produced by a calibration run, never edited by hand.** These are an asset, not a
  tuning knob. Hand-editing one silently rewrites what "moderate" means.
- `source_manifest_hash` records the cohort the statistics came from. A file without one
  cannot be reproduced and must not be marked calibrated.
- Every file currently ships with `calibrated: false`. While any primary ROI of a concern
  is uncalibrated, `util.calibration.assert_public_ready` refuses to publish that concern
  (D2), and the pipeline runs internal-only.
- Changing the capture protocol invalidates every file here. Cohort statistics are not
  transferable across protocols — that is the whole reason spatial parameters are
  fractions of the scale anchor rather than fixed pixel counts.
