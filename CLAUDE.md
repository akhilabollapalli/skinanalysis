# CLAUDE.md — RGB Selfie Skin Analysis

Project instructions for Claude Code / Cowork sessions in this repository.
Read this before writing or modifying any code.

---

## 1. What this project is

A computer-vision pipeline that analyzes seven cosmetic skin concerns from a consumer RGB
selfie: **pores, wrinkles, fine lines, acne, dark spots/hyperpigmentation, visual
texture/roughness, and redness**.

Hydration is **out of scope** and must not be added as a measured output. RGB-selfie hydration
estimation is not reliable enough to present as a measurement (see
`docs/RGB_Selfie_Skin_Analysis_Research_Architecture.docx` §11).

The architecture and every design decision in this repo derive from that architecture document.
When code and this file disagree with the document, raise it — do not silently diverge.

Where the document left a choice open, `docs/DECISIONS.md` closes it. Those decisions (D1–D15)
are binding and are referenced by ID throughout this file and the configs. Read it before
changing any threshold, signature, or output shape.

### V1 non-goals

- **No longitudinal claims** (D12). No "your redness improved", no trend, no history delta, no
  comparison of severity across capture sessions, no treatment-response claim. V1 answers exactly
  one question: *what concerns are visibly detectable in this scan?*
- No hydration output, measured or estimated.
- No numeric score of any kind.
- No on-device inference path in V1 (D10 — server-side, though QC and landmarks must stay
  portable enough to move on-device later without changing any concern API).

---

## 2. The three non-negotiable rules

These are the rules that make this project legally and scientifically defensible. Breaking any
one of them is a blocking bug, not a style preference.

### Rule 1 — Commercial-open licensing gate

**A production asset is eligible only if its authoritative source explicitly permits commercial
use.**

- Check rights *separately* for: (1) dataset images, (2) annotations, (3) source code,
  (4) pretrained weights, (5) underlying source media.
- A permissive **code** license does **not** make the **weights** or **training images**
  commercially usable.
- Never infer rights from: a public GitHub repo, a Kaggle/Roboflow mirror, a paper being open
  access, or "everyone else uses it".
- Reject anything marked research-only, academic-only, non-commercial, "contact the authors",
  permission-required, or license-unclear.
- Every external asset must have a row in `LICENSES/asset_manifest.csv` before it is used.

**Currently EXCLUDED from production** (architecture/method reference only):
FFHQ-Wrinkle (CC BY-NC-SA), AcneSCU, ACNE04, CelebAMask-HQ, PorePatch, ACNE-DET.

**Currently ELIGIBLE**: FairFace (CC BY 4.0, attribution required), SCIN (custom license, its
conditions apply), SFHQ (MIT — infrastructure/QC stress tests only, never skin-concern truth).

> Consequence to remember: because CelebAMask-HQ is excluded, **all common BiSeNet face-parsing
> weights are excluded too**. V1 builds the skin mask from MediaPipe landmark polygons plus
> classical skin/hair/occlusion rejection. Do not `pip install` or download a face parser without
> clearing its weights first.

**MediaPipe Face Landmarker is cleared** (D3). All three bundled models — BlazeFace short-range,
FaceMesh V2, Blendshape V2 — are Apache-2.0 per Google's official model cards, weights included,
each with its own manifest row. Blendshape is cleared but **unused**: run the landmarker with
`output_face_blendshapes=False` and `output_facial_transformation_matrixes=False`.

Weights rows carry `model_card_url`, `sha256`, `version` and `download_date`. `PIN_*` placeholders
xfail today and must be real before the Phase 8 release audit — an unpinned bundle means the
audited artifact and the shipped artifact are not provably the same file.

### Rule 2 — V1 is classical CV, and disabled concerns stay disabled

| Concern | V1 status | Implementation |
|---|---|---|
| Redness | **ACTIVE** | CIELAB a* local excess + ROI baseline + asymmetry |
| Dark spots / pigmentation | **ACTIVE** | CIELAB L* local deficit + shadow rejection + components |
| Texture / roughness | **ACTIVE** | GLCM + gradients + local variance + HF ratio |
| Wrinkles | **ACTIVE BASELINE** | Multi-scale Gabor + Hessian/ridge + morphology |
| Fine lines | **EXPERIMENTAL — flag off** | Geometry derived from line map, internal only |
| Acne | **EXPERIMENTAL — flag off** | Interface only; no training on restricted data |
| Pores | **EXPERIMENTAL — flag off** | Native-res ROI tiling + candidate interface only |

Do **not** enable a flagged-off concern to "fill the UI". A concern turns on only when
commercially clear labels exist *and* it passes its acceptance gate. Their schemas must already
match the final contract so enabling them later changes no app-facing code.

### Rule 3 — The user never sees numbers, and sees nothing at all before calibration

User-facing output is **ordinal only**:

```
concern name + {not_detected | mild | moderate | high} + affected regions
```

Everything else — z-scores, probabilities, area fractions, lesion counts, densities, GLCM
values, confidence — is **internal**. Log it for validation; never return it in the
app-facing payload. There is no 0–100 score and no 0–5 score in this product.

**D2 extends this.** Measurement and publication are separate operations:

- `pipeline.analyze_scan_internal()` always works. It is what validation and threshold-fitting
  use, and it returns raw metrics freely.
- `ScanResultInternal.to_public()` raises `CalibrationRequiredError` until
  `severity_thresholds.yaml meta.calibrated` is true **and** every published concern has
  calibrated population reference statistics for its primary ROIs (D1 stage B).

There is deliberately **no `Severity.UNCALIBRATED`**. Any such member is a value UI code could
eventually render. Uncalibrated is not a finding; it is a state in which the product has no
output. Do not add one, and do not catch `CalibrationRequiredError` to substitute a default.

---

## 3. Repository layout

```
src/skin_analysis/
  capture/qc.py           Capture quality gate. Fails closed -> RETAKE, no analysis runs.
  face/landmarks.py       MediaPipe Face Landmarker wrapper (468/478 pts).
  face/rois.py            Polygons from landmarks; compose() intersects with the skin
                          mask. Refuses unverified polygons in production (D15).
  face/skin_mask.py       Skin mask = ROI polygon ∩ skin/occlusion rejection.
  features/redness.py         ACTIVE
  features/pigmentation.py    ACTIVE
  features/texture.py         ACTIVE
  features/wrinkles.py        ACTIVE BASELINE
  features/fine_lines.py      EXPERIMENTAL (flag off)
  features/acne.py            EXPERIMENTAL (flag off)
  features/pores.py           EXPERIMENTAL (flag off)
  decision/standardize.py Population z_ref against frozen cohort stats (D1 stage B).
  decision/calibrator.py  Standardized value -> ordinal band. Must be monotonic.
  decision/severity.py    Per-ROI severities -> concern severity (D6 max-with-support, D7).
  rules/adapter.py        PublicScanResult -> RecommendationEngine protocol (D9).
  util/config.py          YAML loading, capture-profile resolution (D11).
  util/calibration.py     The publication gate. Only route to a public payload (D2).
  util/scale.py           Fraction-of-anchor -> pixels (D1). No fixed pixel windows.
  pipeline.py             Orchestration. The only public entry point.
  schemas.py              Dataclasses for every boundary. Source of truth for the contract.
config/                   capture_thresholds.yaml, severity_thresholds.yaml, rois.yaml
LICENSES/                 asset_manifest.csv, THIRD_PARTY_NOTICES.md
docs/DECISIONS.md         D1-D15. Binding. Read before changing a threshold or signature.
tests/                    incl. test_license_manifest.py (CI-blocking),
                          test_face_pipeline.py (Stage B definition of done)
data/                     gitignored EXCEPT data/reference/ (aggregate cohort stats, no images).
```

### The three image copies (D5)

They deliberately do not share preprocessing:

| Copy | Preprocessing | Consumers |
|---|---|---|
| `color` | sRGB -> Lab/D65. **No gray-world, no illuminant estimation, no CLAHE** (D4). | redness, pigmentation |
| `texture` | Luminance, fixed normalization only. **No CLAHE** — GLCM downstream of adaptive gain partly measures the gain. | texture |
| `ridge` | Optional *validated* enhancement for Gabor/Hessian. | wrinkles |

The term "color-calibrated" is retired (D4). Without a chart or RAW reference it overclaims; it
is the **standardized color-analysis copy**. Because nothing corrects a colour cast, the QC
white-balance check is load-bearing — tighten it before loosening any colour threshold.

---

## 4. Coding conventions

- Python 3.10+. Type hints on every public function.
- Feature modules expose exactly one entry point:
  `def analyze(image, skin_mask, rois, config, *, context: FeatureContext) -> FeatureResultInternal`
  `context` is **keyword-only**, so an old positional call site fails loudly rather than silently
  binding the wrong argument.
- **`FeatureContext` holds capture and runtime facts ONLY** (D14): `anchor_px`, a `QCVerdict` of
  booleans, capture profile, protocol version, run mode. It must never become a bag for feature
  outputs, calibration values, thresholds, or mutable state — static thresholds stay in `config`,
  reference statistics stay in the calibration layer. `QCVerdict` carries booleans, not QC
  metrics: a measurement that scaled with a QC margin would depend on the room, not the skin.
- **No magic numbers in feature code.** Every threshold comes from `config/*.yaml`. A constant
  inline in a `.py` file is a bug — thresholds must be tunable and auditable without a code change.
- **No fixed pixel parameters either** (D1). Spatial values are fractions of the scale anchor
  (`*_frac_of_iod`, `*_frac_of_iod2`) resolved through `util/scale.py`. A fixed window means
  different things at two resolutions, so cohort statistics computed with one would not transfer
  between devices. `tests/test_repeatability.py` enforces this.
- **Two normalizations, never conflated** (D1). Feature modules compute `z_local` only — the
  within-image score answering *"does this area differ from surrounding skin?"*. That is not a
  severity signal; a uniformly affected face has weak local contrast by construction. Population
  standardization is `decision/standardize.py`, and calling its `robust_z()` from a feature is a bug.
- Feature modules must be pure: no I/O, no network, no global state. Given the same inputs they
  return the same output.
- The vision layer contains **zero** product-recommendation logic. Recommendations live behind
  `rules/adapter.py` only.
- Prefer OpenCV + NumPy + scikit-image (all commercially permissive). Before adding any new
  dependency, add it to the asset manifest with its license.
- Fail closed: if capture QC fails, return `RETAKE` and run no concern logic at all.

---

## 5. Testing expectations

- `test_license_manifest.py` must pass — it is the CI gate for Rule 1.
- **Determinism and repeatability are different properties** (D13). Do not conflate them:

  | Property | Definition | Where |
  |---|---|---|
  | Determinism | Same input array twice -> byte-identical output | CI, every commit |
  | Repeatability | *Distinct captures* of one subject, same session and protocol -> same ordinal severity | Release validation only |

  Repeatability cannot be a CI check because the repo may not hold real face images. It runs as
  `python scripts/run_validation.py --suite repeatability --corpus <local path>` and gates on
  severity agreement, raw-metric CV, ROI availability agreement, and candidate-map stability.
  Repeatability matters more than any single-image accuracy number in this product.
- Report validation sliced by **skin tone, device, lighting, age, makeup/facial hair** — never as
  one aggregate number. A model that works on average and fails on one skin tone is a failed model.
- Use synthetic or SFHQ images in tests. Never commit real face images to the repo.

---

## 6. Things Claude should refuse to do here

- Train on, download, or vendor any EXCLUDED dataset or its derived weights.
- Add a numeric score to user-facing output.
- Add a `Severity.UNCALIBRATED` member, or catch `CalibrationRequiredError` to substitute a
  default severity. Both defeat D2.
- Apply gray-world or any illuminant estimate to the skin channels (D4). Record the illumination
  vector as QC evidence; never use it to modify pixels.
- Apply CLAHE upstream of texture (D5).
- Hardcode a pixel-valued spatial parameter (D1).
- Make a cross-session or trend claim (D12).
- Hand-edit a file under `data/reference/`, or edit a frozen reference set in place (D1). Those
  files are produced by a calibration run and are part of the model; a new protocol gets a new
  versioned directory so old scans stay reproducible.
- Set `config/rois.yaml meta.verified: true` without visual review across poses, hairstyles and
  facial hair (D15), or run production with it false.
- Widen `FeatureContext` beyond capture/runtime facts (D14).
- Enable acne/fine-lines/pores without cleared labels and a passed gate.
- Add hydration as a measured output.
- Present a classical proxy as a clinical or diagnostic measurement. These are **cosmetic
  appearance** estimates. This product does not diagnose.
- Use generative super-resolution to create detail that is then measured. Measurements must trace
  back to real sensor pixels.

---

## 7. Current phase

**Phase 0 → 1**: repository skeleton, licensing manifest, capture QC, MediaPipe landmarks/ROIs,
then the four active classical features. See architecture doc §20 for the full phase gate list.

Stage A (decisions D1–D15, contracts, configs, gates) is complete. Stage B is the face layer, in
this order — note that visual debug comes **before** the skin mask, because a bad polygon can hide
behind a plausible mask:

    B1  face/landmarks.py          verify coordinates + asset loading
    B2  config/rois.yaml           define all 11 polygons
    B3  scripts/debug_roi.py       visually verify -> only then meta.verified: true
    B4  face/skin_mask.py          exclude eyes/brows/lips/nostrils/hair/background
    B5  rois.compose()             ROI ∩ skin mask; measurable-pixel ratios per ROI
    B6  anchor_px                  validate scale behaviour across face sizes
    B7  QC instrumentation         collect blur/exposure/occlusion metrics
    B8  threshold tuning
    B9  capture/qc.check

`tests/test_face_pipeline.py` is the definition of done for Stage B. Its xfails flip to passes as
B1–B9 land; none may be deleted to make the suite green.
