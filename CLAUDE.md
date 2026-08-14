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

### Rule 3 — The user never sees numbers

User-facing output is **ordinal only**:

```
concern name + {not_detected | mild | moderate | high} + affected regions
```

Everything else — z-scores, probabilities, area fractions, lesion counts, densities, GLCM
values, confidence — is **internal**. Log it for validation; never return it in the
app-facing payload. There is no 0–100 score and no 0–5 score in this product.

---

## 3. Repository layout

```
src/skin_analysis/
  capture/qc.py           Capture quality gate. Fails closed -> RETAKE, no analysis runs.
  face/landmarks.py       MediaPipe Face Landmarker wrapper (468/478 pts).
  face/rois.py            Anatomical ROI polygons from landmarks.
  face/skin_mask.py       Skin mask = ROI polygon ∩ skin/occlusion rejection.
  features/redness.py         ACTIVE
  features/pigmentation.py    ACTIVE
  features/texture.py         ACTIVE
  features/wrinkles.py        ACTIVE BASELINE
  features/fine_lines.py      EXPERIMENTAL (flag off)
  features/acne.py            EXPERIMENTAL (flag off)
  features/pores.py           EXPERIMENTAL (flag off)
  decision/severity.py    Raw metric -> ordinal severity via calibrated thresholds.
  rules/adapter.py        concern+severity+regions -> existing recommendation engine.
  pipeline.py             Orchestration. The only public entry point.
  schemas.py              Dataclasses for every boundary. Source of truth for the contract.
config/                   capture_thresholds.yaml, severity_thresholds.yaml, rois.yaml
LICENSES/                 asset_manifest.csv, THIRD_PARTY_NOTICES.md
tests/                    incl. test_license_manifest.py (CI-blocking)
data/                     gitignored. Never commit face images.
```

---

## 4. Coding conventions

- Python 3.10+. Type hints on every public function.
- Feature modules expose exactly one entry point:
  `def analyze(image, skin_mask, rois, config) -> FeatureResult`
- **No magic numbers in feature code.** Every threshold comes from `config/*.yaml`. A constant
  inline in a `.py` file is a bug — thresholds must be tunable and auditable without a code change.
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
- Every active feature needs a **repeatability** test: the same subject captured twice under the
  same conditions must produce the same ordinal severity. Repeatability matters more than any
  single-image accuracy number in this product.
- Report validation sliced by **skin tone, device, lighting, age, makeup/facial hair** — never as
  one aggregate number. A model that works on average and fails on one skin tone is a failed model.
- Use synthetic or SFHQ images in tests. Never commit real face images to the repo.

---

## 6. Things Claude should refuse to do here

- Train on, download, or vendor any EXCLUDED dataset or its derived weights.
- Add a numeric score to user-facing output.
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
