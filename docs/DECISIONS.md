# Architecture Decisions

Binding decisions (D1-D15) that resolve ambiguities the architecture document
(`docs/RGB_Selfie_Skin_Analysis_Research_Architecture.docx`) left open. Each is LOCKED: changing
one requires a documented revision here plus the migration it implies, not a quiet edit in code.

CLAUDE.md remains the operating rules. This file records *why* the code looks the way it does
where the document offered a choice.

Decided 2026-08-14.

---

## D1 — Reference distribution is HYBRID, and the two normalizations are named differently

Calling both stages `z` is what made §13 ambiguous. They are separate quantities with separate
references and separate homes in the codebase.

### A. `z_local` — within-image, feature detection

    z_local(x) = (f(x) - median(f(N_x))) / (1.4826 * MAD(f(N_x)) + eps)

`N_x` is the local skin neighborhood around pixel/patch `x`, inside the same ROI. Answers *"does
this area differ from surrounding skin?"* Used for redness candidates, pigmentation candidates,
local texture deviation, local morphology contrast.

**This is not a severity score.** A uniformly red face has weak local contrast by construction.

Lives in `features/*.py`.

### B. `z_ref` — population, severity standardization

Feature modules first aggregate candidates into raw measurements:

| Concern | Raw measurements |
|---|---|
| Redness | `affected_area_ratio`, `median_positive_delta_a`, `p90_delta_a` |
| Pigmentation | `spot_area_ratio`, `median_delta_L`, `spot_count` |
| Wrinkles | `line_density`, `line_length`, `ridge_contrast` |
| Texture | GLCM contrast/homogeneity/energy/entropy, gradient energy, local variance, HF ratio |

Then:

    z_ref = (m - Median_ref[ROI]) / (1.4826 * MAD_ref[ROI] + eps)

`ref` means exactly one thing: **a frozen, commercially cleared calibration cohort captured under
the defined V1 acquisition protocol.**

`ref` explicitly does **not** mean "people demographically like this user", and it does **not**
define what healthy or normal skin is. It is feature standardization ahead of the ordinal
decision, nothing more.

    raw metrics -> reference-standardized features -> ordinal calibrator -> NOT_DETECTED|MILD|MODERATE|HIGH

### Code structure consequence

    features/redness.py        z_local only
    features/pigmentation.py   z_local only
    decision/standardize.py    z_ref  (population only)
    decision/calibrator.py     ordinal decision
    decision/severity.py       per-concern aggregation policy (D6)

`decision/standardize.robust_z()` is **population-only**. A feature module calling it is a bug.

(The package stays `decision/` rather than `severity/` to match the CLAUDE.md §3 layout; the file
split is as specified above.)

### Reference artifacts are source, not output

They are effectively part of the model: they sit directly between a raw measurement and the
MILD/MODERATE/HIGH a user is shown. Two checkouts of identical code that disagree on these files
return different results, so they are version-controlled, diffable, and reviewed like code.

    data/reference/
      v1/
        manifest.json
        redness.json
        pigmentation.json
        texture.json
        wrinkles.json

Each concern file, with `median` / `mad` / `quantiles` keyed ROI -> measurement:

```json
{
  "schema_version": "1",
  "protocol_version": "v1",
  "capture_profile": "default",
  "n_subjects": 0,
  "median": {},
  "mad": {},
  "quantiles": {},
  "source_manifest_hash": null,
  "calibrated": false
}
```

Three rules:

- **Produced by a calibration run, never hand-edited.** Editing one silently rewrites what
  "moderate" means for every user.
- **Immutable once frozen.** A production reference is never edited in place. A new protocol gets
  `data/reference/v2/`, so old scans and releases stay reproducible.
- **Minimum cohort size** (`min_cohort_subjects`, currently 200). Statistics from two or three
  people produce a median and MAD that look entirely normal and mean nothing. Below the floor the
  set does not count as calibrated and D2 keeps refusing.

Images stay ignored everywhere under `data/` regardless of this exception.

### Scale-relative windows, and what `anchor_px` means

Fixed pixel windows are retired. `window_px: 129` does not mean the same thing at two
resolutions, so cohort statistics computed with one would not transfer between devices.

> **`anchor_px`** = the canonical facial scale measured from landmark geometry in the current
> image (inter-ocular distance), used to convert scale-relative window and kernel sizes into
> pixels.

Features resolve parameters through `util/scale.py`:

```python
window_px = scale.to_px(config["local"]["window_frac_of_iod"], context.anchor_px, odd=True)
```

Because it is derived per capture, it cannot live in static YAML - which is what forced the
`FeatureContext` parameter of D14.

## D2 — No public payload until calibrated

While calibration is incomplete, the pipeline produces internal results only:

```json
{
  "calibration_state": "uncalibrated",
  "redness": {"raw": {"affected_area_ratio": 0.081, "median_delta_a": 3.7}}
}
```

`analyze_scan_internal(image)` always works. `ScanResultInternal.to_public()` raises
`CalibrationRequiredError` when any concern being published has not passed calibration:

```python
def to_public(self) -> PublicScanResult:
    calibration_registry.assert_public_ready(protocol_version=self.protocol_version)
    ...
```

There is **no** `Severity.UNCALIBRATED`. Adding one creates a value UI code can eventually render.
Uncalibrated is not a finding; it is a state in which the product has no output.

CI enforces it: `test_uncalibrated_pipeline_cannot_create_public_payload`.

## D3 — MediaPipe Face Landmarker cleared (Apache-2.0)

The Face Landmarker bundle's three models — **BlazeFace short-range**, **FaceMesh V2**,
**Blendshape V2** — are each Apache-2.0 per Google's official model cards, as is the MediaPipe
repository. The `face/` layer is unblocked; Stage A/B does not wait on an audit.

Manifest consequence: `weights=see-model-card` is not a record. Each model gets its own row with
`asset, source_url, model_card_url, license, license_version, sha256, download_date, version,
commercial_allowed`.

Runtime consequence: skin analysis needs landmarks only.

```python
output_face_blendshapes=False
output_facial_transformation_matrixes=False
```

FaceMesh V2 supplies the 478 3D landmarks the ROI layer requires.

## D4 — No gray-world. Fixed sRGB -> D65 Lab + local normalization

    camera-rendered sRGB
      -> reject severe illumination / color-cast captures at QC
      -> sRGB -> Lab (D65, OpenCV standard conversion)
      -> local delta-a*, local delta-L*
      -> candidate maps

**No gray-world, no illuminant estimation, no post-hoc white-balance fit.**

Rationale: gray-world pushes the average of the selected pixels toward neutral, and the selected
pixels here are deliberately *skin* — not a neutral target. Forcing average skin toward gray
partially removes the red/yellow chromatic information the product exists to measure. Smartphone
JPEG/HEIC has additionally already passed through camera white balance and nonlinear rendering,
so rigorous re-white-balancing after capture is harder than working from RAW and would itself
vary per capture — which is fatal to repeatability, this product's primary quality metric.

Store an illumination / color-cast **QC vector**. Never modify the skin channels with it. Color
constancy may return later as a separately validated preprocessing step.

### Terminology

The term **"color-calibrated copy" is retired.** Without a calibration chart or RAW reference,
"calibrated" overclaims. It is the **standardized color-analysis copy**.

Consequence: `capture_thresholds.yaml white_balance.max_gray_world_deviation` is now
**load-bearing** — the only defense against color casts, since nothing downstream corrects them.
Tune it before loosening any redness or pigmentation threshold. Absolute a*/L* are not comparable
across devices; only local excess/deficit is.

## D5 — No CLAHE before texture. Three image derivatives, not two

    original decoded RGB
      |
      +-- COLOR COPY     no CLAHE; sRGB -> Lab/D65        (redness, pigmentation)
      +-- TEXTURE COPY   luminance; fixed normalization only; NO CLAHE   (texture)
      +-- RIDGE COPY     optional validated enhancement; Gabor / Hessian (wrinkles)

GLCM must run on a stable intensity transform. CLAHE is local adaptive gain, so GLCM downstream of
it partly measures CLAHE's response to the neighborhood instead of the skin — a measurement of
the preprocessing.

    texture:
      clahe: false      # V1

Wrinkles may test CLAHE later as an ablation. If it is ever adopted, its clip limit and tile grid
join the frozen capture protocol and invalidate every existing calibration.

## D6 — MAX-WITH-SUPPORT with concern-specific strong-support escape

A rigid "must appear in two ROIs" would suppress genuinely localized pigmentation and acne. The
rule is highest severity plus *evidence* support, where support has two independent routes:

```python
if n_rois_at_max >= 2:
    final = max_severity
elif strong_single_roi_support:      # concern-specific threshold
    final = max_severity
else:
    final = downgrade_one_band(max_severity)
```

Strong single-ROI support, per concern:

| Concern | Strong-support evidence |
|---|---|
| Redness | red-area fraction + delta-a* magnitude |
| Pigmentation | spot area + spot contrast |
| Wrinkles | line density + line length |
| Texture | (diffuse by nature — no single-ROI escape) |
| Acne (future) | lesion burden / affected area |

    regions = all measurable ROIs whose severity >= min_reportable

independent of which ROI set produced the concern level.

This is a **concern-specific policy**, not one universal function. It lives in
`decision/severity.py` keyed by concern, with thresholds in config.

## D7 — Partial ROI reporting in V1

Report from the measurable primary ROIs. Return `UNMEASURABLE` for a concern only when **none** of
that concern's required primary ROIs is usable.

    forehead:    UNMEASURABLE
    left_cheek:  MODERATE
    right_cheek: MODERATE

    -> redness: MODERATE, regions [left_cheek, right_cheek]

Internally preserve `"unmeasurable_regions": ["forehead"]`. Surface it only where it materially
affects interpretation.

The comparability objection is real and is the reason D12 exists. If longitudinal is ever added,
this decision tightens to require compatible ROI sets.

## D8 — Nasolabial is wrinkle-only

| Use | Nasolabial ROI |
|---|---|
| Wrinkles | allowed |
| Pigmentation | excluded |
| Redness severity | excluded |
| Texture primary scoring | excluded |
| Any asymmetry measurement | excluded |

The ROI stays geometrically defined for future research. The structural detector may use it
because the fold *is* a morphology target. For color and texture metrics the fold's cast shadow
is too ambiguous against genuine chromatic change at V1's fidelity.

## D9 — The vision package terminates at `PublicScanResult`

```python
@dataclass(frozen=True)
class PublicConcern:
    concern: Concern
    severity: Severity
    regions: tuple[ROI, ...]

@dataclass(frozen=True)
class PublicScanResult:
    schema_version: str
    capture_quality: dict[str, Any]
    concerns: tuple[PublicConcern, ...]

@runtime_checkable
class RecommendationEngine(Protocol):
    def recommend(self, result: PublicScanResult) -> RecommendationResult: ...
```

    RAW INTERNAL RESULT
            x   never crosses this line
            |
            v
    PublicScanResult          concern + severity + regions
            |
            v
    RecommendationEngine

This repository knows nothing about how recommendations are implemented. A conforming engine may
later be a local rule engine, a REST service, an existing backend, or a database-driven ruleset,
and none of that changes the computer-vision system.

The enforcement is structural: `recommend` accepts `PublicScanResult` and there is no overload
taking `ScanResultInternal`. The recommendation layer cannot be tuned against a measurement it
was never handed, because no field exists to reach through.

**Destination: still open.** Q9 stays unresolved as "target implementation TBD" until Stage C5.
Only the interface is frozen, and freezing it now is what lets Stage B proceed without it.

## D10 — Server-side V1

    PHONE   capture -> basic capture QC -> high-resolution upload
    SERVER  decode -> full QC -> MediaPipe -> ROI -> CIELAB -> GLCM -> Gabor/Hessian
            -> severity -> public payload
    PHONE   recommendation UI

Budget after upload reaches the server: **P50 < 1.5 s, P95 < 3.0 s**. End-to-end UX target
including network: **< ~5 s** on normal connectivity.

Do not sacrifice native-resolution morphology to chase sub-500 ms in V1.

Consequences:
- Vectorize across patches; share one Lab conversion and one gradient pyramid across features
  rather than recomputing per module. Purity (CLAUDE.md §4) still holds — shared intermediates are
  passed in, never cached in module globals.
- Design so **QC + landmarks can later migrate on-device without changing any concern API.**
- Latency is a release gate, reported with the validation slices.

## D11 — Canonical QC face crop, then device-profile schema with `default` only

Remove the resolution dependence before parameterizing it:

    full image -> canonical QC face crop (fixed width) -> Laplacian variance

Now the blur metric is comparable across devices instead of needing a per-device number.

```yaml
capture_profiles:
  default:
    qc_face_width_px: 768
    min_laplacian_var: ...
    exposure: ...
  overrides: {}   # added only after validation proves a systematic device bias
```

Start with the `default` profile plus **device metadata logging**. Do not invent a cohort of
hard-coded iPhone/Samsung profiles — the architecture document does not define a phone cohort
firmly enough to justify one. Target cohort is an explicit validation TODO.

## D12 — Longitudinal comparison is OUT of V1

V1 non-goals, mirrored into CLAUDE.md:

- No claim that a concern improved or worsened across scans.
- No longitudinal trend reporting.
- No comparison of severity across capture sessions.
- No treatment-response claims.

V1 answers exactly one question: **what concerns are visibly detectable in this scan?**

This is what keeps D1 tractable and lets D7 report from partial ROI sets. A future longitudinal V2
requires same capture protocol, compatible device/color pipeline, matched ROI set, repeatability
validation, session normalization — and probably its own calibration study. It reopens D1, D4,
and D7 together.

## D13 — Determinism in CI, true repeatability at release

| Property | Definition | Where |
|---|---|---|
| Determinism | Same input array twice -> byte-identical output | CI, every commit, `test_pipeline_is_deterministic_for_identical_input` |
| Repeatability | Distinct captures of one subject, same session and protocol -> same ordinal severity | Release validation, local gitignored corpus |

```bash
python scripts/run_validation.py --suite repeatability --corpus /local/path
```

Release gate evaluates: severity agreement, raw-metric coefficient of variation, ROI availability
agreement, candidate-map stability.

The repo cannot hold real face images (CLAUDE.md §5), so true repeatability cannot be a CI check.
Naming the two separately stops the cheap test from being mistaken for the meaningful one.


---

## D14 — `FeatureContext` is capture/runtime facts only

Feature modules take the context keyword-only:

```python
def analyze(image, skin_mask, rois, config, *, context: FeatureContext) -> FeatureResultInternal
```

Keyword-only so an old positional call site fails loudly instead of silently binding the wrong
argument.

```python
@dataclass(frozen=True)
class FeatureContext:
    anchor_px: float          # capture-derived scale; see D1
    qc: QCVerdict             # booleans only, never QC metrics
    capture_profile: str
    protocol_version: str
    run_mode: RunMode
```

**Locked contract: capture and runtime facts ONLY.** This must not become a generic bag for
feature outputs, calibration values, thresholds, or mutable state. Static thresholds stay in
`config`; reference statistics stay in the calibration layer. The moment anything else is added,
two modules can start disagreeing about it.

`QCVerdict` carries booleans, not metrics, deliberately. A feature needs to know *whether* the
capture cleared a check, never by how much - a measurement that scaled with a QC margin would
depend on the room rather than the skin. `shadow_pass` and `color_cast_pass` are there because
V1 corrects neither (D4).

`image_width` / `image_height` may be added if a feature genuinely needs them, and only then.

## D15 — Unverified ROI polygons are refused in production

```python
if run_mode is RunMode.PRODUCTION and not config["meta"]["verified"]:
    raise UnverifiedROIError(...)
```

Development mode is allowed through, because defining and inspecting the polygons is the workflow
that produces the verification in the first place.

One forgotten polygon silently moves every measurement taken inside it, and nothing downstream
would look wrong. `config/rois.yaml meta.verified` flips to true only after review with
`scripts/debug_roi.py` across poses, hairstyles and facial hair.

### Stage B order, and why debug comes third

    B1  face/landmarks.py          verify coordinates + asset loading
    B2  config/rois.yaml           define all 11 polygons
    B3  scripts/debug_roi.py       visually verify -> only then meta.verified: true
    B4  face/skin_mask.py          exclude eyes/brows/lips/nostrils/hair/background
    B5  rois.compose()             ROI INTERSECT skin mask; measurable-pixel ratios per ROI
    B6  anchor_px                  validate scale behaviour across face sizes
    B7  QC instrumentation         collect blur/exposure/occlusion metrics
    B8  threshold tuning
    B9  capture/qc.check

Debug runs **immediately after the polygons are defined, before the skin mask is relied on.**
Three things must be judged independently:

1. Is the anatomical polygon correct?
2. Is the semantic skin mask correct?
3. Is their intersection correct?

Checking only the intersection is a trap - a bad polygon can hide behind a plausible skin mask
and stay hidden until a feature produces a finding nobody can explain. Hence the separable
overlay modes: `landmarks`, `polygons`, `skin_mask`, `intersection`, `measurable`.

`tests/test_face_pipeline.py` is the definition of done for this stage. Its xfails flip to passes
as B1-B9 land; none may be deleted to make the suite green.
