# Phase 2 after collection: severity labelling

Internal planning note (drafted by the annotation-lead agent), not shown to
participants. Photos alone are not a calibration cohort — this is what has to
happen after collection, before `scripts/calibrate.py` can produce anything.

## What has to happen next

`scripts/calibrate.py` cannot fit a threshold without labels, and
`config/calibration_gates.yaml` already fixes their shape: one row per
`(subject_id, concern, roi)` with severity in `{not_detected, mild, moderate, high}`.
One label per **subject** per ROI, not per photo — five captures of one person
produce one label, and a second row for the same key means a rater disagreement,
not extra evidence.

The subject_id recorded by the capture form must survive into labelling — if that
link breaks, labels can't be joined back to the corpus.

## Who labels, and how

- **Two independent raters minimum per concern.** A single rater's opinion is not
  ground truth.
- **A written adjudication path before labelling starts.** Two raters produce ties;
  decide now whether a third rater breaks them or the two raters reach a recorded
  consensus (keeping the original disagreement).
- **Write the rubric before the first label.** Per concern: definition at this
  capture scale, what counts/doesn't (shadow, hair, makeup, glasses, specular
  highlight), zoom instructions, a reference image per ordinal level. Also the
  known confusables: freckles vs. dark spots, moles vs. post-inflammatory
  hyperpigmentation, beard shadow vs. texture.
- **Raters need an "unsure/unmeasurable" option** — a real schema gap today, since
  `severity_values` only has the four bands. Forcing a call on an ambiguous region
  injects noise that later reads as model error. Add the value before labelling;
  unsure rows get excluded from fitting, never silently binned.
- **Human labels first, then a preliminary fit, then model-assisted refinement with
  human sign-off on every changed label.** The model's own output must never become
  the truth it's scored against.

## Why inter-rater consistency matters

The `exact_agreement: 0.70` / `adjacent_agreement: 0.90` gates in
`calibration_gates.yaml` mean nothing if two humans only agree with each other 0.65
of the time. Rater disagreement that skews by skin tone (common for redness/
pigmentation on darker skin) becomes model bias, and later shows up as a
`max_slice_spread` failure whose real cause is in the labels, not the code.

Report inter-rater agreement **before** any model results — weighted kappa per
concern/ROI, sliced by self-reported skin tone, never one aggregate number. A
concern with poor rater agreement isn't ready to calibrate regardless of fit score.

## Recruitment consequences to plan around now

- `min_labelled_per_band: 8` per band, per (concern, ROI), on each side of a 0.30
  holdout split — roughly ~50+ labelled subjects with real severity spread per
  concern, more for any rare band (high wrinkles, high redness).
- Skin-tone spread is a **recruitment target set in advance**, not a later fix — too
  few subjects on a tone slice makes `max_slice_spread` silently unfireable.
- Device and lighting diversity likewise; one phone in one room won't transfer.
- Decide the target number of repeat captures per subject (feeds repeatability)
  before collection, not after.

## Consent scope

The current consent text covers one session of photos from adult volunteers. Paid
participants, a follow-up session invite, or any other change of use falls outside
what it currently says and needs the text revised and re-agreed.

## Needs an actual lawyer, not a draft

1. Face photos are plausibly biometric/special-category data (GDPR Art. 9, Illinois
   BIPA, Texas CUBI, India's DPDP Act depending on where participants are) — these
   can require specific consent wording, notice, retention schedule, and deletion
   deadlines beyond a plain-language checkbox. BIPA carries a private right of
   action and statutory damages.
2. The 18+ gate is currently just a checkbox attestation — fine as a first filter,
   not a guarantee. Don't extend to minors without counsel; the consent model
   changes entirely (parental consent).
3. "Delete on request" needs an actual defined deadline, and Drive's trash/version
   history affects what "deleted" really means operationally.
4. Paid participants shift this toward a contractual relationship (tax, releases,
   possibly different consent standards) — get review before offering money.
5. Whether Google Drive on a personal/Workspace account meets the security bar these
   regimes imply for biometric-adjacent data.
6. "Not sold, not shared with unrelated third parties" is a real commitment — it
   constrains a future acquisition, investor data room, or an outsourced annotation
   vendor (annotators ARE a third party).
