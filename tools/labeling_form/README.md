# Severity labeling tool

A second small Google Apps Script web app, same pattern as `tools/capture_form/`:
a rater opens a link, enters their name, sees one confirmed-passing photo at a
time, and picks a severity (not_detected / mild / moderate / high / unsure) for
every face region each active concern actually uses. Submitting writes straight
into a `labels` Sheet in the exact shape `scripts/calibrate.py` expects.

Nothing here touches the production pipeline. Standalone tool, like the capture
form.

## What it does and doesn't do

- Reads photos directly from the same `SkinAnalysis-Cohort` Drive folder the
  capture form writes to — no separate upload step.
- Only shows photos from `APPROVED` in `Code.gs`, a **curated list of filenames
  already confirmed to pass the real QC gate** (`scripts/qc_report.py`), not
  "whatever's in the subject's folder." A subject folder can hold several failed
  retake attempts alongside the one that passed — only a local `qc_report.py` run
  can tell which file that is, so this list has to be refreshed by hand (I
  regenerate it, you redeploy) as more subjects clear the gate. **Currently 8
  subjects**, last verified 2026-08-21.
- The region list per concern is hardcoded from `config/severity_thresholds.yaml`'s
  real `primary_rois` (D8) — texture doesn't get crows-feet, wrinkles doesn't get
  cheeks, etc. If that config ever changes, `CONCERN_ROIS` in `Index.html` needs a
  matching update; nothing enforces they stay in sync automatically.
- Each rater identifies themselves once (stored in their browser) and gets served
  whichever approved subjects *they personally* haven't labeled yet — two different
  people opening the same link build two independent label sets, which is what lets
  inter-rater agreement be checked later (see
  `tools/capture_form/NEXT_PHASE_LABELING.md`, "two independent raters minimum").
- A "Submit" writes the WHOLE region grid for one subject in one call — no partial
  saves. A half-finished subject would otherwise look "labeled" (a row exists) while
  some regions silently never got judged.
- **"Unsure / cannot tell" is a real option**, not forced into the four real bands.
  `calibrate.py`'s `severity_values` doesn't currently accept `unsure` — **filter
  those rows out before feeding the exported CSV to `calibrate.py`**, don't just
  drop the column. (This was flagged as a real schema gap in
  `NEXT_PHASE_LABELING.md` before any labeling started; this tool takes that
  seriously instead of forcing a guess.)

**Not built**: an adjudication flow for when two raters disagree (see
`NEXT_PHASE_LABELING.md` — decide a process before labeling starts, e.g. a third
rater breaks ties). This tool collects independent labels; reconciling them is a
separate step.

## Deploy (same steps as tools/capture_form)

1. [script.google.com](https://script.google.com) → **New project** (a *separate*
   project from the capture form — different app, same Drive account).
2. Paste this folder's `Code.gs` in.
3. Add an HTML file named `Index`, paste this folder's `Index.html` in.
4. **Deploy** → **New deployment** → **Web app** → Execute as **Me**, access
   **Anyone with the link** → **Deploy**, approve permissions (Drive + Sheets, same
   as before).
5. Send the link to each rater separately from the capture-form link.

## Refreshing the approved list

As more subjects pass the capture form's real QC gate, send me the update and I'll
regenerate `APPROVED` in `Code.gs` — paste the new version in, redeploy (**New
version** → **Deploy**), same link keeps working, raters just see more subjects
appear.

## Pulling labels out for calibration

Export the `labels` Sheet (inside `SkinAnalysis-Cohort` in Drive) as CSV. Filter out
any `severity == "unsure"` rows, then it's directly usable as the `--annotations`
input `scripts/calibrate.py` expects (`subject_id, concern, roi, severity`).
