# Cohort photo capture form

A small Google Apps Script web app: participants open a link, agree to a consent
screen, fill in a few fields, take a photo, and get instant retake feedback if it's
blurry, too dark, or too bright. Photos that pass go straight into your own Google
Drive, with one manifest row per photo already in the shape `scripts/run_validation.py`
and `scripts/calibrate.py` expect.

Nothing here touches the production pipeline or gets deployed with it. It is a
standalone data-collection tool.

## What it checks, and what it can't

This form went through two real-data-driven revisions: build it, run the actual
70-photo cohort through the real server-side gate (`scripts/qc_report.py`), see
what it missed, close what's closeable client-side. Current state:

- **Blur** — Laplacian variance + Tenengrad, measured on the **real detected face
  crop** (native resolution, cropped to the face detector's bounding box), not a
  guess. Went through two revisions, each measured against the real 70-photo
  corpus's actual pass/fail (`scripts/qc_report.py`):
  1. First version used a blind center crop of the whole frame (assumes the face is
     roughly centered) — 80% agreement with the real gate.
  2. Once the face detector existed anyway (for size/roll/yaw, below), it made sense
     to crop blur measurement to the ACTUAL detected face box instead of guessing —
     90% agreement. Re-measured using the real production landmark detector
     (Python, offline, not BlazeFace) as ground truth for where the face box should
     be, then swept thresholds against the real 'blur' pass/fail flag.
  `minLaplacianVarFaceBox` / `minTenengradFaceBox` in `Index.html` are this second,
  more accurate measurement. The original center-crop numbers
  (`minLaplacianVar`/`minTenengrad`) are kept ONLY as a fallback for when face
  detection is unavailable (slow/failed load) — see `analyzeImage`.
- **Exposure** — mean luma range and clipped-pixel fraction, the real
  `capture_thresholds.yaml` values, computed on the same center crop.
- **Colour cast** — mean R/G/B channel deviation from gray, over the *whole frame*.
  This one's an **exact** port of `util/color.py`'s `gray_world_deviation()`, not a
  proxy — the server measures it over the whole frame too, so there's no crop
  mismatch to approximate around. Real `max_gray_world_deviation: 0.18` threshold.
- **Face size + roll + yaw** — a real client-side face detector (MediaPipe BlazeFace
  short-range, loaded from a CDN, same model family the server already uses,
  Apache-2.0/already-cleared, see CLAUDE.md D3) measures the actual face bounding
  box and two keypoint-derived angles:
  - Face too small: real `min_face_height_frac` / `min_face_px` values, checked
    against the actual detected box — a real measurement, not a proxy. Added after
    the first cohort batch showed this was the second-largest failure cause (38/70)
    with nothing client-side checking for it at all.
  - Roll (head tilt): angle between the eye keypoints — real geometry, real 10°
    limit.
  - Yaw (turned away): a standard nose-offset-from-eye-midpoint heuristic, generous
    threshold on purpose. **Unvalidated** against this project's real `yaw_deg` —
    unlike blur, there wasn't a real turned/frontal-labeled sample to measure a
    threshold against yet. Treat it as a rough catch, not a calibrated one.
  **Fails open**: if the model can't load in ~8s (slow connection, unsupported
  browser), all three of these are silently skipped and the form behaves like the
  blur/exposure/colour-only version — a photo is never blocked by a missing model.
- **No face / multiple faces** — same detector, rejects if it finds zero or more
  than one face in frame.

**Not checked at all**: pitch (nodding up/down — a nose-offset heuristic exists for
this too but is much less reliable across different face shapes, so it was left
out rather than risk false rejections), hair/beard/glasses occlusion, and ROI
visibility (which needs real face-mesh landmarks + an approximate skin mask — a much
bigger addition for two failure causes that were rare in the first batch: 3/70 and
4/70 respectively). The on-screen guide oval is the only mitigation for pitch. The
real gate still runs later when these photos go through the actual pipeline — this
form narrows the gap a lot, it isn't a full client-side reimplementation of
`capture/qc.py`.

**Before sending this version to anyone**: test it yourself on your own phone
first. The face-detector addition is the biggest change made to this form so far
and loads an external model over the network — worth confirming it actually works
on a real device before resending the link to everyone.

## Before sending this to anyone

Replace the placeholder consent text in `Index.html` (`consentTextEl.textContent`)
with the reviewed version — a draft was commissioned from the annotation-lead agent
in parallel with building this; check for it before going live.

## Deploy (about 5 minutes, no coding needed)

1. Go to [script.google.com](https://script.google.com) and sign in with the Google
   account whose Drive should receive the photos.
2. **New project**. Name it e.g. `skin-analysis-capture`.
3. Delete the default `Code.gs` contents and paste in this folder's `Code.gs`.
4. Click the **+** next to Files → **HTML** → name it exactly `Index` → paste in this
   folder's `Index.html` (drop the `.html` extension when naming, Apps Script adds it).
5. **Deploy** → **New deployment** → gear icon → type **Web app**.
   - Execute as: **Me**
   - Who has access: **Anyone with the link**
6. **Deploy**. Approve the permission prompts (it needs Drive + Sheets access to save
   photos and log the manifest — that's the point, it's your own Drive).
7. Copy the web app URL. That's the link to send.

Every submission creates/updates a folder `SkinAnalysis-Cohort` in your Drive, with
one subfolder per subject ID and a `manifest` Google Sheet inside it.

## Pulling the data back out for the pipeline

1. Download the `SkinAnalysis-Cohort` folder from Drive (zip download, or `rclone`/
   `gdrive` CLI if you have many photos).
2. Export the `manifest` Sheet as CSV, save it as `manifest.csv` at the root of
   wherever you put the downloaded photos.
3. That directory is now a corpus directory: `python scripts/run_validation.py
   --suite repeatability --corpus <that path>` and `python scripts/calibrate.py
   --corpus <that path>` both read it directly, same as any other corpus.

Real face photos never get committed to this repo (see CLAUDE.md §5) — keep the
downloaded folder under `data/raw/` (gitignored) or outside the repo entirely.

## Updating after deploy

Edit `Code.gs` / `Index.html` in the Apps Script editor (or paste updated versions
from this repo), then **Deploy** → **Manage deployments** → edit → **New version** →
**Deploy**. The same URL keeps working.
