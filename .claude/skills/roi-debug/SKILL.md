---
name: roi-debug
description: Visualize and debug the face pipeline's landmarks, ROI polygons, and skin mask over an input image. Use when a feature produces false positives, a region looks mis-segmented, or after changing ROI definitions.
---

# ROI & Skin Mask Debugging

Most false positives in this pipeline trace back to the mask, not the feature math. Before
tuning a feature's thresholds, verify what pixels it was actually given.

## Why this matters here

V1 has **no learned face parser** — CelebAMask-HQ is excluded by the licensing gate, which
excludes the BiSeNet weights trained on it. The skin mask is built from MediaPipe landmark
polygons plus classical hair/occlusion rejection, so it is weaker at boundaries than a learned
parser would be. Hair strands, beard, lip edges, nostril shadow, and glasses frames are the
usual culprits.

## Procedure

### 1. Render the overlay

```bash
python scripts/debug_roi.py --image <path> --out debug_overlay.png
```

Produce a panel showing: landmarks, each ROI polygon, the final skin mask, and the excluded
pixels in a distinct color. Look at the **excluded** layer as carefully as the included one.

### 2. Work through the checklist

- [ ] Are eyes, brows, lips, nostrils fully excluded?
- [ ] Is hair (including flyaway strands over the forehead) excluded?
- [ ] Is beard/stubble excluded, or is it feeding the wrinkle/texture branch?
- [ ] Are glasses frames and their shadows excluded?
- [ ] Do specular highlights get rejected rather than read as pale skin?
- [ ] Is deep shadow (under nose, under jaw) marked unmeasurable?
- [ ] Do left/right ROIs cover symmetric areas? Asymmetric coverage fakes an asymmetry finding.
- [ ] Does each ROI still have enough pixels after masking to be meaningful?

### 3. Check per-feature consequences

| Symptom | Likely mask cause |
|---|---|
| Redness false positive on cheeks | lip/nostril bleed, or warm-light shadow |
| Wrinkle false positive on forehead | hair strands read as ridges |
| Texture false positive on chin/jaw | beard or stubble |
| Pigmentation false positive | shadow not rejected as darkness |
| Asymmetry finding with no visible cause | uneven ROI coverage or side-lighting |

### 4. Fix in the right place

- Wrong polygon shape → `config/rois.yaml`
- Wrong exclusion logic → `src/skin_analysis/face/skin_mask.py`
- Bad landmarks entirely → this is a capture QC failure; tighten pose/blur gates rather than
  patching downstream

## Rule of thumb

If a region is ambiguous, exclude it. Losing measurable area is cheap; a confident false finding
shown to a user is not.
