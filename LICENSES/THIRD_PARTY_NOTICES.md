# Third-Party Notices

This file packages the attribution required by assets used in this product. It must be
current before any release (architecture doc §20, phase 8).

Every entry here corresponds to a row in `asset_manifest.csv` with
`attribution_required = yes`.

---

## Software

### MediaPipe — Apache License 2.0
Copyright 2019–present Google LLC. Licensed under the Apache License, Version 2.0.
A copy of the license must ship with any distribution, along with a NOTICE of
modifications if the source is modified.

### OpenCV (4.5+) — Apache License 2.0
Copyright OpenCV team. Licensed under the Apache License, Version 2.0.
Note: releases before 4.5 are 3-clause BSD. Confirm the pinned version before shipping,
and do not link non-free contrib modules.

### NumPy — BSD 3-Clause
Copyright NumPy Developers. Redistribution requires the copyright notice, this list of
conditions, and the disclaimer.

### scikit-image — BSD 3-Clause
Copyright the scikit-image contributors.

### PyYAML — MIT
Copyright Ingy döt Net and contributors.

---

## Datasets

### FairFace — CC BY 4.0
Karkkainen, K. and Joo, J. "FairFace: Face Attribute Dataset for Balanced Race, Gender,
and Age." Attribution required in any product or publication that uses the data or
derivatives, including our re-annotations. Preserve the attribution metadata alongside
any derived labels.

### SCIN — SCIN Data Use License
Skin Condition Image Network (SCIN), Google Research and Stanford Medicine.
Source: https://github.com/google-research-datasets/scin — license text at
https://github.com/google-research-datasets/scin/blob/main/LICENSE
Cite: Ward A, Li J, Wang J, et al. "Creating an Empirical Dermatology Dataset Through
Crowdsourcing With Web Search Advertisements." *JAMA Netw Open.* 2024;7(11):e2446615.
doi:10.1001/jamanetworkopen.2024.46615

The SCIN Data Use License contains **no non-commercial restriction**, and its Section 1(f)
covers "the images, labeling and other data" under one grant — so images and annotations are
cleared together. Binding conditions on this project:

- **No re-identification (Section 3(b)).** "You may not make any attempt to re-identify or
  re-link any of the individual data subjects whose data has been de-identified." Breach
  "results in the immediate termination of all rights granted."
- **Attribution (Section 3(a)(1)), triggered only if we Share** the material or Adapted
  Material: retain creator identification, a copyright notice, a notice referring to the SCIN
  License, a warranty-disclaimer notice, and a URI to the material; indicate modifications.
- **No downstream restrictions (Section 2(a)(5)(b))** on the Licensed or Adapted Material.
- **Not licensed (Section 2(b)):** publicity, privacy and other personality rights, and patent
  and trademark rights.

Scope in this product: dermatology-domain robustness and skin-tone coverage checks only.
SCIN is **not** a calibration cohort and **not** severity ground truth — it carries no severity
labels and no facial-ROI granularity. See the manifest row for the full rationale.

### SFHQ — MIT
Synthetic Faces High Quality dataset. Used for infrastructure and QC stress testing
only, never as skin-concern ground truth.

---

## Excluded assets

The following are **not** used in production and therefore require no notice. They are
listed so their exclusion stays visible during audits:
FFHQ-Wrinkle, AcneSCU, ACNE04, ACNE-DET, CelebAMask-HQ (and any weights trained on it,
including common BiSeNet face parsers), PorePatch.
