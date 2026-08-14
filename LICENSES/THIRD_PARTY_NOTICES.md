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
Google Research SCIN dataset. Used under its specific conditions, which include
attribution and a no-reidentification obligation binding on this project.

### SFHQ — MIT
Synthetic Faces High Quality dataset. Used for infrastructure and QC stress testing
only, never as skin-concern ground truth.

---

## Excluded assets

The following are **not** used in production and therefore require no notice. They are
listed so their exclusion stays visible during audits:
FFHQ-Wrinkle, AcneSCU, ACNE04, ACNE-DET, CelebAMask-HQ (and any weights trained on it,
including common BiSeNet face parsers), PorePatch.
