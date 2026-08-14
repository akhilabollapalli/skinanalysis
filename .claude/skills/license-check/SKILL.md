---
name: license-check
description: Verify commercial-use eligibility of a dataset, model weight, or dependency and record it in the asset manifest. Use before adding any external asset to the project, or when auditing the repo before a release.
---

# License Check

Screen an external asset against this project's commercial-open gate and record the decision.

## When to run this

- Before adding any dataset, pretrained weight, checkpoint, or third-party library
- When someone suggests "let's just use X dataset"
- Before any release (audit every manifest row for staleness)

## Procedure

### 1. Identify the authoritative source

Find the original project repo, official dataset page, or the releasing organization's model
card. These are **not** authoritative: Kaggle re-uploads, Roboflow mirrors, third-party
HuggingFace copies, forks, papers-with-code entries.

If you cannot locate an authoritative source, the decision is **EXCLUDED**. Stop here.

### 2. Check all five layers separately

| Layer | Question |
|---|---|
| Dataset images | Can we use the images commercially? |
| Annotations | Labels often carry a different license from the images |
| Source code | Usually the most permissive — and the least relevant |
| Pretrained weights | Do they inherit the training data's restrictions? |
| Underlying source media | e.g. Flickr photos behind a face dataset |

Never collapse these. Apache-2.0 code with non-commercially-trained weights is a **fail**.

### 3. Read the actual license text

Open the LICENSE file, the dataset terms page, or the model card license section. Quote the
operative sentence verbatim with its URL. Do not paraphrase from memory.

### 4. Decide

- **ELIGIBLE** — explicit commercial grant, obligations listed
- **ELIGIBLE WITH CONDITIONS** — permitted, but conditions become engineering tasks
- **EXCLUDED** — research-only, non-commercial, permission-required, or unclear

**Unclear defaults to EXCLUDED.** Do not hedge.

### 5. Record it

Append a row to `LICENSES/asset_manifest.csv`:

```
asset,source_url,version_or_date,license,layer_rights,attribution_required,allowed_purpose,file_hash,decision,reviewed_on
```

If attribution is owed, add the required notice text to `LICENSES/THIRD_PARTY_NOTICES.md`
exactly as the license specifies.

### 6. Verify CI still passes

```bash
pytest tests/test_license_manifest.py -v
```

## Common traps

- Permissive code license masking restricted weights
- Datasets derived from FFHQ/CelebA — upstream terms still bind
- "Free for academic and research purposes" phrased to sound open
- Fine-tuned weights: restrictions propagate downstream
- A model card `license:` field that contradicts its own linked training data

## Standing decisions

**EXCLUDED**: FFHQ-Wrinkle, AcneSCU, ACNE04, CelebAMask-HQ (and BiSeNet parsers trained on it),
PorePatch, ACNE-DET.

**ELIGIBLE**: FairFace (CC BY 4.0, attribution), SCIN (custom license conditions apply),
SFHQ (MIT, infrastructure testing only).

## Note

This is engineering screening, not legal advice. Escalate genuine ambiguity to counsel.
