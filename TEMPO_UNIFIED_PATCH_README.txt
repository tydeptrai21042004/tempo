# TEMPO experimental protocol revision (2026-10-23)

**Upload this ZIP into the existing GitHub repository root, preserving paths.** The patch contains only new/modified files, not the entire repository or datasets. The uploaded `tempo-main.zip` at commit `10e6662c52f61834cca3714c6390be8f42f7f644` is the patch base. If GitHub has moved on, check conflicts before upload.

## What changed

- Adds a **versioned, declarative unified attack protocol** (`configs/attacks/unified_109.json`): 1 clean + 109 non-clean, starting with the original 1+74 conditions, followed by 35 named, parameterized severity/off-grid scenarios. `stress` and `extended` retain their historical 74 non-clean conditions. `unified` is opt-in via `default_attacks('unified')` and now the default for full Kaggle image/ablation/matched-PSNR comparisons.
- Noise variance cases use normalized [0,1] variance, distinct from legacy pixel-unit sigma. JPEG cases encode actual quality and subsampling choices. Supplemental JPEG40 now calls the same JPEG implementation as the primary suite. Perspective attacks fail rather than silently substituting shear when OpenCV is absent.
- Matched-PSNR runner includes CA-QIM as a disclosed secondary/semi-blind comparison. CA-QIM and informed Cox are excluded from the **blind-only primary matched-PSNR rank** by static fidelity eligibility (both remain benchmarkable and explicitly labeled). DEW remains capacity-limited and excluded from the fixed 4096-bit image primary table.
- Adds preflight audit `scripts/audit_common_protocol.py`: checks shapes/types, SHA256, static baseline disclosures and bitwise equivalence for the first 74 legacy attacks on a fixed RGB sample. This is **not a proof of original-paper implementation fidelity**.
- External baseline importer now validates actual trial labels against the unified attack IDs, requires full commit and source-result hashes, and reports missing coverage. Imports evidence only; it cannot verify that an external codebase faithfully reproduced a publication.
- New external MMWatermark adapter runs verified upstream perturbation functions and writes a **separate provider-prefixed set**, not deceptively relabeled as local attacks. Requires upstream dependency installations; unsupported functions fail closed. No pretrained neural model results are fabricated.
- Updates the repository source manifest so the built-in verification step still passes.

## Running

    python scripts/audit_common_protocol.py --attack-preset unified --output /tmp/tempo_attack_audit.json
    bash scripts/kaggle_auto_video.sh full

For historic comparison, set `IMAGE_ATTACK_PRESET=stress`. Development hyperparameter selection continues using the existing `standard` subset, never test attacks. To run the separate official external attack evaluation, see `scripts/run_official_mmwatermark_tempo.py --help`. Alternatively enable `TEMPO_OFFICIAL_MMWATERMARK=1` in the full Kaggle pipeline after installing upstream optional packages. Configure `TEMPO_OFFICIAL_METHODS`, `TEMPO_MMWATERMARK_SOURCE_COMMIT` and `TEMPO_MMWATERMARK_COMPANION` as needed. This optional stage fails closed rather than reporting local substitute attacks.

## Evidence limitations

- **No new full-host × full-attack benchmark numbers are claimed in this patch.** Kaggle must run the actual experiments.
- Existing baselines are paper-inspired/adapted realizations and require source-level verification against the original publications before claims of mathematically exact reproduction. Six legacy baseline test modules were evaluated locally; passing those tests is not independent original-paper verification.
- A valid full 4096-bit comparator must support the fixed payload and disclose detector information. Native-capacity baselines such as DEW and native TrustMark belong in separate strata.
- `unified` is a new experiment version. Historical 74-attack averages cannot be compared directly to 109-attack averages.
- Reviewer-specific targeted/carrier experiments remain named **supplementary protocols**, not fabricated entries inside the common 109 attack set. Real video compression and realtime scheduling likewise remain distinct from static-image transformations.
- External official MMWatermark distortions use provider-qualified IDs and a separate source SHA. They are not claimed to be the exact same transformations as built-in TEMPO attacks.
