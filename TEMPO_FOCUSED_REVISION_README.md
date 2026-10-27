# TEMPO focused revision — source-only overlay

This package contains **changed/new code only**, based on the uploaded `tempo-main (2).zip`. It does **not** include the entire repository, external benchmark datasets, or fabricated experimental output.

## Apply to your TEMPO repository

Extract the ZIP into the **root of your existing TEMPO repository**, preserving the `scripts/` and `tests/` paths. Files with the same names must be overwritten. Commit the applied patch to GitHub before running a Kaggle notebook that clones GitHub `main`. The old measurement commit remains `db7eeba3b9fac289fbeac27c37463f04bf7f06e6`; use a **separate commit hash for this postprocessing revision**.

### Repair the previous evidence without repeating full benchmarks

From the repository root (Linux/Kaggle):

```bash
export PYTHONPATH="$PWD/src:$PWD"
python scripts/repair_existing_evidence.py \
  --evidence-zip '/path/to/TEMPO_FULL_EVIDENCE_20261027T070005Z.zip' \
  --out-dir '/path/to/TEMPO_REPAIRED_EVIDENCE' \
  --source-data data/user_dataset
```

The input evidence ZIP can be attached as a Kaggle Dataset. **Use an empty output directory.** The repair copies archived reports into that directory; it does not overwrite the original ZIP. The existing recorded dataset files in `data/user_dataset/` must be unchanged. Replay will fail if recovered NC or embedding PSNR differs from the archived matched-case measurements.

Outputs include `reports/result_figure_sources/baboon/gallery_manifest.json`, both Baboon extracted-logo PDFs, the attacked-host PDF, the verified `reports/video_common_baselines/` files, all original manuscript PDF figures and `reports/reviewer_completeness.json`. The 2024 optional adapter is not re-run.

### Run a fresh full Kaggle pipeline

The existing notebook/launcher can continue to clone GitHub and run `bash scripts/kaggle_pipeline.sh full` after the patch is committed. The full pipeline now generates the Baboon replay from the normal matched-PSNR experiment and the common-source video evidence export without enabling the independent Haar/SVD baseline. Extra non-result manuscript figures are included in full mode. `TEMPO_RUN_INDEPENDENT_HAAR_QIM_CONTROL=0` remains the default.

## Critical scientific safeguards

- **External Kodak/MMWatermark remains TEMPO-proposal ONLY.** The bridge always passes `--methods proposal`, and the official direct runner now rejects requests for other methods. The existing 5,328 compatible Kodak trials are not rerun by evidence repair. The five upstream Frost conditions remain excluded unless genuinely fixed, and must be disclosed.
- **All internal 45 non-clean attack conditions + clean remain untouched.** Frozen embedding algorithm, watermark size, attack registry and main numerical benchmark are unchanged.
- The same-source video comparison uses all **five actually measured** methods: four blind methods (TEMPO, DM-QIM, STDM-QIM, ISS) and the **semi-blind** CA-QIM-A2. No method has its decoder altered or relabeled. Cox 1997 remains an informed/non-blind image baseline; it is **not fabricated as a measured video result**.
- Video results use the same original source video hashes, prepared clip hash, codec, payload identities, per-frame sample counts, and method-specific **unmatched PSNR**. The video export is an **audit of existing trials**, not an independent experiment or a matched-distortion ranking.
- The independently implemented 2024-inspired Haar-HD-SVD-QIM adapter is **opt-in** for both image and video. When enabled, its diagnostic compares actual decoded NC with a *hypothetical complement NC* and PSNR; it does **not** invert the output to improve reported results or claim to reproduce the published 2024 algorithm.
- Missing evidence is recorded; **no score-based scientific gate** is added to abort the main experiments. `execution_evidence_complete=true` and `scientifically_complete_for_resubmission=true` mean different things and should not be conflated.

## Modified source files

- `scripts/kaggle_pipeline.sh`: decouple Baboon replay, add video export, gate optional adapter, include final PDFs.
- `scripts/export_result_gallery.py`: match replay to existing five-method CSV, validate NC and PSNR, preserve semi-blind classification and eligibility.
- `scripts/generate_manuscript_figures.py`: label actual five methods, detector types, per-method NC and PSNR in Baboon PDFs.
- `scripts/reviewer_completeness.py`: verify both watermark galleries and source-backed multiclass video evidence, not only a CSV path.
- `scripts/run_multiwatermark_video.py`: make detector classification explicit, never silently label an unknown method as blind.
- `scripts/run_official_mmwatermark_tempo.py`: fail closed if someone attempts an external comparator other than TEMPO.

## New source files

- `scripts/export_common_video_baselines.py`: verify measured rows, original/prepared hashes, codec, payloads, per-frame data and detector classes; export signed derived comparison files.
- `scripts/diagnose_haar_adapter_video.py`: optional diagnostics only.
- `scripts/repair_existing_evidence.py`: repair old evidence archives without rerunning numerical benchmarks.
- `tests/test_multiclass_video_evidence_revision.py`: regression tests including tamper/fidelity checks.

## Verified in this revision environment

- The provided `TEMPO_FULL_EVIDENCE_20261027T070005Z.zip` was reused successfully. The repair exported **10 verified Baboon panels** over both watermarks, produced 13 PDF figure outputs, and verified all **40 common-source video summary rows** with CA-QIM-A2 correctly classified `semi_blind`.
- After the repair, both previously failed execution-evidence checks passed; the revised ledger reported **zero failed automated execution checks**, while `scientifically_complete_for_resubmission` remained `false` because other external/manuscript limitations are not resolved by postprocessing.
- Focused tests: **37 passed**, plus **15 passed** in an overlapping set of revision/reviewer/official tests.
- A full test-suite invocation with `PYTHONPATH=src:.` exceeded the local 200-second test budget after progressing through more than 100 tests. It was not recorded as a passing full-suite run. One earlier `PYTHONPATH=src`-only invocation failed during test collection because the repository root was missing from the Python path.

## Manuscript follow-up (no numeric table alteration)

Clarify that CA-QIM-A2 is semi-blind, that video comparisons are same-source rather than PSNR matched, that the official Kodak robustness catalog was applied to TEMPO only, and that native-720p stream figures do not automatically satisfy the strict real-time deadline criterion. Keep the **original measurement commit** distinct from the **new evidence-postprocessing commit** in Code Availability.
