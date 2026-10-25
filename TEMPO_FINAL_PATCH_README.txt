TEMPO FINAL — repository (tempo-main(6).zip) corrections, 2026-10-25

Purpose
-------
1. Internal image studies run ONLY on the repository's existing 12 BMP host
   images (excluded Lena/Lenna), using the pre-existing fixed 46-condition
   reviewer46 attack manifest; two original 64x64 watermark PNGs; 4096 bits.
   The existing five 4096-bit source-backed common-protocol baselines, matched
   distortion, ablations, hyperparameter tuning, time studies, and figures
   remain handled by scripts/kaggle_pipeline.sh full.
2. External Kodak: DO NOT RUN the native 109-attack Kodak TEMPO benchmark.
   Only use the genuine ORIGINAL MMWatermark-Robustness official perturbations
   applied to TEMPO's frozen proposal, covering ALL 24 Kodak images and BOTH
   64x64 watermarks. Kodak originals are independently downloaded with source
   SHA256 provenance. Every one is centrally cropped, NOT RESIZED, to the
   lossless 512x512 RGB BMP required by the existing MMWatermark official
   integration. The exact crop and source digest are recorded.
3. NO strict_scientific_gate call; PSNR and FPS are kept as real measurements.
   Runtime failures, missing data, missing upstream perturbation conditions,
   invalid checksum, or incomplete coverage continue to fail the run.
4. NO external CSV input or simulated third-party methods.

Files updated/added
-------------------
KAGGLE_TEMPO_STRICT_FULL_ONE_CELL.py : full native + official Kodak only,
  fail-closed execution, archives logs/results/source+crop manifests, overlap
  audit via decoded RGB pixel hashes, no scientific gate or CSV import.
scripts/prepare_kodak_official_512.py : pixel-exact 24 RGB BMP crops.
scripts/run_mmwatermark_in_official_repo.py : validates input files BEFORE
  downloading/cloning the two original upstream repositories.
scripts/benchmark_algorithm_only_720p.py : 300 real frames per watermark; JPEG70 runs but its codec time is excluded from algorithm-only embed + blind extract timer. Not an end-to-end FPS claim.
scripts/summarize_official_kodak.py : validated 4096-bit and full-coverage
  condition matrix; outputs detailed mean/std/min/max by watermark, attack
  and host, along with summary_manifest.json.
tests/test_kaggle_no_gate_official_kodak.py : launcher contract and result
  completeness tests.
tests/test_clean_repository.py : recognizes preexisting README_FIX.md as a
  permitted committed protocol note, rather than misclassifying it as an
  unexpected document.
SOURCE_MANIFEST.sha256 : refreshed for source edits and new source files.
KAGGLE_TEMPO_FINAL_ONE_CELL.py : downloadable one-cell Kaggle launcher.

Install
-------
A. Either replace your GitHub repository contents with the FULL corrected
   source ZIP, or overlay the CHANGED_FILES patch ZIP on your current repo.
B. Push the corrected files to GitHub main branch.
C. Kaggle: Accelerator=None, Internet=On. Copy the entire contents of
   KAGGLE_TEMPO_FINAL_ONE_CELL.py into one Python cell and run.
D. Save TEMPO_ALL_RESULTS_FRESH_<UTC>.zip from /kaggle/working.

Key evidence
------------
reports/all_methods_all_metrics/* : internal 46-condition results.
reports/holdout/* : frozen data split (3 development, 9 test).
extra/algorithm_only_720p/algorithm_only_manifest.json
extra/official_mmwatermark_kodak_24/official_mmwatermark_raw.csv
extra/official_mmwatermark_kodak_24/official_mmwatermark_manifest.json
extra/official_mmwatermark_kodak_24/summaries/by_watermark_attack.csv
extra/official_mmwatermark_kodak_24/summaries/by_watermark_host.csv
extra/official_mmwatermark_kodak_24/summaries/by_watermark_overall.csv
run_metadata/dataset_overlap_audit.json
external_provenance/kodak_full_manifest.json
external_provenance/kodak_official_crops_manifest.json
run_metadata/status.json and *.log

Expected official count (subject to genuine upstream catalog)
-------------------------------------------------------------
24 images × 2 watermarks × (1 clean + N genuine official functions × 5 severities).
If N=23, 5,568 trials. A smaller count is never silently accepted: the
upstream manifest's discovered catalog and failed/unsupported conditions are
retained and full requested coverage is explicitly required.

Scientific scope
----------------
Benchmark compares YOUR proposal under genuine upstream attacks; it does NOT
claim to evaluate the upstream MMWatermark model or match the resolutions of
all published results. The 512x512 center-crop policy is recorded. The native
24-Kodak/109-attack test is intentionally NOT launched. Neither PSNR>=50 dB
nor FPS>=30 is used as a completion gate; factual measured performance is
retained with its timing scope (algorithm-only vs end-to-end).

Verification
------------
Run: python scripts/verify_source_manifest.py
Run: PYTHONPATH=src:. python -m pytest -q tests/test_kodak_bmp_integration.py tests/test_official_repo_entry.py tests/test_official_mmwatermark_compat.py tests/test_kaggle_no_gate_official_kodak.py tests/test_reviewer_unified_46.py
Live Kaggle execution and upstream internet downloads cannot be reproduced
inside the offline sandbox; actual 5,568 trial completion must be checked in
Kaggle manifest before a claim of completion.
