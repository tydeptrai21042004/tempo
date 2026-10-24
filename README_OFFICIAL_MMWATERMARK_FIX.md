# TEMPO × original MMWatermark-Robustness integration — October 2026

## What changes

This patch **clones and verifies the original upstream** `Jielin-Qiu/MMWatermark-Robustness` repository and its original `Jielin-Qiu/MM_Robustness` dependency. After verification, it installs exactly one **untracked, TEMPO-owned** integration entry at `MMWatermark-Robustness/perturbations/tempo_proposal_runner.py`, then executes **inside `MMWatermark-Robustness/perturbations`**. The proposal's embedding/extraction comes from the TEMPO `realtime_watermark` codebase. The 23 named image-corruption functions (subject to successful preflight) are imported from the authentic upstream `image_perturb.py`; their source is not replaced or edited.

IMPORTANT: This is the TEMPO method tested against genuine original benchmark perturbations. **It is not a reproduction of the upstream multimodal model baselines**, not a claim of official author endorsement, and not identical to TEMPO's local 109-attack suite. The upstream repository's CC BY-NC-SA license must be respected when distributing derived material; the patch does not package upstream code or assets.

### Fixed code problems

1. Earlier source allowlist omitted the official `rotate_img`; correcting it adds **5 rotation severity conditions** (1 through 5). With the verified 23 original image-corruption function names, this is 115 requested attacked conditions **plus clean** per host/watermark/method. The earlier 22-function run's reported `full_coverage` only covered its smaller requested set.
2. Original `gaussian(..., multichannel=True)` now works with `skimage` versions that use `channel_axis` (module-local compatibility only).
3. The official `impulse_noise` uses `skimage.util.random_noise`. Recent `skimage` employs an unseeded generator by default, so NumPy's global seed alone did not ensure repeatable observations. The patch forwards a deterministic derived RNG seed to the **original** function (module-local compatibility only).
4. NumPy 2 removed `np.float_` used by older upstream weather noise code; the existing narrowly scoped alias is retained.
5. All upstream attack/severity pairs are preflighted at real 512×512 RGB host dimensions. No synthetic or TEMPO attack is substituted for an unsupported upstream condition. Strict coverage is now the *default* for the original-repo integration: it runs and saves every supported original condition, and then exits nonzero if any requested condition was unsupported. Choose `--allow-partial` only if intentionally reporting incomplete evidence.
6. Verification fails on tracked upstream modifications, unexpected untracked files, a mismatched Git remote, or a mismatched integration source digest. Upstream commits, TEMPO entry digest, number of expected/completed rows, and unsupported failures are recorded.
7. The audit no longer treats an old 22-function MMWatermark manifest that says `full_coverage: true` as full **corrected** scope if it lacks `rotate_img`.
8. Includes the earlier image-exclusion fix (`lenna.bmp` and `lena.bmp`) for standalone FPS and image metrics scripts, and updates `SOURCE_MANIFEST.sha256` so the main TEMPO pipeline can verify changed files.

## Applying patch

Extract this ZIP into your TEMPO repo root, **preserving paths**. Push the changed source and `SOURCE_MANIFEST.sha256` to GitHub. Do not upload a clone of MMWatermark-Robustness into TEMPO GitHub; it is retrieved fresh at execution.

Run tests:

```bash
python -m pytest -q tests/test_official_mmwatermark_compat.py tests/test_official_repo_entry.py tests/test_audit_kaggle_run.py
python scripts/verify_source_manifest.py --allow-missing
```

## Run using your existing held-out dataset + frozen config

```bash
export PYTHONPATH="$PWD/src:$PWD"
python scripts/run_mmwatermark_in_official_repo.py \
  --official-repo /kaggle/working/MMWATERMARK_ORIGINAL_FRESH \
  --companion-repo /kaggle/working/MM_ROBUSTNESS_ORIGINAL_FRESH \
  --host-dir /kaggle/working/TEMPO_FRESH_<RUN_ID>/reports/holdout/test \
  --watermark-dir /kaggle/working/TEMPO_FRESH_<RUN_ID>/selected_data/watermark \
  --config /kaggle/working/TEMPO_FRESH_<RUN_ID>/reports/configs/proposal_selected_all_watermarks.json \
  --out-dir /kaggle/working/mmwatermark_original_results \
  --severities 1 2 3 4 5
```

`<RUN_ID>` is a placeholder to replace with an actual existing run directory. Both upstream destination directories must be empty / absent. The new Kaggle one-cell runner (`KAGGLE_MMWATERMARK_ORIGINAL_ONE_CELL.py`) automatically locates prior real evidence or, if none is found, reruns the native TEMPO pipeline on CPU, then clones upstream and runs its benchmark.

In `reports/external_official_mmwatermark` or `extra/external_official_mmwatermark`, inspect:

- `official_integration_provenance.json`: Git URLs, commits, bridge SHA-256, execution cwd, command.
- `official_mmwatermark_preflight.json`: all validated / unsupported upstream conditions.
- `official_mmwatermark_manifest.json`: host/watermark counts, rows, failures, true `full_coverage`.
- `official_mmwatermark_raw.csv`: real proposal clean/attack metrics (NC, BER, PSNR, latency).

## Validation boundaries

A local offline smoke test confirms that the actual TEMPO embedding/extraction executes under a real Git checkout with the one-file integration bridge and stores correct manifest/CSV row counts. This test uses a **minimal fake upstream perturbation function to test the bridge**; it **is not** evidence of the full original benchmark completing. The public original source was inspected, but internet access to GitHub from the code runtime was unavailable, so actual live cloning and all 23×5 upstream checks must be verified in Kaggle. The runner fails closed and preserves partial manifests/logs rather than making success claims without execution. The Kaggle one-cell runner also pins the same Python interpreter used during setup to avoid mixed-environment imports.

### Re-using old runner cell

The earlier notebook cell calls `scripts/run_official_mmwatermark_tempo.py` directly and therefore does **not** launch from inside upstream. To fulfill the requested execution location, use `KAGGLE_MMWATERMARK_ORIGINAL_ONE_CELL.py`, or enable `TEMPO_OFFICIAL_MMWATERMARK=1` in the patched native `scripts/kaggle_pipeline.sh`. The bridge remains one untracked file in upstream and is hashed for provenance.
