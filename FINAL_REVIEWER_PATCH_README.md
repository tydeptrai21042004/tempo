# TEMPO Final Reviewer Code Revision — 2026-10-29

This overlay adds a **separate final reviewer protocol** without deleting the
preexisting TEMPO pipeline or changing the TEMPO watermarking algorithm.

## Final baseline sets

- **Image robustness, ignore timing for ranking:** Chen–Wornell DM-QIM, Chen–Wornell STDM-QIM, Malvar ISS, Mao CA-QIM-A2 (semi-blind), Cox spread-spectrum (informed/non-blind), and TEMPO. Five distinct method algorithms, four paper lineages, implemented in existing repository as documented *paper-derived common-protocol adapters*. These are **not** claimed to be complete author-code reproductions. Native-paper diagnostic tracks are preserved separately.
- **Real-video causal comparison:** TEMPO and **DEW (Langelaar & Lagendijk, 2001)** only, with **256-bit common payload**, frozen configuration, 30 FPS targets, actual wall-clock producer and bounded queue, four video channels and genuine real-source clips. DEW is implemented with paper-associated DCT coefficient **zeroing**, adapted to DCT of decoded RGB frames; this is not a compressed-bitstream codec implementation. Both rows include actual embedding PSNR; **no matched-PSNR claim is made** for the paper-style DEW run.
- **TEMPO 4096-bit video** is another track, never put in the same matched-payload ranking as 256-bit DEW.
- `expanded_learned_native`, RivaGAN, VideoSeal, RBMark and neural models are **out of scope**. A single classical DEW baseline does **not** satisfy the reviewer's request for contemporary video-method comparisons. State this limitation in the response letter.

## Files added or changed

- `src/realtime_watermark/registry.py`: explicit disjoint reviewer baseline tuples.
- `src/realtime_watermark/stream_pipeline.py`: `calibration_mode='frozen'` and `paper_dew=True`, preventing held-out video first-frame calibration. The legacy calibration behavior remains the default **only for backward-compatible historical scripts**.
- `scripts/live_stream_benchmark.py`: new CLI switches `--calibration-mode frozen --paper-dew --require-real-video` and optional disjoint development calibration source.
- `scripts/run_tempo_dew_video_suite.py`: final real-video benchmark/coverage/manifest, verified inputs, separate 256/4096-bit tracks.
- `scripts/run_final_reviewer_protocol.py`: final image/video reviewer orchestrator with stage ledger, fail-visible outputs and method-count validation.
- `scripts/reviewer_cross_interference.py`: controlled payload/pilot leakage before/after integer conversion and local geometric interpolation. Not a full-video geometric attack study.
- `scripts/export_final_integer_audit.py`: count integer correction events from unique clean embeddings, avoiding 45x double counting; **rejected** embeddings need the separate boundary-stress experiment for numerators/denominators.
- `scripts/reviewer_capacity_full.py`: measured payload-size embedding and extraction timing summaries (existing experiments, not new invented outcomes).
- `scripts/reviewer_joint_theory.py`: observed carrier displacement diagnostics.
- `scripts/run_tempo_evidence.py`: removes learned-native stage from selectable/default stages.
- `scripts/kaggle_final_reviewer_pipeline.sh`, `KAGGLE_FINAL_REVIEWER_ONE_CELL.py`: run the full existing pipeline and supplemental reviewer comparison from a fresh GitHub clone.
- `tests/test_final_reviewer_protocol.py`: baseline/frozen-stream controls.

## GitHub upload (overlay ZIP)

Unzip the overlay into your GitHub repository root, preserving the paths, then commit and push. Do not upload the top-level ZIP itself as a replacement for existing scripts. **All other project files remain intact.**

```bash
unzip -o TEMPO_FINAL_REVIEWER_CODE_CHANGES.zip -d .
git add src scripts tests KAGGLE_FINAL_REVIEWER_ONE_CELL.py FINAL_REVIEWER_PATCH_README.md
git commit -m 'Add paper-based split robustness and DEW causal reviewer protocols'
git push
```

## Run locally with existing prepared data

```bash
python scripts/run_final_reviewer_protocol.py \
  --test-dir reports/holdout/test \
  --development-dir reports/holdout/development \
  --watermark-dir data/user_dataset/watermark \
  --config configs/manuscript_frozen.json \
  --video-corpus reports/video_sources/corpus.json \
  --output reports/final_reviewer --frames 300 --continue-on-error
```

If prepared paths do not exist, run `bash scripts/kaggle_pipeline.sh full` first, or use `bash scripts/kaggle_final_reviewer_pipeline.sh full` on Kaggle; then upload the resulting evidence ZIP. The root `KAGGLE_FINAL_REVIEWER_ONE_CELL.py` automates clone/setup/full evaluation after the GitHub changes are pushed.

`--quick` runs a *diagnostic subset* only, not paper-quality results. The full image robustness run uses the original 45 attacks. Matched-PSNR applies only to methods supported by the calibration implementation; Cox remains in native/common tracks and is excluded from matched PSNR instead of assigning fake coefficients. All method-specific detector assumptions must remain disclosed.

## Scientific caveats

A successful unit or local fixture smoke test is **not** evidence that the full Kaggle corpus, external MMWatermark 100-level protocol, or all 45-attack runs have been executed with this patch. Review the per-stage ledger, dataset hashes, source SHA and source-specific plots before updating numbers in the manuscript. The original 45-attack and external Kodak protocols remain separate.
