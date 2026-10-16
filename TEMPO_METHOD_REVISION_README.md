# TEMPO opt-in mathematical revision — experimental patch, 16 October 2026

## Scope and scientific status

This patch does **not** redesign the 8x8 DCT, 4,096-bit QIM payload, 96 orthogonal pilots, RGB opponent field, integer correction, Arnold scrambling, or blind decoder. `configs/proposal_frozen.json` and its SHA-256 config hash remain exactly compatible with the original uploaded repository. Two method extensions are opt-in:

* **Finite-grid QIM-phase ranking with exact upper-bound pruning**: for signed phase samples `s_l in [-1,1]`, once `k` pilots have been evaluated, `U_k=(sum(s_1...s_k)+N-k)/N` is an upper bound on the final candidate score. Pruning candidates only when `U_k` is smaller than the third fully evaluated candidate preserves the original top-three geometry ranking. This is an exact mathematical screening rule, **not** a new global optimizer; it does not certify the approximate translation shortlist.
* **Selective sub-grid rotation estimate**: when the same coarse search selects a sufficiently coherent, zero-translation, nonzero-rotation hypothesis, refine only its angle within ±0.32 degrees by maximizing the **same** sparse QIM-phase objective. Accept only if phase coherence improves by at least 0.03. This preserves the clean-image fast path and the single winning full-frame warp. A better phase score is NOT proof of improved NC on arbitrary images.

The candidate also lowers the QIM payload step from `17.78404328715303` to `17.0` as an explicit **parameter trade-off**. Step tuning is not claimed as algorithmic novelty.

### Conditional 8-bit guarantee, not a false global-optimality claim

For a jointly feasible in-range continuous block `z`, nearest-integer rounding produces `e` with `||e||_infty <= 0.5`. For either carrier `a`, `|a^T e| <= 0.5 ||a||_1`. Consequently, if the continuous carrier output lies in its intended QIM cell at least `rho > 0.5 ||a||_1` away from a decision boundary, rounding cannot change that bit. The certificate script evaluates those constants. **This does not prove** that the current sequential payload/pilot encoder always finds a jointly guarded point, nor that its integer output globally minimizes squared distortion. Quantization-induced cross-coupling after warp also remains possible.

## Evidence: 9 held-out hosts x 2 watermarks; same seed 2026

Development images: `lenna`, `baboon`, `pepper`, `sailboat`. These were used for initial step selection. The held-out evaluation uses the other 9 BMPs (`Girl`, `airplane`, `athens`, `butfly`, `house`, `manhatan`, `milkdrop`, `safari`, `tiffany`). Each has two 64x64 watermarks. Primary raw evidence: `evidence/holdout_paired.csv`, `evidence/holdout_summary.csv`, `evidence/protocol.json`.

| Metric / attack | Original | Candidate |
|---|---:|---:|
| Clean PSNR mean | 44.8402 dB | 45.2164 dB |
| Clean NC | 1.0000 | 1.0000 |
| JPEG70 NC | 0.99628 | 0.99552 |
| JPEG40 NC | 0.98698 | 0.98204 |
| Rotate 1.25° NC | 0.55642 | 0.80279 |
| Rotate 2.25° NC | 0.56426 | 0.82061 |
| Rotate 3.25° NC | 0.58466 | 0.84825 |
| Translate 0.5 px NC | 0.58415 | 0.57351 |
| Scale 1.02x NC | 0.07585 | 0.07530 |

The attacked mean over those **seven** sampled attacks increases from 0.62123 to 0.72829, but this statistic is dominated by the improved rotation cases and must not be substituted for a broader attack benchmark. All attack families and regressions must be shown separately in the paper. This is neither proof of universal invariance nor a guarantee of improvements for other distributions.

## 720p timing caveat

The `method_revision_720p_timing.py` microbenchmark runs complete `embed -> JPEG70 encode/decode -> extract` on resized 1280x720 images. Recorded repeated measurements under sandbox CPU contention had large outliers even after warmup; on one host / two watermarks (5 repeats) median total latency was ~80.1ms original vs ~77.3ms candidate, but the tail latency varied dramatically. **No reproducible 30-FPS deadline improvement is established.** These measurements are not a true live-stream evaluation, and no dropped-frame or end-to-end capture latency is included. Files: `evidence/timing_720_warm.csv`.

## Reproduce and validate

From the project root:

```bash
pip install -r requirements.txt -r requirements-dev.txt
export PYTHONPATH="$PWD/src:$PWD"
export NUMBA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
python -m pytest -q tests/test_certified_phase_revision.py tests/test_sparse_phase_revision.py tests/test_quantization_guard_certificate.py
python scripts/quantization_guard_certificate.py --step 17
python scripts/method_revision_study.py --split development --quick --out results/method_revision_dev
python scripts/method_revision_study.py --split holdout --quick --out results/method_revision_holdout
python scripts/method_revision_study.py --split holdout --out results/method_revision_stress
python scripts/method_revision_720p_timing.py --repeats 20 --output results/method_revision/timing_720.csv
```

The longer stress / full real-video pipeline was **not completed** in this sandbox. The full unit suite was attempted but ran beyond its time budget during existing high-cost paper-native/Kaggle-support tests; the focused new + existing revision tests passed (8/8). Do not report the complete reviewer matrix as passed.

To use the **new candidate** as an embedding configuration in your own run, load `configs/proposal_revision_candidate.json` and pass its `parameters` to the existing `embed('proposal', ..., method_params=...)`. The original `scripts/kaggle_pipeline.sh` still starts from the original `configs/proposal_frozen.json` and performs a parameter search; it has **not** been silently switched to the experimental candidate.

## Reviewer status requiring additional work

1. **Quantized orthogonality / integer optimality**: mathematical rounding bound exists for jointly feasible continuous blocks, but no globally optimal integer solver or proof of actual sequential orthogonality; keep empirical integer failure/rejection and post-warp interference tests.
2. **Geometric scope**: only off-grid rotation improvement verified. Scaling, subpixel translation, general affine shear, off-grid crops and large angles are *not* fixed.
3. **Fixed DCT pair vulnerability**: no carrier redesign adopted; adversarial coefficient removal remains a limitation, and preliminary single-image carrier migration damaged PSNR. Do not claim attack resistance.
4. **Real-time 720p**: currently not shown to meet the 30-FPS/33.33ms deadline. Include real end-to-end p95, on-time completion, dropped frames, codec details and frame distribution.
5. **Temporal voting / ECC**: unchanged; study correlated errors and temporal fusion limits. Do not claim ECC functionality or cryptographic Arnold security.
6. **Experiments, fairness, reproducibility**: the existing reviewer suite has independent matched-PSNR, ROC, ablation and security runners, but complete execution is still required. Modern genuinely independent video baselines and trained adversarial watermark-removal attacks have not been faithfully executed here.

## Novelty boundary

QIM, multidimensional QIM, local geometric optimization and phase-score upper bounds are established mathematical ingredients. **Novelty has not been independently verified** by a complete prior-art search. The strongest potential contribution for a revised manuscript is the specific orthogonal QIM pilot system with a rigorous quantization margin bound and distortion-aware bounded phase refinement under a real-time budget. Avoid presenting generic scalar optimization or parameter tuning as a new mathematical theory.
