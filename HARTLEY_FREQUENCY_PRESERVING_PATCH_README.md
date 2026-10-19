# TEMPO -- minimal opt-in frequency-preserving sparse Hartley proposal

## Source baseline and scope

Designed **specifically for** `tempo-main(20261019-032434).zip` (uploaded original). Unzip these files at the original repository root, retaining the paths inside the ZIP. This patch **does not** include the large original repository, host images, baseline implementations, changed attacks, checkpoint data, or generated results.

**Research status: EXPERIMENTAL.** The candidate improves algorithmic throughput, modestly improves blur NC, and has essentially unchanged PSNR/SSIM. It does **not** improve NC for every evaluated attack. In particular JPEG70/JPEG40 NC is marginally lower than the original. Keep the original manuscript results and default as the reference until the full protocol confirms a suitable replacement.

## Exact changes compared with uploaded original

| Code location | Original method | Opt-in variant |
|---|---|---|
| `config.py` | Fixed low-frequency DCT payload carrier | Adds `hartley_mix`, `hartley_support`, `hartley_carrier_mode`; original default and config hash unchanged |
| `hartley_carrier.py` (new) | No generalized convolution in embedding | Once-per-configuration finite-grid weighted h-Hartley--cosine carrier construction; not Gaussian blur |
| `sparse_hartley_engine.py` (new) | Full 8x8 dense payload reads/writes | 48 nonzero entries for both embedding and blind extraction; Numba sparse loops |
| `method.py` | Original QIM, integer guard, pilots | Opt-in sparse payload kernel and decoder; original controls/pilots/registration preserved |
| New tests/benchmark | N/A | Invariants, round trip, file/stream regression, paired NC/PSNR/SSIM and timed FPS |

No attack-dependent switches, per-host carrier tuning, training, ECC or extra full-frame filtering. Uses fixed 8x8 DCT carrier, QIM lattice and RGB opponent field already present in TEMPO.

### Mathematical design

- Let `P=(B01+B10)/sqrt(2)` (pilot, symmetric), `Q0=(B01-B10)/sqrt(2)` (payload, antisymmetric).
- Construct `Qgamma` from the finite zero-extended h=1 generalized convolution with the fixed signal `f[0]=1/2`, `f[±1]=1/4` and convolution weight `cos(omega)`; blend with `Q0` at `alpha=0.6`. This applies to the **carrier only, once per configuration**, not to every frame.
- Select 24 mirrored off-diagonal pairs from the strongest entries in `Qgamma` (48 pixels total). This support selection does not use images or attack labels.
- For the `frequency_preserving` mode solve, once and cache, the convex quadratic on this fixed support:

  `min_Z ||Z-Qgamma||_F^2 + sum_uv [(1+u+v)^(-2)]*|DCT2(Z-Q0)[u,v]|^2`

  subject to `Z^T=-Z` and `support(Z) subset S48`, then normalize `||Z||_F=1`. (The unconstrained minimizer is normalized after the linear solve; it is NOT claimed to solve a unit-norm constrained convex problem.)
- The mathematical symmetry gives `<Z,P>=0`. Encoder and decoder use identical sparse coefficients; no new synchronization or pilot changes.
- The generalized convolution's infinite-grid transform factorization theorem is **not** claimed to hold exactly for zero-extended 8x8 finite carriers.

## Reproduce by applying the changed files

```bash
# Start from the original project (not a previous Hartley patch).
cd tempo-main
# Extract the patch ZIP HERE: merge src/, scripts/, tests/, configs/.
python -m pip install -r requirements.txt -r requirements-dev.txt
export PYTHONPATH=src NUMBA_NUM_THREADS=2 JILP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
python -m pytest -q tests/test_frequency_preserving_hartley.py \
   tests/test_hartley_convolution_carrier.py tests/test_sparse_hartley_speed.py \
   tests/test_proposal.py tests/test_registry_and_protocol.py \
   tests/test_video_pipeline.py tests/test_stream_pipeline.py
python scripts/frequency_preserving_hartley_compare.py \
   --output results/frequency_preserving48 --timing-runs 8
```

### Activate only when explicitly selected

```python
from dataclasses import replace
from realtime_watermark.proposals.jilp_qim.config import DEFAULT_CONFIG
from realtime_watermark.registry import embed, extract

cfg = replace(DEFAULT_CONFIG, hartley_mix=0.6, hartley_support=48,
              hartley_carrier_mode='frequency_preserving').validated()
marked, key = embed('proposal', host_rgb, binary_wm_64x64, seed=2026,
                    method_params=cfg.to_dict())
recovered = extract(marked, key)  # fully blind to the original host

# Unchanged original baseline: leave DEFAULT_CONFIG as-is.
marked_original, key_original = embed('proposal', host_rgb, binary_wm_64x64,
                                     seed=2026, method_params=DEFAULT_CONFIG.to_dict())
```

Use `hartley_carrier_mode='sparse'` with `hartley_support=48` for the intermediate sparsity-only ablation. Do not decode a new-carrier key with unpatched code; it lacks the new carrier formula.

## Measured original-vs-candidate (same inputs, 2 CPU threads)

Protocol: 9 held-out host images, 2 binary 64x64 watermarks, 4096 payload bits per sample, fixed seed 2026, no synchronization on attacked-image recovery. All parameters except the carrier variant were held fixed. The benchmark computes bipolar NC, mean RGB PSNR and RGB SSIM; no retraining or attack-specific switching. Timing: warm Numba compilation, 4 host images, 6 timed repeats per host and resolution (24 observations for each), alternate variant order, embedding plus blind extraction, no image compression or stream IO. Run-specific measurements, not guaranteed FPS.

| Metric | Original | New frequency-preserving sparse-48 |
|---|---:|---:|
| RGB PSNR (dB) | 44.84019 | 44.85149 |
| RGB SSIM | 0.98767959 | 0.98771397 |
| Clean NC / clean BER | 1.00000 / 0 | 1.00000 / 0 |
| JPEG70 NC | **0.99628** | 0.99593 |
| JPEG40 NC | **0.98698** | 0.98646 |
| Blur radius 1 NC | 0.74916 | **0.75624** |
| Blur radius 1.5 NC | 0.49913 | **0.51294** |
| Box blur radius 1 NC | 0.81963 | **0.82853** |
| Algorithmic FPS, 512x512 | 219.5 | **281.3** (+28.2%) |
| Algorithmic FPS, 1280x720 | 156.7 | **176.7** (+12.8%) |

**Important:** PSNR improvement +0.0113 dB and SSIM improvement +0.000034 are too small to establish a meaningful visual-quality gain. JPEG NC is lower. The candidate's 720p FPS advantage is computed **without JPEG encoding/decoding, geometric search or camera queue**; a separate JPEG-inclusive test of prior sparse variants yielded much smaller improvements. The 720p relative ordering between sparse and frequency-preserving modes varied across timing repetitions. There is no claim of improved full-system FPS or universal robustness.

For a publication decision, also run attack categories/geometric registration, stronger/jpeg multiple qualities, stream p95 and synchronized JPEG-inclusive FPS, reporting failures as well as successes.

## Delivered file list

- `src/realtime_watermark/proposals/jilp_qim/config.py` (modified)
- `src/realtime_watermark/proposals/jilp_qim/method.py` (modified)
- `src/realtime_watermark/proposals/jilp_qim/hartley_carrier.py` (new)
- `src/realtime_watermark/proposals/jilp_qim/sparse_hartley_engine.py` (new)
- `tests/test_hartley_convolution_carrier.py` (new)
- `tests/test_sparse_hartley_speed.py` (new)
- `tests/test_frequency_preserving_hartley.py` (new)
- `scripts/frequency_preserving_hartley_compare.py` (new)
- `configs/hartley_frequency_preserving48_candidate.json` (new)
- `ORIGINAL_TO_NEW_SOURCE.diff` (unified code differences, original modified files only)
- `FILE_SHA256.json` (payload verification)

Validation: 30 passing proposal/Hartley tests and 22 passing registry/phase/video/stream tests in the testing environment. Original frozen SHA-256 configuration hash preserved: `8f54c9e28722d9698396c0390a654aabdbea0ac8da74b8f81adeb7a51e611d92`.
