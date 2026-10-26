# TEMPO — Low-Latency Blind Image and Video Watermarking

**Reproducible experimental implementation accompanying** *Low-Latency Blind Image and Video Watermarking via Orthogonal DCT Carriers and Minimum-Norm Embedding* (Tran Kim Huong and Dang Ba Ty).

TEMPO studies the quality–robustness–latency trade-off for **4096-bit**, blind RGB image and video watermarking. The method uses one low-frequency 8×8 DCT coefficient pair to form orthogonal payload and synchronization coordinates, binary center-QIM, a closed-form continuous minimum-norm RGB update, a bounded integer-pixel correction, and pilot-triggered finite geometric registration. On video, frame arrivals are causal and a fixed watermark is recovered from each processed frame, optionally aggregated by majority vote. See the manuscript for the precise mathematical assumptions and method statement.

This repository is a reproducibility companion, **not** a new algorithm specification. Source code, seed, selected data files and generated protocol metadata determine each result. Source hash and selected configuration hash accompany each run. A successful Python invocation alone does not establish robustness, general geometric invariance, global integer optimality, or a hard real-time guarantee.

## Reproduce the manuscript protocol

The bundled dataset has **12 RGB 512×512 BMP hosts** (Lenna is excluded) and two PNG watermark patterns. An immutable development/test split assigns 3 images for development and 9 for the principal held-out image evaluation. Each watermark is binarized to 64×64 bits. This protocol is distinct from the independent Kodak 24-image external dataset and from real public video clips.

The manuscript operating point is saved in [`configs/manuscript_frozen.json`](configs/manuscript_frozen.json). It fixes **64 pilot blocks**, **center-QIM**, **opponent RGB field** (eta 0.07), payload step 17.78404328715303, pilot step 15.0 and seed 2026. It is intentionally separate from older `proposal_frozen.json` and `frozen_selected_20261021.json`, which describe other configurations and must not be silently substituted. **Do not copy historical manuscript numerical results onto a fresh run; regenerate tables from that run's measured files.**

### Local installation

Requires Python 3.11+ with NumPy, SciPy, Numba, Pillow, OpenCV, scikit-image, pytest and Matplotlib. Linux/WSL is recommended for shell pipeline scripts.

```bash
python -m pip install -r requirements.txt
python -m pip install matplotlib pytest
export PYTHONPATH="$PWD/src:$PWD"

# quick smoke protocol: intentionally not eligible for final paper tables
bash scripts/kaggle_pipeline.sh quick

# full held-out images, video, reviewer diagnostics and PDF figures
bash scripts/kaggle_pipeline.sh full
```

The full run fetches verified public video clips (Internet needed) and may take hours on CPU. For a Kaggle CPU notebook, upload **TEMPO_FINAL_REVIEWER_CODE.zip** as a Dataset, enable Internet, and paste the entire `KAGGLE_ONE_CELL.py` into a cell. It uses the uploaded corrected source rather than cloning an old branch, records per-stage logs, and packages outputs even after a stage failure. Use `MODE='quick'` to test the environment before the full experiment. Only a full run on all relevant hosts/watermarks is appropriate for quantitative manuscript updates.

To evaluate an *alternative development-selected* configuration without modifying the method, set `TEMPO_CONFIG_MODE=development_search`. Those outputs are a **different operating point** from the paper configuration and must not be merged with manuscript numbers.

## What the benchmark measures

| Evidence | Source scripts | Primary outputs |
| --- | --- | --- |
| All 45 non-clean common attacks + clean, six methods, two watermarks | `run_multiwatermark_metrics.py` | `all_methods_all_metrics/*.csv` |
| Structural ablation, including disabled pilots/recovery/Arnold permutation | `run_multiwatermark_ablation.py` | `ablation/**/*.csv` |
| Development-only joint parameter sensitivity | `reviewer_joint_sensitivity.py` | `joint_sensitivity/joint_parameter_coupling.csv` |
| Matched-PSNR effects, strict same-case paired comparisons | `run_multiwatermark_fair_benchmark.py`, `reviewer_matched_pairs.py` | `matched_psnr/*.csv` |
| Host-cluster confidence intervals and paired contrasts | `reviewer_statistics.py` | `statistical_evidence/*.csv` |
| Off-grid rotation, shear, scaling, targeted public carrier attenuation, integer interference | `reviewer_experiments.py` | `reviewer_experiments/*.csv` |
| Stress tests on shadow/highlight/saturated images and embedding rejection denominators | `reviewer_integer_boundaries.py` | `integer_boundaries/*.csv` |
| Development-calibrated threshold, held-out ROC/FPR | `reviewer_detection_protocol.py` | `detection/*.csv` |
| Payload (256/1024/2304/4096) and cost/NC trade-off | `reviewer_experiments.py` | `reviewer_experiments/capacity_*.csv` |
| Temporal voting limits under correlated vs non-correlated errors | `reviewer_temporal_limits.py` | `temporal_limits/*.csv` |
| Genuine downloaded video, causal arrivals, 720p/JPEG70 deadline and throughput | `video_evidence_corpus.py`, `run_video_corpus_suite.py`, `reviewer_throughput_audit.py` | `video/`, `live_stream/`, `throughput_audit/` |
| Result-section vector PDF figures from archived CSV and decoded images | `export_result_gallery.py`, `generate_manuscript_figures.py` | `manuscript_figures/*.pdf`, `figures_manifest.json` |
| Recent blind transform comparison and genuine-video extension | `run_multiwatermark_fair_benchmark.py`, `run_multiwatermark_video.py` | `modern_baselines_matched_psnr/*.csv`, `modern_video_adapter/*.csv` |

The complete attack matrix is defined in `configs/attacks/reviewer_unified_46.json`; geometric limitations and attack severity are part of that manifest. Group means must not be treated as 45 independent host samples. The real-time protocol distinguishes steady-state kernel timing, service time, 95th-percentile queue-inclusive latency, achieved output FPS, generated-frame completion ratio and codec overhead. Results that miss 30 FPS are still recorded and reported.

**Important distinction:** legacy methods in the common-protocol benchmark are independent adapters, not necessarily exact reproductions of every published method. Blind, semi-blind and informed detectors are identified separately. DEW requires more than one 8×8 block per embedded bit, so it is tested at lower capacity rather than falsely labeled a 4096-bit competitor.

## Manuscript figures (PDF, not screenshots)

Run the figure generator using the **same selected configuration** as the experiment:

```bash
python scripts/generate_manuscript_figures.py \
  --results-root /path/to/tempo-manuscript-results \
  --host-dir /path/to/tempo-selected-data/host \
  --watermark-dir /path/to/tempo-selected-data/watermark \
  --config /path/to/tempo-manuscript-results/configs/proposal_selected_all_watermarks.json \
  --out-dir /path/to/tempo-manuscript-results/manuscript_figures
```

The default generator produces **results-section figures only**: `proposal_hyperparameter_multi_panel.pdf`, `baboon_attacked_host.pdf`, `baboon_extracted_methods.pdf` (plus a separately named PDF for the second watermark), and reviewer ROC, capacity/latency, carrier interaction, and 720p deadline figures. `fig01.pdf`, `fig02.pdf`, and the existing host/watermark montage are left unchanged in the manuscript. To redraw those only when explicitly needed, pass `--include-nonresult-figures`.

Baboon panels are generated **exclusively** from the archived matched-condition replay produced by `export_result_gallery.py`. Its decoded NC must reproduce the stored matched-PSNR CSV for each method or the figure is refused. Other plots require their corresponding observed CSV. Missing sources appear in `figures_manifest.json`; nothing is synthesized to fill them.

To include the PDFs in the LaTeX manuscript, copy the files into its working directory (or use `\graphicspath`). See `docs/REVIEWER_EVIDENCE.md` for the reviewer item mapping and scope caveats.

## Reproduced paper baseline, provenance, and comparison classes

**Source DOI (original paper):** [Sudrajat, Ernawan and Amrullah (2024), *Robust Image Watermarking Based on Lifting Wavelet Transform–Hessenberg Decomposition for Copyright Protection*](https://doi.org/10.52866/ijcsm.2024.05.03.043), *Iraqi Journal for Computer Science and Mathematics* **5**(3), 695–706, Eqs. (10)–(22), page 700.

### The paper-based executable comparison (RUNS BY DEFAULT)

`scripts/run_recent_svd_paper_structure.py` is a **direct independent equation-level reproduction** of this published construction, rather than a generic Haar/SVD-QIM substitute. It implements (1) Haar wavelet decomposition to the 64x64 LL of a 512x512 grayscale host (three levels), (2) Hessenberg similarity reduction, (3) singular-value decomposition of the Hessenberg LL matrix and the actual 64x64 watermark logo, (4) additive singular-spectrum embedding `S_modified = S_host + alpha*S_watermark`, (5) inverse SVD, inverse Hessenberg, inverse Haar, (6) extraction by subtraction of the **original host singular values** and the paper's printed 2024 inverse SVD rule using the received-image singular vectors. The method runs real attack images and exports every measured result, development-only alpha calibration and `paper_2024_fidelity.json`.

The papers' ambiguous/correctness-sensitive details are **not hidden**: for 512-to-64 resolution, R=3 is selected to match the dimensions; Eq. (16) of the paper omits the final `P.T` in its printed inverse Hessenberg formula (the code uses the mathematically required inverse `P@H@P.T`); the native grayscale input is measured on the same 512x512 host converted to grayscale; and original authors' numerical implementation is unavailable. The 2024 article's Eq. (22) may not reconstruct the original logo reliably, even in the clean case. We therefore report the actual outputs and do not replace Eq. (22) with another rule to improve its NC.

**This is an informed image-watermarking method, not a blind 4096-independent-bit competitor.** A 64x64 extracted logo is not equal to 4096 independent blind payload symbols: the singular-value modification carries only up to 64 spectral values. The detector uses original-host singular values. Its results appear ONLY under `recent_svd_paper_structure_secondary/`, never in the matched-capacity blind leaderboard. Native experiment results must not be interpreted as head-to-head superiority at 4096 blind bits.

Optional comparative reconstruction from the 2023 Ernawan et al. paper is accessible through `--reconstruction paper2023_watermark_vectors` for a method-mechanism diagnostic, but does **not** convert the experiment to a reproduction of the 2023 paper's IWT pipeline.

### Strict blind, equal-payload comparisons (separate from above)

The primary matched-PSNR study runs `proposal` against genuine blind QIM/ISS/STDM implementations under 4096 bits, identical hosts and common attacks, with development-only calibration and same-case paired effects. These are **independent common-protocol implementations**, not original-author software snapshots. They are not labeled as verified paper-native result reproductions. The independent `haar_hd_svd_qim_adapter2024` is a *new blind control algorithm*, **not** the method of any published 2024 paper. It is disabled in the default paper reproduction run; to explore it honestly, set `TEMPO_RUN_INDEPENDENT_HAAR_QIM_CONTROL=1` (results remain explicitly labeled `independent adapter`).

The supplied [2026 NSST–HD–SVD paper](https://doi.org/10.71129/ijaci.v2i2.pp115-126) calls itself blind in the abstract, but its printed Algorithm 2, Step 5, explicitly accesses original-host singular values and original watermark vectors. No 4096-bit blind reproduction of that paper is asserted. An author's verified implementation can be ingested only with its real detector class, capacity, configuration and provenance using `scripts/import_external_baseline_results.py`.

See `docs/PAPER_2024_FIDELITY.md` and `docs/RELATED_BASELINES.md` for the source-to-equation audit. Missing faithful modern blind reproduction is an openly reported limitation, not silently satisfied by a surrogate.

## Secondary 2024 LWT–Hessenberg–SVD experiment (informed; optional)

`run_recent_svd_paper_structure.py` provides a **separately labeled, independent, equation-level structural experiment** based on the 2024 LWT–HD–SVD literature. It explicitly uses original host singular values and the original watermark singular vectors at detection. Although its input is a 64×64 logo, the embedded singular spectrum does **not** represent 4096 independently recoverable blind bits. This is neither a paper-exact reconstruction nor a legitimate entry in the 4096-bit matched-blind primary leaderboard.

To run this additional secondary benchmark during the standard pipeline, set `TEMPO_RUN_SECONDARY_2024_INFORMED=1`. The script exports per-host/per-watermark/per-attack NC, BER, PSNR, runtime, a separate development-only alpha calibration file and a protocol manifest. Its stage is optional because the informed model is **not** what Reviewer #5 requests as the primary blind-video control.

## Reproducibility and failure reporting

The package intentionally has **no scientific acceptance gate**. It does **not** suppress poor NC, high BER, missed deadlines, missed target PSNR, a failed current attack or a missing baseline. Test results, tool exceptions, incomplete rows and external-source failures remain in logs and JSON/CSV evidence. An execution failure is not a completed experiment. Code/package provenance, input validation and actual test execution remain in place because they protect correctness rather than gate scientific outcomes.

`reviewer_completeness.json` is a diagnostic inventory; it does not block ZIP creation. The run report contains a source SHA-256 manifest, configuration SHA-256, attack parameters, dataset split, package versions, frame-level timing traces, and figure manifest. **Never state that all reviewer comments are satisfied until each required experiment has produced a complete verified record.**

## Project layout

- `src/realtime_watermark/`: encoder, blind decoder, baselines and attack machinery.
- `scripts/`: manuscript/reviewer experiments, provenance, PDF rendering and runners.
- `configs/`: explicit manuscript operating point, development search and attack manifests.
- `data/user_dataset/`: selected public benchmarking inputs; watermark PNGs.
- `tests/`: implementation and protocol regression tests.
- `docs/`: experimental reviewer map and honest literature-to-baseline guidance.

The source code is released under the repository `LICENSE`; the source papers and independently hosted video/Kodak datasets retain their own terms. Cite the paper when using the evaluation results and cite the original papers for baseline algorithms.

## Applying the GitHub patch and running the full Kaggle clone cell

For an existing GitHub repository matching `tempo-main(10).zip`, the small
`TEMPO_GITHUB_DELTA_ONLY.zip` release contains only newly added and modified
source files. Extract its `tempo-main/` directory on top of your existing Git
working tree, preserving paths, and commit the changes to `main` **before**
running Kaggle. If using GitHub web upload, upload files to their exact relative
paths (not into an extra `tempo-main/` folder). The patch deliberately does not
include unchanged host images.

Paste `KAGGLE_CLONE_ALL_EXPERIMENTS.py` into a single Kaggle notebook cell:
Accelerator=None, Internet=ON. It clones the public repository, checks the
required patch files, records the Git commit, runs the full internal experiments,
executes the separate informed 2024 paper-equation reproduction, and attempts
the official MMWatermark robustness protocol on Kodak-24. The optional **internal**
Kodak 109-attacks experiment is off by default; the official MMWatermark-Kodak
route is enabled. `PROFILE='internal'` or `PROFILE='official_kodak'` can be used
to execute the two expensive tracks in independent Kaggle sessions. `smoke`
checks a small subset and must not be reported as manuscript results.

Kaggle session time limits can interrupt a full run. `run_metadata/stage_status.json`
and periodic `TEMPO_EVIDENCE_*.zip` checkpoints record completed, failed, and
unexecuted stages. The cell does **not** claim that a time-limited run has
completed all experiments. `scripts/reviewer_completeness.py` is diagnostic,
not a scientific success gate. `scripts/strict_scientific_gate.py` is obsolete
and should be deleted from GitHub; no new runner invokes it.

The 2024 DOI method remains a grayscale **informed** side-information baseline,
not a 4096-bit blind video competitor. The modern independent Haar–HD–SVD–QIM
experiment is excluded by default because it is not a reproduction of the
paper. This is a scientific comparison limit, not a code execution gate.
