# Reviewer-to-experiment mapping

All output paths below are relative to the full result folder. `quick` mode is a smoke test and should not appear as a manuscript table.

| Reviewer issue | Code / output | Interpretive boundary |
| --- | --- | --- |
| R1: Table 4 throughput denominator | `reviewer_throughput_audit.py` → `throughput_audit/wallclock_throughput_audit.csv` | Throughput = 4096 × **effective** FPS; do not substitute target/kernel FPS |
| R1: duplicated results | `run_multiwatermark_metrics.py`, `reviewer_matched_pairs.py`, source & config manifests | Different protocols require separate captions, not forced identical runtime |
| R1: reproducibility | `record_run_source.py`, `configs/manuscript_frozen.json`, stage logs, attack manifest | Cite exact input/config/output hashes |
| R1 and R5: matched-PSNR | `run_multiwatermark_fair_benchmark.py` → matched results and `matched_psnr_strict_paired_manifest.json` | Pair only eligible same-host/watermark/attack observations |
| R1: CI | `reviewer_statistics.py` → `paired_proposal_effects.csv` | Resample host clusters, not attack rows |
| R2: post-8-bit orthogonality | `reviewer_experiments.py` → `orthogonality_and_integer.csv`, `geometry_and_targeted_attacks.csv`, PDF `reviewer_orthogonality.pdf` | Numeric coordinate movements do **not** prove causal cross-carrier interference by themselves |
| R2: geometric recovery scope | `reviewer_experiments.py` → `geometry_and_targeted_attacks.csv` | Finite recovery grid only; arbitrary affine invariance not claimed |
| R2: temporal voting | `reviewer_temporal_limits.py` → temporal errors | Voting does not guarantee correction of correlated frame errors |
| R2: coupling in Figure 3 | `reviewer_joint_sensitivity.py` → `joint_parameter_coupling.csv`, `proposal_hyperparameter_multi_panel.pdf` | Development hosts only; no test-host parameter optimization |
| R2: Arnold permutation ablation | `run_multiwatermark_ablation.py` → ablation CSV | Permutation is not cryptographic encryption |
| R2/R5: integer failures / extreme illumination | `reviewer_integer_boundaries.py` → `embedding_attempts.csv`, `rejection_summary.csv` | Report accepted/rejected/uncaught errors with attempt denominators |
| R2: payload capacity | `reviewer_experiments.py` → `capacity_by_payload_attack.csv`, `reviewer_capacity_latency.pdf` | Subset of attacks, not full 45-attack mean |
| R5: off-grid rotations / subpixel / shear | `reviewer_experiments.py` → `geometry_and_targeted_attacks.csv` | Record both attack and raw vs recovered outcome |
| R5: intentional band suppression | `reviewer_experiments.py` → `targeted_carrier_interference.csv` | Public-carrier diagnostic outside generic common-attack ranking |
| R5: ROC/FPR | `reviewer_detection_protocol.py` → `heldout_roc.csv`, `reviewer_roc.pdf` | Threshold chosen on development negatives only; known-message detection, not cryptographic proof |
| R5: native 720p JPEG70 | `run_video_corpus_suite.py` → per-frame trace; `reviewer_realtime_720p.pdf` | Stream FPS, p95 end-to-end latency, missed deadlines and decoded/encoded channel assumptions are separate |
| R5: paper-derived modern transform comparison | `run_recent_svd_paper_structure.py` + `configs/paper_2024_reference.json` | Independent equation-level recreation of DOI 10.52866/ijcsm.2024.05.03.043; grayscale/informed, **not 4096 blind bits**, separate from matched-blind leaderboard |

## Publication-facing reconciliation rule

Read final results, then update abstract, Tables 2–5, off-grid/timing/ablation tables and figure captions. Record both 64/96/128 pilot configuration differences if historical results are retained; they are not the same method operating point. No number should be copied from this README as a measured result. Do not claim hard real-time schedulability based on empirical 30-FPS throughput, nor claim cryptographic authentication from an Arnold permutation or a finite ROC test.

## Newly integrated 2024-inspired comparison

- Common 46-condition image comparison: `modern_baselines_matched_psnr/matched_psnr_{by_host_watermark,per_attack}.csv`
- Strict paired comparison: `modern_baselines_matched_psnr/matched_psnr_strict_paired_{effects,attacks}.csv`
- Independent *real-input-video* adapter and TEMPO: `modern_video_adapter/all_watermarks_video_summary.csv`
- Provenance-verified JPEG30 extracted-watermark source images: `result_figure_sources/baboon/gallery_manifest.json`
- Result figure generator does **not** recreate the method schematic PDFs unless explicitly requested.

The adjective *inspired* is mandatory: the 2024 and 2026 original singular-value embedding/extraction formulas differ from the new blind QIM adapter, including original-host side-information requirements. Never attribute the independent adapter's robustness numbers to the source paper's authors.


### Additional source-paper-structure evidence

The optional `recent_svd_paper_structure_secondary/` CSV + JSON records an independent informed LWT–HD–SVD-style decomposition from the supplied 2024 paper, explicitly excluded from primary matched blind comparisons because its detector requires original-host and watermark SVD side information. It does not prove exact fidelity to the authors' numerical implementation. The modern **blind** method-class adapter remains `haar_hd_svd_qim_adapter2024`, distinctly labeled as an independent 4096-bit QIM design.

**Default real-video cross-method result:** `video_common_blind_baselines/` contains the proposed method and the explicitly labeled DCT DM-QIM blind comparator. The optional independently designed Haar–HD–SVD–QIM control is excluded unless `TEMPO_RUN_INDEPENDENT_HAAR_QIM_CONTROL=1`; it is not a published method reproduction. The informed 2024 DOI reproduction is always a separate secondary experiment.
