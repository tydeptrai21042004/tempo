# TEMPO — 23-variant mathematical/FPS ablation patch (2026-10-21)

Apply the ZIP contents over the matching `tempo-main` repository root **using the same directory structure**. This is a *changed-files-only patch*, not the full source tree. It includes the previously tested selective 0.65 phase gate as the default for selective refinement + certified pruning, plus explicit gate overrides for fair ablations. The former default hash remains unchanged when `phase_fast_threshold` is omitted.

## Production and experimental changes

- `src/realtime_watermark/proposals/jilp_qim/config.py`: optional `phase_fast_threshold` plus a new experimental `joint_embedding='orthogonal_joint_center'` configuration. Default remains the old method.
- `src/realtime_watermark/proposals/jilp_qim/phase_registration.py`: use an explicitly selected phase threshold when supplied; retain 0.65 (selective/pruning) or 0.80 (other modes) if not supplied. This threshold is an empirical control, **not** a certified guarantee that an aligned image is correctly registered.
- `src/realtime_watermark/proposals/jilp_qim/method.py`: propagate threshold through frozen embedding key; opt-in joint center-QIM over blocks shared with pilots; leave ordinary sequential center-QIM unchanged.
- `src/realtime_watermark/proposals/jilp_qim/joint_dct_center_engine.py`: a *new experimental method* that computes payload and pilot target displacements from the **same original block**, then applies a joint continuous minimum-energy RGB update. It closes payload parity on the quantized block; existing pilot closure runs afterward. A joint finite-precision result is not assumed to be globally optimal, and a speedup is not assumed.
- `scripts/proposal_fps_ablation.py`: pre-warmed, controlled fixed-4096-bit ablation benchmarking, timings, per-attack output, stage-time estimates, paired effects, bootstrap host confidence intervals, config files and manifest. Failures are recorded instead of silently excluded.
- `scripts/proposal_fps_video_shortlist.py`: real-video, causal 720p JPEG70 wall-clock follow-up for chosen candidate configs. Requires a real video file (no synthetic fallback).
- `tests/test_fps_method_ablation.py`, `tests/test_phase_gate_fps_regression.py`: regression tests.
- `configs/frozen_selected_20261021.json` and `configs/holdout_split_manifest_20261021.json`: exact frozen 128-pilot center-QIM experimental config and original 3/9 host split from supplied results, including SHA-256 host checks.

## Available variants (23 total)

**Reference / phase (8):** `baseline`, `gate_080`, `gate_070`, `gate_065`, `gate_055`, `no_pruning`, `no_refinement`, `no_phase_search_acceleration`.

**Pilot (6 new, plus baseline):** `pilot_064`, `pilot_096`, `pilot_192`, `pilot_random`, `pilot_no_search`, `pilot_disabled`.

**Embedding (5 new, plus baseline):** `joint_center`, `projection_guarded`, `closure_disabled`, `field_opponent`, `field_luminance`.

**Interactions (4 new, plus baseline):** `joint_center_gate065`, `gate065_pilot096`, `gate065_pilot064`, `joint_pilot096`.

`baseline` already uses a 0.65 gate under the supplied frozen configuration. Therefore `gate_065` is an **explicit control**, not a separate different mathematical method. Negative controls that drop synchronization or integer closure should not be selected based on FPS alone. With 512x512 hosts all 4096 8x8 blocks are occupied, so host-ranked payload scheduling has no genuine candidate-pool degree of freedom and is intentionally excluded from this FPS study.

## Local or Kaggle image ablation

From the repository root (Linux/macOS commands; on Windows set `PYTHONPATH` appropriately):

```bash
python -m pip install -e '.[dev]'
PYTHONPATH=src:. python -m pytest -q

# Search over all 23 variants, on ONLY the original 3 development hosts.
PYTHONPATH=src:. python scripts/proposal_fps_ablation.py \
  --config configs/frozen_selected_20261021.json \
  --host-dir data/user_dataset/host \
  --watermark-dir data/user_dataset/watermark \
  --split development --group all --attack-preset stress \
  --attacks '' --repeats 5 --warmups 1 \
  --out-dir reports/fps_ablation_development

# Select finalists using DEVELOPMENT results only, then confirm on the 9 holdout hosts.
PYTHONPATH=src:. python scripts/proposal_fps_ablation.py \
  --config configs/frozen_selected_20261021.json \
  --host-dir data/user_dataset/host \
  --watermark-dir data/user_dataset/watermark \
  --split holdout --variants baseline,joint_center,gate_080,gate_065,gate065_pilot064 \
  --attack-preset stress --attacks '' --repeats 5 --warmups 1 \
  --out-dir reports/fps_ablation_holdout
```

`--group quick`, `--group phase`, `--group pilot`, `--group embedding`, `--group interaction`, and `--group all` are supported. The default attack subset is clean, JPEG70, rotation 1 degree and translation (4,4); `--attacks ''` evaluates every attack in the selected preset. Set `--variants` to test any specific set. Do **not** inspect holdout results to select or retune variants. The current supplied holdout has already been viewed in earlier exploratory comparisons, so new unseen hosts would be needed for a completely untouched final confirmation.

The script writes `fps_ablation_per_attack.csv`, `fps_ablation_summary.csv`, `fps_ablation_paired_deltas.csv`, `fps_ablation_host_bootstrap_ci.csv`, `fps_ablation_manifest.json`, and one replayable JSON under `configs/` for every variant. All input hashes, seeds, attack names and config hashes are retained.

**Important limitations:** image experiment FPS (`1000 / [embedding+attack+extraction]`) is a *stage-time estimate*, NOT real effective stream FPS. Phase threshold trials may match bitmaps on evaluated attacks but have no out-of-distribution guarantee. Varying pilot counts or switching guarded to center-QIM changes the PSNR operating point; this is a native-parameter ablation, **not** matched-PSNR proof. CPU noise/Numba parallel scheduling affect timing. For publication, run repeats, report per-host CIs, and validate native 720p video separately.

## Actual 720p video, once finalist configs are frozen

```bash
PYTHONPATH=src:. python scripts/proposal_fps_video_shortlist.py \
  --input /path/to/native_720p_real_video.mp4 \
  --watermark-dir data/user_dataset/watermark \
  --configs-dir reports/fps_ablation_development/configs \
  --variants baseline,gate_080,gate_065,joint_center,gate065_pilot064 \
  --out-dir reports/fps_ablation_video --frames 180 --fps 30,60
```

The video follow-up reuses the repository's existing JPEG70 implementation, causal bounded queue, and real-time wall-clock measurement. Report actual output FPS, on-time completion, p95 latency and drops. It does *not* change codec backends or remove metrics, which would be implementation-side optimizations rather than mathematical-method ablations.

## Verification performed when preparing this patch

- Full existing + new pytest suite: **183 passed** (local CPU).
- All 23 variants on 2 source hosts x 2 original watermarks x 4 selected attacks: **368 evaluations**, no execution failures; all clean NC = 1 (preliminary).
- 7 important variants on all 12 hosts x 2 watermarks x 4 attacks: **672 evaluations**, no execution failures; all clean NC = 1 (preliminary; not a new blind holdout).
- On the latter run the baseline and `joint_center` PSNR means were **44.1296 / 44.1309 dB**; their attack NC means were **0.971307 / 0.971205**, and their median embedding times were **1.403 / 1.410 ms** (on this machine). **Joint center-QIM has NOT demonstrated an FPS improvement.** `gate065_pilot064` yielded about 1.349 ms embedding and 3.780 ms attacked extraction versus 1.403 / 4.250 ms for baseline in this small attack selection, but attacked NC decreased from 0.971307 to 0.971273. More than four attack conditions are needed before selecting it.

This experimental patch modifies the algorithm only where explicitly opted in; it does **not** justify replacing the submitted manuscript's reported results without a properly controlled evaluation.
