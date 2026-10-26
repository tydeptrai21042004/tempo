#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-full}"
shift || true
if [[ "$MODE" != "quick" && "$MODE" != "full" ]]; then
  echo "Usage: scripts/kaggle_pipeline.sh [quick|full]" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON:-}"
if [[ -z "$PYTHON_BIN" && -f "$ROOT/.kaggle_python" ]]; then
  PYTHON_BIN="$(cat "$ROOT/.kaggle_python")"
fi
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"

export PYTHONPATH="$ROOT/src:$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

if [[ -d /kaggle/working ]]; then
  WORK_ROOT="/kaggle/working"
elif [[ -d /content ]]; then
  WORK_ROOT="/content"
else
  WORK_ROOT="${WORK_ROOT:-$ROOT}"
fi

REPORT_ROOT="${KAGGLE_REPORT_ROOT:-$WORK_ROOT/tempo-manuscript-results}"
SELECTED_DATA_DIR="${KAGGLE_SELECTED_DATA_DIR:-$WORK_ROOT/tempo-selected-data}"
MASTER_ZIP="${KAGGLE_MASTER_ZIP:-$WORK_ROOT/tempo-manuscript-results.zip}"
SEED="${SEED:-2026}"
WM_SIZE="${WM_SIZE:-64}"
HP_HOSTS="${HP_HOSTS:-3}"
VIDEO_CODEC="${VIDEO_CODEC:-mp4v}"
VIDEO_INPUT="${VIDEO_INPUT:-}"
# Default is the *unmodified* development-selected method. Opt in to the
# certified phase refinement with TEMPO_MATH_REFINEMENT=1, retaining the
# original configuration and its SHA-256 separately for comparison.
TEMPO_MATH_REFINEMENT="${TEMPO_MATH_REFINEMENT:-0}"
if [[ "$TEMPO_MATH_REFINEMENT" != "0" && "$TEMPO_MATH_REFINEMENT" != "1" ]]; then
  echo "TEMPO_MATH_REFINEMENT must be 0 or 1" >&2; exit 2
fi
RESUME_FROM_VIDEO="${KAGGLE_RESUME_FROM_VIDEO:-0}"

if [[ "$MODE" == "quick" ]]; then
  HP_ATTACK_PRESET="reviewer46"
  IMAGE_ATTACK_PRESET="reviewer46"
  VIDEO_FRAMES="${VIDEO_FRAMES:-24}"
  VIDEO_WARMUP_FRAMES="${VIDEO_WARMUP_FRAMES:-2}"
  HOST_LIMIT="${HOST_LIMIT:-2}"
  ABLATION_HOST_LIMIT="${ABLATION_HOST_LIMIT:-1}"
  STREAM_FRAMES="${STREAM_FRAMES:-30}"
  STREAM_RESOLUTIONS="${STREAM_RESOLUTIONS:-512x512}"
  VIDEO_STREAM_CHANNELS="clean,jpeg70,jpeg40"
else
  HP_ATTACK_PRESET="reviewer46"
  IMAGE_ATTACK_PRESET="reviewer46"
  VIDEO_FRAMES="${VIDEO_FRAMES:-300}"
  VIDEO_WARMUP_FRAMES="${VIDEO_WARMUP_FRAMES:-10}"
  HOST_LIMIT="${HOST_LIMIT:-0}"
  ABLATION_HOST_LIMIT="${ABLATION_HOST_LIMIT:-0}"
  STREAM_FRAMES="${STREAM_FRAMES:-300}"
  STREAM_RESOLUTIONS="${STREAM_RESOLUTIONS:-512x512,1280x720}"
  VIDEO_STREAM_CHANNELS="clean,jpeg90,jpeg70,jpeg40"
fi

if [[ "$RESUME_FROM_VIDEO" != "1" ]]; then
  rm -rf "$REPORT_ROOT" "$SELECTED_DATA_DIR" "$MASTER_ZIP"
fi
mkdir -p "$REPORT_ROOT"
printf 'Kaggle pipeline %s: %s\n' "$( [[ "$RESUME_FROM_VIDEO" == "1" ]] && echo resumed || echo started )" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$REPORT_ROOT/full_run.log"
HOST_DIR="$SELECTED_DATA_DIR/host"
WATERMARK_DIR="$SELECTED_DATA_DIR/watermark"
SELECTED_CONFIG="$REPORT_ROOT/configs/proposal_selected_all_watermarks.json"
DEV_HOST_DIR="$REPORT_ROOT/holdout/development"
TEST_HOST_DIR="$REPORT_ROOT/holdout/test"
VIDEO_CORPUS_MANIFEST="${VIDEO_CORPUS_MANIFEST:-configs/video_evidence_sources.json}"
VIDEO_CACHE_DIR="${VIDEO_CACHE_DIR:-$WORK_ROOT/tempo-video-cache}"
VIDEO_PREPARED_DIR="${VIDEO_PREPARED_DIR:-$WORK_ROOT/tempo-video-prepared}"
VIDEO_CORPUS_PATH="$REPORT_ROOT/video_sources/corpus.json"

stage() {
  echo
  echo "======================================================================"
  echo "$1"
  echo "======================================================================"
}

if [[ "$RESUME_FROM_VIDEO" == "1" ]]; then
  for item in "$HOST_DIR" "$WATERMARK_DIR" "$DEV_HOST_DIR" "$TEST_HOST_DIR" "$SELECTED_CONFIG" "$VIDEO_CORPUS_PATH" "$REPORT_ROOT/all_methods_all_metrics/all_watermarks_all_methods_all_metrics.csv"; do
    if [[ ! -e "$item" ]]; then echo "Cannot resume: missing $item" >&2; exit 2; fi
  done
  echo "Resume mode: preserving prior image and ablation evidence, restarting video with validated checkpoints."
else
stage "1. RECORD SOURCE AND EXECUTION ENVIRONMENT"
mkdir -p "$REPORT_ROOT/manifest"
"$PYTHON_BIN" scripts/record_run_source.py --repo "$ROOT" --output "$REPORT_ROOT/manifest/source_files.json"

stage "2. SELECT ALL BMP HOSTS EXCEPT LENA AND BOTH PNG WATERMARKS"
"$PYTHON_BIN" scripts/select_kaggle_dataset.py \
  --source-data data/user_dataset \
  --output "$SELECTED_DATA_DIR" \
  --manifest-dir "$REPORT_ROOT"
HOST_DIR="$SELECTED_DATA_DIR/host"
WATERMARK_DIR="$SELECTED_DATA_DIR/watermark"

stage "2A. PREFLIGHT 4096-BIT HOST AND WATERMARK INPUTS"
"$PYTHON_BIN" scripts/reviewer_preflight.py \
  --host-dir "$HOST_DIR" --watermark-dir "$WATERMARK_DIR" \
  --out "$REPORT_ROOT/validation/dataset_preflight.json" --wm-size "$WM_SIZE"

stage "3. RUN ALL TESTS"
mkdir -p "$REPORT_ROOT/validation"
set +e
"$PYTHON_BIN" -m pytest -q --junitxml="$REPORT_ROOT/validation/junit.xml"
TEST_STATUS=$?
set -e
"$PYTHON_BIN" scripts/generate_validation_report.py \
  --junit "$REPORT_ROOT/validation/junit.xml" \
  --out-dir "$REPORT_ROOT/validation" \
  --exit-code "$TEST_STATUS" \
  --command "$PYTHON_BIN -m pytest -q"
# Unit tests are recorded as diagnostics. They do not impose a scientific
# acceptance gate on experiment artifacts or prevent later stages from running.
if [[ "$TEST_STATUS" -ne 0 ]]; then
  echo "WARNING: unit tests reported status $TEST_STATUS; inspect validation/junit.xml" >&2
fi

stage "4. FREEZE CONFIGURATION (MANUSCRIPT OR OPT-IN DEVELOPMENT SEARCH)"
SELECTED_CONFIG="$REPORT_ROOT/configs/proposal_selected_all_watermarks.json"
mkdir -p "$(dirname "$SELECTED_CONFIG")"
"$PYTHON_BIN" scripts/create_reviewer_holdout.py \
  --host-dir "$HOST_DIR" --out-dir "$REPORT_ROOT/holdout" --development-count "$HP_HOSTS"
DEV_HOST_DIR="$REPORT_ROOT/holdout/development"
TEST_HOST_DIR="$REPORT_ROOT/holdout/test"
CONFIG_MODE="${TEMPO_CONFIG_MODE:-manuscript}"
if [[ "$CONFIG_MODE" == "manuscript" ]]; then
  "$PYTHON_BIN" - configs/manuscript_frozen.json "$SELECTED_CONFIG" <<'PY_CONFIG'
import json,sys
from pathlib import Path
from realtime_watermark.proposals.jilp_qim.config import JILPQIMConfig
c=JILPQIMConfig.load(sys.argv[1]);c.save(sys.argv[2])
assert json.loads(Path(sys.argv[2]).read_text())['config_hash']==c.sha256()
print('Frozen manuscript configuration:',c.sha256())
PY_CONFIG
elif [[ "$CONFIG_MODE" == "development_search" ]]; then
  "$PYTHON_BIN" scripts/search_proposal_hyperparameters.py \
    --host-dir "$DEV_HOST_DIR" --watermark-dir "$WATERMARK_DIR" \
    --base-config configs/proposal_frozen.json --selected-config "$SELECTED_CONFIG" \
    --out-dir "$REPORT_ROOT/hyperparameter_search" --mode "$MODE" \
    --seed "$SEED" --wm-size "$WM_SIZE" --search-hosts "$HP_HOSTS" \
    --external-heldout-dir "$TEST_HOST_DIR" --attack-preset "$HP_ATTACK_PRESET"
else
  echo "TEMPO_CONFIG_MODE must be manuscript or development_search" >&2; exit 2
fi

# Keep the development-selected control configuration and its search log.
# Only the proposed method's active config is changed; baseline algorithms
# and all subsequent held-out image/video test protocols are unmodified.
if [[ "$TEMPO_MATH_REFINEMENT" == "1" ]]; then
  stage "4B. ACTIVATE CERTIFIED QIM-PHASE PRUNING + LOCAL REFINEMENT"
  "$PYTHON_BIN" scripts/apply_math_refinement.py \
    --selected-config "$SELECTED_CONFIG" \
    --original-copy "$REPORT_ROOT/configs/proposal_original_selected.json" \
    --audit "$REPORT_ROOT/configs/math_refinement_audit.json"
fi

stage "4A. DOWNLOAD AND VERIFY REAL VIDEO CORPUS (NO SYNTHETIC FALLBACK)"
VIDEO_AVAILABLE=0
VIDEO_ABLATION_INPUT=""
# A failed public download must not discard independently reproducible image
# experiments. The corpus stage remains incomplete and is recorded explicitly.
if "$PYTHON_BIN" scripts/video_evidence_corpus.py \
  --manifest "$VIDEO_CORPUS_MANIFEST" --cache-dir "$VIDEO_CACHE_DIR" \
  --prepared-dir "$VIDEO_PREPARED_DIR" --out "$VIDEO_CORPUS_PATH" \
  --mode "$MODE" --frames "$VIDEO_FRAMES"; then
  VIDEO_AVAILABLE=1
  FIRST_CORPUS_VIDEO="$($PYTHON_BIN - "$VIDEO_CORPUS_PATH" <<'PYVIDEO'
import json,sys
print(json.load(open(sys.argv[1]))['videos'][0]['prepared_path'])
PYVIDEO
)"
  VIDEO_ABLATION_INPUT="${VIDEO_INPUT:-$FIRST_CORPUS_VIDEO}"
else
  echo 'VIDEO SOURCE UNAVAILABLE: genuine-video evidence incomplete; image studies continue.' >&2
  printf '%s\n' 'public_video_download_failed; not replaced with synthetic frames' \
      > "$REPORT_ROOT/video_source_unavailable.txt"
fi

# Paper-native protocols use method-specific attacks and cannot belong to the
# strictly shared 46-condition matched comparison. Keep separately opt-in.
if [[ "${TEMPO_RUN_PAPER_NATIVE_SEPARATELY:-0}" == "1" ]]; then
stage "5. OPTIONAL PAPER-NATIVE REPRODUCTION (NOT PART OF UNIFIED ATTACK COMPARISON)"
NATIVE_ARGS=()
if [[ "$MODE" == "quick" ]]; then
  NATIVE_ARGS+=(--quick)
fi
"$PYTHON_BIN" scripts/run_paper_native_benchmarks.py \
  --data-dir "$TEST_HOST_DIR" \
  --out "$REPORT_ROOT/paper_native" \
  "${NATIVE_ARGS[@]}"
fi

stage "5A. FREEZE CANONICAL ATTACK AND BASELINE FIDELITY MANIFESTS"
"$PYTHON_BIN" scripts/audit_reviewer_unified_46.py --out "$REPORT_ROOT/validation/reviewer_unified_46.json" --source-copy "$REPORT_ROOT/validation/reviewer_unified_46_manifest.json"
"$PYTHON_BIN" scripts/audit_common_protocol.py \
  --output "$REPORT_ROOT/validation/common_protocol.json" \
  --attack-preset "$IMAGE_ATTACK_PRESET"

stage "6. RUN ALL IMAGE METHODS ON BOTH WATERMARKS"
METRIC_ARGS=(
  --host-dir "$TEST_HOST_DIR"
  --watermark-dir "$WATERMARK_DIR"
  --proposal-config "$SELECTED_CONFIG"
  --out-dir "$REPORT_ROOT/all_methods_all_metrics"
  --seed "$SEED"
  --wm-size "$WM_SIZE"
  --attack-preset "$IMAGE_ATTACK_PRESET"
)
if [[ "$HOST_LIMIT" -gt 0 ]]; then
  METRIC_ARGS+=(--host-limit "$HOST_LIMIT")
fi
"$PYTHON_BIN" scripts/run_multiwatermark_metrics.py "${METRIC_ARGS[@]}"

stage "7. RUN IMAGE AND VIDEO ABLATIONS FOR BOTH WATERMARKS"
ABLATION_ARGS=(
  --host-dir "$TEST_HOST_DIR"
  --watermark-dir "$WATERMARK_DIR"
  --config "$SELECTED_CONFIG"
  --out-dir "$REPORT_ROOT/ablation"
  --seed "$SEED"
  --wm-size "$WM_SIZE"
  --max-frames "$VIDEO_FRAMES"
  --codec "$VIDEO_CODEC"
  --attack-preset "$IMAGE_ATTACK_PRESET"
)
if [[ "$ABLATION_HOST_LIMIT" -gt 0 ]]; then
  ABLATION_ARGS+=(--max-hosts "$ABLATION_HOST_LIMIT")
fi
if [[ "$VIDEO_AVAILABLE" == "1" ]]; then
  ABLATION_ARGS+=(--video-input "$VIDEO_ABLATION_INPUT")
else
  ABLATION_ARGS+=(--skip-video)
fi
"$PYTHON_BIN" scripts/run_multiwatermark_ablation.py "${ABLATION_ARGS[@]}"

fi  # End of pre-video stages; resume preserves all outputs from steps 1--7.

SUITE_RESUME_ARGS=()
if [[ "$RESUME_FROM_VIDEO" == "1" ]]; then SUITE_RESUME_ARGS+=(--resume); fi

stage "9B. RUN SUPPLEMENTARY REVIEWER EXPERIMENTS ON HELD-OUT HOSTS"
REVIEWER_EXTRA_ARGS=()
if [[ "$MODE" == "quick" ]]; then
  REVIEWER_EXTRA_ARGS+=(--max-hosts 1 --max-watermarks 1)
fi
"$PYTHON_BIN" scripts/reviewer_experiments.py \
  --host-dir "$TEST_HOST_DIR" --watermark-dir "$WATERMARK_DIR" \
  --config "$SELECTED_CONFIG" --out-dir "$REPORT_ROOT/reviewer_experiments" \
  --mode "$MODE" --seed "$SEED" --attack-preset "$IMAGE_ATTACK_PRESET" "${REVIEWER_EXTRA_ARGS[@]}"


stage "9B1. INTENSITY BOUNDARY AND EMBEDDING REJECTION TESTS"
BOUNDARY_ARGS=()
if [[ "$MODE" == "quick" ]]; then BOUNDARY_ARGS+=(--quick); fi
"$PYTHON_BIN" scripts/reviewer_integer_boundaries.py \
  --host-dir "$TEST_HOST_DIR" --watermark-dir "$WATERMARK_DIR" \
  --config "$SELECTED_CONFIG" --out-dir "$REPORT_ROOT/integer_boundaries" \
  --seed "$SEED" "${BOUNDARY_ARGS[@]}"

stage "9B2. CALIBRATED KNOWN-WATERMARK DETECTION: DEVELOPMENT TO HELD-OUT"
DETECTION_ARGS=()
if [[ "$MODE" == "quick" ]]; then
  DETECTION_ARGS+=(--negative-keys 2)
else
  DETECTION_ARGS+=(--negative-keys 8)
fi
"$PYTHON_BIN" scripts/reviewer_detection_protocol.py \
  --development-dir "$DEV_HOST_DIR" --test-dir "$TEST_HOST_DIR" \
  --watermark-dir "$WATERMARK_DIR" --config "$SELECTED_CONFIG" \
  --out-dir "$REPORT_ROOT/detection" --seed "$SEED" "${DETECTION_ARGS[@]}"

stage "9B3. JOINT PARAMETER SENSITIVITY ON DEVELOPMENT HOSTS"
JOINT_ARGS=()
if [[ "$MODE" == "quick" ]]; then
  JOINT_ARGS+=(--max-hosts 1 --max-watermarks 1)
fi
"$PYTHON_BIN" scripts/reviewer_joint_sensitivity.py \
  --development-dir "$DEV_HOST_DIR" --watermark-dir "$WATERMARK_DIR" \
  --config "$SELECTED_CONFIG" --out-dir "$REPORT_ROOT/joint_sensitivity" \
  --seed "$SEED" --attack-preset "$HP_ATTACK_PRESET" "${JOINT_ARGS[@]}"

stage "9B4. EMPIRICAL TEMPORAL VOTING LIMITS"
TEMPORAL_ARGS=()
if [[ "$MODE" == "quick" ]]; then
  TEMPORAL_ARGS+=(--max-hosts 1 --max-watermarks 1 --repetitions 1)
else
  TEMPORAL_ARGS+=(--max-hosts 3 --max-watermarks 2 --repetitions 5)
fi
"$PYTHON_BIN" scripts/reviewer_temporal_limits.py \
  --host-dir "$TEST_HOST_DIR" --watermark-dir "$WATERMARK_DIR" \
  --config "$SELECTED_CONFIG" --out-dir "$REPORT_ROOT/temporal_limits" \
  --seed "$SEED" "${TEMPORAL_ARGS[@]}"

stage "9B5. EXTRACTED WATERMARK EVIDENCE GALLERIES"
GALLERY_ARGS=()
if [[ "$MODE" == "quick" ]]; then
  GALLERY_ARGS+=(--max-hosts 1 --max-watermarks 1)
fi
"$PYTHON_BIN" scripts/reviewer_gallery.py \
  --host-dir "$TEST_HOST_DIR" --watermark-dir "$WATERMARK_DIR" \
  --config "$SELECTED_CONFIG" --out-dir "$REPORT_ROOT/galleries" \
  --seed "$SEED" --attack-preset "$IMAGE_ATTACK_PRESET" "${GALLERY_ARGS[@]}"

stage "9C. MATCHED-PSNR BASELINES ON HELD-OUT HOSTS"
MATCHED_ARGS=()
if [[ "$MODE" == "quick" ]]; then
  MATCHED_ARGS+=(--host-limit 1 --watermark-limit 1 --attack-preset "$IMAGE_ATTACK_PRESET" --timing-repeats 2 --timing-warmups 1)
else
  MATCHED_ARGS+=(--attack-preset "$IMAGE_ATTACK_PRESET" --timing-repeats 5 --timing-warmups 2)
fi
"$PYTHON_BIN" scripts/run_multiwatermark_fair_benchmark.py \
  --host-dir "$TEST_HOST_DIR" --calibration-host-dir "$DEV_HOST_DIR" \
  --watermark-dir "$WATERMARK_DIR" --proposal-config "$SELECTED_CONFIG" \
  --out-dir "$REPORT_ROOT/matched_psnr" --wm-size "$WM_SIZE" \
  --seed "$SEED" --target-psnr 44.18 --psnr-tolerance 0.75 "${MATCHED_ARGS[@]}"

stage "9C2. STRICT COMMON-PAIR MATCHED-PSNR EFFECTS"
"$PYTHON_BIN" scripts/reviewer_matched_pairs.py \
  --matched-dir "$REPORT_ROOT/matched_psnr" \
  --out-dir "$REPORT_ROOT/matched_psnr" --seed "$SEED"

# Manuscript result-panel sources must always use the real, primary matched
# experiment (not the independent experimental Haar/SVD control).
if [[ "$MODE" == "full" ]]; then
  stage "9C2G. REPLAY AND VERIFY MATCHED-PROTOCOL BABOON IMAGE PANELS"
  if ! "$PYTHON_BIN" scripts/export_result_gallery.py \
    --matched-dir "$REPORT_ROOT/matched_psnr" \
    --host-dir "$TEST_HOST_DIR" --watermark-dir "$WATERMARK_DIR" \
    --proposal-config "$SELECTED_CONFIG" \
    --out-dir "$REPORT_ROOT/result_figure_sources/baboon"; then
    echo 'Manuscript Baboon source images missing or inconsistent; no replacement images fabricated.' >&2
  fi
fi

if [[ "${TEMPO_RUN_INDEPENDENT_HAAR_QIM_CONTROL:-0}" == "1" ]]; then
stage "9C2A. OPTIONAL INDEPENDENT 4096-BIT BLIND CONTROL (NOT PUBLISHED PAPER)"
# Important: this is an *independent* blind QIM adapter, NOT the original 2024
# paper's side-information singular-value method. Calibration uses dev hosts.
MODERN_ARGS=()
if [[ "$MODE" == "quick" ]]; then
  MODERN_ARGS+=(--host-limit 1 --watermark-limit 1 --timing-repeats 1 --timing-warmups 0 --calibration-iterations 8)
else
  MODERN_ARGS+=(--timing-repeats 3 --timing-warmups 1 --calibration-iterations 8)
fi
if "$PYTHON_BIN" scripts/run_multiwatermark_fair_benchmark.py \
  --host-dir "$TEST_HOST_DIR" --calibration-host-dir "$DEV_HOST_DIR" \
  --watermark-dir "$WATERMARK_DIR" --proposal-config "$SELECTED_CONFIG" \
  --methods 'proposal,dct_dm_qim_chen2001,haar_hd_svd_qim_adapter2024' \
  --out-dir "$REPORT_ROOT/modern_baselines_matched_psnr" --wm-size "$WM_SIZE" \
  --seed "$SEED" --target-psnr 44.18 --psnr-tolerance 0.75 \
  --attack-preset "$IMAGE_ATTACK_PRESET" "${MODERN_ARGS[@]}"; then
  "$PYTHON_BIN" scripts/reviewer_matched_pairs.py \
    --matched-dir "$REPORT_ROOT/modern_baselines_matched_psnr" \
    --out-dir "$REPORT_ROOT/modern_baselines_matched_psnr" --seed "$SEED"
else
  echo 'Modern transform comparison execution failed; no false benchmark rows added.' >&2
fi

fi

stage "9C2B. DOI-TRACEABLE 2024 LWT-HD-SVD PAPER EQUATIONS (GRAY/INFORMED)"
# This is the actual LWT->HD->SVD additive-SW construction in Eqs.(10)-(22)
# of DOI 10.52866/ijcsm.2024.05.03.043, not a blind QIM invention.
# It MUST be a separate native/same-attack secondary table: its decoder
# requires original cover singular values and cannot carry 4096 blind bits.
PAPER_ARGS=()
if [[ "$MODE" == "quick" ]]; then
  PAPER_ARGS+=(--quick --max-hosts 1 --max-watermarks 1 --calibration-steps 3)
fi
if ! "$PYTHON_BIN" scripts/run_recent_svd_paper_structure.py \
    --development-dir "$DEV_HOST_DIR" --test-dir "$TEST_HOST_DIR" \
    --watermark-dir "$WATERMARK_DIR" \
    --reconstruction paper2024_eq22 --levels 3 \
    --out-dir "$REPORT_ROOT/recent_svd_paper_structure_secondary" \
    "${PAPER_ARGS[@]}"; then
  echo '2024 paper-equation reproduction FAILED: see status/logs; no dummy rows.' >&2
  printf '%s\n' 'FAILED: 2024-paper-equation trial runner' > "$REPORT_ROOT/recent_svd_paper_structure_secondary_failed.txt"
fi

stage "9C3. VERIFY THE SAME 46 ATTACKS FOR EVERY COMMON-PROTOCOL METHOD"
"$PYTHON_BIN" scripts/audit_internal_unified_coverage.py \
  --reports "$REPORT_ROOT" --out "$REPORT_ROOT/validation/internal_coverage.json"

stage "9D. HOST-CLUSTERED CONFIDENCE INTERVALS AND PAIRED EFFECTS"
"$PYTHON_BIN" scripts/reviewer_statistics.py \
  --input "$REPORT_ROOT/all_methods_all_metrics/all_watermarks_all_methods_all_metrics.csv" \
  --out-dir "$REPORT_ROOT/statistical_evidence" --seed "$SEED"

if [[ "${TEMPO_OFFICIAL_MMWATERMARK:-0}" == "1" ]]; then
  stage "9E. ORIGINAL MMWATERMARK CLONE + TEMPO PROPOSAL IN UPSTREAM CHECKOUT"
  # This clones both REAL official repositories, installs only a TEMPO-owned
  # untracked entrypoint into the original perturbations directory, and runs
  # the proposal there. No tracked upstream code is modified or reimplemented.
  OFFICIAL_CMD=(
    "$PYTHON_BIN" scripts/run_mmwatermark_in_official_repo.py
    --official-repo "${TEMPO_MMWATERMARK_SOURCE:-$WORK_ROOT/MMWatermark-Robustness}"
    --companion-repo "${TEMPO_MMWATERMARK_COMPANION:-$WORK_ROOT/MM_Robustness}"
    --host-dir "$TEST_HOST_DIR" --watermark-dir "$WATERMARK_DIR"
    --config "$SELECTED_CONFIG"
    --out-dir "$REPORT_ROOT/external_official_mmwatermark"
  )
  if [[ "$MODE" == "quick" ]]; then
    OFFICIAL_CMD+=(--host-limit 1 --watermark-limit 1 --severities 1)
  fi
  if [[ "${TEMPO_MMWATERMARK_ALLOW_PARTIAL:-0}" == "1" ]]; then
    OFFICIAL_CMD+=(--allow-partial)
  fi
  "${OFFICIAL_CMD[@]}"
fi

if [[ "${VIDEO_AVAILABLE:-0}" == "1" || "$RESUME_FROM_VIDEO" == "1" ]]; then
  stage "8-9. RUN REAL VIDEO CORPUS AND LIVE DEADLINE TESTS"
  if "$PYTHON_BIN" scripts/run_video_corpus_suite.py \
    --corpus "$VIDEO_CORPUS_PATH" --watermark-dir "$WATERMARK_DIR" \
    --proposal-config "$SELECTED_CONFIG" --video-out "$REPORT_ROOT/video" \
    --stream-out "$REPORT_ROOT/live_stream" --mode "$MODE" \
    --max-frames "$VIDEO_FRAMES" --stream-frames "$STREAM_FRAMES" \
    --codec "$VIDEO_CODEC" --python "$PYTHON_BIN" \
    --channels "$VIDEO_STREAM_CHANNELS" --resolutions "$STREAM_RESOLUTIONS" "${SUITE_RESUME_ARGS[@]}"; then
    stage "9A0. REAL-INPUT VIDEO COMPARISON (BLIND DCT QIM VS PROPOSED)"
    # Both are executable blind 4096-bit methods; neither is passed off as a
    # faithful modern Hessenberg/SVD paper reproduction.
    if [[ -n "${FIRST_CORPUS_VIDEO:-}" && -f "$FIRST_CORPUS_VIDEO" ]]; then
      VIDEO_COMPARE_FRAMES=120
      if [[ "$MODE" == "quick" ]]; then VIDEO_COMPARE_FRAMES=12; fi
      VIDEO_COMPARE_METHODS='proposal,dct_dm_qim_chen2001'
      VIDEO_COMPARE_DIR="$REPORT_ROOT/video_common_blind_baselines"
      if [[ "${TEMPO_RUN_INDEPENDENT_HAAR_QIM_CONTROL:-0}" == "1" ]]; then
        VIDEO_COMPARE_METHODS='proposal,dct_dm_qim_chen2001,haar_hd_svd_qim_adapter2024'
        VIDEO_COMPARE_DIR="$REPORT_ROOT/video_with_independent_haar_control"
      fi
      "$PYTHON_BIN" scripts/run_multiwatermark_video.py \
        --watermark-dir "$WATERMARK_DIR" --proposal-config "$SELECTED_CONFIG" \
        --out-dir "$VIDEO_COMPARE_DIR" --input "$FIRST_CORPUS_VIDEO" \
        --methods "$VIDEO_COMPARE_METHODS" \
        --max-frames "$VIDEO_COMPARE_FRAMES" --warmup-frames 2 \
        --seed "$SEED" --wm-size "$WM_SIZE" --codec "$VIDEO_CODEC" || \
        echo 'Real-video method comparison incomplete; inspect its output.' >&2
    fi
    stage "9A. VERIFY WALL-CLOCK THROUGHPUT DENOMINATORS"
    THROUGHPUT_ARGS=()
    if [[ "$MODE" == "full" ]]; then THROUGHPUT_ARGS+=(--require-720p); fi
    "$PYTHON_BIN" scripts/reviewer_throughput_audit.py \
      --stream-dir "$REPORT_ROOT/live_stream" \
      --out-dir "$REPORT_ROOT/throughput_audit" "${THROUGHPUT_ARGS[@]}" || \
      echo 'Throughput audit FAILED; not counted as complete.' >&2
  else
    echo 'GENUINE VIDEO SUITE FAILED: retaining logs/partial evidence and continuing image reviewer tests.' >&2
    printf '%s\n' 'real_video_or_stream_execution_failed' > "$REPORT_ROOT/video_execution_incomplete.txt"
  fi
fi


stage "9F. RENDER MANUSCRIPT PDF FIGURES FROM REAL EVIDENCE"
"$PYTHON_BIN" scripts/generate_manuscript_figures.py \
  --results-root "$REPORT_ROOT" --host-dir "$HOST_DIR" \
  --watermark-dir "$WATERMARK_DIR" --config "$SELECTED_CONFIG" \
  --out-dir "$REPORT_ROOT/manuscript_figures" || \
    echo "Figure-generation execution error; see diagnostics" >&2

stage "10. EXPORT EVIDENCE AND PACKAGE RESULTS"
"$PYTHON_BIN" scripts/export_method_evidence.py --out-dir "$REPORT_ROOT/method_evidence"
"$PYTHON_BIN" -m pip freeze > "$REPORT_ROOT/pip_freeze.txt" || true
cat > "$REPORT_ROOT/run_manifest.txt" <<EOF
mode=$MODE
python=$PYTHON_BIN
seed=$SEED
watermark_size=$WM_SIZE
host_rule=all BMP except names containing lena or lenna
watermark_rule=both PNG watermarks; quick mode is a smoke subset
hyperparameter_attack_preset=$HP_ATTACK_PRESET
image_attack_preset=$IMAGE_ATTACK_PRESET
internal_attack_manifest=configs/attacks/reviewer_unified_46.json
internal_attack_conditions=46_including_clean
independent_external_kodak_profile=unified109_or_basic
independent_official_mmwatermark_profile=upstream_catalog
official_mmwatermark_enabled=${TEMPO_OFFICIAL_MMWATERMARK:-0}
video_frames=$VIDEO_FRAMES
video_codec=$VIDEO_CODEC
stream_frames=$STREAM_FRAMES
stream_resolutions=$STREAM_RESOLUTIONS
stream_channels=$VIDEO_STREAM_CHANNELS
config_mode=${TEMPO_CONFIG_MODE:-manuscript}
selected_config=$SELECTED_CONFIG
development_hosts=$DEV_HOST_DIR
heldout_test_hosts=$TEST_HOST_DIR
stream_payload_bits=$((WM_SIZE * WM_SIZE))
stream_pacing=realtime
stream_input=verified_public_downloads
video_corpus_provenance=$VIDEO_CORPUS_PATH
video_source_manifest=$VIDEO_CORPUS_MANIFEST
EOF
"$PYTHON_BIN" scripts/reviewer_completeness.py --results-dir "$REPORT_ROOT" --mode "$MODE" || echo "Completeness diagnostic unavailable; inspect logs" >&2
"$PYTHON_BIN" scripts/package_results.py \
  --results-dir "$REPORT_ROOT" \
  --output "$REPORT_ROOT/watermark_full_reports.zip" \
  --csv-output "$REPORT_ROOT/watermark_all_csv.zip"

"$PYTHON_BIN" - "$REPORT_ROOT" "$MASTER_ZIP" <<'PY'
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import sys

root = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2]).resolve()
with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            archive.write(path, Path("reports") / path.relative_to(root))
print(destination)
PY
sha256sum "$MASTER_ZIP" | tee "$REPORT_ROOT/master_zip_sha256.txt"

# No scientific acceptance gate: incomplete or negative scientific outcomes
# stay in the CSVs and diagnostic ledger, without suppressing the result ZIP.

stage "11. COMPLETE"
echo "Results directory: $REPORT_ROOT"
echo "Master ZIP: $MASTER_ZIP"
echo "Image summary: $REPORT_ROOT/all_methods_all_metrics/summary_by_method_watermark.csv"
echo "Attack summary: $REPORT_ROOT/all_methods_all_metrics/summary_by_attack.csv"
echo "Video summary: $REPORT_ROOT/video/all_watermarks_video_summary.csv"
echo "Per-watermark stream summaries: $REPORT_ROOT/live_stream/*/ALL_STREAM_realtime_summary.csv"
