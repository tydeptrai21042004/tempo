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

REPORT_ROOT="${KAGGLE_REPORT_ROOT:-$WORK_ROOT/Quynh-watermark-full-results}"
SELECTED_DATA_DIR="${KAGGLE_SELECTED_DATA_DIR:-$WORK_ROOT/Quynh-watermark-selected-data}"
MASTER_ZIP="${KAGGLE_MASTER_ZIP:-$WORK_ROOT/Quynh-watermark-full-results.zip}"
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
  HP_ATTACK_PRESET="${HP_ATTACK_PRESET:-smoke}"
  IMAGE_ATTACK_PRESET="${IMAGE_ATTACK_PRESET:-smoke}"
  VIDEO_FRAMES="${VIDEO_FRAMES:-24}"
  VIDEO_WARMUP_FRAMES="${VIDEO_WARMUP_FRAMES:-2}"
  HOST_LIMIT="${HOST_LIMIT:-2}"
  ABLATION_HOST_LIMIT="${ABLATION_HOST_LIMIT:-1}"
  STREAM_FRAMES="${STREAM_FRAMES:-30}"
  STREAM_RESOLUTIONS="${STREAM_RESOLUTIONS:-512x512}"
  VIDEO_STREAM_CHANNELS="${VIDEO_STREAM_CHANNELS:-clean,jpeg70,jpeg40}"
else
  HP_ATTACK_PRESET="${HP_ATTACK_PRESET:-standard}"
  IMAGE_ATTACK_PRESET="${IMAGE_ATTACK_PRESET:-stress}"
  VIDEO_FRAMES="${VIDEO_FRAMES:-300}"
  VIDEO_WARMUP_FRAMES="${VIDEO_WARMUP_FRAMES:-10}"
  HOST_LIMIT="${HOST_LIMIT:-0}"
  ABLATION_HOST_LIMIT="${ABLATION_HOST_LIMIT:-0}"
  STREAM_FRAMES="${STREAM_FRAMES:-300}"
  STREAM_RESOLUTIONS="${STREAM_RESOLUTIONS:-512x512,1280x720}"
  VIDEO_STREAM_CHANNELS="${VIDEO_STREAM_CHANNELS:-clean,jpeg90,jpeg70,jpeg40,gaussian5,blur3,drop10}"
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
stage "1. VERIFY SOURCE"
"$PYTHON_BIN" scripts/verify_source_manifest.py \
  --report-dir "$REPORT_ROOT/manifest"

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
if [[ "$TEST_STATUS" -ne 0 ]]; then
  exit "$TEST_STATUS"
fi

stage "4. SEARCH AND FREEZE JILP-QIM CONFIGURATION"
SELECTED_CONFIG="$REPORT_ROOT/configs/proposal_selected_all_watermarks.json"
mkdir -p "$(dirname "$SELECTED_CONFIG")"
"$PYTHON_BIN" scripts/create_reviewer_holdout.py \
  --host-dir "$HOST_DIR" --out-dir "$REPORT_ROOT/holdout" --development-count "$HP_HOSTS"
DEV_HOST_DIR="$REPORT_ROOT/holdout/development"
TEST_HOST_DIR="$REPORT_ROOT/holdout/test"
"$PYTHON_BIN" scripts/search_proposal_hyperparameters.py \
  --host-dir "$DEV_HOST_DIR" \
  --watermark-dir "$WATERMARK_DIR" \
  --base-config configs/proposal_frozen.json \
  --selected-config "$SELECTED_CONFIG" \
  --out-dir "$REPORT_ROOT/hyperparameter_search" \
  --mode "$MODE" \
  --seed "$SEED" \
  --wm-size "$WM_SIZE" \
  --search-hosts "$HP_HOSTS" \
  --external-heldout-dir "$TEST_HOST_DIR" \
  --attack-preset "$HP_ATTACK_PRESET"

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

stage "4A. DOWNLOAD AND VERIFY REAL VIDEO CORPUS (FAIL CLOSED; NO SYNTHETIC FALLBACK)"
VIDEO_CORPUS_MANIFEST="${VIDEO_CORPUS_MANIFEST:-configs/video_evidence_sources.json}"
VIDEO_CACHE_DIR="${VIDEO_CACHE_DIR:-$WORK_ROOT/tempo-video-cache}"
VIDEO_PREPARED_DIR="${VIDEO_PREPARED_DIR:-$WORK_ROOT/tempo-video-prepared}"
VIDEO_CORPUS_PATH="$REPORT_ROOT/video_sources/corpus.json"
"$PYTHON_BIN" scripts/video_evidence_corpus.py \
  --manifest "$VIDEO_CORPUS_MANIFEST" --cache-dir "$VIDEO_CACHE_DIR" \
  --prepared-dir "$VIDEO_PREPARED_DIR" --out "$VIDEO_CORPUS_PATH" \
  --mode "$MODE" --frames "$VIDEO_FRAMES"
FIRST_CORPUS_VIDEO="$($PYTHON_BIN - "$VIDEO_CORPUS_PATH" <<'PYVIDEO'
import json,sys
print(json.load(open(sys.argv[1]))['videos'][0]['prepared_path'])
PYVIDEO
)"
VIDEO_ABLATION_INPUT="${VIDEO_INPUT:-$FIRST_CORPUS_VIDEO}"

stage "5. RUN PAPER-NATIVE BASELINE TRACKS"
NATIVE_ARGS=()
if [[ "$MODE" == "quick" ]]; then
  NATIVE_ARGS+=(--quick)
fi
"$PYTHON_BIN" scripts/run_paper_native_benchmarks.py \
  --data-dir "$TEST_HOST_DIR" \
  --out "$REPORT_ROOT/paper_native" \
  "${NATIVE_ARGS[@]}"

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
ABLATION_ARGS+=(--video-input "$VIDEO_ABLATION_INPUT")
"$PYTHON_BIN" scripts/run_multiwatermark_ablation.py "${ABLATION_ARGS[@]}"

fi  # End of pre-video stages; resume preserves all outputs from steps 1--7.

SUITE_RESUME_ARGS=()
if [[ "$RESUME_FROM_VIDEO" == "1" ]]; then SUITE_RESUME_ARGS+=(--resume); fi
stage "8-9. RUN REAL VIDEO CORPUS, BOTH WATERMARKS AND LIVE DEADLINE TESTS"
"$PYTHON_BIN" scripts/run_video_corpus_suite.py \
  --corpus "$VIDEO_CORPUS_PATH" --watermark-dir "$WATERMARK_DIR" \
  --proposal-config "$SELECTED_CONFIG" --video-out "$REPORT_ROOT/video" \
  --stream-out "$REPORT_ROOT/live_stream" --mode "$MODE" \
  --max-frames "$VIDEO_FRAMES" --stream-frames "$STREAM_FRAMES" \
  --codec "$VIDEO_CODEC" --python "$PYTHON_BIN" \
  --channels "$VIDEO_STREAM_CHANNELS" --resolutions "$STREAM_RESOLUTIONS" "${SUITE_RESUME_ARGS[@]}"

stage "9A. VERIFY WALL-CLOCK THROUGHPUT DENOMINATORS"
THROUGHPUT_ARGS=()
if [[ "$MODE" == "full" ]]; then THROUGHPUT_ARGS+=(--require-720p); fi
"$PYTHON_BIN" scripts/reviewer_throughput_audit.py \
  --stream-dir "$REPORT_ROOT/live_stream" \
  --out-dir "$REPORT_ROOT/throughput_audit" "${THROUGHPUT_ARGS[@]}"

stage "9B. RUN SUPPLEMENTARY REVIEWER EXPERIMENTS ON HELD-OUT HOSTS"
REVIEWER_EXTRA_ARGS=()
if [[ "$MODE" == "quick" ]]; then
  REVIEWER_EXTRA_ARGS+=(--max-hosts 1 --max-watermarks 1)
fi
"$PYTHON_BIN" scripts/reviewer_experiments.py \
  --host-dir "$TEST_HOST_DIR" --watermark-dir "$WATERMARK_DIR" \
  --config "$SELECTED_CONFIG" --out-dir "$REPORT_ROOT/reviewer_experiments" \
  --mode "$MODE" --seed "$SEED" "${REVIEWER_EXTRA_ARGS[@]}"


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
  --seed "$SEED" "${JOINT_ARGS[@]}"

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
  --seed "$SEED" "${GALLERY_ARGS[@]}"

stage "9C. MATCHED-PSNR BASELINES ON HELD-OUT HOSTS"
MATCHED_ARGS=()
if [[ "$MODE" == "quick" ]]; then
  MATCHED_ARGS+=(--host-limit 1 --watermark-limit 1 --attack-preset smoke --timing-repeats 2 --timing-warmups 1)
else
  MATCHED_ARGS+=(--attack-preset stress --timing-repeats 5 --timing-warmups 2)
fi
"$PYTHON_BIN" scripts/run_multiwatermark_fair_benchmark.py \
  --host-dir "$TEST_HOST_DIR" --calibration-host-dir "$DEV_HOST_DIR" \
  --watermark-dir "$WATERMARK_DIR" --proposal-config "$SELECTED_CONFIG" \
  --out-dir "$REPORT_ROOT/matched_psnr" --wm-size "$WM_SIZE" \
  --seed "$SEED" --target-psnr 45 --psnr-tolerance 0.75 "${MATCHED_ARGS[@]}"

stage "9C2. STRICT COMMON-PAIR MATCHED-PSNR EFFECTS"
"$PYTHON_BIN" scripts/reviewer_matched_pairs.py \
  --matched-dir "$REPORT_ROOT/matched_psnr" \
  --out-dir "$REPORT_ROOT/matched_psnr" --seed "$SEED"

stage "9D. HOST-CLUSTERED CONFIDENCE INTERVALS AND PAIRED EFFECTS"
"$PYTHON_BIN" scripts/reviewer_statistics.py \
  --input "$REPORT_ROOT/all_methods_all_metrics/all_watermarks_all_methods_all_metrics.csv" \
  --out-dir "$REPORT_ROOT/statistical_evidence" --seed "$SEED"

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
video_frames=$VIDEO_FRAMES
video_codec=$VIDEO_CODEC
stream_frames=$STREAM_FRAMES
stream_resolutions=$STREAM_RESOLUTIONS
stream_channels=$VIDEO_STREAM_CHANNELS
selected_config=$SELECTED_CONFIG
development_hosts=$DEV_HOST_DIR
heldout_test_hosts=$TEST_HOST_DIR
stream_payload_bits=$((WM_SIZE * WM_SIZE))
stream_pacing=realtime
stream_input=verified_public_downloads
video_corpus_provenance=$VIDEO_CORPUS_PATH
video_source_manifest=$VIDEO_CORPUS_MANIFEST
EOF
"$PYTHON_BIN" scripts/reviewer_completeness.py --results-dir "$REPORT_ROOT" --mode "$MODE"
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

# Preserve all diagnostic CSVs and the ZIP even when a scientifically honest
# reviewer ledger reports unmet claims. Opt in to a nonzero exit for CI gates.
if [[ "${FAIL_ON_INCOMPLETE:-0}" == "1" ]]; then
  "$PYTHON_BIN" scripts/reviewer_completeness.py --results-dir "$REPORT_ROOT" --mode "$MODE" --strict
fi

stage "11. COMPLETE"
echo "Results directory: $REPORT_ROOT"
echo "Master ZIP: $MASTER_ZIP"
echo "Image summary: $REPORT_ROOT/all_methods_all_metrics/summary_by_method_watermark.csv"
echo "Attack summary: $REPORT_ROOT/all_methods_all_metrics/summary_by_attack.csv"
echo "Video summary: $REPORT_ROOT/video/all_watermarks_video_summary.csv"
echo "Per-watermark stream summaries: $REPORT_ROOT/live_stream/*/ALL_STREAM_realtime_summary.csv"
