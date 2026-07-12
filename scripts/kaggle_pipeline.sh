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

if [[ "$MODE" == "quick" ]]; then
  HP_ATTACK_PRESET="${HP_ATTACK_PRESET:-smoke}"
  IMAGE_ATTACK_PRESET="${IMAGE_ATTACK_PRESET:-smoke}"
  VIDEO_FRAMES="${VIDEO_FRAMES:-24}"
  VIDEO_WARMUP_FRAMES="${VIDEO_WARMUP_FRAMES:-2}"
  HOST_LIMIT="${HOST_LIMIT:-2}"
  ABLATION_HOST_LIMIT="${ABLATION_HOST_LIMIT:-1}"
else
  HP_ATTACK_PRESET="${HP_ATTACK_PRESET:-standard}"
  IMAGE_ATTACK_PRESET="${IMAGE_ATTACK_PRESET:-stress}"
  VIDEO_FRAMES="${VIDEO_FRAMES:-300}"
  VIDEO_WARMUP_FRAMES="${VIDEO_WARMUP_FRAMES:-10}"
  HOST_LIMIT="${HOST_LIMIT:-0}"
  ABLATION_HOST_LIMIT="${ABLATION_HOST_LIMIT:-0}"
fi

rm -rf "$REPORT_ROOT" "$SELECTED_DATA_DIR" "$MASTER_ZIP"
mkdir -p "$REPORT_ROOT"
printf "Kaggle pipeline started: %s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$REPORT_ROOT/full_run.log"

stage() {
  echo
  echo "======================================================================"
  echo "$1"
  echo "======================================================================"
}

stage "1. VERIFY SOURCE"
"$PYTHON_BIN" scripts/verify_source_manifest.py \
  --allow-missing --report-dir "$REPORT_ROOT/manifest"

stage "2. SELECT ALL BMP HOSTS EXCEPT LENA AND BOTH PNG WATERMARKS"
"$PYTHON_BIN" scripts/select_kaggle_dataset.py \
  --source-data data/user_dataset \
  --output "$SELECTED_DATA_DIR" \
  --manifest-dir "$REPORT_ROOT"
HOST_DIR="$SELECTED_DATA_DIR/host"
WATERMARK_DIR="$SELECTED_DATA_DIR/watermark"

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
"$PYTHON_BIN" scripts/search_proposal_hyperparameters.py \
  --host-dir "$HOST_DIR" \
  --watermark-dir "$WATERMARK_DIR" \
  --base-config configs/proposal_frozen.json \
  --selected-config "$SELECTED_CONFIG" \
  --out-dir "$REPORT_ROOT/hyperparameter_search" \
  --mode "$MODE" \
  --seed "$SEED" \
  --wm-size "$WM_SIZE" \
  --search-hosts "$HP_HOSTS" \
  --attack-preset "$HP_ATTACK_PRESET"

stage "5. RUN PAPER-NATIVE BASELINE TRACKS"
NATIVE_ARGS=()
if [[ "$MODE" == "quick" ]]; then
  NATIVE_ARGS+=(--quick)
fi
"$PYTHON_BIN" scripts/run_paper_native_benchmarks.py \
  --data-dir "$HOST_DIR" \
  --out "$REPORT_ROOT/paper_native" \
  "${NATIVE_ARGS[@]}"

stage "6. RUN ALL IMAGE METHODS ON BOTH WATERMARKS"
METRIC_ARGS=(
  --host-dir "$HOST_DIR"
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
  --host-dir "$HOST_DIR"
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
if [[ -n "$VIDEO_INPUT" ]]; then
  ABLATION_ARGS+=(--video-input "$VIDEO_INPUT")
fi
"$PYTHON_BIN" scripts/run_multiwatermark_ablation.py "${ABLATION_ARGS[@]}"

stage "8. RUN FULL VIDEO BENCHMARK FOR BOTH WATERMARKS"
VIDEO_ARGS=(
  --watermark-dir "$WATERMARK_DIR"
  --proposal-config "$SELECTED_CONFIG"
  --out-dir "$REPORT_ROOT/video"
  --seed "$SEED"
  --wm-size "$WM_SIZE"
  --max-frames "$VIDEO_FRAMES"
  --warmup-frames "$VIDEO_WARMUP_FRAMES"
  --codec "$VIDEO_CODEC"
)
if [[ -n "$VIDEO_INPUT" ]]; then
  VIDEO_ARGS+=(--input "$VIDEO_INPUT")
fi
"$PYTHON_BIN" scripts/run_multiwatermark_video.py "${VIDEO_ARGS[@]}"

stage "9. EXPORT EVIDENCE AND PACKAGE RESULTS"
"$PYTHON_BIN" scripts/export_method_evidence.py --out-dir "$REPORT_ROOT/method_evidence"
"$PYTHON_BIN" -m pip freeze > "$REPORT_ROOT/pip_freeze.txt" || true
cat > "$REPORT_ROOT/run_manifest.txt" <<EOF
mode=$MODE
python=$PYTHON_BIN
seed=$SEED
watermark_size=$WM_SIZE
host_rule=all BMP except names containing lena or lenna
watermark_rule=exactly both PNG files
hyperparameter_attack_preset=$HP_ATTACK_PRESET
image_attack_preset=$IMAGE_ATTACK_PRESET
video_frames=$VIDEO_FRAMES
video_codec=$VIDEO_CODEC
selected_config=$SELECTED_CONFIG
EOF
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

stage "10. COMPLETE"
echo "Results directory: $REPORT_ROOT"
echo "Master ZIP: $MASTER_ZIP"
echo "Image summary: $REPORT_ROOT/all_methods_all_metrics/summary_by_method_watermark.csv"
echo "Attack summary: $REPORT_ROOT/all_methods_all_metrics/summary_by_attack.csv"
echo "Video summary: $REPORT_ROOT/video/all_watermarks_video_summary.csv"
