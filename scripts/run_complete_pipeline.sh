#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-quick}"
shift || true
PYTHON="${PYTHON:-python3}"
OUT="${OUT_DIR:-reports/complete_${MODE}}"
CONFIG="${PROPOSAL_CONFIG:-configs/proposal_frozen.json}"
DATA_DIR="${DATA_DIR:-data/user_dataset}"
RUN_COMMON="${RUN_COMMON:-0}"
RUN_VIDEO="${RUN_VIDEO:-0}"
RUN_STREAM="${RUN_STREAM:-0}"

export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

mkdir -p "$OUT" "$(dirname "$CONFIG")"
if [[ ! -f "$CONFIG" ]]; then
  "$PYTHON" - <<PY
from realtime_watermark.proposals.jilp_qim.config import DEFAULT_CONFIG
DEFAULT_CONFIG.save("$CONFIG")
PY
fi

JUNIT="$OUT/junit.xml"
"$PYTHON" -m pytest -q --junitxml="$JUNIT"
"$PYTHON" scripts/generate_validation_report.py --junit "$JUNIT" --out-dir "$OUT/validation" --exit-code 0

NATIVE_ARGS=()
if [[ "$MODE" != "full" ]]; then
  NATIVE_ARGS+=(--quick)
fi
"$PYTHON" scripts/run_paper_native_benchmarks.py \
  --data-dir "$DATA_DIR/host" --out "$OUT/paper_native" "${NATIVE_ARGS[@]}"

METRIC_ARGS=(--data-dir "$DATA_DIR/host" --watermark "$DATA_DIR/watermark/wm.png" --out "$OUT/all_metrics")
if [[ "$MODE" == "full" ]]; then
  METRIC_ARGS+=(--attack-preset reviewer46)
else
  METRIC_ARGS+=(--attack-preset reviewer46 --host-limit 1)
fi
"$PYTHON" scripts/run_all_metrics.py "${METRIC_ARGS[@]}"

# Proposal ablation is retained, but baseline comparison is paper-native.
if [[ "$MODE" == "full" ]]; then
  "$PYTHON" scripts/proposal_structural_ablation.py \
    --config "$CONFIG" --out-dir "$OUT/structural_ablation" \
    --attack-preset reviewer46 --max-hosts 3 --calibration-host-count 2 --calibration-iterations 6
fi

# Optional legacy/common protocol. It is no longer the primary evidence and is
# not run unless explicitly requested.
if [[ "$RUN_COMMON" == "1" ]]; then
  "$PYTHON" scripts/run_watermark_commands.py dataset-image512-all \
    --data-dir "$DATA_DIR" --workdir "$OUT/common_protocol" \
    --proposal-config "$CONFIG" --attack-preset reviewer46
fi

if [[ "$RUN_VIDEO" == "1" ]]; then
  "$PYTHON" scripts/run_watermark_commands.py video-realtime \
    --workdir "$OUT/video_benchmark" --proposal-config "$CONFIG" --max-frames 300
fi

if [[ "$RUN_STREAM" == "1" ]]; then
  "$PYTHON" scripts/live_stream_benchmark.py \
    --workdir "$OUT/live_stream" --proposal-config "$CONFIG" \
    --watermark "$DATA_DIR/watermark/wm.png" --wm-size 16 \
    --frames 120 --resolutions 512x512 --target-fps 30 \
    --channels clean,jpeg70 --pace-mode virtual
fi

"$PYTHON" scripts/export_method_evidence.py --out-dir "$OUT/evidence"
"$PYTHON" scripts/package_results.py --results-dir "$OUT" \
  --output "$OUT/watermark_reports.zip" --csv-output "$OUT/watermark_csv.zip"

echo "Complete paper-native pipeline finished: $OUT"
echo "Optional common protocol: RUN_COMMON=1 ./run.sh full"
echo "Optional video protocol: RUN_VIDEO=1 ./run.sh full"
echo "Optional live-stream protocol: RUN_STREAM=1 ./run.sh full"
