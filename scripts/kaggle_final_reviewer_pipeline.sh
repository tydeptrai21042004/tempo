#!/usr/bin/env bash
# Kaggle end-to-end: existing full TEMPO evidence, then final reviewer-only
# pair of disjoint baseline protocols. Does NOT invoke learned-native attacks.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON:-python3}"
if [[ -d /kaggle/working ]]; then WORK=/kaggle/working; else WORK="${WORK_ROOT:-$ROOT}"; fi
REPORT="${KAGGLE_REPORT_ROOT:-$WORK/tempo-manuscript-results}"
SELECTED="${KAGGLE_SELECTED_DATA_DIR:-$WORK/tempo-selected-data}"
OUTPUT="${TEMPO_FINAL_REPORT:-$WORK/TEMPO_FINAL_REVIEWER_EVIDENCE}"
MODE="${1:-full}"
if [[ "$MODE" != full && "$MODE" != quick ]]; then echo 'Mode must be full or quick' >&2; exit 2; fi
export TEMPO_MATH_REFINEMENT=0
export TEMPO_RUN_INDEPENDENT_HAAR_QIM_CONTROL=0
export TEMPO_RUN_PAPER_NATIVE_SEPARATELY=1
export TEMPO_OFFICIAL_MMWATERMARK="${TEMPO_OFFICIAL_MMWATERMARK:-0}"
export PYTHONPATH="$ROOT/src:$ROOT"
# The older pipeline creates validated source manifest, train/test split and
# all established ablations. Its diagnostic non-paper controls are excluded.
if [[ "${TEMPO_SKIP_LEGACY:-0}" != 1 ]]; then
  bash scripts/kaggle_pipeline.sh "$MODE"
fi
args=(
  --test-dir "$REPORT/holdout/test"
  --development-dir "$REPORT/holdout/development"
  --watermark-dir "$SELECTED/watermark"
  --config "$REPORT/configs/proposal_selected_all_watermarks.json"
  --video-corpus "$REPORT/video_sources/corpus.json"
  --output "$OUTPUT" --frames "${VIDEO_FRAMES:-300}"
  --continue-on-error --resume
)
if [[ "$MODE" == quick ]]; then args+=(--quick); fi
status=0
"$PYTHON_BIN" scripts/run_final_reviewer_protocol.py "${args[@]}" || status=$?
# Keep the partial numerical evidence downloadable even if a stage fails.
"$PYTHON_BIN" - "$OUTPUT" <<'PY'
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import json,sys
root=Path(sys.argv[1]);archive=root.with_suffix('.zip')
allow={'.csv','.tsv','.json','.jsonl','.txt','.log','.md','.pdf'}
with ZipFile(archive,'w',ZIP_DEFLATED,compresslevel=5) as z:
    for item in sorted(root.rglob('*')):
        if item.is_file() and item.suffix.lower() in allow:
            z.write(item,item.relative_to(root))
print('Final reviewer numerical evidence ZIP:',archive)
PY
exit "$status"
