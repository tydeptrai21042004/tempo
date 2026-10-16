#!/usr/bin/env bash
# A single entry point: setup dependencies, download true public video inputs,
# execute all paper experiments, verify frame-level and real-time evidence.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-4}"
"${PYTHON:-python3}" -m pip install -r requirements-kaggle.txt
bash scripts/kaggle_pipeline.sh "${1:-full}"
