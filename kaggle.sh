#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMAND="${1:-full}"
shift || true
case "$COMMAND" in
  setup)
    exec "$ROOT/scripts/kaggle_setup.sh" "$@"
    ;;
  quick|full)
    "$ROOT/scripts/kaggle_setup.sh"
    exec "$ROOT/scripts/kaggle_pipeline.sh" "$COMMAND" "$@"
    ;;
  stream)
    "$ROOT/scripts/kaggle_setup.sh"
    PYTHON_BIN="${PYTHON:-$(cat "$ROOT/.kaggle_python" 2>/dev/null || command -v python3)}"
    export PYTHONPATH="$ROOT/src:$ROOT"
    exec "$PYTHON_BIN" "$ROOT/scripts/live_stream_benchmark.py" "$@"
    ;;
  *)
    echo "Usage: ./kaggle.sh [setup|quick|full|stream]" >&2
    exit 2
    ;;
esac
