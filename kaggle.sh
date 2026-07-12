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
  *)
    echo "Usage: ./kaggle.sh [setup|quick|full]" >&2
    exit 2
    ;;
esac
