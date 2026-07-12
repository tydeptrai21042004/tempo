#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
COMMAND="${1:-help}"
shift || true
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="${PYTHON_FALLBACK:-python3}"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

case "$COMMAND" in
  setup)
    if [[ -d /kaggle/working ]]; then
      exec scripts/kaggle_setup.sh "$@"
    fi
    BASE_PY="${PYTHON_FALLBACK:-python3}"
    rm -rf .venv
    if ! "$BASE_PY" -m venv .venv; then
      BOOTSTRAP="${TMPDIR:-/tmp}/quynh-virtualenv-bootstrap"
      rm -rf "$BOOTSTRAP"
      mkdir -p "$BOOTSTRAP"
      "$BASE_PY" -m pip install --quiet --target "$BOOTSTRAP" "virtualenv>=20.26,<21"
      PYTHONPATH="$BOOTSTRAP" "$BASE_PY" -m virtualenv --python "$BASE_PY" .venv
    fi
    .venv/bin/python -m pip install --upgrade pip setuptools wheel
    .venv/bin/python -m pip install -r requirements-dev.txt -c constraints.txt
    .venv/bin/python -m pip install -e . --no-deps
    ;;
  test)
    mkdir -p validation/current
    set +e
    "$PYTHON" -m pytest -q --junitxml=validation/current/junit.xml "$@"
    STATUS=$?
    set -e
    "$PYTHON" scripts/generate_validation_report.py \
      --junit validation/current/junit.xml --out-dir validation/current \
      --exit-code "$STATUS" --command "python -m pytest -q"
    exit "$STATUS"
    ;;
  quick|full)
    exec scripts/run_complete_pipeline.sh "$COMMAND" "$@"
    ;;
  native)
    exec "$PYTHON" scripts/run_paper_native_benchmarks.py "$@"
    ;;
  metrics)
    exec "$PYTHON" scripts/run_all_metrics.py "$@"
    ;;
  image)
    exec "$PYTHON" scripts/run_watermark_commands.py dataset-image512-all "$@"
    ;;
  image-one)
    exec "$PYTHON" scripts/run_watermark_commands.py image512-all "$@"
    ;;
  video)
    exec "$PYTHON" scripts/run_watermark_commands.py video-realtime "$@"
    ;;
  stream)
    exec "$PYTHON" scripts/live_stream_benchmark.py "$@"
    ;;
  structural)
    exec "$PYTHON" scripts/proposal_structural_ablation.py "$@"
    ;;
  video-ablation)
    exec "$PYTHON" scripts/proposal_video_ablation.py "$@"
    ;;
  kaggle-setup)
    exec scripts/kaggle_setup.sh "$@"
    ;;
  kaggle-quick)
    exec ./kaggle.sh quick "$@"
    ;;
  kaggle-full)
    exec ./kaggle.sh full "$@"
    ;;
  clean-check)
    exec "$PYTHON" scripts/check_clean_package.py
    ;;
  clean)
    rm -rf reports validation/current .pytest_cache
    find . -type d -name __pycache__ -prune -exec rm -rf {} +
    find . -type f \( -name '*.pyc' -o -name '*.nbc' -o -name '*.nbi' \) -delete
    ;;
  help|-h|--help)
    cat <<'HELP'
Usage: ./run.sh COMMAND [options]

Commands:
  setup            Install the frozen development environment
  test             Run all equation, integration, fairness, video, and cleanliness tests
  quick            Run a short complete six-method pipeline
  full             Run the complete six-method image/video pipeline
  native           Run paper-native attacks and metrics without PSNR matching
  metrics          Report the complete generic metric schema for every image method
  image            Run optional common-protocol held-out image benchmarking
  image-one        Run one matched-distortion image benchmark
  video            Run the common framewise video benchmark
  stream           Run the bounded-queue causal live-stream benchmark
  structural       Run image structural ablation
  video-ablation   Compare hard and confidence-weighted temporal voting on video
  kaggle-setup     Check/install only missing Kaggle dependencies
  kaggle-quick     Run a short Kaggle end-to-end smoke pipeline
  kaggle-full      Run all tests, search, image/video benchmarks, and ablations
  clean-check      Enforce the clean-source policy
  clean            Remove generated reports and caches
HELP
    ;;
  *)
    echo "Unknown command: $COMMAND" >&2
    exec "$0" help
    ;;
esac
