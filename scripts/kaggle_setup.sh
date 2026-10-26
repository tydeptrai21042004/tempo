#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${KAGGLE_PYTHON:-$(command -v python3 || command -v python || true)}"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "ERROR: Python was not found." >&2
  exit 1
fi

# Kaggle already ships compatible scientific packages. Install only imports
# that are actually missing; never upgrade the notebook's global stack.
"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import importlib
import subprocess
import sys

requirements = {
    "numpy": "numpy>=1.23",
    "PIL": "Pillow>=9",
    "numba": "numba>=0.58",
    "skimage": "scikit-image>=0.20",
    "scipy": "scipy>=1.9",
    "cv2": "opencv-python-headless>=4.8",
    "pytest": "pytest>=7",
    "matplotlib": "matplotlib>=3.6",
}
missing = []
for module, requirement in requirements.items():
    try:
        importlib.import_module(module)
    except Exception:
        missing.append(requirement)
if missing:
    print("Installing missing packages only:", " ".join(missing))
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", *missing]
    )
else:
    print("All required Kaggle packages are already available.")
PY

printf '%s\n' "$PYTHON_BIN" > "$ROOT/.kaggle_python"

export PYTHONPATH="$ROOT/src:$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

"$PYTHON_BIN" - <<'PY'
import cv2
import numba
import numpy
import PIL
import pytest
import scipy
import skimage
import realtime_watermark

print("Python environment ready")
print("NumPy:", numpy.__version__)
print("SciPy:", scipy.__version__)
print("Pillow:", PIL.__version__)
print("scikit-image:", skimage.__version__)
print("Numba:", numba.__version__)
print("OpenCV:", cv2.__version__)
print("pytest:", pytest.__version__)
print("Package:", realtime_watermark.__file__)
PY

echo "Repository manifest is generated at execution; no stale manifest enforcement."

echo "Kaggle setup complete. Python: $PYTHON_BIN"
