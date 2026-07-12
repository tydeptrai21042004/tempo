from __future__ import annotations

"""Timing summaries shared by image and video validation code.

The helpers deliberately return distributions rather than a single optimistic
FPS number. Unit tests validate the schema and formulas; performance values are
reported by validation commands and are never unit-test pass criteria.
"""

from dataclasses import asdict, dataclass
from typing import Iterable
import os
import platform
import sys

import numpy as np


@dataclass(frozen=True)
class TimingDistribution:
    count: int
    mean_ms: float
    median_ms: float
    std_ms: float
    iqr_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    median_fps: float

    def to_dict(self, prefix: str = "") -> dict[str, float | int]:
        values = asdict(self)
        return {f"{prefix}{key}": value for key, value in values.items()}


def summarize_ms(values: Iterable[float]) -> TimingDistribution:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        nan = float("nan")
        return TimingDistribution(0, nan, nan, nan, nan, nan, nan, nan, nan, nan)
    q25, q75 = np.percentile(arr, [25.0, 75.0])
    median = float(np.median(arr))
    return TimingDistribution(
        count=int(arr.size),
        mean_ms=float(np.mean(arr)),
        median_ms=median,
        std_ms=float(np.std(arr, ddof=0)),
        iqr_ms=float(q75 - q25),
        p95_ms=float(np.percentile(arr, 95.0)),
        p99_ms=float(np.percentile(arr, 99.0)),
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
        median_fps=float(1000.0 / max(median, 1e-12)),
    )


def payload_throughput_bps(payload_bits: int, elapsed_ms: float) -> float:
    return float(payload_bits) * 1000.0 / max(float(elapsed_ms), 1e-12)


def megapixels_per_second(width: int, height: int, elapsed_ms: float) -> float:
    megapixels = float(width) * float(height) / 1_000_000.0
    return megapixels * 1000.0 / max(float(elapsed_ms), 1e-12)


def environment_snapshot() -> dict[str, str | int]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": int(os.cpu_count() or 1),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", "unset"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
        "numexpr_num_threads": os.environ.get("NUMEXPR_NUM_THREADS", "unset"),
    }


__all__ = [
    "TimingDistribution",
    "summarize_ms",
    "payload_throughput_bps",
    "megapixels_per_second",
    "environment_snapshot",
]
