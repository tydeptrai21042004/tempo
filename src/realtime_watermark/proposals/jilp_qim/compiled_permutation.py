"""Cached exact Arnold permutation plans for the binary watermark.

The map is a permutation of pixel *indices*, not an interpolation.  Plans
are generated once per (size, iterations), then reused in embedding and
blind extraction.  It reproduces ``core.arnold_transform`` for any square
watermark, including sizes whose period is not in the lookup table.
"""
from __future__ import annotations

from functools import lru_cache
import numpy as np


def _square_uint8(mat: np.ndarray) -> np.ndarray:
    src = np.asarray(mat, dtype=np.uint8)
    if src.ndim != 2 or src.shape[0] != src.shape[1] or src.shape[0] == 0:
        raise ValueError("Arnold transform requires a nonempty square watermark")
    return src


@lru_cache(maxsize=64)
def _arnold_inverse_lookup(n: int, iterations: int) -> np.ndarray:
    """For each output location, return its source position after k steps."""
    n, iterations = int(n), int(iterations)
    if n <= 0 or iterations < 0:
        raise ValueError("invalid Arnold shape or iteration count")
    row, col = np.indices((n, n), dtype=np.int32)
    src = np.arange(n*n, dtype=np.int32).reshape(n, n)
    rr = (row + col) % n
    cc = (row + 2 * col) % n
    for _ in range(iterations):
        nxt = np.empty_like(src)
        nxt[rr, cc] = src
        src = nxt
    src = np.ascontiguousarray(src.ravel())
    src.flags.writeable = False
    return src


def permute_arnold_exact(mat01: np.ndarray, iterations: int) -> np.ndarray:
    src = _square_uint8(mat01)
    lookup = _arnold_inverse_lookup(src.shape[0], int(iterations))
    return np.take(src.ravel(), lookup).reshape(src.shape)


def inverse_arnold_exact(mat01: np.ndarray, iterations: int, period: int) -> np.ndarray:
    src = _square_uint8(mat01)
    if int(period) <= 0:
        raise ValueError("Arnold period must be positive")
    inverse_steps = (-int(iterations)) % int(period)
    return permute_arnold_exact(src, inverse_steps)


__all__ = ["permute_arnold_exact", "inverse_arnold_exact"]
