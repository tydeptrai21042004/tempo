from __future__ import annotations

import numpy as np


def binary_dither_pair(count: int, seed: int, step: float, *, stream_offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate the binary dither pair described by Chen--Wornell.

    The zero-bit dither is pseudorandom and uniform on [-Delta/2, Delta/2).
    The one-bit dither is displaced by exactly Delta/2 while remaining in the
    same principal interval.  Regeneration depends only on count, seed, step,
    and stream_offset, so no explicit dither sequence needs to be serialized.
    """
    delta = float(step)
    if delta <= 0:
        raise ValueError("step must be positive")
    n = int(count)
    if n < 0:
        raise ValueError("count must be nonnegative")
    rng = np.random.default_rng(int(seed) + int(stream_offset))
    d0 = rng.uniform(-0.5 * delta, 0.5 * delta, size=n).astype(np.float64)
    d1 = np.where(d0 < 0.0, d0 + 0.5 * delta, d0 - 0.5 * delta)
    return d0, d1.astype(np.float64)


def quantize_with_dither(values: np.ndarray, dither: np.ndarray, step: float) -> np.ndarray:
    delta = float(step)
    if delta <= 0:
        raise ValueError("step must be positive")
    x = np.asarray(values, dtype=np.float64)
    d = np.asarray(dither, dtype=np.float64)
    if d.shape not in {x.shape, ()}:
        raise ValueError(f"dither shape {d.shape} is incompatible with values shape {x.shape}")
    return delta * np.rint((x - d) / delta) + d


def binary_dither_embed(
    values: np.ndarray,
    bits: np.ndarray,
    dither_zero: np.ndarray,
    dither_one: np.ndarray,
    step: float,
) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    b = np.asarray(bits, dtype=np.uint8)
    d0 = np.asarray(dither_zero, dtype=np.float64)
    d1 = np.asarray(dither_one, dtype=np.float64)
    if x.shape != b.shape or x.shape != d0.shape or x.shape != d1.shape:
        raise ValueError("values, bits, and dither arrays must have identical shapes")
    d = np.where(b > 0, d1, d0)
    return quantize_with_dither(x, d, step)


def binary_dither_decode(
    values: np.ndarray,
    dither_zero: np.ndarray,
    dither_one: np.ndarray,
    step: float,
) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    d0 = np.asarray(dither_zero, dtype=np.float64)
    d1 = np.asarray(dither_one, dtype=np.float64)
    if x.shape != d0.shape or x.shape != d1.shape:
        raise ValueError("values and dither arrays must have identical shapes")
    q0 = quantize_with_dither(x, d0, step)
    q1 = quantize_with_dither(x, d1, step)
    return (np.abs(x - q1) <= np.abs(x - q0)).astype(np.uint8)


__all__ = [
    "binary_dither_pair",
    "quantize_with_dither",
    "binary_dither_embed",
    "binary_dither_decode",
]
