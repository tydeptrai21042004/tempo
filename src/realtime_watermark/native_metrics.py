from __future__ import annotations

import math
import numpy as np


def mse(reference: np.ndarray, test: np.ndarray) -> float:
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(test, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.mean((a - b) ** 2))


def prd(reference: np.ndarray, test: np.ndarray) -> float:
    """Percentage residual difference as used by Mao et al. (ratio, not percent)."""
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(test, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    numerator = float(np.sum((a - b) ** 2))
    denominator = float(np.sum(a ** 2))
    if denominator <= 0.0:
        return 0.0 if numerator <= 0.0 else float("inf")
    return float(math.sqrt(numerator / denominator))


def symbol_error_rate(reference_symbols: np.ndarray, estimated_symbols: np.ndarray) -> float:
    a = np.asarray(reference_symbols).reshape(-1)
    b = np.asarray(estimated_symbols).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"symbol shape mismatch: {a.shape} vs {b.shape}")
    return float(np.mean(a != b)) if a.size else 0.0


def q_function(x: np.ndarray | float) -> np.ndarray | float:
    from scipy.special import erfc

    value = 0.5 * erfc(np.asarray(x, dtype=np.float64) / math.sqrt(2.0))
    return float(value) if np.ndim(value) == 0 else value


def db10(value: float) -> float:
    if value <= 0:
        return float("-inf")
    return float(10.0 * math.log10(value))


def linear_from_db(value_db: float) -> float:
    return float(10.0 ** (float(value_db) / 10.0))


def bit_error_probability(reference_bits: np.ndarray, estimated_bits: np.ndarray) -> float:
    a = np.asarray(reference_bits, dtype=np.uint8).reshape(-1)
    b = np.asarray(estimated_bits, dtype=np.uint8).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"bit shape mismatch: {a.shape} vs {b.shape}")
    return float(np.mean(a != b)) if a.size else 0.0


def normalized_correlation(reference: np.ndarray, estimated: np.ndarray, *, center: bool = False) -> float:
    a = np.asarray(reference, dtype=np.float64).reshape(-1)
    b = np.asarray(estimated, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"correlation shape mismatch: {a.shape} vs {b.shape}")
    if center:
        a = a - float(np.mean(a))
        b = b - float(np.mean(b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 0.0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.clip(np.dot(a, b) / den, -1.0, 1.0))


def bits_to_symbols(bits: np.ndarray, width: int = 4) -> np.ndarray:
    data = np.asarray(bits, dtype=np.uint8).reshape(-1)
    w = int(width)
    if w <= 0 or w > 16:
        raise ValueError("symbol width must be between 1 and 16 bits")
    padding = (-data.size) % w
    if padding:
        data = np.pad(data, (0, padding), constant_values=0)
    groups = data.reshape(-1, w).astype(np.uint32)
    weights = (1 << np.arange(w - 1, -1, -1, dtype=np.uint32))
    return (groups * weights[None, :]).sum(axis=1).astype(np.uint32)


def binary_symbol_error_rate(reference_bits: np.ndarray, estimated_bits: np.ndarray, width: int = 4) -> float:
    return symbol_error_rate(bits_to_symbols(reference_bits, width), bits_to_symbols(estimated_bits, width))


def image_distortion_statistics(reference: np.ndarray, test: np.ndarray) -> dict[str, float]:
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(test, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    squared = (a - b) ** 2
    return {
        "mse": float(np.mean(squared)),
        "distortion_std_squared": float(np.std(squared)),
        "distortion_max_squared": float(np.max(squared)),
    }


__all__ = [
    "mse", "prd", "symbol_error_rate", "q_function", "db10",
    "linear_from_db", "bit_error_probability", "normalized_correlation",
    "bits_to_symbols", "binary_symbol_error_rate", "image_distortion_statistics",
]
