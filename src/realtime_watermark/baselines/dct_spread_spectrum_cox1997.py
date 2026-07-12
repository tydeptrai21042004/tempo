from __future__ import annotations

from typing import Any

import numpy as np
from scipy.fft import dctn, idctn

from ..common.color import _rgb_to_ycbcr_float, _ycbcr_to_rgb_float
from ..common.types import MethodRef, WatermarkKey

METHOD_ID = "dct_spread_spectrum_cox1997"
DEFAULT_ALPHA = 0.10
NATIVE_WATERMARK_LENGTH = 1000

METHOD_REF = MethodRef(
    id=METHOD_ID,
    display_name="DCT informed spread-spectrum watermarking (Cox et al.)",
    paper=(
        "I. J. Cox, J. Kilian, F. T. Leighton, and T. Shamoon, Secure Spread "
        "Spectrum Watermarking for Multimedia, IEEE Transactions on Image "
        "Processing 6(12), 1997."
    ),
    url="https://doi.org/10.1109/83.650120",
    implementation_note=(
        "Informed/non-blind multiplicative embedding in the largest-magnitude "
        "whole-image luminance-DCT coefficients, excluding DC. The detector "
        "uses the corresponding original coefficients, as required by the paper."
    ),
    fully_blind=False,
)


def _dct2(gray: np.ndarray) -> np.ndarray:
    return dctn(np.asarray(gray, dtype=np.float64), type=2, norm="ortho")


def _idct2(coeffs: np.ndarray) -> np.ndarray:
    return idctn(np.asarray(coeffs, dtype=np.float64), type=2, norm="ortho")


def _top_indices(coeffs: np.ndarray, count: int) -> np.ndarray:
    flat = np.abs(np.asarray(coeffs, dtype=np.float64)).ravel().copy()
    if flat.size <= 1:
        raise ValueError("host is too small for Cox DCT embedding")
    flat[0] = -np.inf
    n = int(count)
    if n <= 0 or n >= flat.size:
        raise ValueError(f"invalid watermark length {n} for {flat.size - 1} AC coefficients")
    chosen = np.argpartition(flat, -n)[-n:]
    # Stable paper-style ordering: descending original coefficient magnitude.
    chosen = chosen[np.argsort(flat[chosen])[::-1]]
    return chosen.astype(np.int64)


def _sequence_from_bits(bits01: np.ndarray, seed: int) -> np.ndarray:
    """Continuous common-adapter sequence carrying the supplied bit signs.

    The native Cox track uses an unconstrained i.i.d. Gaussian sequence. For the
    common binary API, Gaussian magnitudes are combined with the requested bit
    signs so BER/NC can also be reported without changing the multiplicative
    insertion equation.
    """
    bits = np.asarray(bits01, dtype=np.uint8).reshape(-1)
    rng = np.random.default_rng(int(seed) + 63017)
    magnitudes = np.maximum(np.abs(rng.normal(0.0, 1.0, size=bits.size)), 0.35)
    return np.where(bits > 0, magnitudes, -magnitudes).astype(np.float64)


def cox_similarity(reference: np.ndarray, extracted: np.ndarray, *, center: bool = False, sign_only: bool = False) -> float:
    x = np.asarray(reference, dtype=np.float64).reshape(-1)
    y = np.asarray(extracted, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("Cox watermark vectors must have the same shape")
    if center:
        y = y - float(np.mean(y))
    if sign_only:
        y = np.sign(y)
    den = float(np.linalg.norm(y))
    if den <= 0.0:
        return 0.0
    return float(np.dot(x, y) / den)


def _extract_sequence(image_rgb: np.ndarray, key: WatermarkKey) -> np.ndarray:
    y, _cb, _cr = _rgb_to_ycbcr_float(np.asarray(image_rgb, dtype=np.uint8))
    coeffs = _dct2(y).ravel()
    indices = np.asarray(key.params["coefficient_indices"], dtype=np.int64)
    original = np.asarray(key.params["original_coefficients"], dtype=np.float64)
    observed = coeffs[indices]
    denom = float(key.step) * original
    safe = np.where(np.abs(denom) > 1e-12, denom, np.where(denom >= 0, 1e-12, -1e-12))
    return (observed - original) / safe


def prepare_embedding(
    host_rgb: np.ndarray,
    watermark_binary: np.ndarray,
    *,
    seed: int = 2026,
    repeat: int | str = 1,
    step: float | None = None,
    method_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del repeat, method_params
    host = np.asarray(host_rgb, dtype=np.uint8)
    wm = (np.asarray(watermark_binary) > 127).astype(np.uint8)
    y, cb, cr = _rgb_to_ycbcr_float(host)
    coeffs = _dct2(y)
    indices = _top_indices(coeffs, wm.size)
    original = coeffs.ravel()[indices].astype(np.float64)
    sequence = _sequence_from_bits(wm.ravel(), int(seed))
    alpha = float(DEFAULT_ALPHA if step is None else step)
    return {
        "y": y,
        "cb": cb,
        "cr": cr,
        "coeffs": coeffs,
        "indices": indices,
        "original": original,
        "sequence": sequence,
        "alpha": alpha,
        "seed": int(seed),
    }


def embed_prepared(
    host_rgb: np.ndarray,
    watermark_binary: np.ndarray,
    prepared: dict[str, Any],
    *,
    step: float | None = None,
    method_params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, WatermarkKey]:
    del method_params
    host = np.asarray(host_rgb, dtype=np.uint8)
    wm = np.asarray(watermark_binary, dtype=np.uint8)
    alpha = float(prepared["alpha"] if step is None else step)
    coeffs = np.asarray(prepared["coeffs"], dtype=np.float64).copy()
    indices = np.asarray(prepared["indices"], dtype=np.int64)
    original = np.asarray(prepared["original"], dtype=np.float64)
    sequence = np.asarray(prepared["sequence"], dtype=np.float64)
    flat = coeffs.ravel()
    flat[indices] = original * (1.0 + alpha * sequence)
    y_marked = _idct2(coeffs)
    watermarked = _ycbcr_to_rgb_float(y_marked, prepared["cb"], prepared["cr"])
    key = WatermarkKey(
        method_id=METHOD_ID,
        host_shape=tuple(int(x) for x in host.shape),
        watermark_shape=tuple(int(x) for x in wm.shape),
        seed=int(prepared["seed"]),
        repeat=1,
        step=alpha,
        arnold_iter=0,
        arnold_period=1,
        threshold=127,
        params={
            "domain": "whole-image luminance DCT",
            "watermark_length": int(wm.size),
            "coefficient_selection": "largest-magnitude AC DCT coefficients",
            "coefficient_indices": indices.astype(int).tolist(),
            "original_coefficients": original.astype(float).tolist(),
            "reference_watermark_sequence": sequence.astype(float).tolist(),
            "insertion_rule": "v_i_prime=v_i*(1+alpha*x_i)",
            "native_similarity": "dot(x,x_hat)/norm(x_hat)",
            "capacity_bits": int(coeffs.size - 1),
            "schedule_storage": "explicit_top_magnitude_indices",
        },
        schedule=[],
        fully_blind=False,
        side_information=(
            "non-blind/informed: original selected DCT coefficients, selected indices, "
            "candidate watermark sequence, alpha, and payload shape"
        ),
    )
    return watermarked, key


def embed(
    host_rgb: np.ndarray,
    watermark_binary: np.ndarray,
    *,
    seed: int = 2026,
    repeat: int | str = 1,
    step: float | None = None,
    method_params: dict[str, Any] | None = None,
):
    prepared = prepare_embedding(
        host_rgb, watermark_binary, seed=seed, repeat=repeat, step=step, method_params=method_params
    )
    return embed_prepared(host_rgb, watermark_binary, prepared, step=step, method_params=method_params)


def extract(image_rgb: np.ndarray, key: WatermarkKey | dict[str, Any]) -> np.ndarray:
    if isinstance(key, dict):
        key = WatermarkKey(**key)
    sequence = _extract_sequence(image_rgb, key)
    bits = (sequence >= 0.0).astype(np.uint8).reshape(key.watermark_shape)
    return (bits * 255).astype(np.uint8)


def detection_statistics(image_rgb: np.ndarray, key: WatermarkKey | dict[str, Any]) -> dict[str, float]:
    if isinstance(key, dict):
        key = WatermarkKey(**key)
    extracted = _extract_sequence(image_rgb, key)
    reference = np.asarray(key.params["reference_watermark_sequence"], dtype=np.float64)
    return {
        "cox_similarity": cox_similarity(reference, extracted),
        "cox_similarity_centered": cox_similarity(reference, extracted, center=True),
        "cox_similarity_sign": cox_similarity(reference, extracted, center=True, sign_only=True),
        "extracted_sequence_mean": float(np.mean(extracted)),
        "extracted_sequence_std": float(np.std(extracted)),
    }


def native_embed_gray(
    host_gray: np.ndarray,
    watermark_sequence: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Paper-native grayscale Cox embedding with an i.i.d. Gaussian watermark."""
    host = np.asarray(host_gray, dtype=np.uint8)
    if host.ndim != 2:
        raise ValueError("native Cox embedding expects a 2-D grayscale host")
    sequence = np.asarray(watermark_sequence, dtype=np.float64).reshape(-1)
    coeffs = _dct2(host)
    indices = _top_indices(coeffs, sequence.size)
    original = coeffs.ravel()[indices].astype(np.float64)
    marked_coeffs = coeffs.copy()
    marked_coeffs.ravel()[indices] = original * (1.0 + float(alpha) * sequence)
    marked = np.clip(np.rint(_idct2(marked_coeffs)), 0, 255).astype(np.uint8)
    return marked, {
        "alpha": float(alpha),
        "indices": indices.astype(int).tolist(),
        "original_coefficients": original.astype(float).tolist(),
        "watermark_sequence": sequence.astype(float).tolist(),
        "host_shape": list(host.shape),
    }


def native_extract_sequence_gray(image_gray: np.ndarray, key: dict[str, Any]) -> np.ndarray:
    image = np.asarray(image_gray, dtype=np.uint8)
    if image.ndim != 2:
        raise ValueError("native Cox extraction expects a 2-D grayscale image")
    if tuple(image.shape) != tuple(int(x) for x in key["host_shape"]):
        raise ValueError("native Cox extraction requires registration to the original image shape")
    coeffs = _dct2(image).ravel()
    indices = np.asarray(key["indices"], dtype=np.int64)
    original = np.asarray(key["original_coefficients"], dtype=np.float64)
    denom = float(key["alpha"]) * original
    safe = np.where(np.abs(denom) > 1e-12, denom, np.where(denom >= 0, 1e-12, -1e-12))
    return (coeffs[indices] - original) / safe


def native_detection_statistics_gray(image_gray: np.ndarray, key: dict[str, Any]) -> dict[str, float]:
    extracted = native_extract_sequence_gray(image_gray, key)
    reference = np.asarray(key["watermark_sequence"], dtype=np.float64)
    return {
        "cox_similarity": cox_similarity(reference, extracted),
        "cox_similarity_centered": cox_similarity(reference, extracted, center=True),
        "cox_similarity_sign": cox_similarity(reference, extracted, center=True, sign_only=True),
        "sequence_correlation": float(np.corrcoef(reference, extracted)[0, 1]) if np.std(extracted) > 0 else 0.0,
        "extracted_sequence_mean": float(np.mean(extracted)),
        "extracted_sequence_std": float(np.std(extracted)),
    }


__all__ = [
    "METHOD_ID", "METHOD_REF", "DEFAULT_ALPHA", "NATIVE_WATERMARK_LENGTH",
    "cox_similarity", "prepare_embedding", "embed_prepared", "embed", "extract",
    "detection_statistics", "native_embed_gray", "native_extract_sequence_gray",
    "native_detection_statistics_gray",
]
