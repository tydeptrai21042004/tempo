from __future__ import annotations

from typing import Any

import numpy as np

from ..common.block_dct import dct_blocks_from_rgb, dct_blocks_to_rgb, rgb_to_dct_blocks
from ..common.dct_vectors import UPPER_HIGH_FREQUENCY_28
from ..common.embedding import _vote_bits
from ..common.keys import _regenerate_block_indices_from_key
from ..common.types import MethodRef, WatermarkKey
from ..core import make_block_indices

METHOD_ID = "dct_iss_malvar2003"
BLOCK_SIZE = 8
DEFAULT_ALPHA = 8.0
DEFAULT_LAMBDA = 0.90
COEFFICIENTS = UPPER_HIGH_FREQUENCY_28
PN_STREAM_OFFSET = 3803

METHOD_REF = MethodRef(
    id=METHOD_ID,
    display_name="DCT limited-distortion improved spread spectrum (Malvar--Florencio)",
    paper=(
        "H. S. Malvar and D. A. F. Florencio, Improved Spread Spectrum: A New "
        "Modulation Technique for Robust Watermarking, IEEE Transactions on "
        "Signal Processing 51(4), 2003."
    ),
    url="https://doi.org/10.1109/TSP.2003.809385",
    implementation_note=(
        "Paper Section V-A maximum-useful-distortion ISS built from the linear "
        "ISS displacement and unchanged correlation-sign detector. The declared "
        "image adapter uses a seeded unit-norm 28-dimensional upper-mid/high-"
        "frequency 8x8 luminance-DCT vector and lambda=0.90."
    ),
)


def _capacity(host_shape: tuple[int, ...]) -> int:
    return (int(host_shape[0]) // BLOCK_SIZE) * (int(host_shape[1]) // BLOCK_SIZE)


def _pn_sequences(count: int, seed: int, length: int = len(COEFFICIENTS)) -> np.ndarray:
    rng = np.random.default_rng(int(seed) + PN_STREAM_OFFSET)
    u = rng.choice(np.asarray([-1.0, 1.0]), size=(int(count), int(length))).astype(np.float64)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return u


def iss_embed_vectors(
    vectors: np.ndarray,
    bits: np.ndarray,
    pn: np.ndarray,
    alpha: float,
    cancellation: float,
) -> np.ndarray:
    x = np.asarray(vectors, dtype=np.float64)
    b = np.where(np.asarray(bits, dtype=np.uint8) > 0, 1.0, -1.0)
    u = np.asarray(pn, dtype=np.float64)
    if x.shape != u.shape or x.shape[0] != b.size:
        raise ValueError("vectors, bits, and PN sequences have incompatible shapes")
    lam = float(cancellation)
    if not (0.0 <= lam <= 1.0):
        raise ValueError("cancellation must lie in [0,1]")
    projection = np.sum(x * u, axis=1)
    mu = float(alpha) * b - lam * projection
    return x + mu[:, None] * u


def iss_embed_vectors_maximum_useful(
    vectors: np.ndarray,
    bits: np.ndarray,
    pn: np.ndarray,
    alpha: float,
    cancellation: float,
) -> np.ndarray:
    """Paper Section V-A maximum-useful-distortion ISS.

    The linear ISS displacement is applied only while the host projection lies
    in the useful interval.  If the unmodified host already has the requested
    sign with at least the target margin, or lies so far in the wrong direction
    that linear ISS would still decode incorrectly, the displacement is zero.
    This preserves the paper detector while avoiding unnecessary/unbounded
    modifications from the plain linear approximation.
    """
    x = np.asarray(vectors, dtype=np.float64)
    b = np.where(np.asarray(bits, dtype=np.uint8) > 0, 1.0, -1.0)
    u = np.asarray(pn, dtype=np.float64)
    if x.shape != u.shape or x.shape[0] != b.size:
        raise ValueError("vectors, bits, and PN sequences have incompatible shapes")
    lam = float(cancellation)
    if not (0.0 <= lam < 1.0):
        raise ValueError("maximum-useful-distortion ISS requires cancellation in [0,1)")
    projection = np.sum(x * u, axis=1)
    signed_projection = b * projection
    displacement = float(alpha) * b - lam * projection
    lower = -float(alpha) / max(1e-12, 1.0 - lam)
    useful = (signed_projection < float(alpha)) & (signed_projection > lower)
    displacement = np.where(useful, displacement, 0.0)
    return x + displacement[:, None] * u


def iss_decode_vectors(vectors: np.ndarray, pn: np.ndarray) -> np.ndarray:
    r = np.sum(np.asarray(vectors, dtype=np.float64) * np.asarray(pn, dtype=np.float64), axis=1)
    return (r >= 0.0).astype(np.uint8)


def prepare_embedding(
    host_rgb: np.ndarray,
    watermark_binary: np.ndarray,
    *,
    seed: int = 2026,
    repeat: int | str = 1,
    step: float | None = None,
    method_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = dict(method_params or {})
    host = np.asarray(host_rgb, dtype=np.uint8)
    wm = (np.asarray(watermark_binary) > 127).astype(np.uint8)
    payload = int(wm.size)
    capacity = _capacity(host.shape)
    if isinstance(repeat, str):
        repeat_i = max(1, capacity // payload) if repeat.lower() == "auto" else int(repeat)
    else:
        repeat_i = int(repeat)
    if payload * repeat_i > capacity:
        raise ValueError(f"capacity too small: need {payload * repeat_i}, have {capacity}")
    alpha = float(DEFAULT_ALPHA if step is None else step)
    cancellation = float(params.get("cancellation", DEFAULT_LAMBDA))
    if not (0.0 <= cancellation <= 1.0):
        raise ValueError("cancellation must lie in [0,1]")
    coeffs, y, cb, cr, h_crop, w_crop = rgb_to_dct_blocks(host, BLOCK_SIZE)
    indices = make_block_indices(h_crop, w_crop, BLOCK_SIZE, payload, repeat_i, int(seed)).astype(np.int32)
    bits = np.tile(wm.ravel(), repeat_i).astype(np.uint8)
    pn = _pn_sequences(indices.size, int(seed))
    return {
        "coeffs": coeffs,
        "y": y,
        "cb": cb,
        "cr": cr,
        "h_crop": h_crop,
        "w_crop": w_crop,
        "indices": indices,
        "bits": bits,
        "pn": pn,
        "repeat": repeat_i,
        "alpha": alpha,
        "cancellation": cancellation,
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
    host = np.asarray(host_rgb, dtype=np.uint8)
    wm = np.asarray(watermark_binary, dtype=np.uint8)
    alpha = float(prepared["alpha"] if step is None else step)
    cancellation = float(dict(method_params or {}).get("cancellation", prepared["cancellation"]))
    coeffs = np.asarray(prepared["coeffs"], dtype=np.float64).copy()
    indices = np.asarray(prepared["indices"], dtype=np.int32)
    coords = tuple(tuple(int(v) for v in c) for c in COEFFICIENTS)
    vectors = np.stack([coeffs[indices, u, v] for u, v in coords], axis=1)
    embedded = iss_embed_vectors_maximum_useful(
        vectors, prepared["bits"], prepared["pn"], alpha, cancellation
    )
    for column, (u, v) in enumerate(coords):
        coeffs[indices, u, v] = embedded[:, column]
    watermarked = dct_blocks_to_rgb(
        coeffs, prepared["y"], prepared["cb"], prepared["cr"],
        prepared["h_crop"], prepared["w_crop"], BLOCK_SIZE,
    )
    key = WatermarkKey(
        method_id=METHOD_ID,
        host_shape=tuple(int(x) for x in host.shape),
        watermark_shape=tuple(int(x) for x in wm.shape),
        seed=int(prepared["seed"]),
        repeat=int(prepared["repeat"]),
        step=alpha,
        arnold_iter=0,
        arnold_period=1,
        threshold=127,
        params={
            "domain": "8x8 luminance DCT upper-mid/high-frequency coefficient vector",
            "block_size": BLOCK_SIZE,
            "coefficients": [list(x) for x in COEFFICIENTS],
            "pn_length": len(COEFFICIENTS),
            "pn_stream_offset": PN_STREAM_OFFSET,
            "cancellation_lambda": cancellation,
            "capacity_bits": _capacity(host.shape),
            "schedule_storage": "regenerate_from_seed",
            "pn_storage": "regenerate_from_seed",
            "paper_equation": "Section V-A maximum-useful-distortion linear ISS; b_hat=sign(<y,u>)",
            "distortion_policy": "maximum_useful_interval",
            "transform_adapter_disclosure": (
                "ISS defines the modulation after a suitable transform; this repository "
                "declares a 28-dimensional upper-mid/high-frequency block-DCT adapter"
            ),
        },
        schedule=[],
        fully_blind=True,
        side_information="seed, alpha, lambda, DCT coefficient vector, payload shape",
    )
    return watermarked, key


def embed(
    host_rgb: np.ndarray,
    watermark_binary: np.ndarray,
    *,
    seed: int = 2026,
    repeat: int | str = "auto",
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
    coeffs, _h_crop, _w_crop = dct_blocks_from_rgb(image_rgb, int(key.params["block_size"]))
    indices = _regenerate_block_indices_from_key(key, np.asarray(image_rgb).shape)
    coords = tuple(tuple(int(v) for v in c) for c in key.params["coefficients"])
    vectors = np.stack([coeffs[indices, u, v] for u, v in coords], axis=1)
    pn = _pn_sequences(indices.size, int(key.seed), len(coords))
    raw = iss_decode_vectors(vectors, pn)
    payload = int(np.prod(key.watermark_shape))
    rec = _vote_bits(raw, payload).reshape(key.watermark_shape)
    return (rec * 255).astype(np.uint8)


__all__ = [
    "METHOD_ID", "METHOD_REF", "BLOCK_SIZE", "COEFFICIENTS", "DEFAULT_ALPHA", "DEFAULT_LAMBDA",
    "iss_embed_vectors", "iss_embed_vectors_maximum_useful", "iss_decode_vectors", "prepare_embedding", "embed_prepared", "embed", "extract",
]
