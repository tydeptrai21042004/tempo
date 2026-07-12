from __future__ import annotations

from typing import Any

import numpy as np

from ..common.block_dct import dct_blocks_from_rgb, dct_blocks_to_rgb, rgb_to_dct_blocks
from ..common.dct_vectors import MID_FREQUENCY_32
from ..common.dither import binary_dither_decode, binary_dither_embed, binary_dither_pair
from ..common.embedding import _vote_bits
from ..common.keys import _regenerate_block_indices_from_key
from ..common.types import MethodRef, WatermarkKey
from ..core import make_block_indices

METHOD_ID = "dct_stdm_qim_chen2001"
BLOCK_SIZE = 8
COEFFICIENTS = MID_FREQUENCY_32
DEFAULT_STEP = 36.0
SPREAD_STREAM_OFFSET = 27103
DITHER_STREAM_OFFSET = 27191

METHOD_REF = MethodRef(
    id=METHOD_ID,
    display_name="DCT spread-transform dither modulation (Chen--Wornell)",
    paper=(
        "B. Chen and G. W. Wornell, Quantization Index Modulation: A Class of "
        "Provably Good Methods for Digital Watermarking and Information Embedding, "
        "IEEE Transactions on Information Theory 47(4), 2001."
    ),
    url="https://doi.org/10.1109/18.923725",
    implementation_note=(
        "Paper-derived spread-transform dither modulation. Each selected 8x8 "
        "luminance-DCT block is projected onto a seeded unit-norm 32-dimensional "
        "spread vector; the scalar projection is pseudorandomly dither-quantized, "
        "and the quantization displacement is spread back along that vector."
    ),
)


def _capacity(host_shape: tuple[int, ...]) -> int:
    return (int(host_shape[0]) // BLOCK_SIZE) * (int(host_shape[1]) // BLOCK_SIZE)


def _spread_vectors(count: int, seed: int, length: int = len(COEFFICIENTS)) -> np.ndarray:
    rng = np.random.default_rng(int(seed) + SPREAD_STREAM_OFFSET)
    vectors = rng.choice(np.asarray([-1.0, 1.0]), size=(int(count), int(length))).astype(np.float64)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


def _dithers(count: int, seed: int, step: float) -> tuple[np.ndarray, np.ndarray]:
    return binary_dither_pair(count, seed, step, stream_offset=DITHER_STREAM_OFFSET)


def stdm_embed_vectors(
    vectors: np.ndarray,
    bits: np.ndarray,
    directions: np.ndarray,
    step: float,
    dither_zero: np.ndarray,
    dither_one: np.ndarray,
) -> np.ndarray:
    x = np.asarray(vectors, dtype=np.float64)
    v = np.asarray(directions, dtype=np.float64)
    b = np.asarray(bits, dtype=np.uint8)
    if x.shape != v.shape or x.shape[0] != b.size:
        raise ValueError("vectors, directions, and bits have incompatible shapes")
    projection = np.sum(x * v, axis=1)
    target = binary_dither_embed(projection, b, dither_zero, dither_one, step)
    return x + (target - projection)[:, None] * v


def stdm_decode_vectors(
    vectors: np.ndarray,
    directions: np.ndarray,
    step: float,
    dither_zero: np.ndarray,
    dither_one: np.ndarray,
) -> np.ndarray:
    x = np.asarray(vectors, dtype=np.float64)
    v = np.asarray(directions, dtype=np.float64)
    if x.shape != v.shape:
        raise ValueError("vectors and directions must have identical shapes")
    projection = np.sum(x * v, axis=1)
    return binary_dither_decode(projection, dither_zero, dither_one, step)


def prepare_embedding(
    host_rgb: np.ndarray,
    watermark_binary: np.ndarray,
    *,
    seed: int = 2026,
    repeat: int | str = 1,
    step: float | None = None,
    method_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del method_params
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
    delta = float(DEFAULT_STEP if step is None else step)
    coeffs, y, cb, cr, h_crop, w_crop = rgb_to_dct_blocks(host, BLOCK_SIZE)
    indices = make_block_indices(h_crop, w_crop, BLOCK_SIZE, payload, repeat_i, int(seed)).astype(np.int32)
    bits = np.tile(wm.ravel(), repeat_i).astype(np.uint8)
    directions = _spread_vectors(indices.size, int(seed))
    d0, d1 = _dithers(indices.size, int(seed), delta)
    return {
        "coeffs": coeffs,
        "y": y,
        "cb": cb,
        "cr": cr,
        "h_crop": h_crop,
        "w_crop": w_crop,
        "indices": indices,
        "bits": bits,
        "directions": directions,
        "dither_zero": d0,
        "dither_one": d1,
        "repeat": repeat_i,
        "step": delta,
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
    delta = float(prepared["step"] if step is None else step)
    coeffs = np.asarray(prepared["coeffs"], dtype=np.float64).copy()
    indices = np.asarray(prepared["indices"], dtype=np.int32)
    coords = tuple(tuple(int(x) for x in c) for c in COEFFICIENTS)
    vectors = np.stack([coeffs[indices, u, v] for u, v in coords], axis=1)
    directions = np.asarray(prepared["directions"], dtype=np.float64)
    if np.isclose(delta, float(prepared["step"]), atol=1e-12, rtol=0):
        d0 = np.asarray(prepared["dither_zero"], dtype=np.float64)
        d1 = np.asarray(prepared["dither_one"], dtype=np.float64)
    else:
        d0, d1 = _dithers(indices.size, int(prepared["seed"]), delta)
    marked_vectors = stdm_embed_vectors(
        vectors, np.asarray(prepared["bits"], dtype=np.uint8), directions,
        delta, d0, d1,
    )
    for column, (u, v) in enumerate(coords):
        coeffs[indices, u, v] = marked_vectors[:, column]
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
        step=delta,
        arnold_iter=0,
        arnold_period=1,
        threshold=127,
        params={
            "domain": "8x8 luminance DCT spread-transform vector",
            "block_size": BLOCK_SIZE,
            "coefficients": [list(x) for x in COEFFICIENTS],
            "spread_length": len(COEFFICIENTS),
            "spread_stream_offset": SPREAD_STREAM_OFFSET,
            "dither_stream_offset": DITHER_STREAM_OFFSET,
            "spread_storage": "regenerate_from_seed",
            "dither_storage": "regenerate_from_seed",
            "capacity_bits": _capacity(host.shape),
            "schedule_storage": "regenerate_from_seed",
            "paper_equation": "project on v; dither-quantize projection; spread displacement along v",
        },
        schedule=[],
        fully_blind=True,
        side_information="seed, step, DCT vector, payload shape; spread and dither regenerated from seed",
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
    coords = tuple(tuple(int(x) for x in c) for c in key.params["coefficients"])
    vectors = np.stack([coeffs[indices, u, v] for u, v in coords], axis=1)
    directions = _spread_vectors(indices.size, int(key.seed), len(coords))
    d0, d1 = _dithers(indices.size, int(key.seed), float(key.step))
    raw = stdm_decode_vectors(vectors, directions, float(key.step), d0, d1)
    payload = int(np.prod(key.watermark_shape))
    rec = _vote_bits(raw, payload).reshape(key.watermark_shape)
    return (rec * 255).astype(np.uint8)


__all__ = [
    "METHOD_ID", "METHOD_REF", "BLOCK_SIZE", "COEFFICIENTS", "DEFAULT_STEP",
    "stdm_embed_vectors", "stdm_decode_vectors", "prepare_embedding",
    "embed_prepared", "embed", "extract",
]
