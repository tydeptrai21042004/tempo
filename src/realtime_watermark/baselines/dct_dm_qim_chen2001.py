from __future__ import annotations

from typing import Any

import numpy as np

from ..common.block_dct import dct_blocks_from_rgb, dct_blocks_to_rgb, rgb_to_dct_blocks
from ..common.dither import binary_dither_decode, binary_dither_embed, binary_dither_pair
from ..common.embedding import _vote_bits
from ..common.keys import _regenerate_block_indices_from_key
from ..common.types import MethodRef, WatermarkKey
from ..core import make_block_indices

METHOD_ID = "dct_dm_qim_chen2001"
BLOCK_SIZE = 8
COEFFICIENT = (3, 2)
DEFAULT_STEP = 44.0
DITHER_STREAM_OFFSET = 17011

METHOD_REF = MethodRef(
    id=METHOD_ID,
    display_name="DCT pseudorandom dither-modulation QIM (Chen--Wornell)",
    paper=(
        "B. Chen and G. W. Wornell, Quantization Index Modulation: A Class of "
        "Provably Good Methods for Digital Watermarking and Information Embedding, "
        "IEEE Transactions on Information Theory 47(4), 2001."
    ),
    url="https://doi.org/10.1109/18.923725",
    implementation_note=(
        "Blind binary dither modulation on one declared mid-frequency coefficient "
        "per selected 8x8 luminance-DCT block. The zero-bit dither is pseudorandom "
        "uniform on [-Delta/2,Delta/2), and the one-bit dither differs by Delta/2, "
        "matching the paper's practical binary dither construction."
    ),
)


def dm_embed_values(
    values: np.ndarray,
    bits: np.ndarray,
    step: float,
    dither_zero: np.ndarray | None = None,
    dither_one: np.ndarray | None = None,
) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    b = np.asarray(bits, dtype=np.uint8)
    if dither_zero is None or dither_one is None:
        dither_zero, dither_one = binary_dither_pair(x.size, 0, step, stream_offset=0)
        dither_zero = dither_zero.reshape(x.shape)
        dither_one = dither_one.reshape(x.shape)
    return binary_dither_embed(x, b, np.asarray(dither_zero), np.asarray(dither_one), step)


def dm_decode_values(
    values: np.ndarray,
    step: float,
    dither_zero: np.ndarray | None = None,
    dither_one: np.ndarray | None = None,
) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if dither_zero is None or dither_one is None:
        dither_zero, dither_one = binary_dither_pair(x.size, 0, step, stream_offset=0)
        dither_zero = dither_zero.reshape(x.shape)
        dither_one = dither_one.reshape(x.shape)
    return binary_dither_decode(x, np.asarray(dither_zero), np.asarray(dither_one), step)


def _capacity(host_shape: tuple[int, ...]) -> int:
    h, w = int(host_shape[0]), int(host_shape[1])
    return (h // BLOCK_SIZE) * (w // BLOCK_SIZE)


def _dithers(count: int, seed: int, step: float) -> tuple[np.ndarray, np.ndarray]:
    return binary_dither_pair(count, seed, step, stream_offset=DITHER_STREAM_OFFSET)


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
    used_step = float(DEFAULT_STEP if step is None else step)
    coeffs, y, cb, cr, h_crop, w_crop = rgb_to_dct_blocks(host, BLOCK_SIZE)
    indices = make_block_indices(h_crop, w_crop, BLOCK_SIZE, payload, repeat_i, int(seed)).astype(np.int32)
    bits = np.tile(wm.ravel(), repeat_i).astype(np.uint8)
    d0, d1 = _dithers(indices.size, int(seed), used_step)
    return {
        "coeffs": coeffs,
        "y": y,
        "cb": cb,
        "cr": cr,
        "h_crop": h_crop,
        "w_crop": w_crop,
        "indices": indices,
        "bits": bits,
        "dither_zero": d0,
        "dither_one": d1,
        "repeat": repeat_i,
        "step": used_step,
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
    used_step = float(prepared["step"] if step is None else step)
    coeffs = np.asarray(prepared["coeffs"], dtype=np.float64).copy()
    indices = np.asarray(prepared["indices"], dtype=np.int32)
    bits = np.asarray(prepared["bits"], dtype=np.uint8)
    if np.isclose(used_step, float(prepared["step"]), atol=1e-12, rtol=0):
        d0 = np.asarray(prepared["dither_zero"], dtype=np.float64)
        d1 = np.asarray(prepared["dither_one"], dtype=np.float64)
    else:
        d0, d1 = _dithers(indices.size, int(prepared["seed"]), used_step)
    u, v = COEFFICIENT
    coeffs[indices, u, v] = dm_embed_values(
        coeffs[indices, u, v], bits, used_step, d0, d1
    )
    watermarked = dct_blocks_to_rgb(
        coeffs,
        prepared["y"], prepared["cb"], prepared["cr"],
        prepared["h_crop"], prepared["w_crop"], BLOCK_SIZE,
    )
    key = WatermarkKey(
        method_id=METHOD_ID,
        host_shape=tuple(int(x) for x in host.shape),
        watermark_shape=tuple(int(x) for x in wm.shape),
        seed=int(prepared["seed"]),
        repeat=int(prepared["repeat"]),
        step=used_step,
        arnold_iter=0,
        arnold_period=1,
        threshold=127,
        params={
            "domain": "8x8 luminance DCT",
            "block_size": BLOCK_SIZE,
            "coefficient": list(COEFFICIENT),
            "quantizer": "binary uniform pseudorandom dither modulation",
            "dither_zero_distribution": "uniform[-Delta/2,Delta/2)",
            "dither_one_rule": "d1=d0+Delta/2 if d0<0 else d0-Delta/2",
            "dither_stream_offset": DITHER_STREAM_OFFSET,
            "dither_storage": "regenerate_from_seed",
            "capacity_bits": _capacity(host.shape),
            "schedule_storage": "regenerate_from_seed",
            "paper_equation": "s(x;m)=q(x+d(m))-d(m); nearest dithered quantizer detection",
        },
        schedule=[],
        fully_blind=True,
        side_information="seed, step, coefficient position, payload shape; dither regenerated from seed",
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
    u, v = (int(x) for x in key.params["coefficient"])
    d0, d1 = _dithers(indices.size, int(key.seed), float(key.step))
    raw = dm_decode_values(coeffs[indices, u, v], float(key.step), d0, d1)
    payload = int(np.prod(key.watermark_shape))
    rec = _vote_bits(raw, payload).reshape(key.watermark_shape)
    return (rec * 255).astype(np.uint8)


__all__ = [
    "METHOD_ID", "METHOD_REF", "BLOCK_SIZE", "COEFFICIENT", "DEFAULT_STEP",
    "DITHER_STREAM_OFFSET", "dm_embed_values", "dm_decode_values",
    "prepare_embedding", "embed_prepared", "embed", "extract",
]
