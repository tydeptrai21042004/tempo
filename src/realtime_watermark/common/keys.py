from __future__ import annotations

from typing import Any

import numpy as np

from ..core import make_block_indices, make_freq_indices
from .types import WatermarkKey

def _shape2(x: Any) -> tuple[int, int]:
    return (int(x[0]), int(x[1]))

def _shape3(x: Any) -> tuple[int, int, int]:
    return (int(x[0]), int(x[1]), int(x[2]))

def _payload_len(key: WatermarkKey) -> int:
    wh, ww = _shape2(key.watermark_shape)
    return int(wh * ww)

def _regenerate_block_indices_from_key(key: WatermarkKey, image_shape: tuple[int, ...]) -> np.ndarray:
    """Return the keyed block schedule.

    Old JSON keys may contain a serialized schedule. New keys do not store the
    schedule and regenerate it from the compact key: seed, original host shape,
    watermark shape, block size, and repeat.
    """
    if key.schedule:
        return np.asarray(key.schedule, dtype=np.int32)
    host_h, host_w, _ = _shape3(key.host_shape)
    bs = int(key.params["block_size"])
    h_crop = host_h - (host_h % bs)
    w_crop = host_w - (host_w % bs)
    return make_block_indices(h_crop, w_crop, bs, _payload_len(key), int(key.repeat), int(key.seed))

def _regenerate_freq_indices_from_key(key: WatermarkKey) -> np.ndarray:
    """Return the keyed DFT-coordinate schedule for the Hamidi-style baseline."""
    if key.schedule:
        return np.asarray(key.schedule, dtype=np.int32)
    host_h, host_w, _ = _shape3(key.host_shape)
    return make_freq_indices(host_h, host_w, _payload_len(key), int(key.repeat), int(key.seed))

def used_units_from_key(key: WatermarkKey | dict[str, Any]) -> int:
    """Number of embedded blocks/frequency coefficients without serializing the schedule."""
    if isinstance(key, dict):
        key = WatermarkKey(**key)
    bits_per_vector = max(1, int(key.params.get("bits_per_vector", 1)))
    total_bits = int(_payload_len(key) * int(key.repeat))
    return int((total_bits + bits_per_vector - 1) // bits_per_vector)

__all__ = ["_shape2", "_shape3", "_payload_len", "_regenerate_block_indices_from_key", "_regenerate_freq_indices_from_key", "used_units_from_key"]
