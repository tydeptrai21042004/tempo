from __future__ import annotations

import numpy as np

from realtime_watermark.baselines.dct_stdm_qim_chen2001 import (
    COEFFICIENTS,
    stdm_decode_vectors,
    stdm_embed_vectors,
)
from realtime_watermark.common.dither import binary_dither_pair
from realtime_watermark.core import bit_accuracy
from realtime_watermark.registry import embed, extract


def _unit_rows(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_stdm_projection_is_quantized_to_requested_codebook():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(50, 12))
    v = _unit_rows(rng.normal(size=(50, 12)))
    bits = rng.integers(0, 2, size=50, dtype=np.uint8)
    d0, d1 = binary_dither_pair(50, seed=15, step=18.0)
    y = stdm_embed_vectors(x, bits, v, 18.0, d0, d1)
    assert np.array_equal(stdm_decode_vectors(y, v, 18.0, d0, d1), bits)


def test_stdm_changes_only_spread_direction():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(20, 9))
    v = _unit_rows(rng.normal(size=(20, 9)))
    bits = rng.integers(0, 2, size=20, dtype=np.uint8)
    d0, d1 = binary_dither_pair(20, seed=10, step=12.0)
    y = stdm_embed_vectors(x, bits, v, 12.0, d0, d1)
    displacement = y - x
    residual = displacement - np.sum(displacement * v, axis=1)[:, None] * v
    assert np.max(np.abs(residual)) < 1e-10


def test_stdm_declared_vector_has_useful_length():
    assert len(COEFFICIENTS) == 32
    assert (0, 0) not in COEFFICIENTS


def test_stdm_clean_image_roundtrip(host128, wm16):
    marked, key = embed("dct_stdm_qim_chen2001", host128, wm16, seed=19, repeat=1, step=36.0)
    assert bit_accuracy(wm16, extract(marked, key)) == 1.0
    assert key.params["spread_storage"] == "regenerate_from_seed"
    assert key.params["dither_storage"] == "regenerate_from_seed"


def test_stdm_is_deterministic(host128, wm16):
    a, _ = embed("dct_stdm_qim_chen2001", host128, wm16, seed=19, repeat=1, step=36.0)
    b, _ = embed("dct_stdm_qim_chen2001", host128, wm16, seed=19, repeat=1, step=36.0)
    assert np.array_equal(a, b)
