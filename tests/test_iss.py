from __future__ import annotations

import numpy as np
import pytest

from realtime_watermark.baselines.dct_iss_malvar2003 import (
    COEFFICIENTS,
    DEFAULT_LAMBDA,
    iss_decode_vectors,
    iss_embed_vectors, iss_embed_vectors_maximum_useful,
)
from realtime_watermark.core import bit_accuracy
from realtime_watermark.registry import embed, extract


def _unit_rows(array):
    x = np.asarray(array, dtype=float)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_linear_iss_cancels_host_projection_at_lambda_one():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(32, 7))
    u = _unit_rows(rng.normal(size=(32, 7)))
    bits = rng.integers(0, 2, size=32, dtype=np.uint8)
    y = iss_embed_vectors(x, bits, u, alpha=5.0, cancellation=1.0)
    expected = np.where(bits > 0, 5.0, -5.0)
    assert np.allclose(np.sum(y * u, axis=1), expected, atol=1e-10)


def test_lambda_zero_reduces_to_conventional_spread_spectrum():
    x = np.array([[1.0, 2.0]])
    u = np.array([[0.6, 0.8]])
    y = iss_embed_vectors(x, np.array([1], np.uint8), u, alpha=3.0, cancellation=0.0)
    assert np.allclose(y, x + 3.0 * u)


def test_iss_detector_is_correlation_sign():
    u = np.eye(2)
    assert np.array_equal(iss_decode_vectors(np.array([[2.0, 0.0], [0.0, -3.0]]), u), [1, 0])


def test_iss_rejects_invalid_cancellation():
    with pytest.raises(ValueError):
        iss_embed_vectors(np.zeros((1, 2)), np.array([0]), np.ones((1, 2)), 1.0, -0.1)
    with pytest.raises(ValueError):
        iss_embed_vectors(np.zeros((1, 2)), np.array([0]), np.ones((1, 2)), 1.0, 1.1)


def test_iss_uses_longer_medium_frequency_adapter():
    assert len(COEFFICIENTS) == 28
    assert (0, 0) not in COEFFICIENTS
    assert DEFAULT_LAMBDA < 1.0


def test_iss_clean_image_roundtrip(host128, wm16):
    marked, key = embed(
        "dct_iss_malvar2003", host128, wm16, seed=8, repeat=1, step=12.0,
        method_params={"cancellation": DEFAULT_LAMBDA},
    )
    assert bit_accuracy(wm16, extract(marked, key)) == 1.0


def test_iss_key_regenerates_pn(host128, wm16):
    _marked, key = embed("dct_iss_malvar2003", host128, wm16, seed=8, repeat=1, step=12.0)
    assert key.schedule == []
    assert key.params["pn_storage"] == "regenerate_from_seed"
    assert "pn" not in key.params


def test_iss_is_deterministic(host128, wm16):
    a, _ = embed("dct_iss_malvar2003", host128, wm16, seed=22, repeat=1, step=12.0)
    b, _ = embed("dct_iss_malvar2003", host128, wm16, seed=22, repeat=1, step=12.0)
    assert np.array_equal(a, b)


def test_maximum_useful_distortion_skips_unnecessary_change():
    u = np.asarray([[1.0, 0.0]], dtype=np.float64)
    x = np.asarray([[20.0, 3.0]], dtype=np.float64)
    out = iss_embed_vectors_maximum_useful(x, np.asarray([1], dtype=np.uint8), u, 8.0, 0.9)
    assert np.allclose(out, x)
