from __future__ import annotations

import numpy as np
import pytest

from realtime_watermark.baselines.dct_dm_qim_chen2001 import dm_decode_values, dm_embed_values
from realtime_watermark.common.dither import binary_dither_pair
from realtime_watermark.core import bit_accuracy
from realtime_watermark.registry import embed, extract


def test_pseudorandom_dither_pair_matches_paper_rule():
    d0, d1 = binary_dither_pair(1000, seed=31, step=20.0)
    assert np.all(d0 >= -10.0)
    assert np.all(d0 < 10.0)
    assert np.std(d0) > 5.0
    assert np.allclose(np.abs(d1 - d0), 10.0)
    assert np.unique(np.round(d0, 8)).size > 990


def test_dm_equation_roundtrip_for_both_bits():
    x = np.linspace(-200, 200, 401)
    d0, d1 = binary_dither_pair(x.size, seed=7, step=20.0)
    for bit in (0, 1):
        bits = np.full(x.shape, bit, dtype=np.uint8)
        encoded = dm_embed_values(x, bits, 20.0, d0, d1)
        assert np.all(dm_decode_values(encoded, 20.0, d0, d1) == bit)


def test_dm_rejects_nonpositive_step():
    with pytest.raises(ValueError):
        binary_dither_pair(1, 0, 0.0)
    with pytest.raises(ValueError):
        dm_embed_values(np.array([1.0]), np.array([0], np.uint8), 0.0)


def test_dm_clean_image_roundtrip(host128, wm16):
    marked, key = embed("dct_dm_qim_chen2001", host128, wm16, seed=7, repeat=1, step=44.0)
    recovered = extract(marked, key)
    assert bit_accuracy(wm16, recovered) == 1.0
    assert key.params["dither_storage"] == "regenerate_from_seed"


def test_dm_is_deterministic(host128, wm16):
    a, ka = embed("dct_dm_qim_chen2001", host128, wm16, seed=11, repeat=1, step=44.0)
    b, kb = embed("dct_dm_qim_chen2001", host128, wm16, seed=11, repeat=1, step=44.0)
    assert np.array_equal(a, b)
    assert ka.params == kb.params


def test_dm_dither_changes_with_seed(host128, wm16):
    a, _ = embed("dct_dm_qim_chen2001", host128, wm16, seed=11, repeat=1, step=44.0)
    b, _ = embed("dct_dm_qim_chen2001", host128, wm16, seed=12, repeat=1, step=44.0)
    assert not np.array_equal(a, b)


def test_dm_key_is_compact(host128, wm16):
    _marked, key = embed("dct_dm_qim_chen2001", host128, wm16, seed=3, repeat=1, step=44.0)
    assert key.schedule == []
    assert key.params["schedule_storage"] == "regenerate_from_seed"
    assert "host" not in key.side_information.lower()


def test_dm_capacity_is_enforced(wm16):
    host = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        embed("dct_dm_qim_chen2001", host, wm16, repeat=1)
