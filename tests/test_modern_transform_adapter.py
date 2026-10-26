"""Unit and integration tests for the independent recent-transform control."""
import numpy as np
import pytest
from realtime_watermark.registry import embed, extract, method_metadata
from realtime_watermark.baselines.haar_hd_svd_qim_adapter2024 import METHOD_ID
from realtime_watermark.paper_fidelity import fidelity_status


def test_blind_encoder_decoder_key_has_no_original_host_information():
    rng = np.random.default_rng(54)
    x = rng.integers(15, 241, size=(64, 64, 3), dtype=np.uint8)
    wm = rng.integers(0, 2, size=(8, 8), dtype=np.uint8) * 255
    y, key = embed(METHOD_ID, x, wm, seed=2026, step=15)
    dec = extract(y, key)
    assert np.array_equal(wm, dec)
    assert key.fully_blind
    assert 'original_host' not in str(key.params).lower().replace('original_host_side_information', '')
    assert key.params['original_host_side_information'] is False
    assert key.params['paper_exact'] is False
    assert y.dtype == np.uint8 and y.shape == x.shape


def test_capacity_is_4096_bits_with_512_host():
    rng = np.random.default_rng(11)
    x = rng.integers(10, 245, size=(512, 512, 3), dtype=np.uint8)
    wm = rng.integers(0, 2, size=(64, 64), dtype=np.uint8) * 255
    y, key = embed(METHOD_ID, x, wm, seed=2026, step=20)
    d = extract(y, key)
    assert np.array_equal(wm, d)
    assert key.params['capacity_bits'] == 4096
    assert key.repeat == 1
    assert fidelity_status(METHOD_ID)['paper_reproduction_eligible'] is False


def test_out_of_capacity_is_rejected():
    with pytest.raises(ValueError, match='Insufficient 8x8 capacity'):
        embed(METHOD_ID,np.zeros((64,64,3),dtype=np.uint8),np.zeros((16,16),dtype=np.uint8))
