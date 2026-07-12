from __future__ import annotations

import numpy as np
import pytest

from realtime_watermark.baselines import dew_langelaar2001 as dew
from realtime_watermark.core import bit_accuracy, make_demo_watermark, psnr
from realtime_watermark.registry import embed, extract


def test_dew_zigzag_is_complete_and_starts_at_dc():
    coords = dew.zigzag_coordinates(8)
    assert len(coords) == 64
    assert len(set(coords)) == 64
    assert coords[0] == (0, 0)
    assert not dew.coefficient_mask(1)[0, 0]


def test_dew_clean_roundtrip_and_energy_sign(host128):
    wm = make_demo_watermark(8)
    marked, key = embed(
        "dew_langelaar2001", host128, wm, seed=17, step=3000.0,
        method_params={"blocks_per_bit": "auto", "cutoff": 1, "embedding_mode": "minimum_energy_removal"},
    )
    recovered = extract(marked, key)
    assert bit_accuracy(wm, recovered) == 1.0
    differential = dew.differential_energy(marked, key)
    bits = (wm.ravel() > 127).astype(np.uint8)
    assert np.all(differential[bits == 0] > 0)
    assert np.all(differential[bits == 1] < 0)
    assert psnr(host128, marked) > 35.0


def test_dew_is_deterministic(host128):
    wm = make_demo_watermark(8)
    kwargs = dict(seed=23, step=2500.0, method_params={"blocks_per_bit": 4, "cutoff": 1})
    first, key1 = embed("dew", host128, wm, **kwargs)
    second, key2 = embed("dew", host128, wm, **kwargs)
    assert np.array_equal(first, second)
    assert key1.params["blocks_per_bit"] == key2.params["blocks_per_bit"] == 4


def test_dew_rejects_payload_above_paper_style_capacity(host128, wm16):
    with pytest.raises(ValueError, match="at least two DCT blocks per bit"):
        embed("dew", host128, wm16, seed=2)
