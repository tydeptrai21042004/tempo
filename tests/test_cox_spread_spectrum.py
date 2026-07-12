from __future__ import annotations

import numpy as np

from realtime_watermark.baselines.dct_spread_spectrum_cox1997 import (
    cox_similarity, native_detection_statistics_gray, native_embed_gray,
)
from realtime_watermark.core import bit_accuracy
from realtime_watermark.registry import embed, extract


def test_cox_similarity_has_paper_scale_for_clean_native_sequence(host128):
    gray = np.rint(0.299 * host128[..., 0] + 0.587 * host128[..., 1] + 0.114 * host128[..., 2]).astype(np.uint8)
    rng = np.random.default_rng(4)
    sequence = rng.normal(size=256)
    marked, key = native_embed_gray(gray, sequence, alpha=0.10)
    stats = native_detection_statistics_gray(marked, key)
    assert stats["sequence_correlation"] > 0.75
    assert stats["cox_similarity"] > 0.5 * np.sqrt(sequence.size)


def test_cox_common_adapter_is_informed_and_roundtrips(host128, wm16):
    marked, key = embed("dct_spread_spectrum_cox1997", host128, wm16, seed=7, repeat=1, step=0.20)
    recovered = extract(marked, key)
    assert bit_accuracy(wm16, recovered) >= 0.95
    assert key.fully_blind is False
    assert "original" in key.side_information.lower()
    assert len(key.params["original_coefficients"]) == wm16.size


def test_cox_similarity_reference_case():
    x = np.asarray([1.0, -1.0, 2.0])
    assert np.isclose(cox_similarity(x, x), np.linalg.norm(x))
