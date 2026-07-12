from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from realtime_watermark.baselines.ca_qim_a2_mao2024 import (
    BITS_PER_VECTOR,
    COSET_COUNT,
    LATTICE_ALPHA,
    a2_decode_cosets,
    a2_quantize_to_cosets,
    bits_to_symbols,
    canonical_labeling,
    inverse_labeling,
    symbols_to_bits,
)
from realtime_watermark.common.keys import used_units_from_key
from realtime_watermark.core import bit_accuracy
from realtime_watermark.registry import embed, extract


def test_paper_rate_configuration_is_used():
    assert LATTICE_ALPHA == 4
    assert COSET_COUNT == 16
    assert BITS_PER_VECTOR == 4


def test_symbol_bit_conversion_is_invertible():
    bits = np.array([0, 1, 1, 0, 1], dtype=np.uint8)
    symbols, padding = bits_to_symbols(bits)
    assert padding == 3
    assert np.array_equal(symbols_to_bits(symbols, len(bits)), bits)


def test_quantized_points_decode_to_requested_cosets():
    rng = np.random.default_rng(2)
    carriers = rng.normal(size=(100, 2)) * 40
    cosets = rng.integers(0, COSET_COUNT, size=100, dtype=np.uint8)
    quantized = a2_quantize_to_cosets(carriers, cosets, step=12.0)
    assert np.array_equal(a2_decode_cosets(quantized, 12.0), cosets)


def test_canonical_labeling_is_global_maximum_and_deterministic():
    rng = np.random.default_rng(12)
    W = rng.integers(0, 20, size=(COSET_COUNT, COSET_COUNT), dtype=np.int64)
    mapping_a = canonical_labeling(W)
    mapping_b = canonical_labeling(W)
    rows, cols = linear_sum_assignment(-W)
    optimum = int(W[rows, cols].sum())
    score = int(sum(W[int(mapping_a[m]), m] for m in range(COSET_COUNT)))
    assert score == optimum
    assert np.array_equal(mapping_a, mapping_b)
    assert sorted(mapping_a.tolist()) == list(range(COSET_COUNT))


def test_inverse_labeling():
    mapping = np.array([2, 0, 3, 1, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14], dtype=np.uint8)
    inverse = inverse_labeling(mapping)
    assert np.array_equal(inverse[mapping], np.arange(COSET_COUNT))


def test_inverse_labeling_rejects_non_permutation():
    with pytest.raises(ValueError):
        inverse_labeling(np.zeros(COSET_COUNT, dtype=np.uint8))


def test_ca_qim_clean_image_roundtrip(host128, wm16):
    marked, key = embed("ca_qim_a2_mao2024", host128, wm16, seed=9, repeat=1, step=24.0)
    assert bit_accuracy(wm16, extract(marked, key)) == 1.0
    assert key.params["bits_per_vector"] == 4
    assert key.params["coarse_lattice"] == "4A2"
    assert used_units_from_key(key) == wm16.size // 4


def test_ca_qim_compact_key(host128, wm16):
    _marked, key = embed("ca_qim_a2_mao2024", host128, wm16, seed=9, repeat=1, step=24.0)
    assert key.schedule == []
    assert sorted(key.params["message_to_coset"]) == list(range(COSET_COUNT))
    assert key.params["adjacency_shape"] == [COSET_COUNT, COSET_COUNT]
    assert "adjacency_matrix" not in key.params


def test_ca_qim_is_deterministic(host128, wm16):
    a, ka = embed("ca_qim_a2_mao2024", host128, wm16, seed=17, repeat=1, step=24.0)
    b, kb = embed("ca_qim_a2_mao2024", host128, wm16, seed=17, repeat=1, step=24.0)
    assert np.array_equal(a, b)
    assert ka.params["message_to_coset"] == kb.params["message_to_coset"]


def test_a2_rejects_nonpositive_scale():
    with pytest.raises(ValueError):
        a2_decode_cosets(np.zeros((1, 2)), 0.0)


def test_auto_repeat_uses_full_native_capacity(host128, wm16):
    marked, key = embed("ca_qim_a2_mao2024", host128, wm16, seed=29, repeat="auto", step=18.0)
    recovered = extract(marked, key)
    assert key.repeat == 4
    assert np.array_equal(recovered, wm16)


def test_a2_compact_neighborhood_matches_wider_bruteforce():
    from itertools import product
    from realtime_watermark.baselines.ca_qim_a2_mao2024 import (
        A2_BASE, COSET_REPRESENTATIVES, LATTICE_ALPHA,
        nearest_points_all_cosets,
    )
    rng = np.random.default_rng(831)
    carriers = rng.normal(0.0, 150.0, size=(500, 2))
    step = 6.0
    points, distances = nearest_points_all_cosets(carriers, step)

    fine = step * A2_BASE
    inv_fine = np.linalg.inv(fine)
    coordinates = carriers @ inv_fine.T
    reps = COSET_REPRESENTATIVES
    centers = np.rint((coordinates[:, None, :] - reps[None, :, :]) / LATTICE_ALPHA).astype(np.int64)
    wide = np.asarray(list(product(range(-3, 4), repeat=2)), dtype=np.int64)
    lattice_coordinates = reps[None, :, None, :] + LATTICE_ALPHA * (
        centers[:, :, None, :] + wide[None, None, :, :]
    )
    candidates = lattice_coordinates @ fine.T
    candidate_distances = np.sum((candidates - carriers[:, None, None, :]) ** 2, axis=-1)
    best = np.argmin(candidate_distances, axis=-1)
    rows = np.arange(carriers.shape[0])[:, None]
    cosets = np.arange(COSET_REPRESENTATIVES.shape[0])[None, :]
    brute_points = candidates[rows, cosets, best]
    brute_distances = candidate_distances[rows, cosets, best]
    assert np.allclose(distances, brute_distances)
    assert np.allclose(points, brute_points)
