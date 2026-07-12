from __future__ import annotations

from pathlib import Path

import numpy as np

from realtime_watermark.native_metrics import (
    bit_error_probability,
    mse,
    prd,
    symbol_error_rate,
)
from realtime_watermark.paper_native import (
    NATIVE_DISCLOSURES,
    _ber_stdm_theory,
    _iss_theoretical_error,
    run_all_paper_native,
)


def test_native_metrics_reference_values():
    a = np.array([0.0, 1.0, 2.0])
    b = np.array([0.0, 2.0, 4.0])
    assert mse(a, b) == 5.0 / 3.0
    assert prd(a, a) == 0.0
    assert bit_error_probability([0, 1, 1, 0], [0, 0, 1, 1]) == 0.5
    assert symbol_error_rate([0, 1, 2, 3], [0, 1, 0, 3]) == 0.25


def test_native_detector_types_follow_papers():
    assert NATIVE_DISCLOSURES["dct_dm_qim_chen2001"].detector_type == "blind"
    assert NATIVE_DISCLOSURES["dct_stdm_qim_chen2001"].detector_type == "blind"
    assert NATIVE_DISCLOSURES["dct_iss_malvar2003"].detector_type == "blind"
    assert NATIVE_DISCLOSURES["ca_qim_a2_mao2024"].detector_type == "semi_blind"
    assert NATIVE_DISCLOSURES["dct_spread_spectrum_cox1997"].detector_type == "non_blind"


def test_paper_formula_bounds_are_probabilities():
    for dnr in (-5.0, 0.0, 5.0, 10.0, 20.0):
        value = _ber_stdm_theory(dnr)
        assert 0.0 <= value <= 1.0
    for rho in (5.0, 10.0, 20.0):
        for snr in (7.0, 10.0):
            for lam in (0.0, 0.5, 1.0):
                value = _iss_theoretical_error(rho=rho, snr_db=snr, cancellation_lambda=lam)
                assert 0.0 <= value <= 0.5


def test_quick_native_pipeline_writes_primary_outputs(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    hosts = sorted(p for p in (root / "data" / "user_dataset" / "host").iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})[:1]
    assert hosts
    result = run_all_paper_native(
        out_dir=tmp_path,
        host_paths=hosts,
        seed=9,
        quick=True,
    )
    assert result["protocol"]["protocol_version"]
    expected = {
        "qim_stdm_native_channel.csv",
        "iss_native_channel.csv",
        "ca_qim_native_imperceptibility.csv",
        "ca_qim_native_ser.csv",
        "cox_native_image_attacks.csv",
        "native_reproduction_checks.csv",
        "paper_native_protocol.json",
    }
    assert expected <= {p.name for p in tmp_path.iterdir()}
    ca_rows = result["ca_qim"]["imperceptibility"]
    assert {(r["frequency"], r["payload_bits_per_dimension"]) for r in ca_rows} == {
        (frequency, count)
        for frequency in ("low", "mid", "high")
        for count in (1, 2, 3)
    }
    assert all(r["image_mode"] == "grayscale" for r in ca_rows)
    assert all({"mse_frequency", "psnr_frequency", "prd_frequency", "psnr_spatial", "ssim_spatial"} <= set(r) for r in ca_rows)
    assert all(not row.get("psnr_calibrated", False) for row in result["qim_stdm"])
    assert all(not row.get("psnr_calibrated", False) for row in result["iss"])


def test_native_a2_rate_sweep_has_correct_coset_counts_and_roundtrip():
    from realtime_watermark.paper_native import (
        _native_a2_spec,
        _native_ca_embed,
        _native_ca_extract_symbols,
    )
    yy, xx = np.mgrid[0:64, 0:64]
    host = np.clip(110 + 35 * np.sin(xx / 6.0) + 25 * np.cos(yy / 8.0), 0, 255).astype(np.uint8)
    for rate, expected_cosets in ((1, 4), (2, 16), (3, 64)):
        alpha, bits_per_vector, reps = _native_a2_spec(rate)
        assert alpha == 2**rate
        assert bits_per_vector == 2 * rate
        assert len(reps) == expected_cosets
        marked, key = _native_ca_embed(
            host,
            frequency="mid",
            message_count=1,
            code_rate=rate,
            step=8.0,
            seed=17 + rate,
            p_zero=0.9,
        )
        recovered = _native_ca_extract_symbols(marked, key)
        original = np.asarray(key["symbols"], dtype=np.uint16)
        assert key["lattice_alpha"] == 2**rate
        assert len(key["mapping"]) == expected_cosets
        assert symbol_error_rate(original, recovered) < 0.01


def test_native_a2_compact_search_matches_wide_search_for_all_rates():
    from itertools import product
    from realtime_watermark.baselines.ca_qim_a2_mao2024 import A2_BASE
    from realtime_watermark.paper_native import (
        _NATIVE_A2_OFFSETS,
        _native_a2_nearest_all_cosets,
        _native_a2_spec,
    )
    assert _NATIVE_A2_OFFSETS.shape == (9, 2)
    rng = np.random.default_rng(92)
    carriers = rng.normal(0.0, 100.0, size=(80, 2))
    wide = np.asarray(list(product(range(-3, 4), repeat=2)), dtype=np.int64)
    for rate in (1, 2, 3):
        alpha, _bits, reps = _native_a2_spec(rate)
        step = 5.0
        points, distances = _native_a2_nearest_all_cosets(carriers, step, rate)
        fine = step * A2_BASE
        coordinates = carriers @ np.linalg.inv(fine).T
        centers = np.rint((coordinates[:, None, :] - reps[None, :, :]) / alpha).astype(np.int64)
        lc = reps[None, :, None, :] + alpha * (centers[:, :, None, :] + wide[None, None, :, :])
        candidates = lc @ fine.T
        candidate_distances = np.sum((candidates - carriers[:, None, None, :]) ** 2, axis=-1)
        best = np.argmin(candidate_distances, axis=-1)
        rows = np.arange(len(carriers))[:, None]
        cosets = np.arange(len(reps))[None, :]
        assert np.allclose(points, candidates[rows, cosets, best])
        assert np.allclose(distances, candidate_distances[rows, cosets, best])
