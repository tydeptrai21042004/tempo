"""Regression tests for the two minimal mathematical revisions.

These tests use synthetic hosts so that the reviewer patch is testable without
shipping third-party benchmark images or precomputed result tables.
"""
import numpy as np
import pytest

from realtime_watermark.core import bit_accuracy, make_demo_host, make_demo_watermark
from realtime_watermark.proposals.jilp_qim import engine as eng
from realtime_watermark.proposals.jilp_qim import method as proposal
from realtime_watermark.proposals.jilp_qim.config import DEFAULT_CONFIG


def _brute_margin(v, step, bit, rho):
    q0 = int(np.rint(v / step))
    vals = []
    for k in range(q0 - 5, q0 + 6):
        if (k & 1) == bit:
            vals.append(float(np.clip(v, k * step - .5 * step + rho,
                                      k * step + .5 * step - rho)))
    return min(vals, key=lambda a: abs(a - v))


def test_constant_time_qim_projection_matches_original_search():
    rng = np.random.default_rng(2026)
    for _ in range(240):
        step = float(rng.uniform(1, 35))
        v = float(rng.uniform(-250, 250))
        rho = float(rng.uniform(0, .499) * step)
        bit = int(rng.integers(0, 2))
        expected = _brute_margin(v, step, bit, rho)
        actual = eng._project_qim_margin(v, step, bit, rho)
        assert abs(expected - actual) < 1e-10
        q = int(np.rint(actual / step))
        assert (q & 1) == bit


def test_integer_pilot_parity_is_preserved_on_saturated_host():
    host = np.zeros((128, 128, 3), dtype=np.uint8)
    wm = make_demo_watermark(16)
    marked, key = proposal.embed_proposal(host, wm, method_params=DEFAULT_CONFIG.to_dict())
    assert key.params['integer_lattice_unresolved'] == 0
    assert key.params['integer_pilot_unresolved'] == 0
    assert bit_accuracy(wm, proposal.extract_proposal(marked, key)) == 1.0
    _payload, pidx, pbits, _, _, wc = proposal._schedules_from_key(proposal._normalize_jilp_key(key))
    bits = eng._extract_one_carrier_bits(
        marked, pidx.astype(np.int32), eng.PILOT_BA, eng.PILOT_BB,
        float(key.params['step_pilot']), wc, 8, float(key.params['eta']), 0, 0,
    )
    assert np.array_equal(bits, pbits)


def test_sparse_phase_registration_recovers_one_pixel_translation():
    cv2 = pytest.importorskip('cv2')
    host = make_demo_host(256, 256)
    wm = make_demo_watermark(16)
    marked, key = proposal.embed_proposal(host, wm, method_params=DEFAULT_CONFIG.to_dict())
    received = cv2.warpAffine(marked, np.float32([[1,0,0],[0,1,1]]), (256, 256))
    recovered, meta = proposal.extract_jilp_qim(received, key, return_meta=True)
    assert bit_accuracy(wm, recovered) > .95
    assert meta['registration'] == 'sparse_qim_phase'
    assert meta['warp_count'] <= 1


def test_phase_registration_clean_fast_path_has_no_full_warp():
    host = make_demo_host(256, 256)
    wm = make_demo_watermark(16)
    marked, key = proposal.embed_proposal(host, wm, method_params=DEFAULT_CONFIG.to_dict())
    recovered, meta = proposal.extract_jilp_qim(marked, key, return_meta=True)
    assert bit_accuracy(wm, recovered) == 1.0
    assert meta['path'] == 'raw_fast_jilp'
    assert meta['warp_count'] == 0
    assert meta['phase_score'] > .8
