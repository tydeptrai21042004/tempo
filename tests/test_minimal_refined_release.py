"""Release invariants: exact scrambling and a decoder-only mathematical revision."""
from dataclasses import replace
from pathlib import Path
import numpy as np

from realtime_watermark.core import arnold_inverse, arnold_period, arnold_transform, make_demo_host, make_demo_watermark
from realtime_watermark.proposals.jilp_qim.compiled_permutation import (
    inverse_arnold_exact, permute_arnold_exact, _arnold_inverse_lookup,
)
from realtime_watermark.proposals.jilp_qim.config import JILPQIMConfig
from realtime_watermark.proposals.jilp_qim.method import embed_proposal, extract_jilp_qim

ROOT = Path(__file__).resolve().parents[1]


def test_cached_exact_permutation_all_standard_sizes_and_roundtrip():
    for n in (8, 16, 32, 64, 128):
        rng = np.random.default_rng(n)
        matrix = rng.integers(0, 2, (n, n), dtype=np.uint8)
        p = arnold_period(n)
        for k in (0, 1, 7, p, p + 3):
            actual = permute_arnold_exact(matrix, k)
            assert np.array_equal(actual, arnold_transform(matrix, k))
            assert np.array_equal(inverse_arnold_exact(actual, k, p), matrix)
            assert np.array_equal(inverse_arnold_exact(actual, k, p), arnold_inverse(actual, k, p))
        assert not _arnold_inverse_lookup(n, 7).flags.writeable


def test_refined_config_keeps_embedding_identical_and_decodes_clean():
    base = JILPQIMConfig.load(ROOT / 'configs/proposal_frozen.json')
    revised = JILPQIMConfig.load(ROOT / 'configs/proposal_minimal_refined.json')
    assert revised == replace(base, certified_phase_pruning=True, phase_refinement='selective')
    assert revised.sha256() != base.sha256()
    host = make_demo_host(256, 256)
    watermark = make_demo_watermark(16)
    marked0, key0 = embed_proposal(host, watermark, method_params=base.to_dict())
    marked1, key1 = embed_proposal(host, watermark, method_params=revised.to_dict())
    assert np.array_equal(marked0, marked1)
    for marked, key in ((marked0, key0), (marked1, key1)):
        recovered, meta = extract_jilp_qim(marked, key, return_meta=True)
        assert np.array_equal(recovered, watermark)
        assert meta['warp_count'] == 0


def test_new_config_hashes_match_serialized_content():
    for name in ('proposal_frozen.json', 'proposal_minimal_refined.json', 'proposal_revision_candidate.json'):
        path = ROOT/'configs'/name
        import json
        contents = json.loads(path.read_text())
        assert contents['config_hash'] == JILPQIMConfig.load(path).sha256()
