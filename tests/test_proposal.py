from __future__ import annotations

from dataclasses import replace
import json
import numpy as np

from realtime_watermark.core import bit_accuracy
from realtime_watermark.proposals.jilp_qim.config import DEFAULT_CONFIG, JILPQIMConfig
from realtime_watermark.registry import embed, extract


def test_config_hash_is_stable_and_order_independent(tmp_path):
    path = tmp_path / "config.json"
    DEFAULT_CONFIG.save(path)
    loaded = JILPQIMConfig.load(path)
    assert loaded == DEFAULT_CONFIG
    assert loaded.sha256() == DEFAULT_CONFIG.sha256()
    payload = json.loads(path.read_text())
    assert payload["config_hash"] == DEFAULT_CONFIG.sha256()


def test_default_primary_policy_is_random_hard_vote():
    assert DEFAULT_CONFIG.payload_schedule_mode == "random"
    assert DEFAULT_CONFIG.candidate_pool_multiplier == 1.0
    assert DEFAULT_CONFIG.temporal_vote_mode == "hard"


def test_proposal_clean_roundtrip(host128, wm16):
    marked, key = embed("proposal", host128, wm16, seed=10, repeat=1, method_params=DEFAULT_CONFIG.to_dict())
    assert bit_accuracy(wm16, extract(marked, key)) == 1.0
    assert key.params["config_hash_sha256"] == DEFAULT_CONFIG.sha256()
    assert key.params["frozen_config"] == DEFAULT_CONFIG.to_dict()


def test_no_false_content_adaptive_label_when_capacity_equals_payload(host128, wm16):
    config = replace(DEFAULT_CONFIG, payload_schedule_mode="content", candidate_pool_multiplier=1.75)
    _marked, key = embed("proposal", host128, wm16, seed=12, repeat=1, method_params=config.to_dict())
    assert key.params["candidate_pool_used"] == wm16.size
    assert key.params["content_adaptive_block_selection"] is False
    assert key.params["structural_components"]["content_adaptive_schedule"] is False
    assert key.params["schedule_storage"] == "regenerate_from_seed"


def test_content_schedule_uses_explicit_key_only_when_active(host192, wm16):
    config = replace(DEFAULT_CONFIG, payload_schedule_mode="content", candidate_pool_multiplier=1.5)
    _marked, key = embed("proposal", host192, wm16, seed=13, repeat=1, method_params=config.to_dict())
    assert key.params["candidate_pool_used"] > wm16.size
    assert key.params["content_adaptive_block_selection"] is True
    assert key.params["schedule_storage"] == "explicit_host_ranked"
    assert "payload_indices" in key.params
    assert "explicit" in key.side_information


def test_random_and_content_modes_are_identical_without_spare_capacity(host128, wm16):
    random_cfg = replace(DEFAULT_CONFIG, payload_schedule_mode="random", candidate_pool_multiplier=1.0)
    content_cfg = replace(DEFAULT_CONFIG, payload_schedule_mode="content", candidate_pool_multiplier=1.0)
    a, _ = embed("proposal", host128, wm16, seed=14, repeat=1, method_params=random_cfg.to_dict())
    b, _ = embed("proposal", host128, wm16, seed=14, repeat=1, method_params=content_cfg.to_dict())
    assert np.array_equal(a, b)


def test_candidate_pool_metadata_is_not_inflated_by_pilots(host128, wm16):
    _marked, key = embed("proposal", host128, wm16, seed=15, repeat=1, method_params=DEFAULT_CONFIG.to_dict())
    assert key.params["candidate_pool_used"] == wm16.size
    assert key.params["candidate_pool_used"] != wm16.size + key.params["pilot_count"]


def test_repair_free_embedding_reports_zero_unresolved(host128, wm16):
    _marked, key = embed("proposal", host128, wm16, seed=16, repeat=1, method_params=DEFAULT_CONFIG.to_dict())
    assert key.params["clean_verification_iters"] == 0
    assert key.params["clean_repaired_bits"] == 0
    assert key.params["integer_lattice_unresolved"] == 0


def test_compact_random_schedule_roundtrip_with_spare_host_capacity():
    from realtime_watermark.core import make_demo_host, make_demo_watermark
    from realtime_watermark.registry import embed, extract

    host = make_demo_host(256, 256)
    watermark = make_demo_watermark(16)
    marked, key = embed(
        "proposal", host, watermark, seed=77, repeat=1,
        method_params=DEFAULT_CONFIG.to_dict(),
    )
    assert key.params["capacity"] > watermark.size
    assert key.params["schedule_storage"] == "regenerate_from_seed"
    assert np.array_equal(extract(marked, key), watermark)
