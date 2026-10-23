from __future__ import annotations

from realtime_watermark.core import default_attacks
from realtime_watermark.paper_fidelity import common_primary_static_eligible, fidelity_status
from realtime_watermark.registry import ALL_METHODS, COMMON_IMAGE_METHODS, COMMON_VIDEO_METHODS, STREAM_BENCHMARK_METHODS, normalize_method_id

EXPECTED = {
    "dct_dm_qim_chen2001",
    "dct_stdm_qim_chen2001",
    "dct_iss_malvar2003",
    "ca_qim_a2_mao2024",
    "dct_spread_spectrum_cox1997",
    "dew_langelaar2001",
    "proposal",
}


def test_registry_contains_seven_methods_while_legacy_high_payload_protocols_remain_unchanged():
    assert set(ALL_METHODS) == EXPECTED
    assert set(COMMON_IMAGE_METHODS) == EXPECTED - {"dew_langelaar2001"}
    assert set(COMMON_VIDEO_METHODS) == EXPECTED - {"dct_spread_spectrum_cox1997", "dew_langelaar2001"}
    assert set(STREAM_BENCHMARK_METHODS) == EXPECTED - {"dct_spread_spectrum_cox1997"}


def test_aliases_resolve():
    assert normalize_method_id("dm_qim") == "dct_dm_qim_chen2001"
    assert normalize_method_id("stdm") == "dct_stdm_qim_chen2001"
    assert normalize_method_id("iss") == "dct_iss_malvar2003"
    assert normalize_method_id("ca_qim") == "ca_qim_a2_mao2024"
    assert normalize_method_id("cox") == "dct_spread_spectrum_cox1997"
    assert normalize_method_id("dew") == "dew_langelaar2001"
    assert normalize_method_id("jilp_qim") == "proposal"


def test_primary_blind_ranking_excludes_methods_requiring_side_information():
    expected_tiers = {
        "ca_qim_a2_mao2024": "semi_blind",
        "dct_spread_spectrum_cox1997": "non_blind",
    }
    for method in EXPECTED:
        disclosure = fidelity_status(method)
        if method in {"ca_qim_a2_mao2024", "dct_spread_spectrum_cox1997"}:
            assert not common_primary_static_eligible(method)
        else:
            assert common_primary_static_eligible(method)
        assert disclosure["blindness_tier"] == expected_tiers.get(method, "blind")
        assert disclosure["unresolved_core_stages"] == []


def test_attack_presets_are_reproducible_and_nested():
    smoke = [name for name, _ in default_attacks("smoke")]
    standard = [name for name, _ in default_attacks("standard")]
    stress = [name for name, _ in default_attacks("stress")]
    assert smoke == ["clean", "jpeg90", "gaussian_sigma1"]
    assert set(smoke) <= set(standard) <= set(stress)
    assert len(standard) == 25


def test_method_reference_blindness_matches_detector_protocol():
    from realtime_watermark.registry import METHOD_REFS
    assert METHOD_REFS["ca_qim_a2_mao2024"].fully_blind is False
    assert METHOD_REFS["dct_spread_spectrum_cox1997"].fully_blind is False
    for method in EXPECTED - {"ca_qim_a2_mao2024", "dct_spread_spectrum_cox1997"}:
        assert METHOD_REFS[method].fully_blind is True
