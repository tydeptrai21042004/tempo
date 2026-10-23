from __future__ import annotations

"""Static scientific-fidelity disclosures for the cleaned benchmark."""

from copy import deepcopy

PAPER_FIDELITY: dict[str, dict] = {
    "dct_dm_qim_chen2001": {
        "native_status": "paper_exact_pseudorandom_binary_dither_modulation_declared_dct_adapter",
        "common_adapter_status": "paper_modulation_common_image_adapter",
        "paper_reproduction_eligible": False,
        "native_formula_reproduction_eligible": True,
        "common_primary_eligible": True,
        "blindness_tier": "blind",
        "verified_capacity": "one bit per selected 8x8 DCT block",
        "verified": [
            "pseudorandom zero-bit dither uniform on [-Delta/2,Delta/2)",
            "one-bit dither displaced by Delta/2 according to the paper rule",
            "nearest-dithered-quantizer embedding and detection",
        ],
        "adapted": ["8x8 luminance-DCT host transform", "one declared coefficient", "common payload schedule"],
        "unresolved_core_stages": [],
        "eligibility_reason": "Primary common-protocol modulation baseline; no complete multimedia-system reproduction is claimed.",
    },
    "dct_stdm_qim_chen2001": {
        "native_status": "paper_exact_spread_transform_dither_modulation_declared_dct_adapter",
        "common_adapter_status": "paper_modulation_common_image_adapter",
        "paper_reproduction_eligible": False,
        "native_formula_reproduction_eligible": True,
        "common_primary_eligible": True,
        "blindness_tier": "blind",
        "verified_capacity": "one bit per selected 8x8 DCT block",
        "verified": [
            "seeded unit-norm spread-transform projection",
            "pseudorandom binary dither quantization of the projection",
            "quantization displacement spread back along the projection vector",
            "nearest-dithered-quantizer detector",
        ],
        "adapted": ["32-dimensional medium-frequency 8x8 luminance-DCT vector", "common block scheduler"],
        "unresolved_core_stages": [],
        "eligibility_reason": "Primary common-protocol STDM realization with the transform adapter explicitly declared.",
    },
    "dct_iss_malvar2003": {
        "native_status": "paper_section_VA_maximum_useful_iss_declared_dct_adapter",
        "common_adapter_status": "paper_modulation_common_image_adapter",
        "paper_reproduction_eligible": False,
        "native_formula_reproduction_eligible": True,
        "common_primary_eligible": True,
        "blindness_tier": "blind",
        "verified_capacity": "one bit per selected 8x8 DCT block",
        "verified": [
            "Section V-A maximum-useful-distortion rule around the linear ISS displacement",
            "unchanged correlation-sign detector",
            "seeded unit-norm bipolar spreading sequences",
        ],
        "adapted": [
            "28-dimensional upper-mid/high-frequency 8x8 luminance-DCT vector",
            "fixed lambda=0.90 common-protocol operating point",
            "common block scheduler",
        ],
        "unresolved_core_stages": [],
        "eligibility_reason": "Primary common-protocol limited-distortion ISS baseline; the paper deliberately leaves the image transform open, so the adapter is disclosed.",
    },
    "ca_qim_a2_mao2024": {
        "native_status": "paper_native_grayscale_A2_rate_and_SER_tracks_semi_blind",
        "common_adapter_status": "paper_core_common_image_adapter",
        "paper_reproduction_eligible": False,
        "native_formula_reproduction_eligible": False,
        "native_metric_schema_reproduction_eligible": True,
        "common_primary_eligible": False,
        "blindness_tier": "semi_blind",
        "verified_capacity": "four bits per A2/4A2 carrier vector",
        "verified": [
            "paper-native grayscale 8x8 DCT track",
            "A2 self-nested rate sweep for Tables II-III and fixed R=2 SER track",
            "paper bit-zero probability 0.9 in the native track",
            "maximum-weight canonical labeling",
            "nearest-point CA-QIM embedding and nearest-coset decoding",
            "image-dependent canonical labeling retained as semi-blind detector side information",
        ],
        "adapted": [
            "one declared DCT coefficient pair per selected block",
            "seeded common carrier schedule",
            "balanced common watermark instead of forcing the paper's 0.9 zero prior",
            "native-capacity repetition with soft minimum-distance combining for a fixed 4096-bit user payload",
        ],
        "unresolved_core_stages": [],
        "eligibility_reason": "Paper-native grayscale/metric track is separate from the optional common balanced-payload adapter.",
    },
    "dct_spread_spectrum_cox1997": {
        "native_status": "paper_native_informed_whole_image_DCT_spread_spectrum",
        "common_adapter_status": "paper_core_informed_binary_sign_adapter",
        "paper_reproduction_eligible": False,
        "native_formula_reproduction_eligible": False,
        "native_metric_schema_reproduction_eligible": True,
        "common_primary_eligible": False,
        "blindness_tier": "non_blind",
        "verified_capacity": "native 1000-sample Gaussian watermark; common adapter supports one sign bit per selected AC coefficient",
        "verified": [
            "i.i.d. Gaussian native watermark sequence",
            "largest-magnitude whole-image DCT coefficient selection excluding DC",
            "multiplicative insertion v_i' = v_i(1 + alpha x_i)",
            "original selected coefficients used by the informed detector",
            "paper similarity response dot(x,x_hat)/norm(x_hat)",
        ],
        "adapted": [
            "RGB common adapter watermarks luminance",
            "binary sign payload is supplementary to the native continuous watermark detector",
        ],
        "unresolved_core_stages": [],
        "eligibility_reason": "Paper-supported informed baseline; detector access to original selected coefficients is preserved and disclosed.",
    },
    "dew_langelaar2001": {
        "native_status": "paper_supported_keyed_region_differential_energy_watermarking",
        "common_adapter_status": "paper_core_decoded_frame_DCT_adapter",
        "paper_reproduction_eligible": False,
        "native_formula_reproduction_eligible": False,
        "common_primary_eligible": True,
        "blindness_tier": "blind",
        "verified_capacity": "one bit per keyed even-sized DCT-block region; at least two blocks per bit",
        "verified": [
            "keyed permutation of 8x8 luminance-DCT blocks",
            "two energy subregions per label bit",
            "D=E_A-E_B sign detector",
            "paper coefficient-zeroing mode",
        ],
        "adapted": [
            "decoded RGB frames rather than direct JPEG/MPEG coefficient parsing",
            "minimum-energy-removal mode for PSNR-controlled common streaming",
        ],
        "unresolved_core_stages": [],
        "eligibility_reason": "Paper-supported blind real-time DCT-energy baseline; compressed-domain parsing is not claimed.",
    },
    "rcme_qim": {
        "native_status": "research_candidate_not_yet_validated",
        "common_adapter_status": "research_candidate",
        "paper_reproduction_eligible": False,
        "common_primary_eligible": False,
        "blindness_tier": "blind",
        "verified_capacity": "one payload bit per selected 8x8 block",
        "verified": ["unit tests for direct decoding and reliability budget only"],
        "adapted": ["reuses TEMPO DCT carrier and integer safeguard"],
        "unresolved_core_stages": ["matched-protocol benchmarking of NC/PSNR/FPS", "JPEG variance assumption"],
        "eligibility_reason": "Opt-in mathematical research prototype; not an established improved proposal.",
    },
    "proposal": {
        "native_status": "repository_proposal",
        "common_adapter_status": "repository_proposal",
        "paper_reproduction_eligible": False,
        "common_primary_eligible": True,
        "blindness_tier": "blind",
        "verified_capacity": "one payload bit per selected 8x8 block",
        "verified": ["repository proposal implementation and regression tests"],
        "adapted": [],
        "unresolved_core_stages": [],
        "eligibility_reason": "Proposed method under evaluation.",
    },
}

for method_id, entry in PAPER_FIDELITY.items():
    entry["native_method_id"] = method_id
    entry["common_adapter_method_id"] = method_id
    entry["evidence_files"] = [f"src/realtime_watermark/baselines/evidence/{method_id}.yaml"] if method_id != "proposal" else []
    entry["status"] = entry["common_adapter_status"]
    entry["common_benchmark_primary_eligible"] = entry["common_primary_eligible"]
    entry["unresolved"] = list(entry["unresolved_core_stages"])


def fidelity_status(method_id: str) -> dict:
    if method_id not in PAPER_FIDELITY:
        raise KeyError(f"No fidelity disclosure for {method_id}")
    return deepcopy(PAPER_FIDELITY[method_id])


def fidelity_tier(method_id: str) -> str:
    return str(PAPER_FIDELITY[method_id]["common_adapter_status"])


def common_primary_static_eligible(method_id: str) -> bool:
    return bool(PAPER_FIDELITY[method_id].get("common_primary_eligible", False))


def paper_reproduction_eligible(method_id: str) -> bool:
    return bool(PAPER_FIDELITY[method_id].get("paper_reproduction_eligible", False))


__all__ = [
    "PAPER_FIDELITY", "fidelity_status", "fidelity_tier",
    "common_primary_static_eligible", "paper_reproduction_eligible",
]
