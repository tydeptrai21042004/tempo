from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping
import json
import time

import numpy as np

from .baselines.dct_spread_spectrum_cox1997 import detection_statistics as cox_detection_statistics
from .core import (
    attack_category, bit_accuracy, ber, default_attacks, load_binary_watermark, load_rgb,
    nc_binary, psnr, save_csv, save_json, ssim_gray, watermark_statistics,
)
from .native_metrics import (
    binary_symbol_error_rate, image_distortion_statistics, mse, prd,
)
from .paper_native import NATIVE_DISCLOSURES
from .registry import COMMON_IMAGE_METHODS, embed, extract, normalize_method_id
from .proposals.jilp_qim.method import extract_jilp_qim

ALL_METRICS_PROTOCOL_VERSION = "1.1"


def _json_key_bytes(key: Any) -> int:
    return len(json.dumps(asdict(key), separators=(",", ":"), default=str).encode("utf-8"))


def _detector_type(method_id: str) -> str:
    if method_id in NATIVE_DISCLOSURES:
        return NATIVE_DISCLOSURES[method_id].detector_type
    return "blind"


def _image_metrics(reference: np.ndarray, test: np.ndarray, prefix: str) -> dict[str, float]:
    stats = image_distortion_statistics(reference, test)
    return {
        f"{prefix}_mse": stats["mse"],
        f"{prefix}_psnr": psnr(reference, test),
        f"{prefix}_prd": prd(reference, test),
        f"{prefix}_ssim": ssim_gray(reference, test),
        f"{prefix}_distortion_std_squared": stats["distortion_std_squared"],
        f"{prefix}_distortion_max_squared": stats["distortion_max_squared"],
    }


def _native_score(method_id: str, attacked: np.ndarray, key: Any, wm: np.ndarray, recovered: np.ndarray) -> tuple[str, float]:
    if method_id == "dct_spread_spectrum_cox1997":
        return "cox_similarity_centered", float(cox_detection_statistics(attacked, key)["cox_similarity_centered"])
    if method_id == "ca_qim_a2_mao2024":
        return "one_minus_SER_4bit", float(1.0 - binary_symbol_error_rate(wm > 127, recovered > 127, 4))
    return "bipolar_NC", float(nc_binary(wm, recovered))


def metric_applicability_rows(methods: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in methods:
        method_id = normalize_method_id(raw)
        disclosure = NATIVE_DISCLOSURES.get(method_id)
        detector = _detector_type(method_id)
        common = {
            "method_id": method_id,
            "detector_type": detector,
            "requires_original_host": detector == "non_blind",
            "requires_image_dependent_side_information": detector in {"semi_blind", "non_blind"},
            "generic_BER": True,
            "generic_NC": True,
            "generic_SER_4bit": True,
            "generic_MSE": True,
            "generic_PSNR": True,
            "generic_PRD": True,
            "generic_SSIM": True,
            "generic_timing": True,
            "paper_analytical_error_formula": method_id in {"dct_stdm_qim_chen2001", "dct_iss_malvar2003"},
            "paper_native_metric": disclosure.paper_metric if disclosure else "NC, BER, PSNR, timing, and robustness statistics",
            "paper_native_attack_model": disclosure.paper_attack_model if disclosure else "complete application attack suite",
        }
        rows.append(common)
    return rows


def run_all_methods_all_metrics(
    *,
    host_paths: Iterable[str | Path],
    watermark_path: str | Path | None,
    out_dir: str | Path,
    methods: Iterable[str] | None = None,
    seed: int = 2026,
    wm_size: int = 32,
    attack_preset: str = "standard",
    mode: str = "512",
    method_params_by_method: Mapping[str, dict[str, Any]] | None = None,
    repeat_by_method: Mapping[str, int | str] | None = None,
    watermark_label: str | None = None,
) -> dict[str, Any]:
    """Report one complete generic metric schema for every retained method.

    This is a supplementary application-level table. It does not replace each
    method's paper-native track and it never calibrates methods to a common PSNR.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    selected = [normalize_method_id(x) for x in (methods or COMMON_IMAGE_METHODS)]
    wm = load_binary_watermark(watermark_path, size=int(wm_size))
    wm_stats = watermark_statistics(wm)
    rows: list[dict[str, Any]] = []
    params_map = {normalize_method_id(k): dict(v) for k, v in (method_params_by_method or {}).items()}
    repeat_map = {normalize_method_id(k): v for k, v in (repeat_by_method or {}).items()}

    for host_path in host_paths:
        host = load_rgb(host_path, mode=mode)
        for method_id in selected:
            t0 = time.perf_counter()
            marked, key = embed(
                method_id, host, wm, seed=seed,
                repeat=repeat_map.get(method_id, "auto"), step=None,
                method_params=params_map.get(method_id),
            )
            embed_ms = (time.perf_counter() - t0) * 1000.0
            embedding_metrics = _image_metrics(host, marked, "embedding")
            clean_extract_start = time.perf_counter()
            if method_id == "proposal":
                clean_recovered, clean_meta = extract_jilp_qim(
                    marked, key, sync=True, return_meta=True
                )
            else:
                clean_recovered = extract(marked, key)
                clean_meta = {"path": "method_default", "candidate": "", "score": float("nan"), "pilot_valid": 0}
            clean_extract_ms = (time.perf_counter() - clean_extract_start) * 1000.0
            clean_accuracy = bit_accuracy(wm, clean_recovered)

            for attack_name, attack_fn in default_attacks(attack_preset):
                attacked = np.asarray(attack_fn(marked), dtype=np.uint8)
                t1 = time.perf_counter()
                if method_id == "proposal":
                    recovered, extraction_meta = extract_jilp_qim(
                        attacked, key, sync=True, return_meta=True
                    )
                else:
                    recovered = extract(attacked, key)
                    extraction_meta = {
                        "path": "method_default", "candidate": "",
                        "score": float("nan"), "pilot_valid": 0,
                        "shift_y": 0, "shift_x": 0,
                    }
                extract_ms = (time.perf_counter() - t1) * 1000.0
                native_name, native_value = _native_score(method_id, attacked, key, wm, recovered)
                detector = _detector_type(method_id)
                row: dict[str, Any] = {
                    "protocol_version": ALL_METRICS_PROTOCOL_VERSION,
                    "method_id": method_id,
                    "watermark_label": watermark_label or Path(watermark_path).name if watermark_path else "generated",
                    "host": Path(host_path).name,
                    "attack": attack_name,
                    "attack_category": attack_category(attack_name),
                    "detector_type": detector,
                    "fully_blind": detector == "blind",
                    "requires_original_host": detector == "non_blind",
                    "requires_image_dependent_side_information": detector in {"semi_blind", "non_blind"},
                    "payload_bits": int(wm.size),
                    "payload_bits_per_pixel": int(wm.size) / max(host.shape[0] * host.shape[1], 1),
                    "watermark_zero_bits": wm_stats["zero_bits"],
                    "watermark_one_bits": wm_stats["one_bits"],
                    "watermark_one_fraction": wm_stats["one_fraction"],
                    "watermark_bit_balance": wm_stats["bit_balance"],
                    "step_or_alpha": float(key.step),
                    "repeat": int(key.repeat),
                    "key_bytes": _json_key_bytes(key),
                    "clean_bit_accuracy": clean_accuracy,
                    "clean_extract_ms": clean_extract_ms,
                    "embed_ms": embed_ms,
                    "extract_ms": extract_ms,
                    "embed_fps": 1000.0 / max(embed_ms, 1e-9),
                    "extract_fps": 1000.0 / max(extract_ms, 1e-9),
                    "embed_payload_bps": int(wm.size) * 1000.0 / max(embed_ms, 1e-9),
                    "extract_payload_bps": int(wm.size) * 1000.0 / max(extract_ms, 1e-9),
                    "embed_megapixels_per_second": (host.shape[0] * host.shape[1]) / max(embed_ms, 1e-9) / 1000.0,
                    "extract_megapixels_per_second": (host.shape[0] * host.shape[1]) / max(extract_ms, 1e-9) / 1000.0,
                    "bit_accuracy": bit_accuracy(wm, recovered),
                    "ber": ber(wm, recovered),
                    "empirical_error_probability": ber(wm, recovered),
                    "nc_bipolar": nc_binary(wm, recovered),
                    "ser_4bit": binary_symbol_error_rate(wm > 127, recovered > 127, 4),
                    "native_detection_metric": native_name,
                    "native_detection_value": native_value,
                    "paper_formula_error_probability": "",
                    "paper_formula_applicable": method_id in {"dct_stdm_qim_chen2001", "dct_iss_malvar2003"} and attack_name.startswith("gaussian"),
                    "paper_formula_note": "Use the separate paper-native channel report for formula-comparable conditions.",
                    "psnr_calibrated": False,
                    "common_ecc": False,
                    "side_information": key.side_information,
                    "configuration_sha256": str(key.params.get("config_hash_sha256", "")),
                    "clean_extraction_path": clean_meta.get("path", ""),
                    "integer_lattice_unit_updates": int(key.params.get("integer_lattice_unit_updates", 0)),
                    "integer_lattice_unresolved": int(key.params.get("integer_lattice_unresolved", 0)),
                    "integer_lattice_corrected_blocks": int(key.params.get("integer_lattice_corrected_blocks", 0)),
                    "integer_lattice_initial_parity_failures": int(key.params.get("integer_lattice_initial_parity_failures", 0)),
                    "integer_lattice_initial_guard_failures": int(key.params.get("integer_lattice_initial_guard_failures", 0)),
                    "integer_lattice_payload_blocks": int(key.params.get("integer_lattice_payload_blocks", wm.size)),
                    "pilot_lattice_unit_updates": int(key.params.get("pilot_lattice_unit_updates", 0)),
                    "pilot_lattice_corrected_blocks": int(key.params.get("pilot_lattice_corrected_blocks", 0)),
                    "pilot_lattice_initial_parity_failures": int(key.params.get("pilot_lattice_initial_parity_failures", 0)),
                    "pilot_lattice_initial_guard_failures": int(key.params.get("pilot_lattice_initial_guard_failures", 0)),
                    "pilot_lattice_unresolved": int(key.params.get("pilot_lattice_unresolved", 0)),
                    "pilot_overlap_unresolved": int(key.params.get("pilot_overlap_unresolved", 0)),
                    "extraction_path": extraction_meta.get("path", ""),
                    "sync_candidate": extraction_meta.get("candidate", ""),
                    "pilot_score": extraction_meta.get("score", float("nan")),
                    "pilot_valid": extraction_meta.get("pilot_valid", 0),
                    "synchronization_confident": bool(extraction_meta.get("synchronization_confident", False)),
                    "estimated_shift_y": extraction_meta.get("shift_y", 0),
                    "estimated_shift_x": extraction_meta.get("shift_x", 0),
                }
                row.update(embedding_metrics)
                row.update(_image_metrics(host, attacked, "attacked_vs_host"))
                row.update(_image_metrics(marked, attacked, "attacked_vs_watermarked"))
                rows.append(row)

    applicability = metric_applicability_rows(selected)
    summary_rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["method_id"]), str(row["host"]), str(row["watermark_label"])), []).append(row)
    for (method_id, host_name, wm_name), group in groups.items():
        non_clean = [r for r in group if r["attack"] != "clean"]
        ncs = np.asarray([float(r["nc_bipolar"]) for r in non_clean], dtype=np.float64)
        bers = np.asarray([float(r["ber"]) for r in non_clean], dtype=np.float64)
        worst = min(non_clean, key=lambda r: float(r["nc_bipolar"])) if non_clean else {"attack": ""}
        recovery_rows = [r for r in non_clean if r.get("extraction_path") in {"wide_sync_jilp", "local_sync_jilp"}]
        fast_rows = [r for r in non_clean if r.get("extraction_path") == "raw_fast_jilp"]
        first = group[0]
        summary_rows.append({
            "method_id": method_id,
            "host": host_name,
            "watermark_label": wm_name,
            "payload_bits": first["payload_bits"],
            "embedding_psnr": first["embedding_psnr"],
            "embedding_ssim": first["embedding_ssim"],
            "clean_bit_accuracy": first["clean_bit_accuracy"],
            "embed_ms": first["embed_ms"],
            "clean_extract_ms": first["clean_extract_ms"],
            "embed_fps": first["embed_fps"],
            "min_attack_nc": float(np.min(ncs)) if non_clean else float("nan"),
            "p10_attack_nc": float(np.percentile(ncs, 10)) if non_clean else float("nan"),
            "median_attack_nc": float(np.median(ncs)) if non_clean else float("nan"),
            "mean_attack_nc": float(np.mean(ncs)) if non_clean else float("nan"),
            "std_attack_nc": float(np.std(ncs)) if non_clean else float("nan"),
            "success_rate_nc_ge_050": float(np.mean(ncs >= 0.50)) if non_clean else float("nan"),
            "success_rate_nc_ge_070": float(np.mean(ncs >= 0.70)) if non_clean else float("nan"),
            "success_rate_nc_ge_090": float(np.mean(ncs >= 0.90)) if non_clean else float("nan"),
            "max_attack_ber": float(np.max(bers)) if non_clean else float("nan"),
            "p90_attack_ber": float(np.percentile(bers, 90)) if non_clean else float("nan"),
            "worst_attack": worst["attack"],
            "integer_lattice_unit_updates": first.get("integer_lattice_unit_updates", 0),
            "integer_lattice_unresolved": first.get("integer_lattice_unresolved", 0),
            "integer_lattice_corrected_blocks": first.get("integer_lattice_corrected_blocks", 0),
            "integer_lattice_initial_parity_failures": first.get("integer_lattice_initial_parity_failures", 0),
            "integer_lattice_initial_guard_failures": first.get("integer_lattice_initial_guard_failures", 0),
            "integer_lattice_payload_blocks": first.get("integer_lattice_payload_blocks", first["payload_bits"]),
            "recovery_invocation_rate": float(len(recovery_rows) / max(len(non_clean), 1)),
            "fast_path_extract_ms_median": float(np.median([r["extract_ms"] for r in fast_rows])) if fast_rows else float("nan"),
            "recovery_path_extract_ms_median": float(np.median([r["extract_ms"] for r in recovery_rows])) if recovery_rows else float("nan"),
            "recovery_path_extract_ms_p95": float(np.percentile([r["extract_ms"] for r in recovery_rows], 95)) if recovery_rows else float("nan"),
        })
    save_csv(out / "all_methods_all_metrics.csv", rows)
    save_csv(out / "all_methods_summary.csv", summary_rows)
    save_csv(out / "metric_applicability.csv", applicability)
    protocol = {
        "protocol_version": ALL_METRICS_PROTOCOL_VERSION,
        "purpose": "supplementary complete metric schema across all methods",
        "primary_paper_native_results_remain_separate": True,
        "forced_common_psnr": False,
        "common_ecc": False,
        "metrics": [
            "bit_accuracy", "BER", "empirical_error_probability", "bipolar_NC",
            "4-bit_SER", "MSE", "PSNR", "PRD", "SSIM", "distortion statistics",
            "embedding/extraction timing", "key bytes", "native detector score",
        ],
        "detector_types": {m: _detector_type(m) for m in selected},
        "method_params_supplied": sorted(params_map),
        "watermark_label": watermark_label or (Path(watermark_path).name if watermark_path else "generated"),
        "watermark_statistics": wm_stats,
    }
    save_json(out / "all_metrics_protocol.json", protocol)
    return {"rows": rows, "summary": summary_rows, "applicability": applicability, "protocol": protocol}


__all__ = [
    "ALL_METRICS_PROTOCOL_VERSION", "metric_applicability_rows",
    "run_all_methods_all_metrics",
]
