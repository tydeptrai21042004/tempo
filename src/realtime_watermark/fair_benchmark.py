from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json
import math
import os
import resource
import time

import numpy as np

from .common.timing import (
    environment_snapshot,
    megapixels_per_second,
    payload_throughput_bps,
    summarize_ms,
)
from .core import (
    bit_accuracy,
    ber,
    default_attacks,
    load_binary_watermark,
    load_rgb,
    make_demo_host,
    nc_binary,
    psnr,
    save_csv,
    save_json,
    ssim_gray,
)
from .methods import extract, warmup_numba
from .paper_fidelity import common_primary_static_eligible, fidelity_status, fidelity_tier
from .registry import IMAGE_BENCHMARK_METHODS, METHOD_REFS, normalize_method_id, prepare_embedding, embed_prepared
from .proposals.jilp_qim.method import extract_jilp_qim
from .proposals.jilp_qim.config import JILPQIMConfig

STEP_BOUNDS: dict[str, tuple[float, float]] = {
    "dct_dm_qim_chen2001": (2.0, 160.0),
    "dct_stdm_qim_chen2001": (1.0, 160.0),
    "dct_iss_malvar2003": (0.25, 80.0),
    "ca_qim_a2_mao2024": (1.0, 160.0),
    "proposal": (0.5, 100.0),
}

FIDELITY_TIER = {m: fidelity_tier(m) for m in IMAGE_BENCHMARK_METHODS}


def make_independent_calibration_host(width: int, height: int, seed: int = 90210) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    yy, xx = np.mgrid[0:height, 0:width]
    x = xx / max(width - 1, 1)
    y = yy / max(height - 1, 1)
    base = 90 + 70 * x + 35 * y + 24 * np.sin(11 * x + 4 * y) + 18 * np.cos(7 * y - 3 * x)
    texture = rng.normal(0.0, 6.0, size=(height, width))
    rgb = np.stack((base + texture, 0.85 * base + 20 * np.sin(9 * y), 1.1 * base + 15 * np.cos(8 * x)), axis=-1)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def _array_sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _embed_once(
    method_id: str,
    host: np.ndarray,
    wm: np.ndarray,
    seed: int,
    step: float,
    method_params: dict[str, Any] | None = None,
):
    prepared = prepare_embedding(
        method_id, host, wm, seed=seed, repeat="auto", step=step, method_params=method_params
    )
    return embed_prepared(
        method_id, host, wm, prepared, seed=seed, repeat="auto", step=step,
        method_params=method_params,
    )


def _median_psnr_for_step(
    method_id: str,
    hosts: Iterable[np.ndarray],
    wm: np.ndarray,
    seed: int,
    step: float,
    method_params: dict[str, Any] | None = None,
) -> float:
    values = [
        psnr(host, _embed_once(method_id, host, wm, seed, step, method_params)[0])
        for host in hosts
    ]
    if not values:
        raise ValueError("at least one calibration host is required")
    return float(np.median(values))


def calibrate_step_to_psnr(
    method_id: str,
    host: np.ndarray,
    wm: np.ndarray,
    *,
    target_psnr: float,
    seed: int,
    iterations: int = 8,
    method_params: dict[str, Any] | None = None,
) -> tuple[float, float]:
    return calibrate_step_on_hosts(
        method_id, [host], wm, target_psnr=target_psnr, seed=seed,
        iterations=iterations, method_params=method_params,
    )


def calibrate_step_on_hosts(
    method_id: str,
    hosts: Iterable[np.ndarray],
    wm: np.ndarray,
    *,
    target_psnr: float,
    seed: int,
    iterations: int = 8,
    method_params: dict[str, Any] | None = None,
) -> tuple[float, float]:
    method_id = normalize_method_id(method_id)
    host_list = [np.asarray(h, dtype=np.uint8) for h in hosts]
    if not host_list:
        raise ValueError("at least one calibration host is required")
    lo, hi = STEP_BOUNDS[method_id]
    candidates: list[tuple[float, float, float]] = []
    # Logarithmic exploration is retained, but every observation is kept so a
    # mildly non-monotonic method still selects the closest measured point.
    for _ in range(max(1, int(iterations))):
        step = math.sqrt(lo * hi)
        value = _median_psnr_for_step(method_id, host_list, wm, seed, step, method_params)
        candidates.append((abs(value - float(target_psnr)), step, value))
        if value > target_psnr:
            lo = step
        else:
            hi = step
    _, best_step, best_value = min(candidates, key=lambda x: x[0])
    return float(best_step), float(best_value)


def calibrate_steps_on_hosts(
    methods: Iterable[str], hosts: Iterable[np.ndarray], wm: np.ndarray, *,
    target_psnr: float, seed: int, iterations: int = 8,
    method_params_by_method: dict[str, dict[str, Any]] | None = None,
    frozen_steps_by_method: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    host_list = [np.asarray(h, dtype=np.uint8) for h in hosts]
    result: dict[str, dict[str, float]] = {}
    for method in methods:
        method_id = normalize_method_id(method)
        method_params = (method_params_by_method or {}).get(method_id)
        provided = {normalize_method_id(k): float(v) for k, v in (frozen_steps_by_method or {}).items()}
        if method_id in provided:
            step = provided[method_id]
            observed = _median_psnr_for_step(method_id, host_list, wm, seed, step, method_params)
            source = "externally_frozen_configuration"
        else:
            step, observed = calibrate_step_on_hosts(
                method_id, host_list, wm, target_psnr=target_psnr, seed=seed,
                iterations=iterations, method_params=method_params,
            )
            source = "heldout_calibration"
        result[method_id] = {
            "step": float(step), "calibration_median_psnr": float(observed),
            "step_source": source,
        }
    return result


def _key_bytes(key: Any) -> int:
    return len(json.dumps(asdict(key), separators=(",", ":"), default=str).encode("utf-8"))


def _effective_method_params(method_id: str, step: float, method_params: dict[str, Any]) -> dict[str, Any]:
    method_id = normalize_method_id(method_id)
    if method_id == "proposal":
        return JILPQIMConfig.from_mapping(method_params).with_step(step).to_dict()
    return dict(method_params)


def _configuration_hash(method_id: str, step: float, method_params: dict[str, Any]) -> str:
    method_id = normalize_method_id(method_id)
    effective = _effective_method_params(method_id, step, method_params)
    if method_id == "proposal":
        return JILPQIMConfig.from_mapping(effective).sha256()
    payload = {"method_id": method_id, "step": float(step), "method_params": effective}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _eligibility(method_id: str, clean_accuracy: float, measured_psnr: float, target_psnr: float, tolerance: float) -> tuple[bool, str]:
    if not common_primary_static_eligible(method_id):
        return False, str(fidelity_status(method_id).get("eligibility_reason", "non-primary fidelity tier"))
    if clean_accuracy < 1.0 - 1e-12:
        return False, f"clean recovery gate failed ({clean_accuracy:.6f} < 1.000000)"
    if abs(measured_psnr - target_psnr) > tolerance:
        return False, f"matched-PSNR gate failed (|{measured_psnr:.3f}-{target_psnr:.3f}|>{tolerance:.3f})"
    return True, "passed declared-fidelity, clean-recovery, and matched-distortion gates"


def _load_calibration_hosts(
    *,
    host_shape: tuple[int, ...],
    mode: str,
    seed: int,
    calibration_host_path: str | Path | None,
    calibration_host_paths: Iterable[str | Path] | None,
    calibration_host_count: int,
) -> tuple[list[np.ndarray], list[str]]:
    paths = [Path(x) for x in (calibration_host_paths or [])]
    if calibration_host_path is not None:
        paths.append(Path(calibration_host_path))
    if paths:
        hosts = [load_rgb(path, mode=mode) for path in paths]
        sources = [str(path) for path in paths]
    else:
        count = max(1, int(calibration_host_count))
        hosts = [
            make_independent_calibration_host(host_shape[1], host_shape[0], seed=seed + 7001 + 97 * i)
            for i in range(count)
        ]
        sources = [f"independent_synthetic_calibration_host_{i}" for i in range(count)]
    for item in hosts:
        if item.shape != host_shape:
            raise ValueError(f"calibration/evaluation host shapes differ: {item.shape} vs {host_shape}")
    return hosts, sources


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.
    return value / (1024.0 if value < 10_000_000 else 1024.0 * 1024.0)


def run_fair_compare(
    *,
    host_path: str | Path | None,
    watermark_path: str | Path | None,
    out_dir: str | Path,
    calibration_host_path: str | Path | None = None,
    calibration_host_paths: Iterable[str | Path] | None = None,
    calibration_host_count: int = 3,
    mode: str = "full",
    methods: list[str] | None = None,
    wm_size: int = 64,
    seed: int = 2026,
    target_psnr: float = 45.0,
    psnr_tolerance: float = 0.75,
    calibration_iterations: int = 8,
    timing_repeats: int = 30,
    timing_warmups: int = 5,
    attack_preset: str = "standard",
    include_secondary_nonexact: bool = True,
    method_params_by_method: dict[str, dict[str, Any]] | None = None,
    frozen_steps_by_method: dict[str, float] | None = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    host = make_demo_host(1920, 1080) if host_path is None and mode == "full" else (
        make_demo_host(512, 512) if host_path is None else load_rgb(host_path, mode=mode)
    )
    calibration_hosts, calibration_sources = _load_calibration_hosts(
        host_shape=host.shape,
        mode=mode,
        seed=seed,
        calibration_host_path=calibration_host_path,
        calibration_host_paths=calibration_host_paths,
        calibration_host_count=calibration_host_count,
    )
    evaluation_hash = _array_sha256(host)
    calibration_hashes = [_array_sha256(item) for item in calibration_hosts]
    if evaluation_hash in calibration_hashes:
        raise ValueError("Calibration and evaluation hosts must be disjoint; an identical SHA-256 hash was found.")

    wm = load_binary_watermark(watermark_path, size=wm_size)
    selected = IMAGE_BENCHMARK_METHODS if not methods or "all" in methods else [normalize_method_id(x) for x in methods]
    if not include_secondary_nonexact:
        selected = [m for m in selected if common_primary_static_eligible(m)]
    warmup_numba()

    summary_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    attack_rows: list[dict[str, Any]] = []
    frozen_steps: dict[str, float] = {}

    for method_id in selected:
        method_id = normalize_method_id(method_id)
        method_params = dict((method_params_by_method or {}).get(method_id, {}))
        c0 = time.perf_counter()
        provided_steps = {normalize_method_id(k): float(v) for k, v in (frozen_steps_by_method or {}).items()}
        if method_id in provided_steps:
            step = float(provided_steps[method_id])
            calibration_psnr = _median_psnr_for_step(
                method_id, calibration_hosts, wm, seed, step, method_params
            )
            step_source = "externally_frozen_configuration"
        else:
            step, calibration_psnr = calibrate_step_on_hosts(
                method_id, calibration_hosts, wm, target_psnr=target_psnr, seed=seed,
                iterations=calibration_iterations, method_params=method_params,
            )
            step_source = "heldout_calibration"
        calibration_ms = (time.perf_counter() - c0) * 1000.0
        frozen_steps[method_id] = step
        effective_method_params = _effective_method_params(method_id, step, method_params)
        config_hash = _configuration_hash(method_id, step, method_params)

        # Untimed warm-up for the requested method and configuration.
        for _ in range(max(0, int(timing_warmups))):
            prepared = prepare_embedding(method_id, host, wm, seed=seed, repeat="auto", step=step, method_params=method_params)
            warmed, warmed_key = embed_prepared(method_id, host, wm, prepared, seed=seed, repeat="auto", step=step, method_params=method_params)
            _ = extract(warmed, warmed_key)
            if method_id == "proposal":
                _ = extract_jilp_qim(warmed, warmed_key, sync=False)

        setup_times: list[float] = []
        online_times: list[float] = []
        total_times: list[float] = []
        extract_sync_times: list[float] = []
        extract_fast_times: list[float] = []
        watermarked = key = recovered = None
        for _ in range(max(1, int(timing_repeats))):
            t0 = time.perf_counter_ns()
            prepared = prepare_embedding(method_id, host, wm, seed=seed, repeat="auto", step=step, method_params=method_params)
            t1 = time.perf_counter_ns()
            watermarked, key = embed_prepared(method_id, host, wm, prepared, seed=seed, repeat="auto", step=step, method_params=method_params)
            t2 = time.perf_counter_ns()
            recovered = extract(watermarked, key)
            t3 = time.perf_counter_ns()
            if method_id == "proposal":
                _ = extract_jilp_qim(watermarked, key, sync=False)
                t4 = time.perf_counter_ns()
                extract_fast_times.append((t4 - t3) / 1_000_000.0)
            else:
                extract_fast_times.append((t3 - t2) / 1_000_000.0)
            setup_times.append((t1 - t0) / 1_000_000.0)
            online_times.append((t2 - t1) / 1_000_000.0)
            total_times.append((t2 - t0) / 1_000_000.0)
            extract_sync_times.append((t3 - t2) / 1_000_000.0)
        assert watermarked is not None and key is not None and recovered is not None
        if method_id == "proposal":
            key_hash = str(key.params.get("config_hash_sha256", ""))
            if key_hash != config_hash:
                raise RuntimeError(
                    f"Proposal configuration identity mismatch: benchmark={config_hash}, key={key_hash}"
                )

        setup_dist = summarize_ms(setup_times)
        online_dist = summarize_ms(online_times)
        total_dist = summarize_ms(total_times)
        sync_dist = summarize_ms(extract_sync_times)
        fast_dist = summarize_ms(extract_fast_times)
        for stage, dist in (
            ("prepare", setup_dist),
            ("online_embed", online_dist),
            ("total_embed", total_dist),
            ("extract_sync", sync_dist),
            ("extract_fast", fast_dist),
        ):
            timing_rows.append({"method_id": method_id, "stage": stage, **dist.to_dict()})

        measured_psnr = psnr(host, watermarked)
        clean_acc = bit_accuracy(wm, recovered)
        eligible, eligibility_reason = _eligibility(method_id, clean_acc, measured_psnr, float(target_psnr), float(psnr_tolerance))
        payload_bits = int(wm.size)
        total_ms = total_dist.median_ms
        extract_ms = sync_dist.median_ms
        key_bytes = _key_bytes(key)

        method_attack_rows: list[dict[str, Any]] = []
        for attack_name, attack_fn in default_attacks(attack_preset):
            attacked = watermarked if attack_name == "clean" else attack_fn(watermarked)
            a0 = time.perf_counter_ns()
            rec = extract(attacked, key)
            a1 = time.perf_counter_ns()
            row = {
                "method_id": method_id,
                "display_name": METHOD_REFS[method_id].display_name,
                "fidelity_tier": fidelity_tier(method_id),
                "primary_rank_eligible": eligible,
                "phase": attack_name,
                "target_psnr": float(target_psnr),
                "frozen_step": float(step),
                "configuration_sha256": config_hash,
                "step_source": step_source,
                "watermarked_psnr": measured_psnr,
                "nc_bipolar": nc_binary(wm, rec),
                "nc": nc_binary(wm, rec),
                "bit_accuracy": bit_accuracy(wm, rec),
                "ber": ber(wm, rec),
                "extract_ms": (a1 - a0) / 1_000_000.0,
            }
            attack_rows.append(row)
            method_attack_rows.append(row)

        non_clean = [r for r in method_attack_rows if r["phase"] != "clean"]
        fidelity = fidelity_status(method_id)
        summary_rows.append({
            "schema_version": "2.0",
            "method_id": method_id,
            "display_name": METHOD_REFS[method_id].display_name,
            "fidelity_tier": fidelity_tier(method_id),
            "native_method_id": fidelity["native_method_id"],
            "common_adapter_method_id": fidelity["common_adapter_method_id"],
            "paper_reproduction_eligible": bool(fidelity["paper_reproduction_eligible"]),
            "static_common_primary_eligible": common_primary_static_eligible(method_id),
            "primary_rank_eligible": eligible,
            "eligibility_reason": eligibility_reason,
            "blindness_tier": fidelity["blindness_tier"],
            "verified_capacity": fidelity["verified_capacity"],
            "host_shape": f"{host.shape[1]}x{host.shape[0]}",
            "payload_bits": payload_bits,
            "target_psnr": float(target_psnr),
            "psnr_tolerance": float(psnr_tolerance),
            "calibrated_step_frozen": float(step),
            "configuration_sha256": config_hash,
            "step_source": step_source,
            "method_params_json": json.dumps(effective_method_params, sort_keys=True, separators=(",", ":")),
            "calibration_observed_median_psnr": float(calibration_psnr),
            "calibration_host_count": len(calibration_hosts),
            "evaluation_watermarked_psnr": measured_psnr,
            "watermarked_ssim": ssim_gray(host, watermarked),
            "clean_nc_bipolar": nc_binary(wm, recovered),
            "clean_bit_accuracy": clean_acc,
            "clean_ber": ber(wm, recovered),
            "calibration_ms_not_timed": calibration_ms,
            "prepare_ms_median": setup_dist.median_ms,
            "prepare_ms_iqr": setup_dist.iqr_ms,
            "online_embed_ms_median": online_dist.median_ms,
            "online_embed_ms_iqr": online_dist.iqr_ms,
            "total_embed_ms_median": total_dist.median_ms,
            "total_embed_ms_iqr": total_dist.iqr_ms,
            "total_embed_ms_p95": total_dist.p95_ms,
            "total_embed_ms_p99": total_dist.p99_ms,
            "extract_sync_ms_median": sync_dist.median_ms,
            "extract_sync_ms_p95": sync_dist.p95_ms,
            "extract_fast_ms_median": fast_dist.median_ms,
            "extract_fast_ms_p95": fast_dist.p95_ms,
            "total_embed_fps_median": total_dist.median_fps,
            "extract_sync_fps_median": sync_dist.median_fps,
            "extract_fast_fps_median": fast_dist.median_fps,
            "embed_payload_bits_per_second": payload_throughput_bps(payload_bits, total_ms),
            "extract_payload_bits_per_second": payload_throughput_bps(payload_bits, extract_ms),
            "embed_megapixels_per_second": megapixels_per_second(host.shape[1], host.shape[0], total_ms),
            "extract_megapixels_per_second": megapixels_per_second(host.shape[1], host.shape[0], extract_ms),
            "timing_warmups": int(timing_warmups),
            "timing_repeats": int(timing_repeats),
            "repeat_fixed": 1,
            "key_bytes": key_bytes,
            "key_to_payload_byte_ratio": key_bytes / max(payload_bits / 8.0, 1e-9),
            "process_cumulative_peak_rss_mb": _peak_rss_mb(),
            "fully_blind": bool(key.fully_blind),
            "side_information": key.side_information,
            "min_attack_nc_bipolar": min((float(r["nc_bipolar"]) for r in non_clean), default=float("nan")),
            "mean_attack_nc_bipolar": float(np.mean([r["nc_bipolar"] for r in non_clean])) if non_clean else float("nan"),
            "max_attack_ber": max((float(r["ber"]) for r in non_clean), default=float("nan")),
            # Backward-compatible columns.
            "embed_ms": total_dist.median_ms,
            "extract_ms": sync_dist.median_ms,
            "embed_fps": total_dist.median_fps,
            "extract_fps": sync_dist.median_fps,
            "watermarked_psnr": measured_psnr,
            "clean_nc": nc_binary(wm, recovered),
            "avg_attack_nc": float(np.mean([r["nc_bipolar"] for r in non_clean])) if non_clean else float("nan"),
            "min_attack_nc": min((float(r["nc_bipolar"]) for r in non_clean), default=float("nan")),
        })

    save_csv(out / "fair_summary_all.csv", summary_rows)
    save_csv(out / "summary_by_method.csv", summary_rows)
    save_csv(out / "fair_summary_primary.csv", [r for r in summary_rows if r["primary_rank_eligible"]])
    save_csv(out / "fair_timing_distribution.csv", timing_rows)
    save_csv(out / "fair_per_attack_all.csv", attack_rows)
    save_csv(out / "per_attack_results.csv", attack_rows)
    save_csv(out / "fair_per_attack_primary.csv", [r for r in attack_rows if r["primary_rank_eligible"]])
    save_json(out / "fair_protocol.json", {
        "schema_version": "2.0",
        "protocol": "multi-host held-out calibration plus frozen matched-distortion evaluation",
        "payload_bits": int(wm.size),
        "repeat": 1,
        "same_host_calibration_forbidden": True,
        "target_psnr": float(target_psnr),
        "psnr_tolerance": float(psnr_tolerance),
        "attack_preset": attack_preset,
        "calibration_sources": calibration_sources,
        "calibration_host_sha256": calibration_hashes,
        "evaluation_host_sha256": evaluation_hash,
        "calibration_evaluation_disjoint_enforced": True,
        "frozen_steps": frozen_steps,
        "method_params_by_method": method_params_by_method or {},
        "externally_frozen_steps": frozen_steps_by_method or {},
        "timing_primary": "total embedding includes all host/message-dependent preparation",
        "timing_statistics": "median, IQR, p95, p99, min, max; warm-ups excluded",
        "process_isolation": "Use scripts/run_isolated_fair_benchmark.py for fresh-process method timing.",
        "metric": "bipolar {-1,+1} NC plus bit accuracy and BER",
        "video_note": "All included baselines use the same explicitly labeled framewise common-video extension.",
        "environment": environment_snapshot(),
        "methods": selected,
    })
    return {"summary": summary_rows, "timing": timing_rows, "attacks": attack_rows, "frozen_steps": frozen_steps}


__all__ = [
    "run_fair_compare",
    "calibrate_step_to_psnr",
    "calibrate_step_on_hosts",
    "calibrate_steps_on_hosts",
    "STEP_BOUNDS",
    "FIDELITY_TIER",
    "make_independent_calibration_host",
]
