from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable
import time

import numpy as np
from PIL import Image

from realtime_watermark.core import (
    bit_accuracy, default_attacks, load_binary_watermark, load_rgb, make_demo_host,
    make_demo_watermark, nc_binary, psnr, save_csv, save_image, save_json,
)
from realtime_watermark.methods import warmup_numba
from realtime_watermark.fair_benchmark import make_independent_calibration_host
from realtime_watermark.proposals.jilp_qim.config import JILPQIMConfig
from realtime_watermark.proposals.jilp_qim.method import extract_jilp_qim
from realtime_watermark.registry import embed

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def evaluate_variant(
    host: np.ndarray,
    wm: np.ndarray,
    *,
    label: str,
    params: dict[str, Any],
    seed: int,
    attack_preset: str,
    host_id: str,
    host_file: str,
    changed_component: str | None,
) -> dict[str, Any]:
    config = JILPQIMConfig.from_mapping(params)
    t0 = time.perf_counter()
    watermarked, key = embed("proposal", host, wm, seed=seed, repeat=1, method_params=config.to_dict())
    t1 = time.perf_counter()
    clean, clean_meta = extract_jilp_qim(
        watermarked, key, sync=True, return_meta=True
    )
    t2 = time.perf_counter()
    attack_rows: list[dict[str, Any]] = []
    for attack_name, attack_fn in default_attacks(attack_preset):
        attacked = watermarked if attack_name == "clean" else attack_fn(watermarked)
        e0 = time.perf_counter()
        recovered, extraction_meta = extract_jilp_qim(
            attacked, key, sync=True, return_meta=True
        )
        e1 = time.perf_counter()
        attack_rows.append({
            "host_id": host_id,
            "host_file": host_file,
            "variant": label,
            "changed_component": changed_component or "none",
            "configuration_sha256": config.sha256(),
            "attack": attack_name,
            "nc": nc_binary(wm, recovered),
            "bit_accuracy": bit_accuracy(wm, recovered),
            "extract_ms": (e1 - e0) * 1000.0,
            "extraction_path": extraction_meta.get("path", ""),
            "sync_candidate": extraction_meta.get("candidate", ""),
            "pilot_score": extraction_meta.get("score", float("nan")),
            "pilot_valid": extraction_meta.get("pilot_valid", 0),
            "estimated_shift_y": extraction_meta.get("shift_y", 0),
            "estimated_shift_x": extraction_meta.get("shift_x", 0),
        })
    non_clean = [r for r in attack_rows if r["attack"] != "clean"]
    recovery_rows = [r for r in non_clean if r.get("extraction_path") == "wide_sync_jilp"]
    fast_rows = [r for r in non_clean if r.get("extraction_path") == "raw_fast_jilp"]
    return {
        "host_id": host_id,
        "host_file": host_file,
        "variant": label,
        "changed_component": changed_component or "none",
        "configuration_sha256": config.sha256(),
        "step": key.step,
        "pilot_step": key.params.get("step_pilot"),
        "pilot_count": key.params.get("pilot_count"),
        "candidate_pool_multiplier": key.params.get("candidate_pool_multiplier"),
        "payload_schedule_mode": key.params.get("payload_schedule_mode"),
        "schedule_storage": key.params.get("schedule_storage"),
        "content_adaptive_active": key.params.get("content_adaptive_block_selection"),
        "candidate_pool_used": key.params.get("candidate_pool_used"),
        "side_information": key.side_information,
        "key_bytes": len(str(asdict(key)).encode("utf-8")),
        "psnr": psnr(host, watermarked),
        "clean_nc": nc_binary(wm, clean),
        "clean_bit_accuracy": bit_accuracy(wm, clean),
        "avg_attack_nc": float(np.mean([r["nc"] for r in non_clean])) if non_clean else float("nan"),
        "min_attack_nc": float(min(r["nc"] for r in non_clean)) if non_clean else float("nan"),
        "embed_ms": (t1 - t0) * 1000.0,
        "extract_clean_ms": (t2 - t1) * 1000.0,
        "clean_extraction_path": clean_meta.get("path", ""),
        "integer_lattice_unit_updates": int(key.params.get("integer_lattice_unit_updates", 0)),
        "integer_lattice_unresolved": int(key.params.get("integer_lattice_unresolved", 0)),
        "integer_lattice_corrected_blocks": int(key.params.get("integer_lattice_corrected_blocks", 0)),
        "integer_lattice_initial_parity_failures": int(key.params.get("integer_lattice_initial_parity_failures", 0)),
        "integer_lattice_initial_guard_failures": int(key.params.get("integer_lattice_initial_guard_failures", 0)),
        "integer_lattice_payload_blocks": int(key.params.get("integer_lattice_payload_blocks", wm.size)),
        "recovery_invocation_rate": float(len(recovery_rows) / max(len(non_clean), 1)),
        "fast_path_attack_ms_median": float(np.median([r["extract_ms"] for r in fast_rows])) if fast_rows else float("nan"),
        "recovery_path_attack_ms_median": float(np.median([r["extract_ms"] for r in recovery_rows])) if recovery_rows else float("nan"),
        "recovery_path_attack_ms_p95": float(np.percentile([r["extract_ms"] for r in recovery_rows], 95)) if recovery_rows else float("nan"),
        "attack_rows": attack_rows,
        "key": asdict(key),
    }


def find_hosts(host: str | None, host_dir: str | None, max_hosts: int | None) -> list[tuple[str, str | None]]:
    if host_dir:
        paths = sorted(
            p for p in Path(host_dir).rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if max_hosts is not None:
            paths = paths[:max(1, int(max_hosts))]
        if not paths:
            raise FileNotFoundError(f"No host images found under {host_dir}")
        return [(p.stem, str(p)) for p in paths]
    if host:
        p = Path(host)
        return [(p.stem, str(p))]
    count = max(1, int(max_hosts or 1))
    return [(f"demo_host_{index}", None) for index in range(count)]


def load_host(path: str | None, host_size: int, demo_index: int = 0) -> np.ndarray:
    if path is None:
        if int(demo_index) == 0:
            return make_demo_host(host_size, host_size)
        return make_independent_calibration_host(
            host_size, host_size, seed=17011 + 997 * int(demo_index)
        )
    host = load_rgb(path, mode="full")
    if host.shape[:2] == (host_size, host_size):
        return host
    return np.asarray(
        Image.fromarray(host, mode="RGB").resize((host_size, host_size), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )


def load_watermark(path: str | None, wm_size: int) -> np.ndarray:
    return load_binary_watermark(path, size=wm_size) if path else make_demo_watermark(wm_size)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for variant in sorted({str(row["variant"]) for row in rows}):
        group = [row for row in rows if row["variant"] == variant]
        first = group[0]
        avg = lambda name: float(np.mean([float(row[name]) for row in group]))
        output.append({
            "variant": variant,
            "changed_component": first["changed_component"],
            "configuration_sha256": first["configuration_sha256"],
            "n_hosts": len(group),
            "step": first["step"],
            "payload_schedule_mode": first["payload_schedule_mode"],
            "content_adaptive_active": first["content_adaptive_active"],
            "psnr": avg("psnr"),
            "clean_nc": avg("clean_nc"),
            "avg_attack_nc": avg("avg_attack_nc"),
            "min_attack_nc": avg("min_attack_nc"),
            "embed_ms": avg("embed_ms"),
            "extract_clean_ms": avg("extract_clean_ms"),
            "integer_lattice_unit_updates": avg("integer_lattice_unit_updates"),
            "integer_lattice_unresolved": avg("integer_lattice_unresolved"),
            "integer_lattice_corrected_blocks": avg("integer_lattice_corrected_blocks"),
            "integer_lattice_initial_parity_failures": avg("integer_lattice_initial_parity_failures"),
            "integer_lattice_initial_guard_failures": avg("integer_lattice_initial_guard_failures"),
            "integer_lattice_payload_blocks": avg("integer_lattice_payload_blocks"),
            "recovery_invocation_rate": avg("recovery_invocation_rate"),
            "fast_path_attack_ms_median": avg("fast_path_attack_ms_median"),
            "recovery_path_attack_ms_median": avg("recovery_path_attack_ms_median"),
            "recovery_path_attack_ms_p95": avg("recovery_path_attack_ms_p95"),
        })
    return output


def run_variants(
    *,
    variants: Iterable[tuple[str, dict[str, Any], str | None]],
    out_dir: str,
    prefix: str,
    host: str | None,
    host_dir: str | None,
    max_hosts: int | None,
    watermark: str | None,
    host_size: int,
    wm_size: int,
    seed: int,
    seed_stride: int,
    attack_preset: str,
    manifest: dict[str, Any],
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    warmup_numba()
    wm = load_watermark(watermark, wm_size)
    save_image(out / "input_watermark.png", wm)
    specs = find_hosts(host, host_dir, max_hosts)
    variant_list = list(variants)
    rows: list[dict[str, Any]] = []
    attacks: list[dict[str, Any]] = []
    for host_index, (host_id, host_file) in enumerate(specs):
        host_arr = load_host(host_file, host_size, demo_index=host_index)
        if host_index == 0:
            save_image(out / "input_host.png", host_arr)
        host_seed = int(seed) + int(seed_stride) * host_index
        for label, params, changed_component in variant_list:
            result = evaluate_variant(
                host_arr, wm, label=label, params=params, seed=host_seed,
                attack_preset=attack_preset, host_id=host_id,
                host_file=host_file or "demo", changed_component=changed_component,
            )
            rows.append({k: v for k, v in result.items() if k not in {"attack_rows", "key"}})
            attacks.extend(result["attack_rows"])
            if len(specs) == 1:
                save_json(out / f"{label}_key.json", result["key"])
    save_csv(out / f"{prefix}_by_host_summary.csv", rows)
    save_csv(out / f"{prefix}_summary.csv", aggregate(rows))
    save_csv(out / f"{prefix}_per_attack.csv", attacks)
    save_json(out / f"{prefix}_manifest.json", manifest)
