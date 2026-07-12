from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import hashlib
import json
import time

import numpy as np

from .core import (
    bit_accuracy, ber, default_attacks, load_binary_watermark, load_rgb, make_demo_host,
    nc_binary, psnr, save_csv, save_image, save_json, ssim_gray,
)
from .methods import ALL_METHODS, METHOD_REFS, METHOD_ALIASES, normalize_method_id, embed, extract, method_metadata, warmup_numba, used_units_from_key
from .registry import IMAGE_BENCHMARK_METHODS
from .paper_fidelity import common_primary_static_eligible, fidelity_status, fidelity_tier


def run_compare(
    *,
    host_path: str | Path | None,
    watermark_path: str | Path | None,
    out_dir: str | Path,
    mode: str = "full",
    methods: list[str] | None = None,
    wm_size: int = 64,
    repeat: int | str = "auto",
    seed: int = 2026,
    step_overrides: dict[str, float] | None = None,
    target_psnr: float | None = None,
    psnr_tolerance: float = 0.75,
    save_images: bool = False,
    attack_preset: str = "standard",
    method_params_by_method: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if host_path is None:
        host = make_demo_host(1920, 1080) if mode == "full" else make_demo_host(512, 512)
        host_label = f"demo_{host.shape[1]}x{host.shape[0]}"
    else:
        host = load_rgb(host_path, mode=mode)
        host_label = str(host_path)
    wm = load_binary_watermark(watermark_path, size=wm_size)
    wm_label = "demo" if watermark_path is None else str(watermark_path)
    methods = methods or IMAGE_BENCHMARK_METHODS
    if "all" in methods:
        methods = IMAGE_BENCHMARK_METHODS

    warmup_numba()

    attack_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    keys: dict[str, Any] = {}

    if save_images:
        save_image(out / "input_host.png", host)
        save_image(out / "input_watermark.png", wm)

    for method_id in methods:
        requested_method_id = method_id
        method_id = normalize_method_id(method_id)
        if method_id not in ALL_METHODS:
            raise KeyError(f"Unknown method {requested_method_id}. Choices: {ALL_METHODS}; legacy aliases: {sorted(METHOD_ALIASES)}")
        print(f"[RUN] {method_id}")
        mdir = out / method_id
        mdir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        frozen_step = None if step_overrides is None else step_overrides.get(method_id)
        method_params = dict((method_params_by_method or {}).get(method_id, {}))
        watermarked, key = embed(
            method_id, host, wm, seed=seed, repeat=repeat, step=frozen_step,
            method_params=method_params,
        )
        effective_step = float(key.step)
        generic_config_payload = {"method_id": method_id, "step": effective_step, "method_params": method_params}
        generic_config_hash = hashlib.sha256(
            json.dumps(generic_config_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()
        config_hash = str(key.params.get("config_hash_sha256") or generic_config_hash)
        effective_params_for_report = key.params.get("frozen_config", method_params)
        t1 = time.perf_counter()
        rec_clean = extract(watermarked, key)
        t2 = time.perf_counter()

        embed_ms = (t1 - t0) * 1000.0
        clean_extract_ms = (t2 - t1) * 1000.0
        wpsnr = psnr(host, watermarked)
        wssim = ssim_gray(host, watermarked)
        clean_nc = nc_binary(wm, rec_clean)
        clean_ber = ber(wm, rec_clean)
        clean_acc = bit_accuracy(wm, rec_clean)
        static_eligible = common_primary_static_eligible(method_id)
        distortion_ok = target_psnr is None or abs(wpsnr - float(target_psnr)) <= float(psnr_tolerance)
        clean_ok = clean_acc >= 1.0 - 1e-12
        primary_eligible = bool(static_eligible and distortion_ok and clean_ok)
        if not static_eligible:
            eligibility_reason = str(fidelity_status(method_id).get("eligibility_reason", "non-primary fidelity tier"))
        elif not clean_ok:
            eligibility_reason = f"clean recovery gate failed ({clean_acc:.6f} < 1.000000)"
        elif not distortion_ok:
            eligibility_reason = f"matched-PSNR gate failed (target={target_psnr}, measured={wpsnr:.3f})"
        else:
            eligibility_reason = "passed static fidelity, clean-recovery, and distortion gates"

        if save_images:
            save_image(mdir / "watermarked.png", watermarked)
            save_image(mdir / "extracted_clean.png", rec_clean)
        save_json(mdir / "key.json", asdict(key))
        keys[method_id] = asdict(key)

        method_attack_rows: list[dict[str, Any]] = []
        for attack_name, attack_fn in default_attacks(attack_preset):
            if attack_name == "clean":
                attacked = watermarked
                attack_ms = 0.0
            else:
                a0 = time.perf_counter()
                attacked = attack_fn(watermarked)
                a1 = time.perf_counter()
                attack_ms = (a1 - a0) * 1000.0
            e0 = time.perf_counter()
            rec = extract(attacked, key)
            e1 = time.perf_counter()
            row = {
                "method_id": method_id,
                "display_name": METHOD_REFS[method_id].display_name,
                "fidelity_tier": fidelity_tier(method_id),
                "primary_rank_eligible": primary_eligible,
                "phase": attack_name,
                "host_mode": mode,
                "host_shape": f"{host.shape[1]}x{host.shape[0]}",
                "repeat_effective": key.repeat,
                "step": key.step,
                "configuration_sha256": config_hash,
                "watermarked_psnr": wpsnr,
                "watermarked_ssim": wssim,
                "attack_psnr_to_watermarked": psnr(watermarked, attacked) if attack_name != "clean" else float("inf"),
                "nc_bipolar": nc_binary(wm, rec),
                "bit_accuracy": bit_accuracy(wm, rec),
                "ber": ber(wm, rec),
                "embed_ms": embed_ms,
                "extract_ms": (e1 - e0) * 1000.0,
                "attack_ms": attack_ms,
                "embed_fps": 1000.0 / max(embed_ms, 1e-9),
            }
            attack_rows.append(row)
            method_attack_rows.append(row)
            if save_images and attack_name != "clean":
                save_image(mdir / f"extracted_{attack_name}.png", rec)

        non_clean = [r for r in method_attack_rows if r["phase"] != "clean"]
        summary_rows.append({
            "method_id": method_id,
            "display_name": METHOD_REFS[method_id].display_name,
            "configuration_sha256": config_hash,
            "method_params_json": json.dumps(effective_params_for_report, sort_keys=True, separators=(",", ":")),
            "fidelity_tier": fidelity_tier(method_id),
            "static_common_primary_eligible": static_eligible,
            "primary_rank_eligible": primary_eligible,
            "eligibility_reason": eligibility_reason,
            "host_mode": mode,
            "host_shape": f"{host.shape[1]}x{host.shape[0]}",
            "repeat_effective": key.repeat,
            "used_blocks_or_coeffs": used_units_from_key(key),
            "watermarked_psnr": wpsnr,
            "watermarked_ssim": wssim,
            "clean_nc_bipolar": clean_nc,
            "clean_bit_accuracy": clean_acc,
            "clean_ber": clean_ber,
            "embed_ms": embed_ms,
            "extract_clean_ms": clean_extract_ms,
            "embed_fps": 1000.0 / max(embed_ms, 1e-9),
            "min_attack_nc_bipolar": min(r["nc_bipolar"] for r in non_clean),
            "avg_attack_nc_bipolar": float(np.mean([r["nc_bipolar"] for r in non_clean])) ,
            "max_attack_ber": max(r["ber"] for r in non_clean),
        })

    save_csv(out / "per_attack_results.csv", attack_rows)
    save_csv(out / "summary_by_method.csv", summary_rows)
    save_csv(out / "per_attack_primary.csv", [r for r in attack_rows if r.get("primary_rank_eligible")])
    save_csv(out / "summary_primary.csv", [r for r in summary_rows if r.get("primary_rank_eligible")])
    save_json(out / "method_metadata.json", method_metadata())
    save_json(out / "run_config.json", {
        "host": host_label,
        "host_shape": list(host.shape),
        "watermark": wm_label,
        "watermark_shape": list(wm.shape),
        "mode": mode,
        "seed": seed,
        "repeat": repeat,
        "step_overrides_frozen": step_overrides or {},
        "method_params_by_method": method_params_by_method or {},
        "target_psnr": target_psnr,
        "psnr_tolerance": psnr_tolerance,
        "metric": "bipolar {-1,+1} NC; bit accuracy/BER primary",
        "methods": methods,
        "attack_preset": attack_preset,
    })
    return {"summary": summary_rows, "attacks": attack_rows}
