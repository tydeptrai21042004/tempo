#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from realtime_watermark.core import (
    bit_accuracy,
    default_attacks,
    load_binary_watermark,
    load_rgb,
    nc_binary,
    psnr,
)
from realtime_watermark.methods import warmup_numba
from realtime_watermark.proposals.jilp_qim.config import JILPQIMConfig
from realtime_watermark.registry import embed, extract


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-stage JILP-QIM search using all selected PNG watermarks."
    )
    parser.add_argument("--host-dir", required=True)
    parser.add_argument("--watermark-dir", required=True)
    parser.add_argument("--base-config", default="configs/proposal_frozen.json")
    parser.add_argument("--selected-config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mode", choices=["quick", "full"], default="full")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--wm-size", type=int, default=64)
    parser.add_argument("--search-hosts", type=int, default=3)
    parser.add_argument("--external-heldout-dir", default=None, help="test directory recorded but never used during selection")
    parser.add_argument(
        "--attack-preset",
        choices=["smoke", "standard", "extended", "stress", "unified", "reviewer46"],
        default="reviewer46",
    )
    parser.add_argument("--mean-psnr-drop", type=float, default=1.0)
    parser.add_argument("--min-psnr-drop", type=float, default=1.5)
    args = parser.parse_args()

    host_dir = Path(args.host_dir)
    watermark_dir = Path(args.watermark_dir)
    out = Path(args.out_dir)
    selected_config_path = Path(args.selected_config)
    out.mkdir(parents=True, exist_ok=True)

    host_paths = sorted(path for path in host_dir.glob("*.bmp") if path.is_file())
    watermark_paths = sorted(path for path in watermark_dir.glob("*.png") if path.is_file())
    if len(host_paths) < args.search_hosts:
        raise RuntimeError(
            f"Need at least {args.search_hosts} development hosts; found {len(host_paths)}."
        )
    if len(watermark_paths) != 2:
        raise RuntimeError(f"Expected exactly two PNG watermarks; found {len(watermark_paths)}.")

    search_host_paths = host_paths[: args.search_hosts]
    heldout_host_paths = (sorted(Path(args.external_heldout_dir).glob("*.bmp"))
                          if args.external_heldout_dir else host_paths[args.search_hosts :])
    if args.external_heldout_dir:
        import hashlib
        dev_hashes={hashlib.sha256(p.read_bytes()).digest() for p in search_host_paths}
        test_hashes={hashlib.sha256(p.read_bytes()).digest() for p in heldout_host_paths}
        if not heldout_host_paths or (dev_hashes & test_hashes):
            raise RuntimeError("External held-out host set is empty or overlaps the development hosts")
    hosts = [load_rgb(path, mode="512") for path in search_host_paths]
    watermarks = [load_binary_watermark(path, size=args.wm_size) for path in watermark_paths]
    attacks = list(default_attacks(args.attack_preset))
    base = JILPQIMConfig.load(args.base_config)

    warmup_numba()
    cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    def score(result: dict[str, Any]) -> float:
        return (
            float(result["mean_attack_nc"])
            + 0.35 * float(result["mean_pair_worst_nc"])
            + 0.10 * float(result["global_worst_nc"])
        )

    def evaluate(config: JILPQIMConfig, stage: str) -> dict[str, Any]:
        config = config.validated()
        config_hash = config.sha256()
        if config_hash in cache:
            return cache[config_hash]

        pair_psnr: list[float] = []
        clean_accuracy: list[float] = []
        all_attack_nc: list[float] = []
        pair_worst_nc: list[float] = []
        per_watermark: dict[str, list[float]] = {path.name: [] for path in watermark_paths}

        for host_index, host in enumerate(hosts):
            for watermark_index, watermark in enumerate(watermarks):
                pair_seed = args.seed + 1009 * host_index + 1000003 * watermark_index
                marked, key = embed(
                    "proposal",
                    host,
                    watermark,
                    seed=pair_seed,
                    repeat=1,
                    method_params=config.to_dict(),
                )
                recovered_clean = extract(marked, key)
                pair_psnr.append(float(psnr(host, marked)))
                clean_accuracy.append(float(bit_accuracy(watermark, recovered_clean)))

                current: list[float] = []
                for attack_name, attack_fn in attacks:
                    if attack_name == "clean":
                        continue
                    attacked = np.asarray(attack_fn(marked), dtype=np.uint8)
                    recovered = extract(attacked, key)
                    value = float(nc_binary(watermark, recovered))
                    current.append(value)
                    all_attack_nc.append(value)
                    per_watermark[watermark_paths[watermark_index].name].append(value)
                if not current:
                    raise RuntimeError("The search attack preset contains no non-clean attack.")
                pair_worst_nc.append(float(min(current)))

        result: dict[str, Any] = {
            "stage": stage,
            "configuration_sha256": config_hash,
            "search_host_count": len(hosts),
            "watermark_count": len(watermarks),
            "host_watermark_pair_count": len(hosts) * len(watermarks),
            "nonclean_attack_count": len([name for name, _ in attacks if name != "clean"]),
            "mean_psnr": float(np.mean(pair_psnr)),
            "min_psnr": float(np.min(pair_psnr)),
            "max_psnr": float(np.max(pair_psnr)),
            "mean_clean_bit_accuracy": float(np.mean(clean_accuracy)),
            "min_clean_bit_accuracy": float(np.min(clean_accuracy)),
            "mean_attack_nc": float(np.mean(all_attack_nc)),
            "mean_pair_worst_nc": float(np.mean(pair_worst_nc)),
            "global_worst_nc": float(np.min(all_attack_nc)),
            "parameters": config.to_dict(),
        }
        for name, values in per_watermark.items():
            safe = _safe_name(name)
            result[f"mean_attack_nc__{safe}"] = float(np.mean(values))
            result[f"worst_attack_nc__{safe}"] = float(np.min(values))
        result["robustness_score"] = score(result)
        cache[config_hash] = result
        rows.append(result)
        print(
            f"[{stage}] {config_hash[:10]} PSNR={result['mean_psnr']:.3f} "
            f"clean={result['min_clean_bit_accuracy']:.4f} "
            f"meanNC={result['mean_attack_nc']:.4f} worst={result['global_worst_nc']:.4f}",
            flush=True,
        )
        return result

    reference = evaluate(base, "reference")
    mean_psnr_floor = float(reference["mean_psnr"]) - args.mean_psnr_drop
    min_psnr_floor = float(reference["min_psnr"]) - args.min_psnr_drop

    def eligible(result: dict[str, Any]) -> bool:
        return (
            float(result["min_clean_bit_accuracy"]) >= 1.0 - 1e-12
            and float(result["mean_psnr"]) >= mean_psnr_floor
            and float(result["min_psnr"]) >= min_psnr_floor
        )

    if args.mode == "quick":
        projections = (base.projection_mode, "center_qim")
        fields = (base.field_mode,)
        pilot_counts = (base.pilot_count,)
        pilot_schedules = (base.pilot_schedule, "random")
        step_multipliers = (0.90, 1.00, 1.10)
        pilot_multipliers = (1.00,)
        rho_values = (base.rho_frac,)
        eta_values = (base.eta,)
    else:
        projections = ("minimum_distortion", "center_qim")
        fields = ("opponent", "luminance")
        pilot_counts = (64, 96, 128)
        pilot_schedules = ("random", "stratified")
        step_multipliers = (0.85, 1.00, 1.15)
        pilot_multipliers = (0.85, 1.00, 1.15)
        rho_values = (0.35, 0.45)
        eta_values = (0.05, 0.07, 0.10)

    # The sparse Hartley embedding kernel implements only minimum-distortion
    # integer closure. Center-QIM is a valid control for the original dense
    # carrier, but it is not an implemented sparse Hartley candidate.
    if base.hartley_support > 0:
        projections = ("minimum_distortion",)

    stage1: list[JILPQIMConfig] = []
    for projection_mode in projections:
        for field_mode in fields:
            for pilot_count in pilot_counts:
                for pilot_schedule in pilot_schedules:
                    stage1.append(
                        replace(
                            base,
                            projection_mode=projection_mode,
                            field_mode=field_mode,
                            pilot_count=pilot_count,
                            pilot_schedule=pilot_schedule,
                            payload_schedule_mode="random",
                            candidate_pool_multiplier=1.0,
                            enable_pilots=True,
                            enable_sync_search=True,
                            temporal_vote_mode="hard",
                        ).validated()
                    )
    for config in stage1:
        evaluate(config, "stage1_structural")

    eligible_stage1 = [row for row in rows if row["stage"] in {"reference", "stage1_structural"} and eligible(row)]
    if not eligible_stage1:
        raise RuntimeError("No Stage-1 configuration satisfied clean recovery and quality gates.")
    best_stage1_row = max(
        eligible_stage1,
        key=lambda row: (
            float(row["robustness_score"]),
            float(row["mean_attack_nc"]),
            float(row["mean_psnr"]),
        ),
    )
    best_stage1 = JILPQIMConfig.from_mapping(best_stage1_row["parameters"])

    stage2: list[JILPQIMConfig] = []
    for step_multiplier in step_multipliers:
        for pilot_multiplier in pilot_multipliers:
            for rho_frac in rho_values:
                for eta in eta_values:
                    stage2.append(
                        replace(
                            best_stage1,
                            step=float(best_stage1.step) * step_multiplier,
                            pilot_step=float(best_stage1.pilot_step) * pilot_multiplier,
                            rho_frac=rho_frac,
                            eta=eta,
                        ).validated()
                    )
    for config in stage2:
        evaluate(config, "stage2_numerical")

    eligible_rows = [row for row in rows if eligible(row)]
    if not eligible_rows:
        raise RuntimeError("No searched configuration satisfied clean recovery and quality gates.")
    selected_row = max(
        eligible_rows,
        key=lambda row: (
            float(row["robustness_score"]),
            float(row["mean_attack_nc"]),
            float(row["mean_pair_worst_nc"]),
            float(row["mean_psnr"]),
        ),
    )
    selected = JILPQIMConfig.from_mapping(selected_row["parameters"])
    selected.save(selected_config_path)

    flat_rows: list[dict[str, Any]] = []
    for result in rows:
        flat = {key: value for key, value in result.items() if key != "parameters"}
        for key, value in result["parameters"].items():
            flat[f"parameter_{key}"] = value
        flat["eligible"] = eligible(result)
        flat_rows.append(flat)
    fieldnames = sorted({key for row in flat_rows for key in row})
    with (out / "hyperparameter_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)

    artifact = {
        "selection_data": {
            "search_hosts": [str(path) for path in search_host_paths],
            "heldout_hosts_not_used_for_selection": [str(path) for path in heldout_host_paths],
            "watermarks_used_for_every_candidate": [str(path) for path in watermark_paths],
        },
        "protocol": {
            "mode": args.mode,
            "seed": args.seed,
            "watermark_size": args.wm_size,
            "attack_preset": args.attack_preset,
            "forced_fixed_psnr": False,
            "common_ecc": False,
            "clean_recovery_required_for_both_watermarks": True,
            "reference_configuration_sha256": base.sha256(),
            "reference_mean_psnr": reference["mean_psnr"],
            "reference_min_psnr": reference["min_psnr"],
            "mean_psnr_floor": mean_psnr_floor,
            "min_psnr_floor": min_psnr_floor,
            "stage1_candidate_count": len(stage1),
            "stage2_candidate_count": len(stage2),
            "unique_evaluated_count": len(rows),
            "selection_score": "mean_attack_nc + 0.35*mean_pair_worst_nc + 0.10*global_worst_nc",
        },
        "selected": selected_row,
    }
    (out / "selected_configuration.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"Selected configuration: {selected_config_path}")
    print(f"Selected SHA-256: {selected.sha256()}")


if __name__ == "__main__":
    main()
