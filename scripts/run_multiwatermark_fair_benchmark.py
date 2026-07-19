#!/usr/bin/env python3
from __future__ import annotations

"""Run the matched-PSNR protocol over multiple evaluation hosts and watermarks."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from realtime_watermark.core import save_csv, save_json
from realtime_watermark.fair_benchmark import run_fair_compare
from realtime_watermark.proposals.jilp_qim.config import JILPQIMConfig
from realtime_watermark.registry import normalize_method_id

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_METHODS = (
    "dct_dm_qim_chen2001",
    "dct_stdm_qim_chen2001",
    "dct_iss_malvar2003",
    "proposal",
)


def _find_images(directory: str | Path, *, excludes: list[str] | None = None) -> list[Path]:
    root = Path(directory)
    excluded = [item.lower() for item in (excludes or []) if item]
    paths = [
        path for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not any(token in path.name.lower() for token in excluded)
    ]
    if not paths:
        raise FileNotFoundError(f"No image files found under {root}")
    return paths


def _parse_methods(raw: str) -> list[str]:
    return [normalize_method_id(item.strip()) for item in raw.split(",") if item.strip()]


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows if row.get(field) not in {None, ""}]
    return float(np.mean(values)) if values else float("nan")


def _aggregate(
    pair_summaries: list[dict[str, Any]], attack_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method_id in sorted({str(row["method_id"]) for row in pair_summaries}):
        pairs = [row for row in pair_summaries if row["method_id"] == method_id]
        attacks = [
            row for row in attack_rows
            if row["method_id"] == method_id and row["phase"] != "clean"
        ]
        ncs = np.asarray([float(row["nc_bipolar"]) for row in attacks], dtype=np.float64)
        recovery = [row for row in attacks if row.get("extraction_path") == "wide_sync_jilp"]
        fast = [row for row in attacks if row.get("extraction_path") == "raw_fast_jilp"]
        output.append({
            "method_id": method_id,
            "host_watermark_pairs": len(pairs),
            "attack_observations": len(attacks),
            "mean_evaluation_psnr": _mean(pairs, "evaluation_watermarked_psnr"),
            "mean_clean_bit_accuracy": _mean(pairs, "clean_bit_accuracy"),
            "mean_attack_nc": float(np.mean(ncs)) if ncs.size else float("nan"),
            "median_attack_nc": float(np.median(ncs)) if ncs.size else float("nan"),
            "p10_attack_nc": float(np.percentile(ncs, 10)) if ncs.size else float("nan"),
            "minimum_attack_nc": float(np.min(ncs)) if ncs.size else float("nan"),
            "success_rate_nc_ge_050": float(np.mean(ncs >= 0.50)) if ncs.size else float("nan"),
            "success_rate_nc_ge_070": float(np.mean(ncs >= 0.70)) if ncs.size else float("nan"),
            "success_rate_nc_ge_090": float(np.mean(ncs >= 0.90)) if ncs.size else float("nan"),
            "mean_total_embed_ms": _mean(pairs, "total_embed_ms_median"),
            "mean_aligned_extract_ms": _mean(pairs, "extract_fast_ms_median"),
            "recovery_invocation_rate": float(len(recovery) / max(len(attacks), 1)),
            "fast_path_extract_ms_median": float(np.median([float(r["extract_ms"]) for r in fast])) if fast else float("nan"),
            "recovery_path_extract_ms_median": float(np.median([float(r["extract_ms"]) for r in recovery])) if recovery else float("nan"),
            "recovery_path_extract_ms_p95": float(np.percentile([float(r["extract_ms"]) for r in recovery], 95)) if recovery else float("nan"),
            "mean_integer_lattice_unit_updates": _mean(pairs, "integer_lattice_unit_updates"),
            "total_integer_lattice_unresolved": int(sum(int(row.get("integer_lattice_unresolved", 0)) for row in pairs)),
            "mean_integer_lattice_corrected_blocks": _mean(pairs, "integer_lattice_corrected_blocks"),
            "mean_integer_lattice_initial_parity_failures": _mean(pairs, "integer_lattice_initial_parity_failures"),
            "mean_integer_lattice_initial_guard_failures": _mean(pairs, "integer_lattice_initial_guard_failures"),
            "mean_integer_lattice_payload_blocks": _mean(pairs, "integer_lattice_payload_blocks"),
        })
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Matched-PSNR evaluation over all selected hosts and watermarks."
    )
    parser.add_argument("--host-dir", required=True)
    parser.add_argument("--watermark-dir", required=True)
    parser.add_argument("--calibration-host-dir", default=None)
    parser.add_argument("--proposal-config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--exclude-substring", action="append", default=["lenna", "lena"])
    parser.add_argument("--host-limit", type=int, default=0)
    parser.add_argument("--watermark-limit", type=int, default=0)
    parser.add_argument("--wm-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--seed-stride", type=int, default=1009)
    parser.add_argument("--target-psnr", type=float, default=45.0)
    parser.add_argument("--psnr-tolerance", type=float, default=0.75)
    parser.add_argument("--calibration-iterations", type=int, default=8)
    parser.add_argument("--calibration-host-count", type=int, default=3)
    parser.add_argument("--timing-repeats", type=int, default=10)
    parser.add_argument("--timing-warmups", type=int, default=3)
    parser.add_argument(
        "--attack-preset", choices=["smoke", "standard", "extended", "stress"],
        default="stress",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    methods = _parse_methods(args.methods)
    hosts = _find_images(args.host_dir, excludes=args.exclude_substring)
    watermarks = _find_images(args.watermark_dir)
    if args.host_limit > 0:
        hosts = hosts[: args.host_limit]
    if args.watermark_limit > 0:
        watermarks = watermarks[: args.watermark_limit]
    calibration_paths = (
        _find_images(args.calibration_host_dir) if args.calibration_host_dir else None
    )
    proposal = JILPQIMConfig.load(args.proposal_config)
    method_params = {"proposal": proposal.to_dict()}

    all_summaries: list[dict[str, Any]] = []
    all_attacks: list[dict[str, Any]] = []
    all_timing: list[dict[str, Any]] = []
    frozen_by_watermark: dict[str, dict[str, float]] = {}

    for watermark_index, watermark in enumerate(watermarks):
        frozen_steps: dict[str, float] | None = None
        watermark_seed = int(args.seed) + args.seed_stride * watermark_index
        for host_index, host in enumerate(hosts):
            pair_seed = watermark_seed
            pair_out = out / "pairs" / watermark.stem / host.stem
            result = run_fair_compare(
                host_path=host,
                watermark_path=watermark,
                out_dir=pair_out,
                calibration_host_paths=calibration_paths,
                calibration_host_count=args.calibration_host_count,
                mode="512",
                methods=methods,
                wm_size=args.wm_size,
                seed=pair_seed,
                target_psnr=args.target_psnr,
                psnr_tolerance=args.psnr_tolerance,
                calibration_iterations=args.calibration_iterations,
                timing_repeats=args.timing_repeats,
                timing_warmups=args.timing_warmups,
                attack_preset=args.attack_preset,
                include_secondary_nonexact=True,
                method_params_by_method=method_params,
                frozen_steps_by_method=frozen_steps,
            )
            if frozen_steps is None:
                frozen_steps = {key: float(value) for key, value in result["frozen_steps"].items()}
                frozen_by_watermark[watermark.name] = dict(frozen_steps)
            for row in result["summary"]:
                enriched = dict(row)
                enriched.update({
                    "host": host.name,
                    "host_index": host_index,
                    "watermark": watermark.name,
                    "watermark_index": watermark_index,
                    "evaluation_seed": pair_seed,
                })
                all_summaries.append(enriched)
            for row in result["attacks"]:
                enriched = dict(row)
                enriched.update({
                    "host": host.name,
                    "host_index": host_index,
                    "watermark": watermark.name,
                    "watermark_index": watermark_index,
                    "evaluation_seed": pair_seed,
                })
                all_attacks.append(enriched)
            for row in result["timing"]:
                enriched = dict(row)
                enriched.update({
                    "host": host.name,
                    "host_index": host_index,
                    "watermark": watermark.name,
                    "watermark_index": watermark_index,
                    "evaluation_seed": pair_seed,
                })
                all_timing.append(enriched)

    aggregate = _aggregate(all_summaries, all_attacks)
    save_csv(out / "matched_psnr_by_host_watermark.csv", all_summaries)
    save_csv(out / "matched_psnr_per_attack.csv", all_attacks)
    save_csv(out / "matched_psnr_timing.csv", all_timing)
    save_csv(out / "matched_psnr_summary_by_method.csv", aggregate)
    save_json(out / "matched_psnr_protocol.json", {
        "protocol": "multi-host multi-watermark held-out matched-PSNR evaluation",
        "target_psnr": float(args.target_psnr),
        "psnr_tolerance": float(args.psnr_tolerance),
        "attack_preset": args.attack_preset,
        "methods": methods,
        "hosts": [str(path) for path in hosts],
        "watermarks": [str(path) for path in watermarks],
        "calibration_hosts": [str(path) for path in calibration_paths] if calibration_paths else "deterministic synthetic held-out hosts",
        "calibration_host_count": int(args.calibration_host_count),
        "proposal_configuration_sha256": proposal.sha256(),
        "frozen_steps_by_watermark": frozen_by_watermark,
        "timing_repeats": int(args.timing_repeats),
        "timing_warmups": int(args.timing_warmups),
    })
    print(f"Matched-PSNR benchmark complete: {out}")
    print(f"Hosts={len(hosts)}, watermarks={len(watermarks)}, methods={len(methods)}")


if __name__ == "__main__":
    main()
