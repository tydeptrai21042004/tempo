#!/usr/bin/env python3
from __future__ import annotations

"""Run the matched-PSNR benchmark for several seeds and aggregate uncertainty."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repeated-seed matched-PSNR experiment runner.")
    parser.add_argument("--host-dir", required=True)
    parser.add_argument("--watermark-dir", required=True)
    parser.add_argument("--calibration-host-dir", default=None)
    parser.add_argument("--proposal-config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", default="2026,2127,2228,2329,2430")
    parser.add_argument("--methods", default="dct_dm_qim_chen2001,dct_stdm_qim_chen2001,dct_iss_malvar2003,proposal")
    parser.add_argument("--wm-size", type=int, default=64)
    parser.add_argument("--target-psnr", type=float, default=45.0)
    parser.add_argument("--psnr-tolerance", type=float, default=0.75)
    parser.add_argument("--calibration-iterations", type=int, default=8)
    parser.add_argument("--calibration-host-count", type=int, default=3)
    parser.add_argument("--timing-repeats", type=int, default=10)
    parser.add_argument("--timing-warmups", type=int, default=3)
    parser.add_argument("--attack-preset", choices=["smoke", "standard", "extended", "stress", "unified"], default="stress")
    parser.add_argument("--host-limit", type=int, default=0)
    parser.add_argument("--watermark-limit", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)

    for seed in seeds:
        seed_out = out / f"seed_{seed}"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_multiwatermark_fair_benchmark.py"),
            "--host-dir", args.host_dir,
            "--watermark-dir", args.watermark_dir,
            "--proposal-config", args.proposal_config,
            "--out-dir", str(seed_out),
            "--methods", args.methods,
            "--wm-size", str(args.wm_size),
            "--seed", str(seed),
            "--target-psnr", str(args.target_psnr),
            "--psnr-tolerance", str(args.psnr_tolerance),
            "--calibration-iterations", str(args.calibration_iterations),
            "--calibration-host-count", str(args.calibration_host_count),
            "--timing-repeats", str(args.timing_repeats),
            "--timing-warmups", str(args.timing_warmups),
            "--attack-preset", args.attack_preset,
            "--host-limit", str(args.host_limit),
            "--watermark-limit", str(args.watermark_limit),
        ]
        if args.calibration_host_dir:
            command += ["--calibration-host-dir", args.calibration_host_dir]
        subprocess.run(command, cwd=ROOT, env=env, check=True)

    aggregate_command = [
        sys.executable,
        str(ROOT / "scripts" / "aggregate_seed_results.py"),
        "--input-glob", str(out / "seed_*" / "matched_psnr_summary_by_method.csv"),
        "--out", str(out / "multiseed_confidence_intervals.csv"),
        "--group-by", "method_id",
        "--metrics",
        "mean_attack_nc,mean_evaluation_psnr,mean_total_embed_ms,mean_aligned_extract_ms,recovery_invocation_rate,mean_integer_lattice_unit_updates",
        "--bootstrap-samples", str(args.bootstrap_samples),
        "--bootstrap-seed", str(seeds[0]),
    ]
    subprocess.run(aggregate_command, cwd=ROOT, env=env, check=True)
    print(f"Multi-seed experiment complete: {out}")


if __name__ == "__main__":
    main()
