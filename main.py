#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realtime_watermark.benchmark import run_compare
from realtime_watermark.fair_benchmark import run_fair_compare
from realtime_watermark.methods import ALL_METHODS
from realtime_watermark.proposals.jilp_qim.config import DEFAULT_CONFIG, JILPQIMConfig


def parse_methods(s: str):
    if s.strip().lower() == "all":
        return ["all"]
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Watermarking benchmark with six paper-supported baselines and JILP-QIM.")
    p.add_argument("--host", default=None, help="RGB host image path. If omitted, a deterministic demo host is generated.")
    p.add_argument("--watermark", default=None, help="Binary watermark image path. If omitted, a deterministic demo watermark is generated.")
    p.add_argument("--mode", choices=["512", "full"], default="full", help="Resize host to 512x512 or use native/full size, e.g. 1920x1080.")
    p.add_argument("--methods", default="all", help="Comma-separated method IDs or 'all'.")
    p.add_argument("--wm-size", type=int, default=64, help="Binary watermark size, e.g. 64 for 64x64.")
    p.add_argument("--repeat", default="auto", help="Repetition factor or 'auto'. Auto uses repeat=1 at 512 and up to repeat=3 for large hosts.")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", default="results/run")
    p.add_argument("--save-images", action="store_true", help="Save watermarked/extracted images for visual inspection.")
    p.add_argument("--attack-preset", choices=["smoke", "standard", "extended", "stress"], default="standard", help="Attack set: smoke for CI, standard for quick tables, extended for broad tests, stress for the full stress suite.")
    p.add_argument("--fair", action="store_true", help="Run the PSNR-matched fair protocol (repeat=1, setup/online timing split).")
    p.add_argument("--target-psnr", type=float, default=45.0, help="Target PSNR used by --fair calibration.")
    p.add_argument("--timing-repeats", type=int, default=30, help="Measured timing repetitions used by --fair after warm-up.")
    p.add_argument("--timing-warmups", type=int, default=5, help="Untimed warm-up repetitions used by --fair.")
    p.add_argument("--calibration-host", action="append", default=None, help="Held-out calibration host path. Repeat for multiple calibration hosts.")
    p.add_argument("--calibration-host-count", type=int, default=3, help="Number of deterministic synthetic calibration hosts when no paths are supplied.")
    p.add_argument("--exclude-nonexact", action="store_true", help="Exclude non-exact fallback methods from the fair output.")
    p.add_argument("--proposal-config", default=None, help="Frozen JILP-QIM JSON configuration used in every benchmark stage.")
    p.add_argument("--list-methods", action="store_true")
    args = p.parse_args()

    if args.list_methods:
        print("Available methods:")
        for m in ALL_METHODS:
            print("  -", m)
        return

    repeat = args.repeat
    if repeat != "auto":
        repeat = int(repeat)
    proposal_config = JILPQIMConfig.load(args.proposal_config) if args.proposal_config else DEFAULT_CONFIG
    method_params = {"proposal": proposal_config.to_dict()}
    if args.fair:
        run_fair_compare(
            host_path=args.host, watermark_path=args.watermark, out_dir=args.out,
            mode=args.mode, methods=parse_methods(args.methods), wm_size=args.wm_size,
            seed=args.seed, target_psnr=args.target_psnr, timing_repeats=args.timing_repeats,
            timing_warmups=args.timing_warmups, calibration_host_paths=args.calibration_host,
            calibration_host_count=args.calibration_host_count, attack_preset=args.attack_preset,
            include_secondary_nonexact=not args.exclude_nonexact,
            method_params_by_method=method_params,
            frozen_steps_by_method={"proposal": proposal_config.step},
        )
    else:
        run_compare(
            host_path=args.host, watermark_path=args.watermark, out_dir=args.out,
            mode=args.mode, methods=parse_methods(args.methods), wm_size=args.wm_size,
            repeat=repeat, seed=args.seed, save_images=args.save_images,
            attack_preset=args.attack_preset,
            method_params_by_method=method_params,
        )
    print(f"\nDone. Results saved to: {Path(args.out).resolve()}")
    print("Main files: fair_summary_all.csv, fair_summary_primary.csv, fair_per_attack_all.csv, fair_per_attack_primary.csv, and fair_protocol.json in --fair mode")


if __name__ == "__main__":
    main()
