#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realtime_watermark.all_metrics import run_all_methods_all_metrics

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run every generic metric for every retained watermark method without PSNR calibration.")
    parser.add_argument("--data-dir", default="data/user_dataset/host")
    parser.add_argument("--watermark", default="data/user_dataset/watermark/wm.png")
    parser.add_argument("--out", default="reports/all_metrics")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--wm-size", type=int, default=32)
    parser.add_argument("--attack-preset", choices=["smoke", "standard", "extended", "stress", "unified"], default="standard")
    parser.add_argument("--host-limit", type=int, default=0)
    args = parser.parse_args()
    root = Path(args.data_dir)
    hosts = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if args.host_limit > 0:
        hosts = hosts[: args.host_limit]
    if not hosts:
        raise FileNotFoundError(f"No host images under {root}")
    run_all_methods_all_metrics(
        host_paths=hosts, watermark_path=args.watermark, out_dir=args.out,
        seed=args.seed, wm_size=args.wm_size, attack_preset=args.attack_preset,
    )
    print(f"All-metrics results saved to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
