#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realtime_watermark.paper_native import run_all_paper_native

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baseline paper-native metrics and attack models without PSNR matching."
    )
    parser.add_argument("--data-dir", default="data/user_dataset/host")
    parser.add_argument("--out", default="reports/paper_native")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    root = Path(args.data_dir)
    hosts = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not hosts:
        raise FileNotFoundError(f"No host images found under {root}")
    run_all_paper_native(out_dir=args.out, host_paths=hosts, seed=args.seed, quick=args.quick)
    print(f"Paper-native results saved to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
