#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def find_hosts(host_dir: Path) -> list[Path]:
    hosts = [
        path for path in sorted(host_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and valid_image(path)
    ]
    if len(hosts) < 4:
        raise SystemExit(f"Need at least four valid host images; found {len(hosts)} in {host_dir}")
    return hosts


def copy_split(paths: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(paths, start=1):
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in source.name)
        shutil.copy2(source, destination / f"{index:03d}_{safe_name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create deterministic, disjoint development and evaluation splits.")
    parser.add_argument("--host-dir", default="data/user_dataset/host")
    parser.add_argument("--watermark", default="data/user_dataset/watermark/wm.png")
    parser.add_argument("--out-dir", default="runtime/experiment")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--calibration-count", type=int, default=3)
    parser.add_argument("--development-count", type=int, default=3)
    parser.add_argument("--validation-count", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    host_dir = Path(args.host_dir).resolve()
    watermark = Path(args.watermark).resolve()
    out = Path(args.out_dir).resolve()

    if not watermark.is_file() or not valid_image(watermark):
        raise SystemExit(f"Invalid watermark image: {watermark}")

    hosts = find_hosts(host_dir)
    requested = args.calibration_count + args.development_count + args.validation_count
    if requested >= len(hosts):
        raise SystemExit(
            "calibration + development + validation counts must leave at least one evaluation host"
        )

    shuffled = list(hosts)
    random.Random(args.seed).shuffle(shuffled)
    c0 = args.calibration_count
    c1 = c0 + args.development_count
    c2 = c1 + args.validation_count
    splits = {
        "calibration": shuffled[:c0],
        "development": shuffled[c0:c1],
        "validation": shuffled[c1:c2],
        "evaluation": shuffled[c2:],
    }

    if out.exists():
        shutil.rmtree(out)
    split_root = out / "splits"
    for name, paths in splits.items():
        copy_split(paths, split_root / name)

    wm_dir = out / "watermark"
    wm_dir.mkdir(parents=True, exist_ok=True)
    wm_target = wm_dir / "wm.png"
    shutil.copy2(watermark, wm_target)

    benchmark = out / "image_benchmark_dataset"
    benchmark_host = benchmark / "host"
    benchmark_watermark = benchmark / "watermark"
    benchmark_host.mkdir(parents=True, exist_ok=True)
    benchmark_watermark.mkdir(parents=True, exist_ok=True)
    for split_name in ("calibration", "evaluation"):
        for index, source in enumerate(splits[split_name], start=1):
            shutil.copy2(source, benchmark_host / f"{split_name}_{index:03d}_{source.name}")
    shutil.copy2(watermark, benchmark_watermark / "wm.png")

    rows: list[dict[str, object]] = []
    for split_name, paths in splits.items():
        for source in paths:
            rows.append({
                "split": split_name,
                "source": str(source),
                "sha256": sha256(source),
            })

    with (out / "split_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["split", "source", "sha256"])
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "schema_version": "1.0",
        "seed": args.seed,
        "host_count": len(hosts),
        "watermark": str(wm_target),
        "image_benchmark_dataset": str(benchmark),
        "splits": {name: [str(path) for path in paths] for name, paths in splits.items()},
        "policy": {
            "development": "optional structure-only ablation",
            "validation": "optional proposal-design validation",
            "evaluation": "frozen final evaluation only",
            "calibration": "distortion/strength calibration only",
        },
    }
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
