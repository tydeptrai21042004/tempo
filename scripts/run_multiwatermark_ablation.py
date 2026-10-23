#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _safe(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run JILP-QIM image and video ablations for both PNG watermarks.")
    parser.add_argument("--host-dir", required=True)
    parser.add_argument("--watermark-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--wm-size", type=int, default=64)
    parser.add_argument("--max-hosts", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--video-input", default=None)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument(
        "--attack-preset",
        choices=["smoke", "standard", "extended", "stress", "unified"],
        default="stress",
    )
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    watermarks = sorted(path for path in Path(args.watermark_dir).glob("*.png") if path.is_file())
    if len(watermarks) != 2:
        raise RuntimeError(f"Expected exactly two PNG watermarks; found {len(watermarks)}.")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for index, watermark in enumerate(watermarks, start=1):
        tag = f"watermark_{index:02d}_{_safe(watermark.stem)}"
        image_out = out / "image_structural" / tag
        command = [
            sys.executable,
            str(root / "scripts" / "proposal_structural_ablation.py"),
            "--config",
            args.config,
            "--out-dir",
            str(image_out),
            "--host-dir",
            args.host_dir,
            "--watermark",
            str(watermark),
            "--host-size",
            "512",
            "--wm-size",
            str(args.wm_size),
            "--seed",
            str(args.seed + 1000003 * (index - 1)),
            "--attack-preset",
            args.attack_preset,
            "--include-security",
            "--no-calibrate",
        ]
        if args.max_hosts > 0:
            command += ["--max-hosts", str(args.max_hosts)]
        subprocess.run(command, check=True)

        if not args.skip_video:
            video_out = out / "video_temporal" / tag
            command = [
                sys.executable,
                str(root / "scripts" / "proposal_video_ablation.py"),
                "--config",
                args.config,
                "--out-dir",
                str(video_out),
                "--watermark",
                str(watermark),
                "--wm-size",
                str(args.wm_size),
                "--seed",
                str(args.seed + 1000003 * (index - 1)),
                "--max-frames",
                str(args.max_frames),
                "--codec",
                args.codec,
            ]
            if args.video_input:
                command += ["--input", args.video_input]
            subprocess.run(command, check=True)

    print(f"Ablation complete: {out}")


if __name__ == "__main__":
    main()
