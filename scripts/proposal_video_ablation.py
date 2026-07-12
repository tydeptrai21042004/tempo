#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realtime_watermark.core import make_demo_watermark, save_csv, save_image
from realtime_watermark.demo_assets import ensure_demo_video
from realtime_watermark.proposals.jilp_qim.config import DEFAULT_CONFIG, JILPQIMConfig
from realtime_watermark.video_pipeline import embed_video_method, extract_video_method


def main() -> None:
    parser = argparse.ArgumentParser(description="Real temporal-voting ablation on encoded video.")
    parser.add_argument("--input", default=None)
    parser.add_argument("--watermark", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out-dir", default="reports/proposal_video_ablation")
    parser.add_argument("--wm-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--step", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--warmup-frames", type=int, default=10)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = JILPQIMConfig.load(args.config) if args.config else DEFAULT_CONFIG
    if args.step is not None:
        base = base.with_step(args.step)
    watermark = Path(args.watermark) if args.watermark else out / f"wm{args.wm_size}.png"
    if not watermark.exists():
        save_image(watermark, make_demo_watermark(args.wm_size))
    video = ensure_demo_video(
        out / "inputs", input_path=args.input, auto_download=False,
        width=512, height=512, frames=max(args.max_frames, 12),
    )

    rows = []
    for vote_mode in ("hard", "confidence_weighted"):
        config = replace(base, temporal_vote_mode=vote_mode)
        variant = out / vote_mode
        variant.mkdir(parents=True, exist_ok=True)
        marked = variant / "watermarked.mp4"
        key = variant / "key.json"
        recovered = variant / "recovered.png"
        er = embed_video_method(
            method_id="proposal", input_video_path=video, output_video_path=marked,
            watermark_path=watermark, key_path=key, wm_size=args.wm_size,
            seed=args.seed, step=config.step, codec=args.codec,
            max_frames=args.max_frames, warmup_frames=args.warmup_frames,
            method_params=config.to_dict(),
        )["stats"]
        xr = extract_video_method(
            input_video_path=marked, key_path=key, output_watermark_path=recovered,
            reference_watermark_path=watermark, max_frames=args.max_frames,
            warmup_frames=args.warmup_frames,
        )["stats"]
        rows.append({
            "temporal_vote_mode": vote_mode,
            "configuration_sha256": config.sha256(),
            "frame_count": er["frame_count"],
            "precodec_psnr": er["avg_frame_psnr_before_video_codec"],
            "bit_accuracy": xr["bit_accuracy"],
            "nc": xr["nc"],
            "extract_kernel_fps": xr["kernel_fps"],
            "extract_end_to_end_fps": xr["end_to_end_fps"],
        })
    save_csv(out / "temporal_vote_ablation.csv", rows)


if __name__ == "__main__":
    main()
