#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realtime_watermark.proposals.jilp_qim.config import DEFAULT_CONFIG, JILPQIMConfig
from realtime_watermark.video_pipeline import (
    REALTIME_VIDEO_METHODS,
    embed_video_method,
    extract_video_method,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Common framewise MP4 protocol for four reproducible baselines and JILP-QIM."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("embed", help="Embed a 64x64 binary watermark into an MP4 video.")
    pe.add_argument("--method", default="proposal", choices=REALTIME_VIDEO_METHODS)
    pe.add_argument("--input", required=True, help="Input MP4/video path.")
    pe.add_argument("--output", required=True, help="Output watermarked MP4 path.")
    pe.add_argument("--watermark", default=None, help="Optional watermark image. It is resized/binarized to 64x64 by default.")
    pe.add_argument("--key", default=None, help="Output key JSON. Defaults to output path with .key.json suffix.")
    pe.add_argument("--wm-size", type=int, default=64, help="Binary watermark side length. Default: 64.")
    pe.add_argument("--seed", type=int, default=2026)
    pe.add_argument("--repeat", default="auto", help="Repeat factor for per-frame mode, or 'auto'.")
    pe.add_argument("--step", type=float, default=None, help="Embedding strength/margin. Defaults are method-specific.")
    pe.add_argument("--codec", default="mp4v", help="FourCC codec. Default: mp4v.")
    pe.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for tests/quick runs.")
    pe.add_argument("--warmup-frames", type=int, default=10, help="Initial processed frames excluded from timing distributions.")
    pe.add_argument("--proposal-config", default=None)

    px = sub.add_parser("extract", help="Blindly extract the watermark from a watermarked MP4.")
    px.add_argument("--input", required=True, help="Watermarked MP4 path.")
    px.add_argument("--key", required=True, help="Key JSON created by the embed command.")
    px.add_argument("--output-watermark", required=True, help="Recovered 64x64 watermark PNG path.")
    px.add_argument("--report", default=None, help="Optional extraction report JSON path.")
    px.add_argument("--reference-watermark", default=None, help="Optional reference watermark for bit accuracy/NC in the report.")
    px.add_argument("--max-frames", type=int, default=None)
    px.add_argument("--warmup-frames", type=int, default=10)

    pr = sub.add_parser("roundtrip", help="Embed then extract in one command for testing.")
    pr.add_argument("--method", default="proposal", choices=REALTIME_VIDEO_METHODS)
    pr.add_argument("--input", required=True)
    pr.add_argument("--output", required=True)
    pr.add_argument("--watermark", default=None)
    pr.add_argument("--key", default=None)
    pr.add_argument("--output-watermark", required=True)
    pr.add_argument("--report", default=None)
    pr.add_argument("--wm-size", type=int, default=64)
    pr.add_argument("--seed", type=int, default=2026)
    pr.add_argument("--repeat", default="auto")
    pr.add_argument("--step", type=float, default=None)
    pr.add_argument("--codec", default="mp4v")
    pr.add_argument("--max-frames", type=int, default=None)
    pr.add_argument("--warmup-frames", type=int, default=10)
    pr.add_argument("--proposal-config", default=None)
    return p


def _print_embed(stats: dict, method_id: str) -> None:
    print(f"Embedded video with method: {method_id}")
    print(f"  mode: {stats['mode']}")
    print(f"  frames: {stats['frame_count']}")
    print(f"  encoded size: {stats['encoded_width']}x{stats['encoded_height']}")
    print(f"  kernel ms/frame: {stats['watermark_kernel_ms_median'] or stats['avg_process_ms_per_frame']:.3f}")
    print(f"  kernel FPS: {stats.get('kernel_fps') or stats['process_fps']:.2f}")
    if stats.get("end_to_end_fps") is not None:
        print(f"  end-to-end FPS: {stats['end_to_end_fps']:.2f}")
        print(f"  real-time factor: {stats['real_time_factor']:.3f}")
        print(f"  payload throughput: {stats['payload_bits_per_second']:.2f} bit/s")
    print(f"  watermark coverage: {stats['watermark_coverage_fraction']:.3f}")
    print(f"  output: {Path(stats['output_video']).resolve()}")
    print(f"  key: {Path(stats['key_path']).resolve()}")


def _print_extract(stats: dict) -> None:
    print("Extracted watermark from video")
    print(f"  mode: {stats['mode']}")
    print(f"  frames: {stats['frame_count']}")
    print(f"  kernel extract ms/frame: {stats['watermark_kernel_ms_median'] or stats['avg_process_ms_per_frame']:.3f}")
    print(f"  kernel extract FPS: {stats.get('kernel_fps') or stats['process_fps']:.2f}")
    if stats.get("end_to_end_fps") is not None:
        print(f"  end-to-end extract FPS: {stats['end_to_end_fps']:.2f}")
        print(f"  real-time factor: {stats['real_time_factor']:.3f}")
    print(f"  coverage: {stats['watermark_coverage_fraction']:.3f}")
    if stats.get("bit_accuracy") is not None:
        print(f"  bit accuracy: {stats['bit_accuracy']:.4f}")
        print(f"  NC: {stats['nc']:.4f}")
    print(f"  recovered watermark: {Path(stats['recovered_watermark_path']).resolve()}")


def _method_params(method_id: str, path: str | None):
    if method_id != "proposal":
        return None
    config = JILPQIMConfig.load(path) if path else DEFAULT_CONFIG
    return config.to_dict()


def main() -> None:
    args = build_parser().parse_args()

    if args.cmd == "embed":
        result = embed_video_method(
            method_id=args.method,
            input_video_path=args.input,
            output_video_path=args.output,
            watermark_path=args.watermark,
            key_path=args.key,
            wm_size=args.wm_size,
            seed=args.seed,
            repeat=args.repeat,
            step=args.step,
            codec=args.codec,
            max_frames=args.max_frames,
            warmup_frames=args.warmup_frames,
            method_params=_method_params(args.method, args.proposal_config),
        )
        _print_embed(result["stats"], args.method)

    elif args.cmd == "extract":
        result = extract_video_method(
            input_video_path=args.input,
            key_path=args.key,
            output_watermark_path=args.output_watermark,
            report_path=args.report,
            reference_watermark_path=args.reference_watermark,
            max_frames=args.max_frames,
            warmup_frames=args.warmup_frames,
        )
        _print_extract(result["stats"])

    elif args.cmd == "roundtrip":
        key_path = args.key or str(Path(args.output).with_suffix(".key.json"))
        embed_result = embed_video_method(
            method_id=args.method,
            input_video_path=args.input,
            output_video_path=args.output,
            watermark_path=args.watermark,
            key_path=key_path,
            wm_size=args.wm_size,
            seed=args.seed,
            repeat=args.repeat,
            step=args.step,
            codec=args.codec,
            max_frames=args.max_frames,
            warmup_frames=args.warmup_frames,
            method_params=_method_params(args.method, args.proposal_config),
        )
        extract_result = extract_video_method(
            input_video_path=args.output,
            key_path=key_path,
            output_watermark_path=args.output_watermark,
            report_path=args.report,
            reference_watermark_path=args.watermark,
            max_frames=args.max_frames,
            warmup_frames=args.warmup_frames,
        )
        print("Roundtrip complete")
        print(f"  method: {args.method}")
        print(f"  embed mode: {embed_result['stats']['mode']}")
        print(f"  embed kernel FPS: {embed_result['stats']['process_fps']:.2f}")
        print(f"  embed end-to-end FPS: {embed_result['stats'].get('end_to_end_fps') or embed_result['stats']['process_fps']:.2f}")
        print(f"  extract kernel FPS: {extract_result['stats']['process_fps']:.2f}")
        print(f"  extract end-to-end FPS: {extract_result['stats'].get('end_to_end_fps') or extract_result['stats']['process_fps']:.2f}")
        if extract_result["stats"].get("bit_accuracy") is not None:
            print(f"  bit accuracy: {extract_result['stats']['bit_accuracy']:.4f}")
            print(f"  NC: {extract_result['stats']['nc']:.4f}")
        print(f"  watermarked video: {Path(args.output).resolve()}")
        print(f"  key: {Path(key_path).resolve()}")
        print(f"  recovered watermark: {Path(args.output_watermark).resolve()}")


if __name__ == "__main__":
    main()
