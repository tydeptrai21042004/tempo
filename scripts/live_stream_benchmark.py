#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib

def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realtime_watermark.core import load_binary_watermark, make_demo_watermark, save_csv, save_json
from realtime_watermark.proposals.jilp_qim.config import DEFAULT_CONFIG, JILPQIMConfig
from realtime_watermark.stream_pipeline import STREAM_METHODS, load_stream_frames, run_stream_benchmark


def _csv_list(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _methods(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(STREAM_METHODS)
    selected = _csv_list(value)
    unknown = [x for x in selected if x not in STREAM_METHODS]
    if unknown:
        raise ValueError(f"Unknown stream methods {unknown}; choices: {STREAM_METHODS}")
    return selected


def _resolutions(value: str) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for item in _csv_list(value):
        w, h = item.lower().split("x", 1)
        result.append((int(w), int(h)))
    return result


def main() -> None:
    p = argparse.ArgumentParser(
        description="Deadline-constrained causal live-stream watermark benchmark with bounded queues."
    )
    p.add_argument("--input", default=None, help="Optional input video. A deterministic moving stream is generated when omitted.")
    p.add_argument("--watermark", default=None)
    p.add_argument("--workdir", default="reports/live_stream")
    p.add_argument("--methods", default="all")
    p.add_argument("--channels", default="clean,jpeg70")
    p.add_argument("--resolutions", default="512x512")
    p.add_argument("--target-fps", default="30")
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--wm-size", type=int, default=16, help="16 gives a baseline-friendly 256-bit payload.")
    p.add_argument("--queue-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--target-psnr", type=float, default=45.0)
    p.add_argument("--calibration-iterations", type=int, default=7)
    p.add_argument("--pace-mode", choices=["virtual", "realtime"], default="virtual")
    p.add_argument("--proposal-config", default=None)
    p.add_argument("--calibration-mode", choices=["legacy_test_frame", "frozen", "development"],
                   default="legacy_test_frame", help="Use frozen for primary reviewer comparison; no test-video tuning")
    p.add_argument("--calibration-input", default=None, help="Separate development video, only with development mode")
    p.add_argument("--paper-dew", action="store_true", help="Use the paper-associated coefficient-zeroing DEW implementation")
    p.add_argument("--require-real-video", action="store_true", help="Fail instead of using synthetic video frames")
    args = p.parse_args()

    if args.require_real_video and not args.input:
        raise ValueError("A real --input video is required; synthetic fallback forbidden")
    if args.calibration_mode == "development" and not args.calibration_input:
        raise ValueError("--calibration-input required for development calibration")
    out = Path(args.workdir)
    out.mkdir(parents=True, exist_ok=True)
    watermark = (
        load_binary_watermark(args.watermark, size=args.wm_size)
        if args.watermark else make_demo_watermark(args.wm_size)
    )
    proposal = JILPQIMConfig.load(args.proposal_config) if args.proposal_config else DEFAULT_CONFIG
    methods = _methods(args.methods)
    channels = _csv_list(args.channels)
    target_rates = [float(x) for x in _csv_list(args.target_fps)]

    combined_summary: list[dict] = []
    combined_trace: list[dict] = []
    combined_calibration: list[dict] = []
    runs: list[dict] = []
    for width, height in _resolutions(args.resolutions):
        frames = load_stream_frames(
            args.input, width=width, height=height, max_frames=args.frames
        )
        development_frames = (load_stream_frames(args.calibration_input, width=width, height=height,
                              max_frames=1) if args.calibration_mode == "development" else None)
        for fps in target_rates:
            run_out = out / f"{width}x{height}_{fps:g}fps"
            result = run_stream_benchmark(
                frames=frames,
                watermark=watermark,
                out_dir=run_out,
                methods=methods,
                channels=channels,
                target_fps=fps,
                queue_size=args.queue_size,
                seed=args.seed,
                target_psnr=args.target_psnr,
                calibration_iterations=args.calibration_iterations,
                pace_mode=args.pace_mode,
                proposal_params=proposal.to_dict(),
                calibration_mode=args.calibration_mode,
                calibration_frames=development_frames,
                paper_dew=args.paper_dew,
            )
            combined_summary.extend(result["summary"])
            combined_trace.extend(result["trace"])
            for row in result["calibration"]:
                combined_calibration.append({**row, "resolution": f"{width}x{height}", "target_fps": fps})
            runs.append({
                "resolution": f"{width}x{height}", "target_fps": fps,
                "frames_available": len(frames), "output": str(run_out),
            })

    save_csv(out / "ALL_STREAM_realtime_summary.csv", combined_summary)
    save_csv(out / "ALL_STREAM_realtime_trace.csv", combined_trace)
    save_csv(out / "ALL_STREAM_calibration.csv", combined_calibration)
    save_json(out / "ALL_STREAM_run_manifest.json", {
        "runs": runs,
        "methods": methods,
        "channels": channels,
        "pace_mode": args.pace_mode,
        "queue_size": args.queue_size,
        "calibration_mode": args.calibration_mode,
        "calibration_source": args.calibration_input,
        "calibration_source_sha256": _sha256_file(Path(args.calibration_input)) if args.calibration_input else None,
        "paper_dew": bool(args.paper_dew),
        "source_type": "real_video" if args.input else "synthetic_diagnostic_only",
        "frames_requested": args.frames,
        "watermark_size": args.wm_size,
        "payload_bits": int(watermark.size),
        "source_video": str(args.input) if args.input else "deterministic_synthetic_stream",
        "source_sha256": _sha256_file(Path(args.input)) if args.input else None,
        "watermark_sha256": _sha256_file(Path(args.watermark)) if args.watermark else None,
        "proposal_configuration_sha256": proposal.sha256(),
        "timing_scope": "wall-clock producer and bounded queue; not a formal worst-case scheduling guarantee",
    })
    print(f"Live-stream benchmark saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
