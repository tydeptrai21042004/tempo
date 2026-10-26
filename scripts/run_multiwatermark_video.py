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

from realtime_watermark.core import (
    ber,
    bit_accuracy,
    load_binary_watermark,
    watermark_statistics,
    nc_binary,
    psnr,
    save_csv,
    save_json,
    ssim_gray,
)
from realtime_watermark.demo_assets import ensure_demo_video
from realtime_watermark.native_metrics import binary_symbol_error_rate, mse, prd
from realtime_watermark.paper_native import NATIVE_DISCLOSURES
from realtime_watermark.proposals.jilp_qim.config import JILPQIMConfig
from realtime_watermark.registry import COMMON_VIDEO_METHODS, MODERN_ADAPTER_METHODS
from realtime_watermark.video_pipeline import embed_video_method, extract_video_method


def _safe(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all video-supported methods for both selected PNG watermarks without PSNR calibration."
    )
    parser.add_argument("--watermark-dir", required=True)
    parser.add_argument("--proposal-config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--input", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--wm-size", type=int, default=64)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--methods", default="all", help="all or comma-separated registered video methods")
    args = parser.parse_args()

    watermark_paths = sorted(
        path for path in Path(args.watermark_dir).glob("*.png") if path.is_file()
    )
    if len(watermark_paths) != 2:
        raise RuntimeError(f"Expected exactly two PNG watermarks; found {len(watermark_paths)}.")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    proposal = JILPQIMConfig.load(args.proposal_config)
    proposal_hash = proposal.sha256()
    video = ensure_demo_video(
        out / "inputs",
        input_path=args.input,
        auto_download=False,
        width=512,
        height=512,
        frames=max(args.max_frames, 12),
    )

    methods = list(COMMON_VIDEO_METHODS) if args.methods == "all" else [x.strip() for x in args.methods.split(",") if x.strip()]
    # New adapters are opt-in; the historical default video protocol is unchanged.
    if not methods or any(x not in set(COMMON_VIDEO_METHODS) | set(MODERN_ADAPTER_METHODS) for x in methods):
        raise ValueError(f"Unsupported video method selection: {args.methods}")

    rows: list[dict[str, object]] = []
    for watermark_index, watermark_path in enumerate(watermark_paths, start=1):
        watermark_tag = f"watermark_{watermark_index:02d}_{_safe(watermark_path.stem)}"
        watermark_seed = args.seed + 1000003 * (watermark_index - 1)
        original = load_binary_watermark(watermark_path, size=args.wm_size)

        for method_id in methods:
            method_out = out / watermark_tag / method_id
            method_out.mkdir(parents=True, exist_ok=True)
            output_video = method_out / "watermarked.mp4"
            key_path = method_out / "key.json"
            recovered_path = method_out / "recovered.png"
            extraction_report = method_out / "extraction_report.json"

            if method_id == "proposal":
                step = proposal.step
                method_params = proposal.to_dict()
                expected_hash = proposal_hash
            else:
                step = None
                method_params = None
                expected_hash = None

            embedding = embed_video_method(
                method_id=method_id,
                input_video_path=video,
                output_video_path=output_video,
                watermark_path=watermark_path,
                key_path=key_path,
                wm_size=args.wm_size,
                seed=watermark_seed,
                step=step,
                codec=args.codec,
                max_frames=args.max_frames,
                warmup_frames=args.warmup_frames,
                method_params=method_params,
            )["stats"]
            extraction = extract_video_method(
                input_video_path=output_video,
                key_path=key_path,
                output_watermark_path=recovered_path,
                report_path=extraction_report,
                per_frame_report_path=method_out / "per_frame.csv",
                reference_watermark_path=watermark_path,
                max_frames=args.max_frames,
                warmup_frames=args.warmup_frames,
            )["stats"]

            if expected_hash is not None:
                actual_hash = str(embedding.get("configuration_sha256", ""))
                if actual_hash != expected_hash:
                    raise RuntimeError(
                        "Proposal configuration mismatch in video benchmark: "
                        f"expected={expected_hash}, actual={actual_hash}"
                    )

            # Constant recovered images are valid failure outcomes: do not discard
            # them or fabricate missing robustness metrics. Input checks stay strict.
            recovered = load_binary_watermark(
                recovered_path, size=args.wm_size, allow_degenerate=True
            )
            recovered_stats = watermark_statistics(recovered)
            disclosure = NATIVE_DISCLOSURES.get(method_id)
            detector_type = disclosure.detector_type if disclosure is not None else "blind"
            accuracy = float(bit_accuracy(original, recovered))
            bit_error = float(ber(original, recovered))
            rows.append(
                {
                    "watermark_index": watermark_index,
                    "watermark_name": watermark_path.name,
                    "watermark_path": str(watermark_path),
                    "method_id": method_id,
                    "detector_type": detector_type,
                    "video_protocol": "common_framewise_adapter_native_default_strength",
                    "forced_psnr_calibration": False,
                    "common_ecc": False,
                    "configured_step": step if step is not None else "method_default",
                    "configuration_sha256": embedding.get("configuration_sha256"),
                    "frame_count": embedding["frame_count"],
                    "precodec_mean_psnr": embedding["avg_frame_psnr_before_video_codec"],
                    "bit_accuracy": accuracy,
                    "ber": bit_error,
                    "recovered_is_constant": bool(
                        recovered_stats["zero_bits"] == 0 or recovered_stats["one_bits"] == 0
                    ),
                    "recovered_one_fraction": recovered_stats["one_fraction"],
                    "empirical_error_probability": bit_error,
                    "nc": float(nc_binary(original, recovered)),
                    "ser_4bit": float(binary_symbol_error_rate(original > 127, recovered > 127, 4)),
                    "recovered_watermark_mse": float(mse(original, recovered)),
                    "recovered_watermark_psnr": float(psnr(original, recovered)),
                    "recovered_watermark_prd": float(prd(original, recovered)),
                    "recovered_watermark_ssim": float(ssim_gray(original, recovered)),
                    "embed_kernel_fps": embedding["kernel_fps"],
                    "embed_algorithm_fps": embedding["algorithm_fps"],
                    "embed_end_to_end_fps": embedding["end_to_end_fps"],
                    "extract_kernel_fps": extraction["kernel_fps"],
                    "extract_algorithm_fps": extraction["algorithm_fps"],
                    "extract_end_to_end_fps": extraction["end_to_end_fps"],
                    "payload_bits_per_second_embed": embedding["payload_bits_per_second"],
                    "payload_bits_per_second_embed_kernel_limited": embedding["payload_bits_per_second"],
                    "payload_bits_per_second_extract_end_to_end": extraction.get("end_to_end_payload_bits_per_second"),
                    "serialized_key_bytes": embedding.get("serialized_key_bytes"),
                    "side_information_bytes_per_frame": embedding["side_information_bytes_per_frame"],
                    "amortized_serialized_key_bytes_per_frame": embedding["side_information_bytes_per_frame"],
                    "side_information_metric_label": embedding.get("side_information_metric_label"),
                    "per_frame_bit_accuracy_mean": extraction.get("per_frame_bit_accuracy_mean"),
                    "per_frame_bit_accuracy_min": extraction.get("per_frame_bit_accuracy_min"),
                    "per_frame_bit_accuracy_p10": extraction.get("per_frame_bit_accuracy_p10"),
                    "per_frame_nc_mean": extraction.get("per_frame_nc_mean"),
                    "per_frame_csv": extraction.get("per_frame_report_path"),
                    "per_frame_nc_min": extraction.get("per_frame_nc_min"),
                    "per_frame_nc_p10": extraction.get("per_frame_nc_p10"),
                    "frames_to_vote_nc_090": extraction.get("frames_to_vote_nc_090"),
                    "frames_to_vote_nc_095": extraction.get("frames_to_vote_nc_095"),
                    "frames_to_exact_recovery": extraction.get("frames_to_exact_recovery"),
                    "watermarked_video": str(output_video),
                    "key_json": str(key_path),
                    "recovered_watermark": str(recovered_path),
                }
            )

    save_csv(out / "all_watermarks_video_summary.csv", rows)
    save_json(
        out / "all_watermarks_video_protocol.json",
        {
            "input_video": str(video),
            "input_sha256": _sha256_file(video),
            "watermarks": [str(path) for path in watermark_paths],
            "watermark_sha256": {path.name: _sha256_file(path) for path in watermark_paths},
            "methods": methods,
            "excluded_image_method": "dct_spread_spectrum_cox1997",
            "excluded_reason": (
                "Cox is informed/non-blind and requires frame-specific original-host coefficients."
            ),
            "forced_psnr_calibration": False,
            "common_ecc": False,
            "proposal_configuration_sha256": proposal_hash,
            "frames": args.max_frames,
            "codec": args.codec,
            "warmup_frames": args.warmup_frames,
        },
    )
    print(f"Video benchmark complete: {out}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
