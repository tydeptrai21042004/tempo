#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realtime_watermark.benchmark import run_compare
from realtime_watermark.core import load_binary_watermark, load_rgb, make_demo_watermark, psnr, save_csv, save_image, save_json
from realtime_watermark.demo_assets import ensure_demo_image_assets, ensure_demo_video
from realtime_watermark.fair_benchmark import (
    calibrate_step_to_psnr,
    calibrate_steps_on_hosts,
    make_independent_calibration_host,
    run_fair_compare,
)
from realtime_watermark.paper_fidelity import common_primary_static_eligible, fidelity_tier
from realtime_watermark.proposals.jilp_qim.config import DEFAULT_CONFIG, JILPQIMConfig
from realtime_watermark.registry import IMAGE_BENCHMARK_METHODS, embed as registry_embed
from realtime_watermark.video_pipeline import REALTIME_VIDEO_METHODS, embed_video_method, extract_video_method

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _find_hosts(data_dir: Path) -> list[Path]:
    host_root = data_dir / "host"
    root = host_root if host_root.exists() else data_dir
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _find_watermark(data_dir: Path, explicit: str | None, wm_size: int) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    candidates = [
        p for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        and ("watermark" in str(p.parent).lower() or "wm" in p.name.lower())
    ]
    if not candidates:
        raise FileNotFoundError(f"No watermark image found under {data_dir}")
    preferred = {f"wm{wm_size}.png", "wm.png", "watermark.png"}
    return sorted(candidates, key=lambda p: (p.name.lower() not in preferred, str(p)))[0]


def _methods(value: str, video: bool = False) -> list[str]:
    available = list(REALTIME_VIDEO_METHODS if video else IMAGE_BENCHMARK_METHODS)
    if value.strip().lower() == "all":
        return available
    selected = [x.strip() for x in value.split(",") if x.strip()]
    unknown = [x for x in selected if x not in available]
    if unknown:
        raise ValueError(f"Unknown methods {unknown}; choices are {available}")
    return selected


def _proposal_config(path: str | None) -> JILPQIMConfig:
    return JILPQIMConfig.load(path) if path else DEFAULT_CONFIG


def _proposal_params(path: str | None) -> dict[str, dict[str, Any]]:
    config = _proposal_config(path)
    return {"proposal": config.to_dict()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproducible watermark benchmark commands.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    image = sub.add_parser("image512-all", help="Run the existing matched-distortion image benchmark.")
    image.add_argument("--workdir", default="reports/image512_all")
    image.add_argument("--host", default=None)
    image.add_argument("--watermark", default=None)
    image.add_argument("--calibration-host", action="append", default=[])
    image.add_argument("--proposal-config", default=None)
    image.add_argument("--methods", default="all")
    image.add_argument("--seed", type=int, default=2026)
    image.add_argument("--attack-preset", choices=["smoke", "standard", "extended", "stress", "unified"], default="standard")
    image.add_argument("--target-psnr", type=float, default=45.0)
    image.add_argument("--psnr-tolerance", type=float, default=0.75)
    image.add_argument("--timing-repeats", type=int, default=30)
    image.add_argument("--timing-warmups", type=int, default=5)

    dataset = sub.add_parser("dataset-image512-all", help="Calibrate on held-out hosts, then evaluate every remaining host.")
    dataset.add_argument("--data-dir", required=True)
    dataset.add_argument("--workdir", default="reports/dataset_image512_all")
    dataset.add_argument("--watermark", default=None)
    dataset.add_argument("--proposal-config", default=None)
    dataset.add_argument("--methods", default="all")
    dataset.add_argument("--seed", type=int, default=2026)
    dataset.add_argument("--wm-size", type=int, default=64)
    dataset.add_argument("--attack-preset", choices=["smoke", "standard", "extended", "stress", "unified"], default="standard")
    dataset.add_argument("--target-psnr", type=float, default=45.0)
    dataset.add_argument("--psnr-tolerance", type=float, default=0.75)
    dataset.add_argument("--calibration-hosts", type=int, default=3)
    dataset.add_argument("--max-hosts", type=int, default=None)

    video = sub.add_parser("video-realtime", help="Run the existing common framewise video protocol.")
    video.add_argument("--workdir", default="reports/video_realtime")
    video.add_argument("--input", default=None)
    video.add_argument("--watermark", default=None)
    video.add_argument("--proposal-config", default=None)
    video.add_argument("--methods", default="all")
    video.add_argument("--seed", type=int, default=2026)
    video.add_argument("--wm-size", type=int, default=64)
    video.add_argument("--max-frames", type=int, default=300)
    video.add_argument("--codec", default="mp4v")
    video.add_argument("--warmup-frames", type=int, default=10)
    video.add_argument("--target-psnr", type=float, default=45.0)
    video.add_argument("--psnr-tolerance", type=float, default=1.0)
    video.add_argument("--auto-download-video", action="store_true")
    return parser


def run_image512_all(args: argparse.Namespace) -> None:
    out = Path(args.workdir)
    out.mkdir(parents=True, exist_ok=True)
    demo_host, demo_wm = ensure_demo_image_assets(out / "inputs", host_size=512, wm_size=64)
    host = Path(args.host) if args.host else demo_host
    watermark = Path(args.watermark) if args.watermark else demo_wm
    calibration_paths = [Path(x) for x in args.calibration_host]
    proposal_config = _proposal_config(args.proposal_config)
    params = {"proposal": proposal_config.to_dict()}
    result = run_fair_compare(
        host_path=host,
        calibration_host_paths=calibration_paths or None,
        calibration_host_count=3,
        watermark_path=watermark,
        out_dir=out,
        mode="512",
        methods=_methods(args.methods),
        wm_size=64,
        seed=args.seed,
        target_psnr=args.target_psnr,
        psnr_tolerance=args.psnr_tolerance,
        timing_repeats=args.timing_repeats,
        timing_warmups=args.timing_warmups,
        attack_preset=args.attack_preset,
        method_params_by_method=params,
        frozen_steps_by_method={"proposal": proposal_config.step},
    )
    print(f"Completed {len(result['summary'])} methods; results: {out.resolve()}")


def run_dataset_image512_all(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    out = Path(args.workdir)
    out.mkdir(parents=True, exist_ok=True)
    hosts = _find_hosts(data_dir)
    if args.max_hosts is not None:
        hosts = hosts[: int(args.max_hosts)]
    n_cal = int(args.calibration_hosts)
    if len(hosts) <= n_cal:
        raise ValueError(f"Need more than {n_cal} hosts so calibration and evaluation remain disjoint")
    calibration_paths = hosts[:n_cal]
    evaluation_paths = hosts[n_cal:]
    watermark_path = _find_watermark(data_dir, args.watermark, args.wm_size)
    wm = load_binary_watermark(watermark_path, size=args.wm_size)
    calibration_hosts = [load_rgb(p, mode="512") for p in calibration_paths]
    selected = _methods(args.methods)
    proposal_config = _proposal_config(args.proposal_config)
    params = {"proposal": proposal_config.to_dict()}
    frozen = calibrate_steps_on_hosts(
        selected, calibration_hosts, wm,
        target_psnr=args.target_psnr,
        seed=args.seed,
        iterations=10,
        method_params_by_method=params,
        frozen_steps_by_method={"proposal": proposal_config.step},
    )
    frozen_steps = {m: float(v["step"]) for m, v in frozen.items()}
    save_json(out / "frozen_calibration.json", {
        "calibration_hosts": [str(p) for p in calibration_paths],
        "evaluation_hosts": [str(p) for p in evaluation_paths],
        "frozen": frozen,
        "method_params": params,
    })

    all_summary: list[dict[str, Any]] = []
    all_attacks: list[dict[str, Any]] = []
    for index, host_path in enumerate(evaluation_paths):
        host_out = out / "hosts" / f"{index:04d}_{host_path.stem}"
        result = run_compare(
            host_path=host_path,
            watermark_path=watermark_path,
            out_dir=host_out,
            mode="512",
            methods=selected,
            wm_size=args.wm_size,
            repeat="auto",
            seed=args.seed,
            step_overrides=frozen_steps,
            target_psnr=args.target_psnr,
            psnr_tolerance=args.psnr_tolerance,
            attack_preset=args.attack_preset,
            method_params_by_method=params,
        )
        for row in result["summary"]:
            all_summary.append({"host": str(host_path), **row})
        for row in result["attacks"]:
            all_attacks.append({"host": str(host_path), **row})
    save_csv(out / "ALL_HOSTS_summary_by_host_method.csv", all_summary)
    save_csv(out / "ALL_HOSTS_per_attack_results.csv", all_attacks)
    print(f"Evaluated {len(evaluation_paths)} held-out hosts; results: {out.resolve()}")


def run_video_realtime(args: argparse.Namespace) -> None:
    out = Path(args.workdir)
    out.mkdir(parents=True, exist_ok=True)
    if args.watermark:
        watermark = Path(args.watermark)
    else:
        watermark = out / f"wm{args.wm_size}.png"
        save_image(watermark, make_demo_watermark(args.wm_size))
    input_video = ensure_demo_video(
        out / "inputs",
        input_path=args.input,
        auto_download=bool(args.auto_download_video),
        width=512,
        height=512,
        frames=max(int(args.max_frames), 12),
    )
    selected = _methods(args.methods, video=True)
    proposal_config = _proposal_config(args.proposal_config)
    proposal_params = {"proposal": proposal_config.to_dict()}

    cv2 = __import__("cv2")
    cap = cv2.VideoCapture(str(input_video))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 512)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 512)
    cap.release()
    calibration_host = make_independent_calibration_host(width, height, seed=args.seed + 8107)
    wm = load_binary_watermark(watermark, size=args.wm_size)
    frozen_steps: dict[str, float] = {}
    observed: dict[str, float] = {}
    for method_id in selected:
        if method_id == "proposal":
            value = float(proposal_config.step)
            calibration_watermarked, calibration_key = registry_embed(
                method_id, calibration_host, wm, seed=args.seed, repeat="auto", step=value,
                method_params=proposal_params.get(method_id),
            )
            psnr_observed = psnr(calibration_host, calibration_watermarked)
            expected_hash = proposal_config.sha256()
            actual_hash = str(calibration_key.params.get("config_hash_sha256", ""))
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"Proposal configuration identity mismatch during video calibration: "
                    f"expected={expected_hash}, key={actual_hash}"
                )
        else:
            value, psnr_observed = calibrate_step_to_psnr(
                method_id, calibration_host, wm,
                target_psnr=args.target_psnr,
                seed=args.seed,
                iterations=10,
                method_params=proposal_params.get(method_id),
            )
        frozen_steps[method_id] = float(value)
        observed[method_id] = float(psnr_observed)

    rows: list[dict[str, Any]] = []
    for method_id in selected:
        method_out = out / method_id
        method_out.mkdir(parents=True, exist_ok=True)
        video_path = method_out / "watermarked.mp4"
        key_path = method_out / "key.json"
        recovered_path = method_out / "recovered.png"
        report_path = method_out / "extract_report.json"
        embed_result = embed_video_method(
            method_id=method_id,
            input_video_path=input_video,
            output_video_path=video_path,
            watermark_path=watermark,
            key_path=key_path,
            wm_size=args.wm_size,
            seed=args.seed,
            step=frozen_steps[method_id],
            codec=args.codec,
            max_frames=args.max_frames,
            warmup_frames=args.warmup_frames,
            method_params=proposal_params.get(method_id),
        )
        extract_result = extract_video_method(
            input_video_path=video_path,
            key_path=key_path,
            output_watermark_path=recovered_path,
            report_path=report_path,
            reference_watermark_path=watermark,
            max_frames=args.max_frames,
            warmup_frames=args.warmup_frames,
        )
        er = embed_result["stats"]
        xr = extract_result["stats"]
        mean_psnr = float(er["avg_frame_psnr_before_video_codec"])
        recovery_ok = float(xr["bit_accuracy"]) >= 1.0 - 1e-12
        distortion_ok = abs(mean_psnr - args.target_psnr) <= args.psnr_tolerance
        eligible = bool(common_primary_static_eligible(method_id) and recovery_ok and distortion_ok)
        rows.append({
            "method_id": method_id,
            "fidelity_tier": fidelity_tier(method_id),
            "video_protocol": "common_framewise_video_adapter",
            "primary_rank_eligible": eligible,
            "eligibility_reason": "passed" if eligible else (
                f"recovery={xr['bit_accuracy']}, precodec_psnr={mean_psnr:.3f}"
            ),
            "configuration_sha256": er.get("configuration_sha256"),
            "target_psnr_before_codec": args.target_psnr,
            "frozen_step": frozen_steps[method_id],
            "calibration_observed_psnr": observed[method_id],
            "avg_frame_psnr_before_video_codec": mean_psnr,
            "frame_count": er["frame_count"],
            "embed_kernel_fps": er["kernel_fps"],
            "embed_end_to_end_fps": er["end_to_end_fps"],
            "extract_kernel_fps": xr["kernel_fps"],
            "extract_end_to_end_fps": xr["end_to_end_fps"],
            "bit_accuracy": xr["bit_accuracy"],
            "nc": xr["nc"],
            "serialized_key_bytes": er.get("serialized_key_bytes"),
            "amortized_serialized_key_bytes_per_frame": er["side_information_bytes_per_frame"],
            "side_information_metric_label": er.get("side_information_metric_label"),
            "per_frame_bit_accuracy_mean": xr.get("per_frame_bit_accuracy_mean"),
            "per_frame_bit_accuracy_min": xr.get("per_frame_bit_accuracy_min"),
            "per_frame_nc_mean": xr.get("per_frame_nc_mean"),
            "per_frame_nc_min": xr.get("per_frame_nc_min"),
            "per_frame_nc_p10": xr.get("per_frame_nc_p10"),
            "frames_to_vote_nc_090": xr.get("frames_to_vote_nc_090"),
            "frames_to_vote_nc_095": xr.get("frames_to_vote_nc_095"),
            "frames_to_exact_recovery": xr.get("frames_to_exact_recovery"),
            "watermarked_video": str(video_path),
            "key_json": str(key_path),
            "recovered_watermark": str(recovered_path),
        })
    save_csv(out / "video_realtime_summary.csv", rows)
    save_csv(out / "video_realtime_primary.csv", [r for r in rows if r["primary_rank_eligible"]])
    save_json(out / "video_fair_protocol.json", {
        "protocol": "same framewise payload, codec, independent calibration frame, and hard temporal vote",
        "target_psnr": args.target_psnr,
        "psnr_tolerance": args.psnr_tolerance,
        "frozen_steps": frozen_steps,
        "step_sources": {m: ("externally_frozen_configuration" if m == "proposal" else "heldout_calibration") for m in selected},
        "proposal_configuration_sha256": proposal_config.sha256(),
        "calibration_observed_psnr": observed,
        "method_params": proposal_params,
        "methods": selected,
    })
    print(f"Completed common video benchmark; results: {out.resolve()}")


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "image512-all":
        run_image512_all(args)
    elif args.cmd == "dataset-image512-all":
        run_dataset_image512_all(args)
    else:
        run_video_realtime(args)


if __name__ == "__main__":
    main()
