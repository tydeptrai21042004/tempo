from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Any, Iterable
import json
import math
import time

import numpy as np
from PIL import Image

from .common.timing import summarize_ms
from .core import (
    attack_blur,
    attack_brightness_contrast,
    attack_gaussian,
    attack_jpeg,
    bit_accuracy,
    ber,
    make_demo_host,
    nc_binary,
    psnr,
    save_csv,
    save_json,
    watermark_statistics,
)
from .fair_benchmark import calibrate_step_to_psnr
from .registry import STREAM_BENCHMARK_METHODS, embed, extract, normalize_method_id

STREAM_PROTOCOL_VERSION = "1.0"
STREAM_METHODS = tuple(STREAM_BENCHMARK_METHODS)
VOTE_WINDOWS = (1, 3, 5, 10, 15, 30)


@dataclass
class StreamPacket:
    frame_id: int
    arrival_s: float
    frame_rgb: np.ndarray


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _synthetic_frames(width: int, height: int, count: int) -> list[np.ndarray]:
    base = make_demo_host(int(width), int(height)).astype(np.int16)
    yy = np.linspace(0.0, 1.0, int(height), dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, int(width), dtype=np.float32)[None, :]
    frames: list[np.ndarray] = []
    for i in range(int(count)):
        frame = base.copy()
        wave = 12.0 * np.sin(2.0 * np.pi * (2.0 * xx + 1.5 * yy + i / max(1, count)))
        frame[..., 0] = np.clip(frame[..., 0] + wave, 0, 255)
        frame[..., 1] = np.clip(frame[..., 1] + (i % 16), 0, 255)
        frame[..., 2] = np.clip(frame[..., 2] - (i % 11), 0, 255)
        frames.append(frame.astype(np.uint8))
    return frames


def load_stream_frames(
    input_video: str | Path | None,
    *,
    width: int,
    height: int,
    max_frames: int,
) -> list[np.ndarray]:
    if input_video is None:
        return _synthetic_frames(width, height, max_frames)
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required to read live-stream benchmark videos") from exc
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise FileNotFoundError(input_video)
    frames: list[np.ndarray] = []
    try:
        while len(frames) < int(max_frames):
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if frame_rgb.shape[1] != int(width) or frame_rgb.shape[0] != int(height):
                frame_rgb = cv2.resize(frame_rgb, (int(width), int(height)), interpolation=cv2.INTER_AREA)
            frames.append(np.asarray(frame_rgb, dtype=np.uint8))
    finally:
        cap.release()
    if not frames:
        raise ValueError("Input video contains no readable frames")
    return frames


def apply_stream_channel(frame_rgb: np.ndarray, channel: str, *, frame_id: int, seed: int) -> np.ndarray | None:
    name = str(channel).lower().strip()
    if name == "clean":
        return np.asarray(frame_rgb, dtype=np.uint8)
    if name.startswith("jpeg"):
        return attack_jpeg(frame_rgb, int(name.replace("jpeg", "")))
    if name == "gaussian2":
        return attack_gaussian(frame_rgb, 2.0, seed=int(seed) + int(frame_id))
    if name == "gaussian5":
        return attack_gaussian(frame_rgb, 5.0, seed=int(seed) + int(frame_id))
    if name == "blur3":
        return attack_blur(frame_rgb, 1.0)
    if name == "brightness":
        return attack_brightness_contrast(frame_rgb, 1.05, 1.05)
    if name in {"drop05", "drop5"}:
        rng = np.random.default_rng(int(seed) + 100003 * int(frame_id))
        return None if float(rng.random()) < 0.05 else np.asarray(frame_rgb, dtype=np.uint8)
    if name in {"drop10", "drop10pct"}:
        rng = np.random.default_rng(int(seed) + 100003 * int(frame_id))
        return None if float(rng.random()) < 0.10 else np.asarray(frame_rgb, dtype=np.uint8)
    raise ValueError(
        "Unknown stream channel. Use clean, jpeg90, jpeg70, jpeg50, gaussian2, "
        "gaussian5, blur3, brightness, drop05, or drop10."
    )


def _running_vote(recovered: list[np.ndarray]) -> np.ndarray:
    stack = np.stack([(np.asarray(x) > 127).astype(np.uint8) for x in recovered], axis=0)
    return ((np.sum(stack, axis=0) * 2 >= stack.shape[0]).astype(np.uint8) * 255)


def _method_params(method_id: str, proposal_params: dict[str, Any] | None) -> dict[str, Any]:
    method_id = normalize_method_id(method_id)
    if method_id == "proposal":
        return dict(proposal_params or {})
    if method_id == "dew_langelaar2001":
        return {
            "blocks_per_bit": "auto",
            "cutoff": 1,
            "embedding_mode": "minimum_energy_removal",
        }
    return {}


def _process_packet(
    packet: StreamPacket,
    *,
    method_id: str,
    watermark: np.ndarray,
    channel: str,
    seed: int,
    step: float | None,
    method_params: dict[str, Any],
) -> dict[str, Any]:
    t0 = time.perf_counter()
    marked, key = embed(
        method_id, packet.frame_rgb, watermark, seed=seed,
        repeat="auto", step=step, method_params=method_params,
    )
    t1 = time.perf_counter()
    attacked = apply_stream_channel(marked, channel, frame_id=packet.frame_id, seed=seed)
    t2 = time.perf_counter()
    if attacked is None:
        return {
            "frame_id": packet.frame_id,
            "arrival_s": packet.arrival_s,
            "marked": marked,
            "attacked": None,
            "recovered": None,
            "embed_ms": (t1 - t0) * 1000.0,
            "channel_ms": (t2 - t1) * 1000.0,
            "extract_ms": 0.0,
            "service_ms": (t2 - t0) * 1000.0,
            "psnr": psnr(packet.frame_rgb, marked),
            "channel_dropped": True,
        }
    recovered = extract(attacked, key)
    t3 = time.perf_counter()
    return {
        "frame_id": packet.frame_id,
        "arrival_s": packet.arrival_s,
        "marked": marked,
        "attacked": attacked,
        "recovered": recovered,
        "embed_ms": (t1 - t0) * 1000.0,
        "channel_ms": (t2 - t1) * 1000.0,
        "extract_ms": (t3 - t2) * 1000.0,
        "service_ms": (t3 - t0) * 1000.0,
        "psnr": psnr(packet.frame_rgb, marked),
        "channel_dropped": False,
    }


def _run_virtual(
    frames: list[np.ndarray],
    *,
    method_id: str,
    watermark: np.ndarray,
    channel: str,
    target_fps: float,
    queue_size: int,
    seed: int,
    step: float | None,
    method_params: dict[str, Any],
) -> tuple[list[dict[str, Any]], int, float]:
    period_s = 1.0 / float(target_fps)
    available_s = 0.0
    traces: list[dict[str, Any]] = []
    queue_drops = 0
    for frame_id, frame in enumerate(frames):
        arrival_s = frame_id * period_s
        waiting_frames = max(0, int(math.floor(max(0.0, available_s - arrival_s) / period_s)))
        if waiting_frames >= int(queue_size):
            queue_drops += 1
            traces.append({
                "frame_id": frame_id, "arrival_s": arrival_s,
                "queue_dropped": True, "channel_dropped": False,
                "deadline_missed": True, "queue_wait_ms": float("nan"),
            })
            continue
        start_s = max(arrival_s, available_s)
        result = _process_packet(
            StreamPacket(frame_id, arrival_s, frame), method_id=method_id,
            watermark=watermark, channel=channel, seed=seed, step=step,
            method_params=method_params,
        )
        end_s = start_s + result["service_ms"] / 1000.0
        available_s = end_s
        traces.append({
            **result,
            "queue_dropped": False,
            "queue_wait_ms": (start_s - arrival_s) * 1000.0,
            "end_to_end_latency_ms": (end_s - arrival_s) * 1000.0,
            "deadline_missed": bool(end_s > arrival_s + period_s),
        })
    total_duration_s = max(available_s, len(frames) * period_s)
    return traces, queue_drops, total_duration_s


def _run_realtime(
    frames: list[np.ndarray],
    *,
    method_id: str,
    watermark: np.ndarray,
    channel: str,
    target_fps: float,
    queue_size: int,
    seed: int,
    step: float | None,
    method_params: dict[str, Any],
) -> tuple[list[dict[str, Any]], int, float]:
    period_s = 1.0 / float(target_fps)
    queue: Queue[StreamPacket] = Queue(maxsize=max(1, int(queue_size)))
    done = Event()
    lock = Lock()
    traces: list[dict[str, Any]] = []
    queue_drops = 0
    origin = time.perf_counter()

    def producer() -> None:
        nonlocal queue_drops
        for frame_id, frame in enumerate(frames):
            target = origin + frame_id * period_s
            remaining = target - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            packet = StreamPacket(frame_id, time.perf_counter() - origin, frame)
            try:
                queue.put_nowait(packet)
            except Full:
                try:
                    dropped = queue.get_nowait()
                    queue.task_done()
                    with lock:
                        traces.append({
                            "frame_id": dropped.frame_id,
                            "arrival_s": dropped.arrival_s,
                            "queue_dropped": True,
                            "channel_dropped": False,
                            "deadline_missed": True,
                            "queue_wait_ms": float("nan"),
                        })
                    queue_drops += 1
                except Empty:
                    pass
                queue.put_nowait(packet)
        done.set()

    def consumer() -> None:
        while not done.is_set() or not queue.empty():
            try:
                packet = queue.get(timeout=0.05)
            except Empty:
                continue
            start = time.perf_counter()
            result = _process_packet(
                packet, method_id=method_id, watermark=watermark, channel=channel,
                seed=seed, step=step, method_params=method_params,
            )
            end = time.perf_counter()
            with lock:
                traces.append({
                    **result,
                    "queue_dropped": False,
                    "queue_wait_ms": max(0.0, (start - origin - packet.arrival_s) * 1000.0),
                    "end_to_end_latency_ms": max(0.0, (end - origin - packet.arrival_s) * 1000.0),
                    "deadline_missed": bool(end - origin > packet.arrival_s + period_s),
                })
            queue.task_done()

    producer_thread = Thread(target=producer, daemon=True)
    consumer_thread = Thread(target=consumer, daemon=True)
    producer_thread.start(); consumer_thread.start()
    producer_thread.join(); consumer_thread.join()
    traces.sort(key=lambda row: int(row["frame_id"]))
    return traces, queue_drops, time.perf_counter() - origin


def summarize_stream_run(
    traces: list[dict[str, Any]],
    *,
    method_id: str,
    channel: str,
    watermark: np.ndarray,
    target_fps: float,
    resolution: tuple[int, int],
    pace_mode: str,
    queue_size: int,
    queue_drops: int,
    total_duration_s: float,
    calibrated_step: float | None,
    calibration_psnr: float | None,
) -> dict[str, Any]:
    processed = [r for r in traces if not r.get("queue_dropped") and not r.get("channel_dropped") and r.get("recovered") is not None]
    channel_drops = sum(bool(r.get("channel_dropped")) for r in traces)
    recovered_frames = [r["recovered"] for r in processed]
    single_nc = [nc_binary(watermark, r["recovered"]) for r in processed]
    single_ber = [ber(watermark, r["recovered"]) for r in processed]
    embed_ms = [float(r["embed_ms"]) for r in processed]
    extract_ms = [float(r["extract_ms"]) for r in processed]
    service_ms = [float(r["service_ms"]) for r in processed]
    latency_ms = [float(r["end_to_end_latency_ms"]) for r in processed]
    psnrs = [float(r["psnr"]) for r in processed]
    deadlines = [bool(r.get("deadline_missed")) for r in traces if not r.get("queue_dropped")]
    running_ncs: list[float] = []
    for i in range(1, len(recovered_frames) + 1):
        running_ncs.append(nc_binary(watermark, _running_vote(recovered_frames[:i])))
    windows: dict[str, float] = {}
    for window in VOTE_WINDOWS:
        if recovered_frames:
            count = min(int(window), len(recovered_frames))
            windows[f"nc_after_{window}_frames"] = nc_binary(watermark, _running_vote(recovered_frames[:count]))
        else:
            windows[f"nc_after_{window}_frames"] = float("nan")
    def first_threshold(threshold: float) -> float:
        for i, value in enumerate(running_ncs):
            if value >= threshold:
                return 1000.0 * (i + 1) / float(target_fps)
        return float("nan")
    frame_count = len(traces)
    processed_count = len(processed)
    effective_fps = processed_count / max(float(total_duration_s), 1e-12)
    pixels = int(resolution[0]) * int(resolution[1])
    service_median = _percentile(service_ms, 50)
    service_p95 = _percentile(service_ms, 95)
    summary = {
        "protocol_version": STREAM_PROTOCOL_VERSION,
        "method_id": method_id,
        "channel": channel,
        "resolution": f"{resolution[0]}x{resolution[1]}",
        "target_fps": float(target_fps),
        "pace_mode": pace_mode,
        "queue_size": int(queue_size),
        "payload_bits": int(watermark.size),
        "frames_generated": int(frame_count),
        "frames_processed": int(processed_count),
        "frames_dropped_queue": int(queue_drops),
        "frames_dropped_channel": int(channel_drops),
        "processed_frame_ratio": float(processed_count / max(frame_count, 1)),
        "deadline_miss_rate": float(np.mean(deadlines)) if deadlines else float("nan"),
        "effective_output_fps": float(effective_fps),
        "real_time_factor": float(effective_fps / float(target_fps)),
        "payload_throughput_bps": float(effective_fps * watermark.size),
        "megapixels_per_second": float(effective_fps * pixels / 1e6),
        "calibrated_step": calibrated_step,
        "calibration_psnr": calibration_psnr,
        "mean_embedding_psnr": float(np.mean(psnrs)) if psnrs else float("nan"),
        "min_embedding_psnr": float(np.min(psnrs)) if psnrs else float("nan"),
        "embed_ms_median": _percentile(embed_ms, 50),
        "embed_ms_p95": _percentile(embed_ms, 95),
        "extract_ms_median": _percentile(extract_ms, 50),
        "extract_ms_p95": _percentile(extract_ms, 95),
        "service_ms_median": service_median,
        "service_ms_p95": service_p95,
        "sustainable_fps_median": 1000.0 / max(service_median, 1e-9),
        "sustainable_fps_p95": 1000.0 / max(service_p95, 1e-9),
        "p95_meets_frame_deadline": bool(service_p95 <= 1000.0 / float(target_fps)) if service_ms else False,
        "latency_ms_median": _percentile(latency_ms, 50),
        "latency_ms_p95": _percentile(latency_ms, 95),
        "latency_ms_p99": _percentile(latency_ms, 99),
        "single_frame_nc_mean": float(np.mean(single_nc)) if single_nc else float("nan"),
        "single_frame_nc_min": float(np.min(single_nc)) if single_nc else float("nan"),
        "single_frame_nc_p10": _percentile(single_nc, 10),
        "single_frame_ber_mean": float(np.mean(single_ber)) if single_ber else float("nan"),
        "single_frame_success_nc_050": float(np.mean(np.asarray(single_nc) >= 0.50)) if single_nc else float("nan"),
        "single_frame_success_nc_070": float(np.mean(np.asarray(single_nc) >= 0.70)) if single_nc else float("nan"),
        "single_frame_success_nc_090": float(np.mean(np.asarray(single_nc) >= 0.90)) if single_nc else float("nan"),
        "time_to_nc_090_ms": first_threshold(0.90),
        "time_to_nc_095_ms": first_threshold(0.95),
        **windows,
    }
    return summary


def run_stream_benchmark(
    *,
    frames: list[np.ndarray],
    watermark: np.ndarray,
    out_dir: str | Path,
    methods: Iterable[str] = STREAM_METHODS,
    channels: Iterable[str] = ("clean", "jpeg70"),
    target_fps: float = 30.0,
    queue_size: int = 2,
    seed: int = 2026,
    target_psnr: float = 45.0,
    calibration_iterations: int = 7,
    pace_mode: str = "virtual",
    proposal_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not frames:
        raise ValueError("At least one frame is required")
    mode = str(pace_mode).lower().strip()
    if mode not in {"virtual", "realtime"}:
        raise ValueError("pace_mode must be 'virtual' or 'realtime'")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wm = np.asarray(watermark, dtype=np.uint8)
    if np.unique(wm).size < 2:
        raise ValueError("Stream watermark must contain both binary symbols")
    h, w = frames[0].shape[:2]
    summaries: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []

    for raw_method in methods:
        method_id = normalize_method_id(raw_method)
        if method_id not in STREAM_METHODS:
            raise KeyError(f"Unsupported stream method {raw_method}; choices: {STREAM_METHODS}")
        params = _method_params(method_id, proposal_params)
        step, observed = calibrate_step_to_psnr(
            method_id, frames[0], wm, target_psnr=float(target_psnr), seed=int(seed),
            iterations=int(calibration_iterations), method_params=params,
        )
        calibration_rows.append({
            "method_id": method_id, "target_psnr": float(target_psnr),
            "calibrated_step": float(step), "observed_psnr": float(observed),
            "method_params_json": json.dumps(params, sort_keys=True),
        })
        # Exact-configuration warm-up before measured streaming.
        marked, key = embed(method_id, frames[0], wm, seed=seed, repeat="auto", step=step, method_params=params)
        _ = extract(marked, key)

        for channel in channels:
            runner = _run_realtime if mode == "realtime" else _run_virtual
            traces, queue_drops, duration = runner(
                frames, method_id=method_id, watermark=wm, channel=channel,
                target_fps=float(target_fps), queue_size=int(queue_size), seed=int(seed),
                step=float(step), method_params=params,
            )
            summary = summarize_stream_run(
                traces, method_id=method_id, channel=channel, watermark=wm,
                target_fps=float(target_fps), resolution=(w, h), pace_mode=mode,
                queue_size=int(queue_size), queue_drops=queue_drops,
                total_duration_s=duration, calibrated_step=float(step),
                calibration_psnr=float(observed),
            )
            summaries.append(summary)
            for row in traces:
                compact = {k: v for k, v in row.items() if k not in {"marked", "attacked", "recovered"}}
                if row.get("recovered") is not None:
                    compact["single_frame_nc"] = nc_binary(wm, row["recovered"])
                    compact["single_frame_ber"] = ber(wm, row["recovered"])
                    compact["single_frame_bit_accuracy"] = bit_accuracy(wm, row["recovered"])
                compact.update({
                    "protocol_version": STREAM_PROTOCOL_VERSION,
                    "method_id": method_id,
                    "channel": channel,
                    "target_fps": float(target_fps),
                    "resolution": f"{w}x{h}",
                    "pace_mode": mode,
                })
                trace_rows.append(compact)

    save_csv(out / "stream_realtime_summary.csv", summaries)
    save_csv(out / "stream_realtime_trace.csv", trace_rows)
    save_csv(out / "stream_calibration.csv", calibration_rows)
    protocol = {
        "protocol_version": STREAM_PROTOCOL_VERSION,
        "kind": "deadline_constrained_causal_online_stream",
        "future_frames_available": False,
        "bounded_queue": int(queue_size),
        "drop_policy": "drop oldest in realtime mode; equivalent bounded-backlog replay in virtual mode",
        "target_fps": float(target_fps),
        "target_psnr": float(target_psnr),
        "channels": list(channels),
        "methods": [normalize_method_id(x) for x in methods],
        "watermark": watermark_statistics(wm),
        "vote_windows": list(VOTE_WINDOWS),
        "pace_mode": mode,
        "scientific_scope": "aligned low-payload streaming benchmark favorable to classical real-time baselines",
    }
    save_json(out / "stream_protocol.json", protocol)
    return {"summary": summaries, "trace": trace_rows, "calibration": calibration_rows, "protocol": protocol}


__all__ = [
    "STREAM_PROTOCOL_VERSION", "STREAM_METHODS", "VOTE_WINDOWS", "load_stream_frames",
    "apply_stream_channel", "summarize_stream_run", "run_stream_benchmark",
]
