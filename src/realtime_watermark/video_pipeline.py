from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
import hashlib
import json
import time

import numpy as np

from .common.timing import payload_throughput_bps, summarize_ms
from .common.types import WatermarkKey
from .core import (
    arnold_inverse,
    bit_accuracy,
    load_binary_watermark,
    nc_binary,
    psnr,
    save_image,
    save_json,
)
from .methods import warmup_numba
from .registry import COMMON_VIDEO_METHODS, embed, extract, normalize_method_id
from .proposals.jilp_qim.method import _extract_payload_bits_confidence, _normalize_jilp_key

REALTIME_VIDEO_METHODS = tuple(COMMON_VIDEO_METHODS)


@dataclass
class VideoWatermarkStats:
    mode: str
    input_video: str
    output_video: str | None
    key_path: str | None
    frame_count: int
    input_width: int
    input_height: int
    encoded_width: int
    encoded_height: int
    fps_input: float
    avg_process_ms_per_frame: float
    process_fps: float
    watermark_shape: tuple[int, int]
    watermark_coverage_fraction: float
    avg_frame_psnr_before_video_codec: float | None = None
    recovered_watermark_path: str | None = None
    bit_accuracy: float | None = None
    nc: float | None = None
    protocol_kind: str = "common_framewise_video_adapter"
    timing_boundary: str = "reported separately for kernel and end-to-end stages"
    decode_ms_median: float | None = None
    padding_ms_median: float | None = None
    color_conversion_ms_median: float | None = None
    watermark_kernel_ms_median: float | None = None
    temporal_vote_ms_median: float | None = None
    encode_write_ms_median: float | None = None
    algorithm_total_ms_median: float | None = None
    end_to_end_ms_median: float | None = None
    kernel_fps: float | None = None
    algorithm_fps: float | None = None
    end_to_end_fps: float | None = None
    real_time_factor: float | None = None
    payload_bits: int = 0
    native_bits_per_frame: int = 0
    common_bits_per_frame: int = 0
    payload_bits_per_second: float | None = None
    frames_per_complete_watermark: float | None = 1.0
    time_to_first_complete_watermark_seconds: float | None = None
    complete_watermarks_per_second: float | None = None
    processed_frame_ratio: float = 1.0
    side_information_bytes_per_frame: float | None = None
    warmup_frames: int = 0
    timing_distributions: dict[str, Any] | None = None
    configuration_sha256: str | None = None


def _require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "The video pipeline requires OpenCV. Install opencv-python-headless."
        ) from exc
    return cv2


def _json_load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pad_bgr_to_even(frame_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    h, w = frame_bgr.shape[:2]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h == 0 and pad_w == 0:
        return frame_bgr, {"top": 0, "bottom": 0, "left": 0, "right": 0}
    padded = np.pad(frame_bgr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return padded.astype(np.uint8), {
        "top": 0, "bottom": int(pad_h), "left": 0, "right": int(pad_w)
    }


def proposal_frame_capacity(
    frame_shape: tuple[int, int, int] | tuple[int, int], block_size: int = 8
) -> int:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    return (h // int(block_size)) * (w // int(block_size))


def video_frame_capacity(
    method_id: str, frame_shape: tuple[int, int, int] | tuple[int, int]
) -> int:
    method_id = normalize_method_id(method_id)
    block_count = proposal_frame_capacity(frame_shape, 8)
    return block_count * (4 if method_id == "ca_qim_a2_mao2024" else 1)


def choose_video_mode(
    frame_shape: tuple[int, int, int], wm_shape: tuple[int, int], block_size: int = 8
) -> str:
    capacity = proposal_frame_capacity(frame_shape, block_size)
    payload = int(wm_shape[0] * wm_shape[1])
    if capacity >= payload:
        return "per_frame_full"
    if capacity > 0:
        return "stream_chunked_unsupported"
    raise ValueError("Frame is too small for 8x8 watermark blocks.")


def choose_video_mode_for_method(
    method_id: str, frame_shape: tuple[int, int, int], wm_shape: tuple[int, int]
) -> str:
    payload = int(wm_shape[0] * wm_shape[1])
    return "per_frame_full" if video_frame_capacity(method_id, frame_shape) >= payload else "stream_chunked_unsupported"


def _configuration_hash(method_id: str, step: float | None, method_params: dict[str, Any]) -> str:
    payload = {
        "method_id": normalize_method_id(method_id),
        "step": None if step is None else float(step),
        "method_params": method_params,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _trim_times(values: list[float], warmups: int) -> list[float]:
    usable = values[min(max(0, int(warmups)), len(values)):]
    return usable or values or [0.0]


def _dist(values: list[float], warmups: int) -> dict[str, float]:
    return summarize_ms(_trim_times(values, warmups)).to_dict()


def _open_video(path: str | Path):
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 0:
        fps = 30.0
    return cv2, cap, width, height, fps


def embed_video_method(
    *,
    method_id: str,
    input_video_path: str | Path,
    output_video_path: str | Path,
    watermark_path: str | Path | None,
    key_path: str | Path | None = None,
    wm_size: int = 64,
    seed: int = 2026,
    repeat: int | str = 1,
    step: float | None = None,
    codec: str = "mp4v",
    max_frames: int | None = None,
    warmup_frames: int = 10,
    method_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del repeat  # every frame carries exactly one complete common payload
    method_id = normalize_method_id(method_id)
    if method_id not in REALTIME_VIDEO_METHODS:
        raise KeyError(f"Unsupported video method: {method_id}")
    method_params = dict(method_params or {})
    warmup_numba()

    cv2, cap, width, height, fps = _open_video(input_video_path)
    wm = load_binary_watermark(watermark_path, size=wm_size)
    mode = choose_video_mode_for_method(method_id, (height, width, 3), tuple(wm.shape))
    if mode != "per_frame_full":
        cap.release()
        raise ValueError(
            f"{method_id} capacity {video_frame_capacity(method_id, (height, width, 3))} "
            f"is smaller than payload {wm.size}; the clean release does not silently switch protocols."
        )

    destination = Path(output_video_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    key_destination = Path(key_path) if key_path else destination.with_suffix(".key.json")
    encoded_width = width + (width % 2)
    encoded_height = height + (height % 2)
    fourcc = cv2.VideoWriter_fourcc(*str(codec)[:4].ljust(4, "v"))
    writer = cv2.VideoWriter(str(destination), fourcc, fps, (encoded_width, encoded_height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video writer for {destination}")

    decode_times: list[float] = []
    color_times: list[float] = []
    kernel_times: list[float] = []
    padding_times: list[float] = []
    write_times: list[float] = []
    total_times: list[float] = []
    frame_psnrs: list[float] = []
    first_key: WatermarkKey | None = None
    per_frame_ca_mappings: list[list[int]] = []
    frame_count = 0

    try:
        while max_frames is None or frame_count < int(max_frames):
            total_start = time.perf_counter_ns()
            t0 = time.perf_counter_ns()
            ok, frame_bgr = cap.read()
            t1 = time.perf_counter_ns()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            t2 = time.perf_counter_ns()
            marked_rgb, frame_key = embed(
                method_id, frame_rgb, wm, seed=seed, repeat="auto", step=step,
                method_params=method_params,
            )
            t3 = time.perf_counter_ns()
            marked_bgr = cv2.cvtColor(marked_rgb, cv2.COLOR_RGB2BGR)
            t4 = time.perf_counter_ns()
            padded, padding = _pad_bgr_to_even(marked_bgr)
            t5 = time.perf_counter_ns()
            writer.write(padded)
            t6 = time.perf_counter_ns()

            if first_key is None:
                first_key = frame_key
            if method_id == "ca_qim_a2_mao2024":
                # CA-QIM canonical labeling is host dependent. Retaining only
                # the first frame mapping would decode later frames with the
                # wrong coset-to-message permutation, so the compact 16-entry
                # mapping is stored for each frame.
                per_frame_ca_mappings.append(
                    [int(x) for x in frame_key.params["message_to_coset"]]
                )
            frame_psnrs.append(psnr(frame_rgb, marked_rgb))
            decode_times.append((t1 - t0) / 1e6)
            color_times.append(((t2 - t1) + (t4 - t3)) / 1e6)
            kernel_times.append((t3 - t2) / 1e6)
            padding_times.append((t5 - t4) / 1e6)
            write_times.append((t6 - t5) / 1e6)
            total_times.append((t6 - total_start) / 1e6)
            frame_count += 1
    finally:
        cap.release()
        writer.release()

    if first_key is None or frame_count == 0:
        raise ValueError("Input video contains no readable frames")

    config_hash = str(first_key.params.get("config_hash_sha256") or _configuration_hash(method_id, first_key.step, method_params))
    key_payload = {
        "schema_version": "2.0",
        "protocol_kind": "common_framewise_video_adapter",
        "method_id": method_id,
        "configuration_sha256": config_hash,
        "method_params": method_params,
        "frame_key": asdict(first_key),
        "per_frame_ca_message_to_coset": (
            per_frame_ca_mappings if per_frame_ca_mappings else None
        ),
        "per_frame_key_reason": (
            "host-dependent 16-coset canonical labeling"
            if per_frame_ca_mappings else "frame-invariant key"
        ),
        "watermark_shape": list(wm.shape),
        "payload_bits": int(wm.size),
        "frame_count": int(frame_count),
        "fps": float(fps),
        "input_shape": [height, width, 3],
        "encoded_shape": [encoded_height, encoded_width, 3],
        "codec": str(codec),
        "padding": padding,
        "temporal_vote_mode": str(first_key.params.get("temporal_vote_mode", "hard")),
        "scramble_payload": bool(first_key.params.get("scramble_payload", False)),
        "scientific_scope": "common framewise extension; not a native-video paper reproduction",
    }
    key_destination.parent.mkdir(parents=True, exist_ok=True)
    key_destination.write_text(json.dumps(key_payload, indent=2, default=str), encoding="utf-8")

    d_decode = _dist(decode_times, warmup_frames)
    d_color = _dist(color_times, warmup_frames)
    d_kernel = _dist(kernel_times, warmup_frames)
    d_padding = _dist(padding_times, warmup_frames)
    d_write = _dist(write_times, warmup_frames)
    d_total = _dist(total_times, warmup_frames)
    algorithm_ms = d_color["median_ms"] + d_kernel["median_ms"] + d_padding["median_ms"]
    key_bytes = key_destination.stat().st_size
    stats = VideoWatermarkStats(
        mode="per_frame_full",
        input_video=str(input_video_path),
        output_video=str(destination),
        key_path=str(key_destination),
        frame_count=frame_count,
        input_width=width,
        input_height=height,
        encoded_width=encoded_width,
        encoded_height=encoded_height,
        fps_input=fps,
        avg_process_ms_per_frame=d_kernel["median_ms"],
        process_fps=1000.0 / max(d_kernel["median_ms"], 1e-9),
        watermark_shape=tuple(int(x) for x in wm.shape),
        watermark_coverage_fraction=1.0,
        avg_frame_psnr_before_video_codec=float(np.mean(frame_psnrs)),
        decode_ms_median=d_decode["median_ms"],
        padding_ms_median=d_padding["median_ms"],
        color_conversion_ms_median=d_color["median_ms"],
        watermark_kernel_ms_median=d_kernel["median_ms"],
        encode_write_ms_median=d_write["median_ms"],
        algorithm_total_ms_median=algorithm_ms,
        end_to_end_ms_median=d_total["median_ms"],
        kernel_fps=1000.0 / max(d_kernel["median_ms"], 1e-9),
        algorithm_fps=1000.0 / max(algorithm_ms, 1e-9),
        end_to_end_fps=1000.0 / max(d_total["median_ms"], 1e-9),
        real_time_factor=(1000.0 / max(d_total["median_ms"], 1e-9)) / fps,
        payload_bits=int(wm.size),
        native_bits_per_frame=0,
        common_bits_per_frame=int(wm.size),
        payload_bits_per_second=payload_throughput_bps(int(wm.size), d_kernel["median_ms"]),
        frames_per_complete_watermark=1.0,
        time_to_first_complete_watermark_seconds=1.0 / fps,
        complete_watermarks_per_second=fps,
        side_information_bytes_per_frame=key_bytes / max(frame_count, 1),
        warmup_frames=min(int(warmup_frames), frame_count),
        timing_distributions={
            "decode": d_decode, "color": d_color, "kernel": d_kernel,
            "padding": d_padding, "write": d_write, "end_to_end": d_total,
        },
        configuration_sha256=config_hash,
    )
    return {"stats": asdict(stats), "key": key_payload}


def _proposal_confidence_vote(
    frames_rgb: list[np.ndarray], frame_key: WatermarkKey
) -> tuple[np.ndarray, float]:
    params = _normalize_jilp_key(frame_key)
    shape = tuple(int(x) for x in frame_key.watermark_shape)
    weighted_ones = np.zeros(int(np.prod(shape)), dtype=np.float64)
    weighted_zeros = np.zeros_like(weighted_ones)
    start = time.perf_counter_ns()
    for frame in frames_rgb:
        bits, positions, confidence = _extract_payload_bits_confidence(frame, params, 0, 0)
        n = min(bits.size, positions.size, confidence.size)
        pos = np.asarray(positions[:n], dtype=np.int64)
        weights = np.maximum(np.asarray(confidence[:n], dtype=np.float64), 1e-9)
        bit_slice = np.asarray(bits[:n], dtype=np.uint8)
        np.add.at(weighted_ones, pos, weights * (bit_slice == 1))
        np.add.at(weighted_zeros, pos, weights * (bit_slice == 0))
    scrambled = (weighted_ones >= weighted_zeros).astype(np.uint8).reshape(shape)
    if bool(frame_key.params.get("scramble_payload", True)):
        recovered = arnold_inverse(
            scrambled,
            int(frame_key.arnold_iter),
            int(frame_key.arnold_period),
        )
    else:
        recovered = scrambled
    elapsed = (time.perf_counter_ns() - start) / 1e6
    return (recovered * 255).astype(np.uint8), elapsed


def _hard_vote(recovered_frames: list[np.ndarray]) -> tuple[np.ndarray, float]:
    start = time.perf_counter_ns()
    stack = np.stack([(x > 127).astype(np.uint8) for x in recovered_frames], axis=0)
    rec = (np.sum(stack, axis=0) * 2 >= stack.shape[0]).astype(np.uint8) * 255
    return rec, (time.perf_counter_ns() - start) / 1e6


def extract_video_method(
    *,
    input_video_path: str | Path,
    key_path: str | Path,
    output_watermark_path: str | Path,
    report_path: str | Path | None = None,
    reference_watermark_path: str | Path | None = None,
    max_frames: int | None = None,
    warmup_frames: int = 10,
) -> dict[str, Any]:
    warmup_numba()
    key_payload = _json_load(key_path)
    method_id = normalize_method_id(str(key_payload["method_id"]))
    frame_key = WatermarkKey(**key_payload["frame_key"])
    raw_frame_keys = key_payload.get("frame_keys") or []  # schema 2.0 compatibility
    frame_keys = [WatermarkKey(**item) for item in raw_frame_keys]
    ca_mappings = key_payload.get("per_frame_ca_message_to_coset") or []
    temporal_mode = str(key_payload.get("temporal_vote_mode", "hard"))
    if temporal_mode not in {"hard", "confidence_weighted"}:
        raise ValueError(f"Unsupported temporal vote mode: {temporal_mode}")

    cv2, cap, width, height, fps = _open_video(input_video_path)
    decoded_frames: list[np.ndarray] = []
    recovered_frames: list[np.ndarray] = []
    decode_times: list[float] = []
    color_times: list[float] = []
    kernel_times: list[float] = []
    total_times: list[float] = []
    frame_count = 0
    try:
        while max_frames is None or frame_count < int(max_frames):
            total_start = time.perf_counter_ns()
            t0 = time.perf_counter_ns()
            ok, frame_bgr = cap.read()
            t1 = time.perf_counter_ns()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # Remove encoder padding before watermark extraction.
            target_h, target_w = frame_key.host_shape[:2]
            frame_rgb = frame_rgb[:target_h, :target_w]
            t2 = time.perf_counter_ns()
            if frame_count < len(ca_mappings):
                current_key = replace(
                    frame_key,
                    params={
                        **frame_key.params,
                        "message_to_coset": [int(x) for x in ca_mappings[frame_count]],
                    },
                )
            else:
                current_key = frame_keys[frame_count] if frame_count < len(frame_keys) else frame_key
            if method_id == "proposal" and temporal_mode == "confidence_weighted":
                decoded_frames.append(frame_rgb)
            else:
                recovered_frames.append(extract(frame_rgb, current_key))
            t3 = time.perf_counter_ns()
            decode_times.append((t1 - t0) / 1e6)
            color_times.append((t2 - t1) / 1e6)
            kernel_times.append((t3 - t2) / 1e6)
            total_times.append((t3 - total_start) / 1e6)
            frame_count += 1
    finally:
        cap.release()
    if frame_count == 0:
        raise ValueError("Input video contains no readable frames")

    if method_id == "proposal" and temporal_mode == "confidence_weighted":
        recovered, vote_ms = _proposal_confidence_vote(decoded_frames, frame_key)
    else:
        recovered, vote_ms = _hard_vote(recovered_frames)

    output = Path(output_watermark_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image(output, recovered)

    reference = None
    accuracy = None
    nc = None
    if reference_watermark_path is not None:
        reference = load_binary_watermark(reference_watermark_path, size=int(frame_key.watermark_shape[0]))
        accuracy = bit_accuracy(reference, recovered)
        nc = nc_binary(reference, recovered)

    d_decode = _dist(decode_times, warmup_frames)
    d_color = _dist(color_times, warmup_frames)
    d_kernel = _dist(kernel_times, warmup_frames)
    d_total = _dist(total_times, warmup_frames)
    vote_per_frame = vote_ms / max(frame_count, 1)
    algorithm_ms = d_color["median_ms"] + d_kernel["median_ms"] + vote_per_frame
    stats = VideoWatermarkStats(
        mode="per_frame_full",
        input_video=str(input_video_path),
        output_video=None,
        key_path=str(key_path),
        frame_count=frame_count,
        input_width=width,
        input_height=height,
        encoded_width=width,
        encoded_height=height,
        fps_input=fps,
        avg_process_ms_per_frame=d_kernel["median_ms"],
        process_fps=1000.0 / max(d_kernel["median_ms"], 1e-9),
        watermark_shape=tuple(int(x) for x in frame_key.watermark_shape),
        watermark_coverage_fraction=1.0,
        recovered_watermark_path=str(output),
        bit_accuracy=accuracy,
        nc=nc,
        decode_ms_median=d_decode["median_ms"],
        color_conversion_ms_median=d_color["median_ms"],
        watermark_kernel_ms_median=d_kernel["median_ms"],
        temporal_vote_ms_median=vote_per_frame,
        algorithm_total_ms_median=algorithm_ms,
        end_to_end_ms_median=d_total["median_ms"] + vote_per_frame,
        kernel_fps=1000.0 / max(d_kernel["median_ms"], 1e-9),
        algorithm_fps=1000.0 / max(algorithm_ms, 1e-9),
        end_to_end_fps=1000.0 / max(d_total["median_ms"] + vote_per_frame, 1e-9),
        real_time_factor=(1000.0 / max(d_total["median_ms"] + vote_per_frame, 1e-9)) / fps,
        payload_bits=int(np.prod(frame_key.watermark_shape)),
        common_bits_per_frame=int(np.prod(frame_key.watermark_shape)),
        payload_bits_per_second=payload_throughput_bps(
            int(np.prod(frame_key.watermark_shape)), d_kernel["median_ms"]
        ),
        side_information_bytes_per_frame=Path(key_path).stat().st_size / max(frame_count, 1),
        warmup_frames=min(int(warmup_frames), frame_count),
        timing_distributions={
            "decode": d_decode, "color": d_color, "kernel": d_kernel,
            "end_to_end": d_total, "temporal_vote_total_ms": vote_ms,
        },
        configuration_sha256=str(key_payload.get("configuration_sha256")),
    )
    result = {"stats": asdict(stats), "key": key_payload}
    if report_path is not None:
        save_json(report_path, result)
    return result


def embed_video_proposal(**kwargs) -> dict[str, Any]:
    kwargs["method_id"] = "proposal"
    return embed_video_method(**kwargs)


def extract_video_proposal(**kwargs) -> dict[str, Any]:
    return extract_video_method(**kwargs)


__all__ = [
    "VideoWatermarkStats", "REALTIME_VIDEO_METHODS", "proposal_frame_capacity",
    "video_frame_capacity", "choose_video_mode", "choose_video_mode_for_method",
    "_pad_bgr_to_even", "embed_video_method", "extract_video_method",
    "embed_video_proposal", "extract_video_proposal", "_hard_vote",
    "_proposal_confidence_vote",
]
