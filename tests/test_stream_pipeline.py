from __future__ import annotations

from pathlib import Path

from realtime_watermark.core import make_demo_watermark
from realtime_watermark.stream_pipeline import (
    STREAM_METHODS, apply_stream_channel, load_stream_frames, run_stream_benchmark,
)


def test_stream_registry_contains_dew_and_proposal():
    assert "dew_langelaar2001" in STREAM_METHODS
    assert "proposal" in STREAM_METHODS
    assert "dct_spread_spectrum_cox1997" not in STREAM_METHODS


def test_virtual_stream_pipeline_writes_causal_reports(tmp_path: Path):
    frames = load_stream_frames(None, width=128, height=128, max_frames=4)
    wm = make_demo_watermark(8)
    result = run_stream_benchmark(
        frames=frames, watermark=wm, out_dir=tmp_path,
        methods=["dew_langelaar2001"], channels=["clean"],
        target_fps=30.0, queue_size=2, seed=5, target_psnr=45.0,
        calibration_iterations=3, pace_mode="virtual",
    )
    summary = result["summary"][0]
    assert summary["frames_generated"] == 4
    assert summary["frames_processed"] >= 1
    assert summary["payload_bits"] == 64
    assert "latency_ms_p95" in summary
    assert "time_to_nc_090_ms" in summary
    assert result["protocol"]["future_frames_available"] is False
    assert (tmp_path / "stream_realtime_summary.csv").exists()
    assert (tmp_path / "stream_realtime_trace.csv").exists()
    assert (tmp_path / "stream_protocol.json").exists()


def test_stream_channels_preserve_shape_and_drop_is_supported(host128):
    assert apply_stream_channel(host128, "jpeg70", frame_id=0, seed=1).shape == host128.shape
    dropped_or_frame = apply_stream_channel(host128, "drop10", frame_id=1, seed=1)
    assert dropped_or_frame is None or dropped_or_frame.shape == host128.shape
