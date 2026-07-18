from __future__ import annotations

from pathlib import Path
import numpy as np

from realtime_watermark.core import make_demo_watermark, save_image
from realtime_watermark.demo_assets import ensure_demo_video
from realtime_watermark.proposals.jilp_qim.config import DEFAULT_CONFIG
from realtime_watermark.registry import embed
from realtime_watermark.video_pipeline import (
    REALTIME_VIDEO_METHODS, _hard_vote, _pad_bgr_to_even,
    _proposal_confidence_vote, embed_video_method, extract_video_method,
    video_frame_capacity,
)


def test_video_method_set_is_the_same_five_method_protocol():
    assert set(REALTIME_VIDEO_METHODS) == {
        "dct_dm_qim_chen2001", "dct_stdm_qim_chen2001", "dct_iss_malvar2003", "ca_qim_a2_mao2024", "proposal"
    }


def test_ca_qim_video_capacity_is_four_bits_per_block():
    assert video_frame_capacity("dct_dm_qim_chen2001", (128, 128, 3)) == 256
    assert video_frame_capacity("ca_qim_a2_mao2024", (128, 128, 3)) == 1024


def test_even_padding():
    frame = np.zeros((127, 129, 3), dtype=np.uint8)
    padded, info = _pad_bgr_to_even(frame)
    assert padded.shape == (128, 130, 3)
    assert info["bottom"] == 1 and info["right"] == 1


def test_hard_vote_majority():
    zero = np.zeros((4, 4), dtype=np.uint8)
    one = np.full((4, 4), 255, dtype=np.uint8)
    rec, _ = _hard_vote([one, one, zero])
    assert np.all(rec == 255)


def test_confidence_vote_applies_inverse_scrambling(host128, wm16):
    marked, key = embed("proposal", host128, wm16, seed=31, repeat=1, method_params={**DEFAULT_CONFIG.to_dict(), "temporal_vote_mode": "confidence_weighted"})
    recovered, _ = _proposal_confidence_vote([marked, marked, marked], key)
    assert np.array_equal(recovered, wm16)


def test_small_video_roundtrip(tmp_path):
    wm_path = tmp_path / "wm.png"
    save_image(wm_path, make_demo_watermark(16))
    video = ensure_demo_video(tmp_path / "inputs", input_path=None, auto_download=False, width=128, height=128, frames=6)
    marked = tmp_path / "marked.mp4"
    key = tmp_path / "key.json"
    recovered = tmp_path / "recovered.png"
    er = embed_video_method(
        method_id="proposal", input_video_path=video, output_video_path=marked,
        watermark_path=wm_path, key_path=key, wm_size=16, seed=32,
        step=DEFAULT_CONFIG.step, max_frames=6, warmup_frames=1,
        method_params=DEFAULT_CONFIG.to_dict(),
    )["stats"]
    report = tmp_path / "extract_report.json"
    result = extract_video_method(
        input_video_path=marked, key_path=key, output_watermark_path=recovered,
        report_path=report, reference_watermark_path=wm_path,
        max_frames=6, warmup_frames=1,
    )
    xr = result["stats"]
    assert er["frame_count"] == 6
    assert er["protocol_kind"] == "common_framewise_video_adapter"
    assert xr["bit_accuracy"] >= 0.90
    assert xr["per_frame_bit_accuracy_mean"] is not None
    assert xr["per_frame_nc_mean"] is not None
    assert xr["serialized_key_bytes"] > 0
    assert xr["end_to_end_payload_bits_per_second"] > 0
    assert Path(xr["per_frame_report_path"]).is_file()
    assert len(result["per_frame"]) == 6
    assert Path(recovered).is_file()


def test_ca_qim_video_stores_and_uses_each_frame_canonical_mapping(tmp_path):
    import json

    wm_path = tmp_path / "wm_ca.png"
    save_image(wm_path, make_demo_watermark(16))
    video = ensure_demo_video(
        tmp_path / "inputs_ca", input_path=None, auto_download=False,
        width=128, height=128, frames=6,
    )
    marked = tmp_path / "ca_marked.mp4"
    key = tmp_path / "ca_key.json"
    recovered = tmp_path / "ca_recovered.png"
    er = embed_video_method(
        method_id="ca_qim_a2_mao2024", input_video_path=video,
        output_video_path=marked, watermark_path=wm_path, key_path=key,
        wm_size=16, seed=44, step=28.0, max_frames=6, warmup_frames=1,
    )["stats"]
    payload = json.loads(key.read_text(encoding="utf-8"))
    mappings = payload["per_frame_ca_message_to_coset"]
    assert len(mappings) == er["frame_count"] == 6
    assert all(sorted(mapping) == list(range(16)) for mapping in mappings)
    xr = extract_video_method(
        input_video_path=marked, key_path=key,
        output_watermark_path=recovered,
        reference_watermark_path=wm_path, max_frames=6, warmup_frames=1,
    )["stats"]
    assert xr["bit_accuracy"] >= 0.95
