"""Ensure 720p streaming traces never retain 2 giant video buffers/frame."""
from __future__ import annotations
import numpy as np
from realtime_watermark import stream_pipeline as sp


def test_realtime_trace_drops_large_image_arrays(monkeypatch):
    def fake_process(packet, **kwargs):
        return {'frame_id':packet.frame_id,'arrival_s':packet.arrival_s,
                'marked':np.zeros((720,1280,3),dtype=np.uint8),
                'attacked':np.zeros((720,1280,3),dtype=np.uint8),
                'recovered':np.zeros((64,64),dtype=np.uint8),
                'service_ms':1.0,'channel_dropped':False}
    monkeypatch.setattr(sp,'_process_packet',fake_process)
    frames=[np.zeros((720,1280,3),dtype=np.uint8) for _ in range(3)]
    rows,drops,duration=sp._run_realtime(
        frames,method_id='proposal',watermark=np.zeros((64,64),dtype=np.uint8),
        channel='jpeg70',target_fps=120,queue_size=2,seed=2026,step=15.,method_params={})
    assert len(rows)==3 and drops==0 and duration>0
    assert all('marked' not in r and 'attacked' not in r for r in rows)
    assert all(r['recovered'].shape==(64,64) for r in rows)


def test_virtual_trace_drops_large_image_arrays(monkeypatch):
    def fake_process(packet, **kwargs):
        return {
            'frame_id': packet.frame_id, 'arrival_s': packet.arrival_s,
            'marked': np.ones((720, 1280, 3), dtype=np.uint8),
            'attacked': np.ones((720, 1280, 3), dtype=np.uint8),
            'recovered': np.zeros((64, 64), dtype=np.uint8),
            'service_ms': 1.0, 'channel_dropped': False,
        }
    monkeypatch.setattr(sp, '_process_packet', fake_process)
    frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(3)]
    rows, drops, duration = sp._run_virtual(
        frames, method_id='proposal',
        watermark=np.zeros((64, 64), dtype=np.uint8),
        channel='jpeg70', target_fps=30, queue_size=2, seed=2026,
        step=15., method_params={},
    )
    assert len(rows) == 3 and drops == 0 and duration > 0
    assert all('marked' not in row and 'attacked' not in row for row in rows)
    assert all(row['recovered'].shape == (64, 64) for row in rows)


def test_realtime_trace_drops_channel_frames_without_retaining_arrays(monkeypatch):
    def fake_process(packet, **kwargs):
        return {
            'frame_id': packet.frame_id, 'arrival_s': packet.arrival_s,
            'marked': np.zeros((720, 1280, 3), dtype=np.uint8),
            'attacked': None, 'recovered': None,
            'service_ms': 1.0, 'channel_dropped': True,
        }
    monkeypatch.setattr(sp, '_process_packet', fake_process)
    rows, drops, _ = sp._run_realtime(
        [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(2)],
        method_id='proposal', watermark=np.zeros((64, 64), dtype=np.uint8),
        channel='drop10', target_fps=10, queue_size=2, seed=2026,
        step=15., method_params={},
    )
    assert len(rows) == 2 and drops == 0
    assert all(row['channel_dropped'] for row in rows)
    assert all('marked' not in row and 'attacked' not in row for row in rows)
