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
