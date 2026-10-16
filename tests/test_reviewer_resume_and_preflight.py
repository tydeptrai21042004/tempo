"""Regressions for preflight and crash-safe reviewer evidence resumption."""
from __future__ import annotations
import json
from pathlib import Path
import pytest
from PIL import Image
import numpy as np
from scripts.reviewer_preflight import inspect
from scripts import reviewer_completeness

ROOT=Path(__file__).resolve().parents[1]


def test_real_dataset_preflight_is_valid():
    report=inspect(ROOT/'data/user_dataset/host',ROOT/'data/user_dataset/watermark')
    # Raw upstream has 13 BMPs including Lena. The selected 12-host directory
    # is the one expected to pass; raw input should fail early by design.
    assert report['status']=='failed' and report['host_count']==13
    assert report['watermark_count']==2 and len(report['watermarks'])==2
    assert all(0.0<r['one_fraction']<1.0 for r in report['watermarks'])


def test_input_preflight_rejects_one_symbol_payload(tmp_path):
    hosts=tmp_path/'hosts'; hosts.mkdir()
    wms=tmp_path/'watermarks';wms.mkdir()
    Image.fromarray(np.zeros((64,64),dtype=np.uint8)).save(wms/'bad.png')
    report=inspect(hosts,wms)
    assert report['status']=='failed'
    assert any('Degenerate watermark' in s for s in report['errors'])


def test_720p_coverage_not_required_for_every_lower_resolution_stream(tmp_path):
    # Two native-HD streams have HD conditions, two SD streams do not.
    # The old reviewer checker mistakenly required 720p from all four.
    root=tmp_path
    stream=root/'live_stream';stream.mkdir()
    for i,hd in enumerate((True,True,False,False)):
        job=stream/f's{i}'; job.mkdir()
        (job/'ALL_STREAM_run_manifest.json').write_text(json.dumps({'pace_mode':'realtime','payload_bits':4096}))
        conditions=[('512x512','clean')]+([('1280x720','clean'),('1280x720','jpeg70')] if hd else [])
        content='resolution,channel,pace_mode,payload_bits,service_ms_p95,effective_output_fps\n'
        content+=''.join(f'{res},{chan},realtime,4096,20,25\n' for res,chan in conditions)
        (job/'ALL_STREAM_realtime_summary.csv').write_text(content)
    result=reviewer_completeness.check(root,mode='full')
    assert result['checks']['720p_deadline_stress'] is True
    assert result['checks']['downloaded_real_video_corpus'] is False
    assert result['execution_evidence_complete'] is False
