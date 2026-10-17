"""Regression tests for scientific holdout and tamper-resistant video evidence."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image
import numpy as np
from scripts.method_revision_study import resolved_hosts
from scripts.create_reviewer_holdout import split_hosts
from scripts.video_evidence_corpus import prepare
import cv2


def local_fixture(tmp_path: Path, *, width=320, height=180, frames=4) -> Path:
    """Build a small genuine codec-encoded video without importing other tests.

    Kaggle can have a third-party ``tests`` package installed, so ``from
    tests.test_video_evidence_corpus import ...`` is not portable.
    """
    video = tmp_path / 'actual_video.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'),
                             15, (width, height))
    assert writer.isOpened(), 'OpenCV mp4v encoder unavailable'
    try:
        for i in range(frames):
            frame = np.empty((height, width, 3), dtype=np.uint8)
            frame[:] = (20 + i * 8, 80 + i * 10, 120 + i * 15)
            cv2.rectangle(frame, (max(0, i * 9), 30),
                          (min(width - 1, i * 9 + 50), min(height - 1, 95)),
                          (120, 210, 30), -1)
            writer.write(frame)
    finally:
        writer.release()
    assert video.is_file() and video.stat().st_size > 1000
    return video


def test_revision_study_fails_closed_without_actual_split(tmp_path):
    with pytest.raises(FileNotFoundError, match='Official split manifest missing'):
        resolved_hosts('holdout', tmp_path/'missing.json')


def test_revision_study_uses_actual_holdout_and_checksums(tmp_path):
    raw=tmp_path/'raw';raw.mkdir()
    for i in range(4):
        Image.fromarray(np.full((8,8,3),i*55,dtype=np.uint8)).save(raw/f'host{i}.bmp')
    output=tmp_path/'holdout'
    split_hosts(raw,output,development_count=2)
    selected=resolved_hosts('holdout',output/'split_manifest.json')
    assert [x.name for x in selected]==['host2.bmp','host3.bmp']
    assert not set(selected)&set(resolved_hosts('development',output/'split_manifest.json'))
    (output/'test/host3.bmp').write_bytes(b'tampered')
    with pytest.raises(RuntimeError,match='hash mismatch'):
        resolved_hosts('holdout',output/'split_manifest.json')


def test_regenerating_holdout_drops_stale_files(tmp_path):
    raw=tmp_path/'raw';raw.mkdir()
    for i in range(4):
        Image.fromarray(np.full((8,8,3),i*45,dtype=np.uint8)).save(raw/f'host{i}.bmp')
    output=tmp_path/'holdout'
    split_hosts(raw,output,development_count=1)
    assert (output/'test/host1.bmp').exists()
    split_hosts(raw,output,development_count=2)
    assert not (output/'test/host1.bmp').exists()


def test_download_retries_and_preserves_real_source_provenance(tmp_path):
    source=local_fixture(tmp_path,frames=4)
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'sources':[{'id':'video','split':'quick',
        'url':'https://example.org/video.mp4','max_download_mb':8}]}))
    calls=[]
    def flaky(url,dest,limit,timeout):
        calls.append(url)
        if len(calls)==1:
            raise TimeoutError('transient download failure')
        dest.write_bytes(source.read_bytes())
    with patch('scripts.video_evidence_corpus._download',side_effect=flaky),patch('scripts.video_evidence_corpus.time.sleep'):
        evidence=prepare(manifest=manifest,cache=tmp_path/'cache',prepared=tmp_path/'prepared',
                         output=tmp_path/'corpus.json',mode='quick',frames=3)
    assert len(calls)==2 and evidence['all_real_videos'] is True
    assert evidence['videos'][0]['original_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()


def test_download_fail_closed_after_three_attempts(tmp_path):
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'sources':[{'id':'video','split':'quick',
        'url':'https://example.org/video.mp4','max_download_mb':8}]}))
    with patch('scripts.video_evidence_corpus._download',side_effect=TimeoutError('offline')) as mock, patch('scripts.video_evidence_corpus.time.sleep'):
        with pytest.raises(TimeoutError,match='offline'):
            prepare(manifest=manifest,cache=tmp_path/'cache',prepared=tmp_path/'prepared',
                    output=tmp_path/'corpus.json',mode='quick',frames=3)
    assert mock.call_count==3
    assert not (tmp_path/'corpus.json').is_file()


def test_tampered_prepared_video_refused_before_expensive_experiments(tmp_path):
    from scripts.run_video_corpus_suite import run
    original=tmp_path/'raw.mp4';original.write_bytes(b'original bytes')
    prepared=tmp_path/'prepared.mp4';prepared.write_bytes(b'old prepared bytes')
    corpus=tmp_path/'corpus.json'
    corpus.write_text(json.dumps({'status':'complete','videos':[{'id':'clip','status':'verified',
        'native_720p':False,'original_path':str(original),
        'original_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
        'prepared_path':str(prepared),'prepared_sha256':hashlib.sha256(prepared.read_bytes()).hexdigest()}]}))
    wms=tmp_path/'watermarks';wms.mkdir()
    (wms/'a.png').write_bytes(b'a');(wms/'b.png').write_bytes(b'b')
    prepared.write_bytes(b'tampered content')
    with pytest.raises(RuntimeError,match='Prepared video changed'):
        run(corpus_path=corpus,watermark_dir=wms,config=tmp_path/'config.json',
            video_out=tmp_path/'video_out',stream_out=tmp_path/'stream_out',
            mode='quick',max_frames=2,stream_frames=2,codec='mp4v',python_bin='python',resume=True)
