from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from scripts.video_evidence_corpus import prepare, sha256, _download
from scripts.run_video_corpus_suite import run as run_suite


def local_fixture(tmp_path: Path, *, width=320, height=180, frames=4) -> Path:
    f=tmp_path/'actual_video.mp4'
    writer=cv2.VideoWriter(str(f),cv2.VideoWriter_fourcc(*'mp4v'),15,(width,height))
    assert writer.isOpened()
    for i in range(frames):
        frame=np.empty((height,width,3),dtype=np.uint8)
        frame[:]=[20+i*8,80+i*10,120+i*15]
        cv2.rectangle(frame,(max(0,i*9),30),(min(width-1,i*9+50),min(height-1,95)),(120,210,30),-1)
        writer.write(frame)
    writer.release()
    assert f.is_file() and f.stat().st_size>1000
    return f


def make_manifest(tmp_path:Path,filename:Path, *, checksum=None, split='quick'):
    cfg={'schema':'tempo_video_corpus_v1','sources':[{'id':'fixture','local_path':str(filename),
        'split':split,'sha256':checksum or sha256(filename),'max_download_mb':8}]}
    path=tmp_path/'manifest.json';path.write_text(json.dumps(cfg))
    return path


def test_prepared_video_has_source_checksum_and_real_video_flag(tmp_path):
    video=local_fixture(tmp_path)
    manifest=make_manifest(tmp_path,video)
    result=prepare(manifest=manifest,cache=tmp_path/'cache',prepared=tmp_path/'prepared',
                   output=tmp_path/'corpus.json',mode='quick',frames=3,allow_local=True)
    assert result['verified_count']==1
    clip=result['videos'][0]
    assert clip['source_kind']=='local_test_fixture' and clip['is_real_source_video'] is False
    assert clip['original_sha256']==sha256(video)
    assert clip['preparation']['frames_written']==3
    assert clip['preparation']['prepared_resolution']=='512x512'
    assert Path(clip['prepared_path']).is_file()


def test_reject_bad_sha256_and_quarantine_download(tmp_path):
    video=local_fixture(tmp_path)
    cfg=make_manifest(tmp_path,video,checksum='0'*64)
    with pytest.raises(ValueError,match='SHA256 mismatch'):
        prepare(manifest=cfg,cache=tmp_path/'cache',prepared=tmp_path/'prepared',
                output=tmp_path/'corpus.json',mode='quick',frames=3,allow_local=True)


def test_local_input_never_silently_accepted_as_public_download(tmp_path):
    video=local_fixture(tmp_path)
    cfg=make_manifest(tmp_path,video)
    with pytest.raises(ValueError,match='--allow-local-fixtures'):
        prepare(manifest=cfg,cache=tmp_path/'cache',prepared=tmp_path/'prepared',
                output=tmp_path/'corpus.json',mode='quick',frames=2)


def test_full_requires_genuinely_native_720p(tmp_path):
    video=local_fixture(tmp_path)
    cfg=make_manifest(tmp_path,video)
    with pytest.raises(RuntimeError,match='native 720p'):
        prepare(manifest=cfg,cache=tmp_path/'cache',prepared=tmp_path/'prepared',
                output=tmp_path/'corpus.json',mode='full',frames=3,allow_local=True)


def test_download_validation_fails_closed_on_bad_content(tmp_path):
    class FakeResponse(io.BytesIO):
        headers={'Content-Length':'4096'}
        def geturl(self):return 'https://example.org/test.mp4'
    with patch('scripts.video_evidence_corpus.urlopen',return_value=FakeResponse(b'<html>'*800)):
        result=tmp_path/'video.mp4'
        _download('https://example.org/test.mp4',result,1024*1024,5)
        with pytest.raises(RuntimeError):
            from scripts.video_evidence_corpus import inspect_video
            inspect_video(result)


def test_download_rejects_non_https(tmp_path):
    with pytest.raises(ValueError,match='HTTPS'):
        _download('http://example.org/test.mp4',tmp_path/'video.mp4',100000,5)


def test_unproven_cached_file_is_redownloaded(tmp_path):
    original=local_fixture(tmp_path,frames=3)
    cfg={'sources':[{'id':'cached','split':'quick','url':'https://example.org/video.mp4','max_download_mb':8}]}
    manifest=tmp_path/'remote.json';manifest.write_text(json.dumps(cfg))
    cache=tmp_path/'cache';cache.mkdir()
    # A same-name file with no download sidecar is not trusted as remote evidence.
    (cache/'cached.mp4').write_bytes(original.read_bytes())
    seen=[]
    def fake_download(url,destination,limit,timeout):
        seen.append(url)
        destination.write_bytes(original.read_bytes())
    with patch('scripts.video_evidence_corpus._download',side_effect=fake_download):
        out=prepare(manifest=manifest,cache=cache,prepared=tmp_path/'prepared',
                    output=tmp_path/'corpus.json',mode='quick',frames=2)
    assert seen==['https://example.org/video.mp4']
    assert out['all_real_videos'] is True
    assert (cache/'cached.download_provenance.json').exists()
    # A valid downloaded video is re-used and rehashed, not fetched again.
    with patch('scripts.video_evidence_corpus._download',side_effect=AssertionError('unexpected network call')):
        prepare(manifest=manifest,cache=cache,prepared=tmp_path/'prepared',
                output=tmp_path/'corpus.json',mode='quick',frames=2)
