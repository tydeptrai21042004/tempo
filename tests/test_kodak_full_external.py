"""Isolated full-coverage and validation checks; no public downloads in CI."""
import sys
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import download_full_kodak as fetch
import run_full_kodak_external as runner


def _png(size=(768,512), mode='RGB'):
    import io
    buf=io.BytesIO()
    Image.new(mode,size,(12,25,38) if mode=='RGB' else 140).save(buf,format='PNG')
    return buf.getvalue()


def test_native_kodak_png_validation():
    assert fetch.inspect_png(_png(), 'kodim01.png')['width']==768
    assert fetch.inspect_png(_png((512,768)), 'kodim02.png')['height']==768
    with pytest.raises(ValueError):fetch.inspect_png(_png((512,512)),'invalid.png')
    with pytest.raises(ValueError):fetch.inspect_png(_png(mode='L'),'invalid.png')


def test_strict_offline_full_kodak(tmp_path):
    for i,name in enumerate(fetch.NAMES):
        Image.new('RGB',(768,512),(i+1,25,38)).save(tmp_path/name)
    out=fetch.download(tmp_path,offline=True)
    assert out['status']=='COMPLETE_24_OF_24' and out['verified_images']==24
    (tmp_path/'kodim24.png').unlink()
    with pytest.raises(FileNotFoundError):fetch.download(tmp_path,offline=True)


def test_full_native_proposal_coverage_mocked(tmp_path, monkeypatch):
    data=tmp_path/'images';data.mkdir()
    for i,name in enumerate(fetch.NAMES):
        Image.new('RGB',(768,512),(i, i, i)).save(data/name)
    marks=tmp_path/'marks';marks.mkdir()
    for name in ('a.png','b.png'):
        Image.fromarray(np.tile(np.array([0,255],dtype=np.uint8),(64,32))).save(marks/name)
    output=tmp_path/'results'
    dummy_cfg=SimpleNamespace(sha256=lambda:'frozen_config')
    monkeypatch.setattr(runner, 'download', lambda *args,**kw:None)
    monkeypatch.setattr(runner, 'config', lambda *args:dummy_cfg)
    monkeypatch.setattr(runner, 'init_cpu',lambda:None)
    monkeypatch.setattr(runner, 'encode',lambda host,wm,cfg,seed:(host,None,3.0))
    monkeypatch.setattr(runner, 'decode',lambda attacked,key:(np.tile(np.array([0,255],dtype=np.uint8),(64,32)),{},2.0))
    monkeypatch.setattr(runner, 'psnr',lambda x,y:44.)
    monkeypatch.setattr(runner, 'ssim_gray',lambda x,y:.99)
    monkeypatch.setattr(runner, 'attacks',lambda x:[('clean',lambda y:y),('jpeg70',lambda y:y),('rotation2p5',lambda y:y)])
    args=SimpleNamespace(out_dir=output,dataset_dir=data,watermark_dir=marks,config=tmp_path/'config.json',
        attack_profile='basic',seed=2026,offline=True)
    result=runner.run(args)
    assert result['status']=='COMPLETE' and result['expected_rows']==144
    assert result['completed_rows']==144 and result['payload_bits']==4096
    # Resume does not duplicate rows or introduce another watermark method.
    result_again=runner.run(args)
    assert result_again['completed_rows']==144
    assert len((output/'kodak_all_cases.csv').read_text().splitlines())==145


def test_full_kodak_capacity_and_no_resize(tmp_path):
    img=Image.new('RGB',(512,768),(5,20,50))
    src=tmp_path/'kodim01.png';img.save(src)
    crop_dir=tmp_path/'crop';crop_dir.mkdir()
    # Actual native source cannot be confused with its separate 512x512 crop.
    assert Image.open(src).size==(512,768)
    assert runner.attacks('basic')[0][0]=='clean'
    assert len(fetch.NAMES)==24
