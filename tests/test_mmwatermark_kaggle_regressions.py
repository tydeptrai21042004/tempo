"""Offline regression tests for the MMW-only Kaggle path and baseline contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
from PIL import Image

from scripts.prepare_mmwatermark_inputs import prepare
from scripts.baseline_preflight import preflight
from realtime_watermark.baselines.dct_dm_qim_chen2001 import dm_embed_values, dm_decode_values
from realtime_watermark.baselines.dct_stdm_qim_chen2001 import stdm_embed_vectors, stdm_decode_vectors
from realtime_watermark.paper_fidelity import fidelity_status
from realtime_watermark.registry import COMMON_IMAGE_METHODS, BASELINE_METHODS

ROOT=Path(__file__).resolve().parents[1]


def _input_repo(tmp_path:Path) -> Path:
    root=tmp_path/'mini_source'
    for sub in ('data/user_dataset/host', 'data/user_dataset/watermark','configs'):
        (root/sub).mkdir(parents=True,exist_ok=True)
    for p in (ROOT/'data/user_dataset/host').glob('*.bmp'):
        shutil.copy2(p,root/'data/user_dataset/host'/p.name)
    for p in (ROOT/'data/user_dataset/watermark').glob('*.png'):
        shutil.copy2(p,root/'data/user_dataset/watermark'/p.name)
    for name in ('holdout_split_manifest_20261021.json', 'frozen_selected_20261021.json'):
        shutil.copy2(ROOT/'configs'/name,root/'configs'/name)
    return root


def test_previous_kaggle_repository_docs_not_a_generated_artifact():
    # Former Kaggle error (211 tests, only this README rejected) is corrected.
    source=(ROOT/'tests/test_clean_repository.py').read_text()
    assert 'README_OFFICIAL_MMWATERMARK_FIX.md' in source
    assert (ROOT/'README_OFFICIAL_MMWATERMARK_FIX.md').is_file()


def test_reproduction_uses_12_real_verified_hosts_and_no_heldout_tuning(tmp_path):
    root=_input_repo(tmp_path)
    report=prepare(root,tmp_path/'run')
    assert report['source']=='repository_frozen_20261021_not_newly_tuned'
    assert report['host_count']==12 and report['heldout_count']==9
    assert report['watermark_count']==2
    holdout=tmp_path/'run/reports/holdout'
    doc=json.loads((holdout/'split_manifest.json').read_text())
    ref=json.loads((root/'configs/holdout_split_manifest_20261021.json').read_text())
    assert doc==ref
    assert {x['sha256'] for x in doc['test']}.isdisjoint({x['sha256'] for x in doc['development']})
    assert (tmp_path/'run/reports/configs/proposal_selected_all_watermarks.json').read_bytes()==(root/'configs/frozen_selected_20261021.json').read_bytes()


def test_host_hash_mismatch_fails_closed_before_benchmark(tmp_path):
    root=_input_repo(tmp_path)
    source=root/'data/user_dataset/host/baboon.bmp'
    blob=bytearray(source.read_bytes())
    blob[-12] ^= 1
    source.write_bytes(blob)
    with pytest.raises(RuntimeError,match='hash/split mismatch'):
        prepare(root,tmp_path/'out')
    assert not (tmp_path/'out/reports/holdout').exists()


def test_dataset_completeness_fails_closed(tmp_path):
    root=_input_repo(tmp_path)
    (root/'data/user_dataset/watermark/wm.png').unlink()
    with pytest.raises(RuntimeError,match='12 real hosts and 2 PNG'):
        prepare(root,tmp_path/'out')


def test_baseline_preflight_never_claims_paper_exact_reproductions():
    root=ROOT/'data/user_dataset'
    result=preflight(root/'host/baboon.bmp',root/'watermark/wm.png')
    assert len(result['results'])==len(COMMON_IMAGE_METHODS)-1
    assert all(x['status']=='EXECUTED_VALID_OUTPUT' for x in result['results'])
    assert all(x['payload_bits']==4096 for x in result['results'])
    assert all(0<=x['clean_bit_accuracy']<=1 for x in result['results'])
    assert all(not x['paper_reproduction_eligible'] for x in result['results'])
    tier={x['method_id']:x['detector_class'] for x in result['results']}
    assert tier['ca_qim_a2_mao2024']=='semi_blind'
    assert tier['dct_spread_spectrum_cox1997']=='non_blind'
    assert all(x['common_primary_eligible'] == (x['detector_class']=='blind') for x in result['results'])
    assert 'dew_langelaar2001' in BASELINE_METHODS and 'dew_langelaar2001' not in COMMON_IMAGE_METHODS


def test_dither_modulation_reference_lattice_and_detector():
    samples=np.asarray([-19.2,2.5,20.8,41.7],dtype=float)
    bits=np.asarray([0,1,1,0],dtype=np.uint8)
    d0=np.asarray([0.25,-0.15,0.4,-0.2]); d1=np.asarray([-2.25,2.35,-2.1,2.3])
    step=5.0
    embedded=dm_embed_values(samples,bits,step,d0,d1)
    expected=np.rint((samples-np.where(bits>0,d1,d0))/step)*step + np.where(bits>0,d1,d0)
    np.testing.assert_allclose(embedded,expected,atol=1e-12)
    np.testing.assert_array_equal(dm_decode_values(embedded,step,d0,d1),bits)


def test_spread_transform_reference_projection_and_detector():
    x=np.asarray([[3.,1.,2.],[-2.,4.,1.],[3.,-2.,7.]],dtype=float)
    v=np.asarray([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],dtype=float)
    bits=np.array([0,1,0],dtype=np.uint8)
    d0=np.array([0.2,0.2,0.2]);d1=np.array([-1.8,-1.8,-1.8])
    marked=stdm_embed_vectors(x,bits,v,4,d0,d1)
    projection=np.sum(x*v,axis=1)
    target=dm_embed_values(projection,bits,4,d0,d1)
    np.testing.assert_allclose(marked,x+(target-projection)[:,None]*v)
    np.testing.assert_array_equal(stdm_decode_vectors(marked,v,4,d0,d1),bits)


def test_kaggle_only_runner_skips_full_native_by_default():
    source=(ROOT/'KAGGLE_MMWATERMARK_ORIGINAL_ONE_CELL.py').read_text()
    assert 'TEMPO_MMW_RUN_NATIVE_FULL' in source
    assert '== "1"' in source
    assert '02_prepare_verified_inputs' in source
    assert '03_baseline_preflight' in source
    assert '04_original_mmwatermark' in source
    assert "'--methods','proposal'" in (ROOT/'scripts/run_mmwatermark_in_official_repo.py').read_text()


def test_unsupported_common_capacity_dew_is_explicitly_rejected():
    from realtime_watermark.registry import embed
    from realtime_watermark.core import load_rgb,load_binary_watermark
    img=load_rgb(ROOT/'data/user_dataset/host/baboon.bmp')
    payload=load_binary_watermark(ROOT/'data/user_dataset/watermark/wm.png',size=64)
    with pytest.raises(ValueError,match='capacity'):
        embed('dew_langelaar2001',img,payload,seed=2026,repeat=1)


def test_dew_lower_capacity_real_image_roundtrip():
    from realtime_watermark.registry import embed,extract
    from realtime_watermark.core import load_rgb,load_binary_watermark,bit_accuracy
    img=load_rgb(ROOT/'data/user_dataset/host/baboon.bmp')
    payload=load_binary_watermark(ROOT/'data/user_dataset/watermark/wm.png',size=16)
    watermarked,key=embed('dew_langelaar2001',img,payload,seed=2026,repeat=1)
    assert key.fully_blind
    assert bit_accuracy(payload,extract(watermarked,key)) >= 0.99


def _kaggle_eligibility_checker():
    # Isolate the pure eligibility function; importing the notebook runner
    # would start clone/benchmark work, which is inappropriate in unit tests.
    import ast
    tree=ast.parse((ROOT/'KAGGLE_MMWATERMARK_ORIGINAL_ONE_CELL.py').read_text())
    func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='eligible')
    namespace={'json':json,'hashlib':hashlib}
    exec(compile(ast.Module(body=[func],type_ignores=[]),'<Kaggle eligibility>','exec'),namespace)
    return namespace['eligible']


def test_previous_run_acceptance_requires_verified_inputs(tmp_path):
    prepared=tmp_path/'reproduction'
    prepare(ROOT,prepared)
    eligibility=_kaggle_eligibility_checker()
    assert eligibility(prepared)
    # A stale/corrupted result directory cannot silently win over fresh inputs.
    victim=prepared/'reports/holdout/test/host_004_baboon.bmp'
    with victim.open('r+b') as f:
        f.seek(-1,2)
        original=f.read(1)
        f.seek(-1,2)
        f.write(bytes([original[0]^1]))
    assert not eligibility(prepared)


def test_previous_run_missing_watermark_is_not_reused(tmp_path):
    prepared=tmp_path/'reproduction'
    prepare(ROOT,prepared)
    eligibility=_kaggle_eligibility_checker()
    assert eligibility(prepared)
    (prepared/'selected_data/watermark/watermark_002_wm.png').unlink()
    assert not eligibility(prepared)
