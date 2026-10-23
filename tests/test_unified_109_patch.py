"""Versioned common protocol and fail-closed external-provider contract."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

from realtime_watermark.attack_manifest import load_manifest, manifest_summary
from realtime_watermark.core import default_attacks, attack_gaussian_variance, attack_jpeg_subsampling
from realtime_watermark.paper_fidelity import common_primary_static_eligible

ROOT = Path(__file__).resolve().parents[1]


def _load_script(path):
    spec = importlib.util.spec_from_file_location('tempo_check_'+path.stem,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unified_has_109_named_attacks_and_legacy_74_unchanged():
    legacy=default_attacks('stress')
    unified=default_attacks('unified')
    assert len(legacy)==75 and len(unified)==110
    assert [name for name,_ in unified[:75]]==[name for name,_ in legacy]
    assert len(set(name for name,_ in unified))==110
    assert manifest_summary()['attacked_conditions']==109
    assert len(manifest_summary()['sha256'])==64
    x=np.random.default_rng(91).integers(0,256,size=(48,48,3),dtype=np.uint8)
    for (name, fn1), (_, fn2) in zip(legacy, unified):
        assert np.array_equal(fn1(x),fn2(x)), name


def test_all_unified_attacks_shape_and_dtype():
    x=np.random.default_rng(2026).integers(0,256,size=(48,48,3),dtype=np.uint8)
    for name,fn in default_attacks('unified'):
        y=fn(x)
        assert y.shape==x.shape and y.dtype==np.uint8, name


def test_new_noise_units_and_codec_modes():
    x=np.full((48,48,3),128,np.uint8)
    assert np.array_equal(attack_gaussian_variance(x,.003,seed=17),attack_gaussian_variance(x,.003,seed=17))
    assert not np.array_equal(attack_jpeg_subsampling(x,70,0),np.zeros_like(x))
    with pytest.raises(ValueError,match='subsampling'):
        attack_jpeg_subsampling(x,70,1)


def test_invalid_manifest_operation_fails_closed():
    from realtime_watermark.attack_manifest import _validate_node,_operators
    with pytest.raises(ValueError,match='Unknown attack'):
        _validate_node({'op':'os.system','args':['$image','rm -rf /']},_operators())


def test_blind_matched_psnr_ranking_excludes_sideinfo_methods():
    assert not common_primary_static_eligible('ca_qim_a2_mao2024')
    assert not common_primary_static_eligible('dct_spread_spectrum_cox1997')
    assert common_primary_static_eligible('dct_dm_qim_chen2001')


def test_external_import_strict_contract():
    module=_load_script(ROOT/'scripts/import_external_baseline_results.py')
    row={k:'x' for k in module.REQUIRED}
    row.update(method_id='external_2026',reference_doi='10.1000/reference',
        implementation_url='https://example.org/official',implementation_commit='a'*40,
        detector_class='blind',host='house.bmp',watermark='wm.png',attack='jpeg40',
        payload_bits='4096',psnr='45',nc='.8',ber='.1',embed_ms='1',extract_ms='2',
        source_result_sha256='b'*64)
    assert len(module.validate([row],{'house.bmp'},{'wm.png'},allowed_attacks={'jpeg40'},strict_provenance=True))==1
    with pytest.raises(ValueError,match='unknown or incompatible'):
        module.validate([dict(row,attack='random_attack')],{'house.bmp'},{'wm.png'},allowed_attacks={'jpeg40'},strict_provenance=True)
    with pytest.raises(ValueError,match='commit'):
        module.validate([dict(row,implementation_commit='fake')],{'house.bmp'},{'wm.png'},allowed_attacks={'jpeg40'},strict_provenance=True)


def test_official_upstream_source_wrong_remote_rejected(tmp_path):
    mod=_load_script(ROOT/'scripts/run_official_mmwatermark_tempo.py')
    repo=tmp_path/'official'
    (repo/'perturbations').mkdir(parents=True)
    (repo/'perturbations'/'image_perturb.py').write_text('def gaussian_noise(image,severity): return image')
    subprocess.run(['git','init','-q',str(repo)],check=True)
    subprocess.run(['git','-C',str(repo),'config','user.email','test@example.test'],check=True)
    subprocess.run(['git','-C',str(repo),'config','user.name','Test'],check=True)
    subprocess.run(['git','-C',str(repo),'add','.'],check=True)
    subprocess.run(['git','-C',str(repo),'commit','-qm','test'],check=True)
    subprocess.run(['git','-C',str(repo),'remote','add','origin','https://example.org/not-official.git'],check=True)
    with pytest.raises(RuntimeError,match='Not the official'):
        mod.prepare_repo(repo,clone=False)
