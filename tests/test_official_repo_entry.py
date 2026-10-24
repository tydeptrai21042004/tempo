"""Offline tests for the official-checkout TEMPO bridge (no fake performance claims)."""
from __future__ import annotations
import hashlib, importlib.util, os, subprocess, sys
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'


def load():
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec=importlib.util.spec_from_file_location('mm_launcher_under_test', SCRIPTS/'run_mmwatermark_in_official_repo.py')
        obj=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(obj)
        return obj
    finally:
        sys.path.remove(str(SCRIPTS))


def git_repo(path: Path, origin: str, files: dict[str,str]):
    path.mkdir(parents=True)
    for name, content in files.items():
        dest=path/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)
    subprocess.run(['git','init','-q',str(path)],check=True)
    for option,value in [('user.email','test@example.test'),('user.name','Test')]:
        subprocess.run(['git','-C',str(path),'config',option,value],check=True)
    subprocess.run(['git','-C',str(path),'add','.'],check=True)
    subprocess.run(['git','-C',str(path),'commit','-qm','fixture'],check=True)
    subprocess.run(['git','-C',str(path),'remote','add','origin',origin],check=True)


def test_bridge_written_into_original_checkout_with_unmodified_tracked_files(tmp_path,monkeypatch):
    launch=load()
    origin=launch.official.OFFICIAL_URL
    official=tmp_path/'original'
    git_repo(official, origin, {'perturbations/image_perturb.py':'def gaussian_noise(x, severity): return x\n'})
    entry=SCRIPTS/'run_official_mmwatermark_tempo.py'
    bridge,digest=launch.install_bridge(official,entry)
    assert bridge == official/'perturbations/tempo_proposal_runner.py'
    assert hashlib.sha256(bridge.read_bytes()).hexdigest() == digest
    assert subprocess.check_output(['git','-C',str(official),'diff','--name-only'],text=True).strip()==''
    assert subprocess.check_output(['git','-C',str(official),'ls-files','--others','--exclude-standard'],text=True).strip()=='perturbations/tempo_proposal_runner.py'
    with pytest.raises(RuntimeError,match='modified/unexpected files'):
        launch.official.prepare_repo(official,clone=False)
    monkeypatch.setenv('TEMPO_OFFICIAL_BRIDGE_SHA256',digest)
    monkeypatch.setattr(launch.official,'__file__',str(bridge))
    assert len(launch.official.prepare_repo(official,clone=False))==40
    (official/'perturbations/image_perturb.py').write_text('TAMPERED')
    with pytest.raises(RuntimeError,match='modified/unexpected files'):
        launch.official.prepare_repo(official,clone=False)


def test_launcher_builds_correct_command_and_strict_coverage():
    mod=load()
    args=SimpleNamespace(official_repo=Path('/official'),companion_repo=Path('/companion'),
        host_dir=Path('/host'),watermark_dir=Path('/watermarks'),config=Path('/config'),
        out_dir=Path('/results'),attacks='',severities=[1,2,3,4,5],seed=2026,
        host_limit=0,watermark_limit=0,allow_partial=False)
    cmd=mod.build_command(args,Path('/official/perturbations/tempo_proposal_runner.py'),'f'*40)
    assert '--strict-official-coverage' in cmd
    assert cmd[cmd.index('--methods')+1]=='proposal'
    assert cmd[cmd.index('--expected-sha')+1]=='f'*40
    assert cmd[cmd.index('--severities')+1:cmd.index('--severities')+6]==['1','2','3','4','5']


def test_original_runner_strict_coverage_preserves_completed_trials(tmp_path):
    """A broken upstream condition must not erase genuinely completed rows."""
    import json
    from PIL import Image
    import numpy as np
    launch=load()
    original=tmp_path/'upstream'
    companion=tmp_path/'MM_Robustness'
    git_repo(original,launch.official.OFFICIAL_URL,{
        'perturbations/image_perturb.py':(
            'def gaussian_noise(x, severity):\n    return x\n'
            'def shot_noise(x, severity):\n    raise RuntimeError("unavailable original attack")\n'
        )
    })
    git_repo(companion,launch.official.COMPANION_URL,{
        'image_perturbation/imagenet_c/imagenet_c/__init__.py':'# fixture\n'
    })
    host_dir=tmp_path/'hosts';watermark_dir=tmp_path/'marks'
    host_dir.mkdir();watermark_dir.mkdir()
    Image.fromarray(np.full((512,512,3), 127, dtype=np.uint8)).save(host_dir/'host.bmp')
    bits=(np.indices((64,64)).sum(axis=0)%2*255).astype(np.uint8)
    Image.fromarray(bits).save(watermark_dir/'mark.png')
    root=Path(__file__).resolve().parents[1]
    args=SimpleNamespace(official_repo=original,companion_repo=companion,
        host_dir=host_dir,watermark_dir=watermark_dir,
        config=root/'configs/frozen_selected_20261021.json',
        out_dir=tmp_path/'out',attacks='gaussian_noise,shot_noise',
        severities=[1],seed=2026,host_limit=0,watermark_limit=0,allow_partial=False)
    import subprocess,os
    with pytest.raises(subprocess.CalledProcessError):
        launch.main(args)
    report=json.loads((tmp_path/'out/official_mmwatermark_manifest.json').read_text())
    assert report['full_coverage'] is False
    assert report['completed_rows']==2 # one clean + one real upstream attack
    assert report['expected_full_requested_rows']==3
    assert report['preflight_unsupported_conditions'][0]['attack']=='shot_noise'
    assert len((tmp_path/'out/official_mmwatermark_raw.csv').read_text().splitlines())==3
