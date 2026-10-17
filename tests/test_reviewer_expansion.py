"""Proof that extension evidence is fail-closed and does not alter proposal code."""
from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))


def load(name):
    file=ROOT/'scripts'/name
    spec=importlib.util.spec_from_file_location('rev_'+name.replace('.','_'),file)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_nondominated_filters_strictly_dominated_and_nonfinite():
    mod=load('reviewer_pareto_frontier.py')
    rows=[{'mean_psnr_db':40.,'mean_nc':.6,'effective_fps':25.,'name':'weak'},
          {'mean_psnr_db':44.,'mean_nc':.8,'effective_fps':30.,'name':'strong'},
          {'mean_psnr_db':46.,'mean_nc':.5,'effective_fps':40.,'name':'tradeoff'},
          {'mean_psnr_db':float('nan'),'mean_nc':1.,'effective_fps':99.,'name':'invalid'}]
    assert {r['name'] for r in mod.nondominated(rows)}=={'strong','tradeoff'}


def test_restricted_exact_integer_study_requires_real_feasibility():
    mod=load('reviewer_integer_optimality.py')
    # Constant block produces zero DCT low-frequency difference; bit=0 valid.
    a=np.full((8,8,3),128,dtype=np.uint8)
    result=mod.search_local(a,a.copy(),mod.carrier(.07),17.,.45,radius=1)
    assert result['original_energy']==0.
    assert result['best_restricted_energy']==0.
    assert result['original_feasible_for_requested_guard']
    assert result['feasible_candidates']>0
    assert not result['strict_restricted_improvement']


def test_missing_pretrained_baseline_is_never_marked_complete(tmp_path):
    run=load('run_reviewer_expansion.py')
    p=run.parser()
    a=p.parse_args(['--only','learned','--quick','--output',str(tmp_path)])
    state=run.run(a)
    assert state['status']=='incomplete'
    assert state['stages']['learned']['status']=='blocked'
    assert not (tmp_path/'learned'/'manifest.json').exists()


def test_pretrained_csv_provenance_refuses_missing_weights(tmp_path):
    mod=load('reviewer_learned_baseline.py')
    path=tmp_path/'provenance.json';path.write_text(json.dumps({'model_name':'test-model'}))
    obs=tmp_path/'observations.csv';obs.write_text('host\nframe1\n')
    args=mod.parser().parse_args(['--provenance',str(path),'--observations',str(obs),
                                  '--out-dir',str(tmp_path/'out')])
    with pytest.raises(ValueError,match='provenance'):
        mod.run(args)


def test_external_image_hash_overlap_is_refused(tmp_path):
    mod=load('reviewer_external_dataset.py')
    from PIL import Image
    import shutil
    development=tmp_path/'dev';external=tmp_path/'external'
    development.mkdir();external.mkdir()
    a=np.full((512,512,3),120,dtype=np.uint8)
    Image.fromarray(a).save(development/'dev.png')
    shutil.copyfile(development/'dev.png',external/'leak.png')
    args=mod.parser().parse_args(['--external-host-dir',str(external),
       '--development-host-dir',str(development), '--out-dir',str(tmp_path/'result'),
       '--watermark-dir',str(ROOT/'data/user_dataset/watermark')])
    with pytest.raises(RuntimeError,match='overlap'):
        mod.run(args)


def test_real_temporal_rejects_missing_video(tmp_path):
    mod=load('reviewer_real_temporal.py')
    with pytest.raises(FileNotFoundError):
        mod.load_video(tmp_path/'never.mp4',3)
