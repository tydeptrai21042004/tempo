"""Regression tests for experimental integrity, not watermark algorithm changes."""
from __future__ import annotations
import csv
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


def load(name):
    path=ROOT/'scripts'/name
    spec=importlib.util.spec_from_file_location('verify_'+name.replace('.','_'),path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rejection_wilson_and_classification():
    m=load('reviewer_integer_boundaries.py')
    a,b=m.wilson(0,10)
    assert a==pytest.approx(0.) and 0<b<1
    a,b=m.wilson(10,10)
    assert a<1 and b==pytest.approx(1.)
    assert m.classify_error(RuntimeError('Pilot parity infeasible in 1 uint8 block'))=='rejected_by_integer_constraints'
    assert m.classify_error(ValueError('Bad shape'))=='unexpected_error'
    cases=m.fixtures((512,512,3),quick=False)
    assert {name for name,_ in cases} >= {'black_0','white_255','shadow_0to3','highlight_252to255'}
    assert all(arr.shape==(512,512,3) and arr.dtype==np.uint8 for _,arr in cases)


def test_strict_paired_psnr_excludes_nonmatching_conditions():
    m=load('reviewer_matched_pairs.py')
    summary=[
        {'host':'h1','watermark':'w','method_id':'proposal','primary_rank_eligible':True},
        {'host':'h1','watermark':'w','method_id':'baseline','primary_rank_eligible':True},
        {'host':'h2','watermark':'w','method_id':'proposal','primary_rank_eligible':True},
        {'host':'h2','watermark':'w','method_id':'baseline','primary_rank_eligible':False},
    ]
    attacks=[{'host':h,'watermark':'w','phase':phase,'method_id':method,'primary_rank_eligible':True,'nc_bipolar':n}
             for h,phase,method,n in [
                ('h1','jpeg30','proposal',.8),('h1','jpeg30','baseline',.3),
                ('h1','blur','proposal',.7),('h2','jpeg30','proposal',.9),('h2','jpeg30','baseline',.2)]]
    rows,effects=m.matched_effects(summary,attacks,nboot=20)
    assert len(rows)==1
    assert rows[0]['host']=='h1' and rows[0]['attack']=='jpeg30'
    assert rows[0]['delta_nc']==pytest.approx(.5)
    assert effects[0]['n_paired_hosts']==1
    assert np.isnan(effects[0]['ci95_host_cluster_lo'])
    with pytest.raises(ValueError,match='Duplicate matched-PSNR'):
        m.matched_effects(summary+summary[:1],attacks)


def test_sync_metrics_and_pilot_actual_counters_in_source():
    method=(ROOT/'src/realtime_watermark/proposals/jilp_qim/method.py').read_text()
    metrics=(ROOT/'src/realtime_watermark/all_metrics.py').read_text()
    experiments=(ROOT/'scripts/reviewer_experiments.py').read_text()
    assert 'params["integer_pilot_unit_updates"]' in method
    assert 'key.params["integer_pilot_unit_updates"]' in metrics
    assert "key_params['integer_pilot_unit_updates']" in experiments
    assert 'sync_harmed_nc' in experiments and 'sync_gain_nc' in experiments
    assert 'key.params.get("pilot_lattice_unit_updates", 0)' not in metrics


def test_completeness_is_not_a_file_presence_only_check(tmp_path):
    audit=load('reviewer_completeness.py')
    boundary=tmp_path/'integer_boundaries'
    boundary.mkdir()
    (boundary/'boundary_manifest.json').write_text('{"status":"complete","attempts":2}')
    with (boundary/'embedding_attempts.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['scenario','attempted','status'])
        writer.writeheader()
        writer.writerow({'scenario':'black_0','attempted':1,'status':'accepted'})
    checks=audit.check(tmp_path,mode='quick')['checks']
    assert checks['integer_rejection_denominators'] is False
    assert checks['matched_psnr_same_case_paired'] is False
    assert checks['720p_deadline_stress'] is False


def test_throughput_audit_rejects_wrong_denominator(tmp_path):
    import json
    streams=tmp_path/'streams'/'wm';streams.mkdir(parents=True)
    (streams/'ALL_STREAM_run_manifest.json').write_text(json.dumps({
        'pace_mode':'realtime','payload_bits':4096,'source_video':'deterministic_synthetic_stream'}))
    with (streams/'ALL_STREAM_realtime_summary.csv').open('w',newline='') as f:
        fields=['effective_output_fps','payload_bits','payload_throughput_bps',
                'service_ms_p95','target_fps','on_time_completion_ratio',
                'resolution','channel','pace_mode','frames_generated',
                'frames_processed','frames_dropped_queue','frames_dropped_channel']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        w.writerow(dict(effective_output_fps=20,payload_bits=4096,
             payload_throughput_bps=30*4096,service_ms_p95=60,target_fps=30,
             on_time_completion_ratio=.5,resolution='1280x720',channel='jpeg70',
             pace_mode='realtime',frames_generated=10,frames_processed=7,
             frames_dropped_queue=3,frames_dropped_channel=0))
    m=load('reviewer_throughput_audit.py')
    with pytest.raises(RuntimeError,match='throughput mismatch'):
        m.audit(tmp_path/'streams',tmp_path/'audited')
    audit=json.loads((tmp_path/'audited'/'wallclock_throughput_manifest.json').read_text())
    assert audit['status']=='failed'
    assert audit['has_real_video'] is False
