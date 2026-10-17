#!/usr/bin/env python3
"""Run new experimental evidence without touching TEMPO proposal implementation.

Full experiments are opt-in. Independent stages never need to wait on another
stage and can be scheduled on different Kaggle sessions using --only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
STAGES={
 'paired':'reviewer_method_paired_study.py',
 'pareto':'reviewer_pareto_frontier.py',
 'sync':'reviewer_sync_decisions.py',
 'integer':'reviewer_integer_optimality.py',
 'latency':'reviewer_latency_breakdown.py',
 'external':'reviewer_external_dataset.py',
 'temporal':'reviewer_real_temporal.py',
 'attack':'reviewer_attack_budget.py',
 'learned':'reviewer_learned_baseline.py',
}

def run(a):
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    names=a.only.split(',') if a.only else list(STAGES)
    if not names or any(x not in STAGES for x in names):
        raise ValueError(f'Unknown stage; choices: {sorted(STAGES)}')
    env=os.environ.copy()
    env['PYTHONPATH']=os.pathsep.join([str(ROOT/'src'),str(ROOT),env.get('PYTHONPATH','')])
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[name]='1'
    manifest={'protocol':'TEMPO_reviewer_expansion_v1','mode':'quick' if a.quick else 'full',
       'stages':{},'source_core_sha256':{},'no_proposal_modifications':True}
    # Identify the source implementation, unchanged throughout execution.
    for p in ('src/realtime_watermark/proposals/jilp_qim/method.py',
              'src/realtime_watermark/proposals/jilp_qim/engine.py',
              'src/realtime_watermark/proposals/jilp_qim/phase_registration.py'):
        manifest['source_core_sha256'][p]=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
    for stage in names:
        script=ROOT/'scripts'/STAGES[stage]
        args=[sys.executable,str(script),'--out-dir',str(out/stage)]
        if stage!='learned':
            args+=['--host-dir',str(a.host_dir),'--watermark-dir',str(a.watermark_dir),
                   '--config',str(a.config),'--seed',str(a.seed)]
            if a.quick:
                args+=['--quick','--max-hosts','1','--max-watermarks','1']
        if stage=='pareto':
            if not a.development_dir:
                manifest['stages'][stage]={'status':'blocked','reason':'Development host directory required for parameter sweep'};continue
            args[args.index('--host-dir')+1]=str(a.development_dir)
        elif stage=='external':
            if not (a.external_host_dir and a.development_dir):
                manifest['stages'][stage]={'status':'blocked','reason':'Requires external and development image directories'};continue
            args+=['--external-host-dir',str(a.external_host_dir),
                   '--development-host-dir',str(a.development_dir)]
        elif stage=='temporal':
            if not a.video:
                manifest['stages'][stage]={'status':'blocked','reason':'Requires actual video path'};continue
            args+=['--video',str(a.video),'--frames',str(3 if a.quick else a.video_frames)]
        elif stage=='latency':
            args+=['--repeats',str(2 if a.quick else a.repeats)]
        elif stage=='integer':
            args+=['--blocks',str(2 if a.quick else 10)]
        elif stage=='learned':
            if not (a.learned_provenance and a.learned_csv):
                manifest['stages'][stage]={'status':'blocked','reason':'Requires independently run pretrained model CSV and provenance JSON'};continue
            args=[sys.executable,str(script),'--out-dir',str(out/stage),
                  '--provenance',str(a.learned_provenance),'--observations',str(a.learned_csv)]
        logpath=out/f'{stage}.log'
        start=time.monotonic()
        with logpath.open('w',encoding='utf8') as log:
            result=subprocess.run(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                                  check=False)
        status={'status':'complete' if result.returncode==0 else 'failed',
                'exit_code':result.returncode,'seconds':round(time.monotonic()-start,3),
                'command':args,'log':str(logpath),'manifest_path':str(out/stage/'manifest.json')}
        if result.returncode==0 and not Path(status['manifest_path']).is_file():
            status['status']='failed';status['reason']='No evidence manifest produced'
        manifest['stages'][stage]=status
        (out/'expansion_ledger.json').write_text(json.dumps(manifest,indent=2))
        print(stage,status['status'],f"{status.get('seconds',0)}s",flush=True)
    completed=[name for name in names if manifest['stages'][name]['status']=='complete']
    missing=[name for name in names if manifest['stages'][name]['status']!='complete']
    manifest['status']='complete' if not missing else 'incomplete'
    manifest['completed_stages']=completed
    manifest['incomplete_stages']=missing
    (out/'expansion_ledger.json').write_text(json.dumps(manifest,indent=2))
    if not a.no_zip:
        archive = shutil.make_archive(str(out), 'zip', root_dir=str(out))
        print('Results ZIP:', archive, flush=True)
    if missing and a.strict:raise RuntimeError('Some experiments are incomplete: '+', '.join(missing))
    return manifest

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--quick',action='store_true')
    p.add_argument('--only',default='',help='Comma-separated subset; stages run independently')
    p.add_argument('--strict',action='store_true',help='Nonzero exit if any selected stage blocked or failed')
    p.add_argument('--no-zip',action='store_true',help='Do not automatically ZIP experiment output logs and CSV files')
    p.add_argument('--host-dir',default=str(ROOT/'data/user_dataset/host'))
    p.add_argument('--watermark-dir',default=str(ROOT/'data/user_dataset/watermark'))
    p.add_argument('--config',default=str(ROOT/'configs/proposal_revision_candidate.json'))
    p.add_argument('--development-dir',default=None,help='Separate development-only image folder')
    p.add_argument('--external-host-dir',default=None,help='Actually new images, not repository host duplicates')
    p.add_argument('--video',default=None,help='Actual video input .mp4')
    p.add_argument('--video-frames',type=int,default=31)
    p.add_argument('--repeats',type=int,default=8)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--learned-provenance',default=None)
    p.add_argument('--learned-csv',default=None)
    return p
if __name__=='__main__':run(parser().parse_args())
