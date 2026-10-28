#!/usr/bin/env python3
"""Full TEMPO experiment launcher without automatic scientific gate enforcement.

Run stages independently with --only, or all listed stages by default.
Runs do not alter the proposal. Experimental failures remain visible in a
ledger; they are not silently replaced with pseudo-measurements.
A compact ZIP contains only numerical evidence and selected PDF figures.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from package_paper_evidence import package

ROOT=Path(__file__).resolve().parents[1]
SCRIPTS={
    'theory':'reviewer_joint_theory.py',
    'capacity':'reviewer_capacity_full.py',
    'security':'reviewer_security_extended.py',
    'key_schedule':'reviewer_key_schedule_security.py',
    'detection':'reviewer_detection_protocol.py',
    'verification':'reviewer_security_verification.py',
    'sync':'reviewer_sync_decisions.py',
    'attack_budget':'reviewer_attack_budget.py',
    'integer':'reviewer_integer_optimality.py',
    'integer_boundaries':'reviewer_integer_boundaries.py',
    'latency':'reviewer_latency_breakdown.py',
    'learned_native':'reviewer_learned_baseline.py',
    'pareto':'reviewer_pareto_frontier.py',
    'joint_sensitivity':'reviewer_joint_sensitivity.py',
    'temporal_limits':'reviewer_temporal_limits.py',
    'video_security':'reviewer_video_security.py',
    'real_temporal':'reviewer_real_temporal.py',
    'baselines':'run_multiwatermark_fair_benchmark.py',
    'external':'reviewer_external_dataset.py',
    'mmwatermark':'run_mmwatermark_in_official_repo.py',
    'video_baselines':'run_multiwatermark_video.py',
    'video_corpus':'run_video_corpus_suite.py',
    'gallery':'reviewer_gallery.py',
    'plots':'generate_tempo_evidence_plots.py',
}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    stages=[s.strip() for s in a.only.split(',') if s.strip()] if a.only else list(SCRIPTS)
    if not stages or any(x not in SCRIPTS for x in stages):
        raise ValueError(f'Unknown stage, supported: {", ".join(SCRIPTS)}')
    if a.development_dir is None or a.test_dir is None:
        split=out/'split'
        split_manifest=split/'split_manifest.json'
        if not split_manifest.exists():
            command=[sys.executable,str(ROOT/'scripts/create_reviewer_holdout.py'),
                '--host-dir',str(a.host_dir),'--out-dir',str(split),
                '--development-count',str(a.development_count)]
            subprocess.run(command,cwd=ROOT,check=True)
        dev=split/'development';test=split/'test'
    else:
        dev=a.development_dir;test=a.test_dir
    if not dev.is_dir() or not test.is_dir():
        raise FileNotFoundError('Separate development and held-out test directories are required')
    dev_hash={sha(p) for p in dev.glob('*.bmp')};test_hash={sha(p) for p in test.glob('*.bmp')}
    if dev_hash & test_hash:
        raise RuntimeError('Development and test contain identical bytes')
    env={**os.environ,'PYTHONPATH':os.pathsep.join((str(ROOT/'src'),str(ROOT),os.environ.get('PYTHONPATH','')))}
    for var in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[var]='1'
    ledger_path=out/'experiment_ledger.json'
    ledger=json.loads(ledger_path.read_text()) if a.resume and ledger_path.is_file() else {
        'suite':'TEMPO-full-paper-evidence-v2','proposal_method_changed':False,
        'config_sha256':sha(a.config),
        'source_sha256':{s:sha(ROOT/'src/realtime_watermark/proposals/jilp_qim'/s)
            for s in ('method.py','engine.py','phase_registration.py')},'stages':{}}
    shared=['--host-dir',str(test),'--watermark-dir',str(a.watermark_dir),
            '--config',str(a.config),'--seed',str(a.seed)]
    if a.quick:shared+=['--quick','--max-hosts','1','--max-watermarks','1']
    for stage in stages:
        script=ROOT/'scripts'/SCRIPTS[stage]
        stage_dir=out/stage
        command=[sys.executable,str(script)]
        blocked=None
        if stage=='plots':
            command+=['--results-dir',str(out),'--out-dir',str(stage_dir)]
        elif stage=='detection':
            command+=['--development-dir',str(dev),'--test-dir',str(test),
                '--watermark-dir',str(a.watermark_dir),'--config',str(a.config),
                '--out-dir',str(stage_dir),'--seed',str(a.seed),
                '--negative-keys',str(2 if a.quick else a.negative_keys)]
            if a.sync_detection:command+=['--sync']
        elif stage=='learned_native':
            if a.learned_provenance is None or a.learned_csv is None:
                blocked='Independent real learned-model results CSV and provenance JSON are required; no simulated model attacks'
            command+=['--provenance',str(a.learned_provenance or ''),
                '--observations',str(a.learned_csv or ''),'--out-dir',str(stage_dir)]
        elif stage=='verification':
            command+=['--development-dir',str(dev),'--test-dir',str(test),
                '--watermark-dir',str(a.watermark_dir),'--config',str(a.config),
                '--out-dir',str(stage_dir),'--seed',str(a.seed),
                '--negative-keys',str(2 if a.quick else a.negative_keys)]
            if a.quick:command+=['--quick']
            if a.sync_detection:command+=['--both-modes']
        elif stage=='mmwatermark':
            command+=['--official-repo',str(a.official_repo),'--companion-repo',str(a.companion_repo),
                '--host-dir',str(a.mm_host_dir or test),'--watermark-dir',str(a.watermark_dir),
                '--config',str(a.config),'--out-dir',str(stage_dir),'--seed',str(a.seed),
                '--protocol','paper100']
            if a.quick:command+=['--host-limit','1','--watermark-limit','1']
        elif stage=='baselines':
            command+=['--host-dir',str(test),'--watermark-dir',str(a.watermark_dir),
                '--calibration-host-dir',str(dev),'--proposal-config',str(a.config),
                '--out-dir',str(stage_dir),'--seed',str(a.seed),
                '--attack-preset','reviewer46']
            if a.quick:
                command+=['--host-limit','1','--watermark-limit','1',
                          '--calibration-iterations','2','--timing-repeats','2','--timing-warmups','1']
        elif stage=='gallery':
            command+=['--host-dir',str(test),'--watermark-dir',str(a.watermark_dir),
                '--config',str(a.config),'--out-dir',str(stage_dir),
                '--max-hosts','1' if a.quick else '2',
                '--max-watermarks','1' if a.quick else '2','--seed',str(a.seed)]
        elif stage=='video_baselines':
            if a.video is None or not a.video.is_file():
                blocked='A real video file is required for video baselines'
            command+=['--watermark-dir',str(a.watermark_dir),
                '--proposal-config',str(a.config),'--out-dir',str(stage_dir),
                '--input',str(a.video or ''),'--seed',str(a.seed),
                '--max-frames',str(12 if a.quick else a.video_frames),
                '--warmup-frames','2' if a.quick else '10', '--methods','all']
        elif stage=='video_corpus':
            if a.video_corpus is None or not a.video_corpus.is_file():
                blocked='A verified multi-video --video-corpus JSON is required; no synthetic fallback'
            command+=['--corpus',str(a.video_corpus or ''),
                '--watermark-dir',str(a.watermark_dir),'--proposal-config',str(a.config),
                '--video-out',str(stage_dir/'video'),'--stream-out',str(stage_dir/'streams'),
                '--mode','quick' if a.quick else 'full',
                '--max-frames',str(12 if a.quick else a.video_frames),
                '--stream-frames',str(12 if a.quick else a.video_frames),
                '--resolutions','512x512' if a.quick else '512x512,1280x720',
                '--channels','clean,jpeg70' if a.quick else 'clean,jpeg90,jpeg70,jpeg40']
            if a.resume:command+=['--resume']
        elif stage in ('video_security','real_temporal'):
            if a.video is None or not a.video.is_file():
                blocked='A real readable --video file is required; synthetic fallback is forbidden'
            command+=shared+['--out-dir',str(stage_dir),'--video',str(a.video or ''),
                '--frames',str(5 if a.quick else a.video_frames)]
        elif stage=='joint_sensitivity':
            command+=['--development-dir',str(dev),'--watermark-dir',str(a.watermark_dir),
                '--config',str(a.config),'--out-dir',str(stage_dir),'--seed',str(a.seed),
                '--max-hosts','1' if a.quick else '3','--max-watermarks','1' if a.quick else '2']
        elif stage=='temporal_limits':
            command+=['--host-dir',str(test),'--watermark-dir',str(a.watermark_dir),
                '--config',str(a.config),'--out-dir',str(stage_dir),
                '--seed',str(a.seed),'--max-hosts','1','--max-watermarks','1' if a.quick else '2',
                '--repetitions','1' if a.quick else '3']
        elif stage=='latency':
            command+=shared+['--out-dir',str(stage_dir),
                '--original-config',str(a.config),
                '--refined-config',str(ROOT/'configs/proposal_minimal_refined.json'),
                '--repeats','2' if a.quick else '8']
        elif stage=='integer_boundaries':
            command+=['--host-dir',str(test),'--watermark-dir',str(a.watermark_dir),
                '--config',str(a.config),'--out-dir',str(stage_dir),
                '--seed',str(a.seed)]
        elif stage=='external':
            if a.external_host_dir is None or not a.external_host_dir.is_dir():
                blocked='External held-out host folder required for this stage'
            command+=shared+['--out-dir',str(stage_dir),
                '--external-host-dir',str(a.external_host_dir or ''),
                '--development-host-dir',str(dev)]
        else:
            command+=shared+['--out-dir',str(stage_dir)]
            if stage=='pareto':
                command[command.index('--host-dir')+1]=str(dev)
                command+=['--data-role','development']
            if stage=='integer':command+=['--blocks','2' if a.quick else '10']
        digest=sha(script)
        previous=ledger['stages'].get(stage,{})
        if stage!='plots' and a.resume and previous.get('status')=='complete' and previous.get('command')==command and previous.get('script_sha256')==digest:
            print(f'SKIP completed {stage}',flush=True);continue
        if blocked:
            record={'status':'blocked','reason':blocked,'script_sha256':digest,'command':command}
        else:
            stage_dir.mkdir(parents=True,exist_ok=True)
            logfile=out/f'{stage}.log'
            with logfile.open('w',encoding='utf8') as log:
                response=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            record={'status':'complete' if response.returncode==0 else 'failed',
                    'exit_code':response.returncode,'script_sha256':digest,'command':command,
                    'log':str(logfile.relative_to(out))}
        ledger['stages'][stage]=record
        ledger_path.write_text(json.dumps(ledger,indent=2))
        print(f'{stage}: {record["status"]}',flush=True)
        if record['status']=='failed' and a.fail_fast:break
    statuses=[ledger['stages'][k].get('status') for k in stages if k in ledger['stages']]
    ledger['status']='complete' if len(statuses)==len(stages) and all(s=='complete' for s in statuses) else 'partial'
    ledger_path.write_text(json.dumps(ledger,indent=2))
    if not a.no_zip:
        archive=a.zip_output or out.with_name(out.name+'_PAPER_EVIDENCE.zip')
        index=package(out,archive)
        print(f'Compact evidence ZIP: {archive} ({index["saved_files"]} selected files)',flush=True)
    return 0 if ledger['status']=='complete' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--only',default='',help='Comma-separated stage names; default is all')
    p.add_argument('--quick',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--no-zip',action='store_true')
    p.add_argument('--zip-output',type=Path)
    p.add_argument('--fail-fast',action='store_true')
    p.add_argument('--config',type=Path,default=ROOT/'configs/manuscript_frozen.json')
    p.add_argument('--host-dir',type=Path,default=ROOT/'data/user_dataset/host')
    p.add_argument('--watermark-dir',type=Path,default=ROOT/'data/user_dataset/watermark')
    p.add_argument('--development-dir',type=Path)
    p.add_argument('--test-dir',type=Path)
    p.add_argument('--development-count',type=int,default=3)
    p.add_argument('--external-host-dir',type=Path)
    p.add_argument('--mm-host-dir',type=Path,help='Optional Kodak BMP directory for original-100 evaluation')
    p.add_argument('--official-repo',type=Path,default=Path('/kaggle/working/MMWatermark-Robustness'))
    p.add_argument('--companion-repo',type=Path,default=Path('/kaggle/working/MM_Robustness'))
    p.add_argument('--video',type=Path)
    p.add_argument('--video-corpus',type=Path,help='Real multi-video corpus JSON for causal video benchmark')
    p.add_argument('--video-frames',type=int,default=31)
    p.add_argument('--negative-keys',type=int,default=16)
    p.add_argument('--sync-detection',action='store_true')
    p.add_argument('--learned-provenance',type=Path)
    p.add_argument('--learned-csv',type=Path)
    p.add_argument('--seed',type=int,default=2026)
    sys.exit(main(p.parse_args()))
