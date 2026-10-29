#!/usr/bin/env python3
"""Final reviewer protocol: 5 distinct image baselines, DEW-only real-video baseline.

Required inputs are existing prepared split folders and the verified real-video
corpus from kaggle_pipeline.sh. No network calls or neural baseline stages.
A stage is complete only after its subprocess succeeds and its expected
numerical outputs exist. Failed stages remain visible in the ledger.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from realtime_watermark.registry import ROBUSTNESS_BASELINES, REALTIME_VIDEO_BASELINES


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def csv_rows(path):
    with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def invoke(label, cmd, expected, out, ledger, *, resume=False, env=None):
    checked=[Path(x) for x in expected]
    prior=ledger['stages'].get(label,{})
    script_path=Path(cmd[1]); script_hash=sha(script_path) if script_path.is_file() else None
    if (resume and prior.get('status')=='complete' and prior.get('command')==[str(x) for x in cmd]
        and prior.get('script_sha256')==script_hash and all(p.is_file() and p.stat().st_size>0 for p in checked)):
        print('RESUME',label,flush=True)
        return True
    log=out/(label+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    print('RUN',label,' '.join(map(str,cmd)),flush=True)
    with log.open('w',encoding='utf-8') as f:
        r=subprocess.run([str(x) for x in cmd],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    ok=r.returncode==0 and all(p.is_file() and p.stat().st_size>0 for p in checked)
    ledger['stages'][label]={'status':'complete' if ok else 'failed','command':[str(x) for x in cmd],
        'script_sha256':script_hash,'expected_files':[str(p) for p in checked],
        'exit_code':r.returncode,'log':str(log),'checked_at':datetime.now(timezone.utc).isoformat()}
    (out/'experiment_ledger.json').write_text(json.dumps(ledger,indent=2))
    if not ok:
        print('FAILED',label,'(see',log,')',flush=True)
        print('\n'.join(log.read_text(errors='replace').splitlines()[-12:]),flush=True)
    return ok


def main(a):
    if set(ROBUSTNESS_BASELINES)&set(REALTIME_VIDEO_BASELINES):
        raise RuntimeError('Image and real-video baseline sets must be disjoint')
    if len(ROBUSTNESS_BASELINES)!=5 or tuple(REALTIME_VIDEO_BASELINES)!=('dew_langelaar2001',):
        raise RuntimeError('Unexpected reviewer baseline registry')
    for path in (a.test_dir,a.development_dir,a.watermark_dir,a.config):
        if not Path(path).exists():raise FileNotFoundError(path)
    output=Path(a.output).resolve();output.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
    for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[n]='1'
    existing=(output/'experiment_ledger.json')
    ledger=json.loads(existing.read_text()) if a.resume and existing.is_file() else {'protocol':'TEMPO_final_reviewer_2026',
        'image_baselines':list(ROBUSTNESS_BASELINES), 'real_video_baselines':list(REALTIME_VIDEO_BASELINES),
        'excluded_by_scope':['expanded_learned_native','RivaGAN','VideoSeal','RBMark'],
        'proposal_algorithm_modified':False,'stages':{},
        'config_sha256':sha(a.config),
        'test_hosts':{p.name:sha(p) for p in sorted(Path(a.test_dir).glob('*.bmp'))},
        'development_hosts':{p.name:sha(p) for p in sorted(Path(a.development_dir).glob('*.bmp'))},
        'watermarks':{p.name:sha(p) for p in sorted(Path(a.watermark_dir).glob('*.png'))}}
    if ledger.get('config_sha256')!=sha(a.config):raise RuntimeError('Resume config changed: use a fresh output directory')
    steps=[]
    python=sys.executable
    if not a.video_only:
        image=output/'01_image_robustness'
        common=image/'common_45_attack'
        command=[python,ROOT/'scripts/run_multiwatermark_metrics.py','--host-dir',a.test_dir,
            '--watermark-dir',a.watermark_dir,'--proposal-config',a.config,
            '--out-dir',common,'--attack-preset','smoke' if a.quick else 'reviewer46']
        if a.quick:command+=['--host-limit','1']
        steps.append(('image_45_attack',command,[common/'all_watermarks_all_methods_all_metrics.csv']))
        native=image/'paper_native'
        command=[python,ROOT/'scripts/run_paper_native_benchmarks.py','--data-dir',a.test_dir,'--out',native,'--seed',a.seed]
        if a.quick: command+=['--quick']
        # Native track output varies per method; require meaningful stage directory files in a separate audit.
        steps.append(('image_paper_native',command,[native/'paper_native_protocol.json']))
        matched=image/'matched_psnr'
        command=[python,ROOT/'scripts/run_multiwatermark_fair_benchmark.py',
                 '--host-dir',a.test_dir,'--watermark-dir',a.watermark_dir,
                 '--calibration-host-dir',a.development_dir,'--proposal-config',a.config,
                 '--out-dir',matched,'--attack-preset','smoke' if a.quick else 'reviewer46',
                 '--methods',','.join(tuple(m for m in ROBUSTNESS_BASELINES if m!='dct_spread_spectrum_cox1997')+('proposal',))]
        if a.quick:command+=['--host-limit','1','--watermark-limit','1','--timing-repeats','1',
                              '--timing-warmups','0','--calibration-iterations','2']
        steps.append(('image_matched_psnr',command,[matched/'matched_psnr_per_attack.csv',matched/'matched_psnr_protocol.json']))
        stats=image/'paired_statistics'
        steps.append(('image_statistics',[python,ROOT/'scripts/reviewer_statistics.py',
            '--input',common/'all_watermarks_all_methods_all_metrics.csv','--out-dir',stats],
            [stats/'paired_proposal_effects.csv',stats/'method_host_cluster_ci.csv']))
        integer=output/'03_mathematical_validation'/'integer'
        steps.append(('integer_counter_audit',[python,ROOT/'scripts/export_final_integer_audit.py',
            '--input',common/'all_watermarks_all_methods_all_metrics.csv','--out-dir',integer],
            [integer/'integer_clean_embedding_summary.json']))
        cross=output/'03_mathematical_validation'/'orthogonality'
        command=[python,ROOT/'scripts/reviewer_cross_interference.py','--host-dir',a.test_dir,'--out-dir',cross]
        if a.quick:command+=['--max-hosts','1','--blocks-per-host','2']
        steps.append(('orthogonality_interference',command,[cross/'cross_interference_summary.csv']))
    if not a.image_only:
        if not a.video_corpus or not a.video_corpus.is_file():
            raise FileNotFoundError('Verified real-video --video-corpus required')
        video=output/'02_real_video'/'tempo_vs_dew'
        command=[python,ROOT/'scripts/run_tempo_dew_video_suite.py',
            '--corpus',a.video_corpus,'--watermark-dir',a.watermark_dir,
            '--config',a.config,'--out-dir',video,'--frames',a.frames,
            '--protocol','development_check' if a.quick else 'full']
        if a.quick:command+=['--no-native-hd','--only-matched','--channels','clean','--fps','1000']
        steps.append(('real_video_tempo_dew',command,[video/'video_comparison_summary.csv',video/'video_comparison_manifest.json']))
    ok=True
    for label,command,expected in steps:
        result=invoke(label,command,expected,output,ledger,resume=a.resume,env=env)
        if result and label=='image_45_attack':
            records=csv_rows(expected[0]); methods={r.get('method_id') for r in records}
            valid=methods.issuperset({*ROBUSTNESS_BASELINES,'proposal'})
            if not valid:
                ledger['stages'][label]['status']='failed'
                ledger['stages'][label]['reason']='Not all five distinct paper-derived image methods produced observations'
                result=False
        if result and label=='real_video_tempo_dew':
            records=csv_rows(expected[0]); pairs={(r['method_id'],r['protocol_track']) for r in records}
            valid={('proposal','matched_payload_256'),('dew_langelaar2001','matched_payload_256')}.issubset(pairs)
            if not a.quick:valid=valid and ('proposal','native_tempo_4096') in pairs
            if not valid:
                ledger['stages'][label]['status']='failed'
                ledger['stages'][label]['reason']='TEMPO/DEW matched comparison incomplete'
                result=False
        ok &= result
        (output/'experiment_ledger.json').write_text(json.dumps(ledger,indent=2))
        if not result and not a.continue_on_error:break
    ledger['matched_psnr_excluded_baselines']={'dct_spread_spectrum_cox1997':'No supported PSNR-step calibration; remains in full robustness and paper-native tracks'}
    ledger['overall_status']='complete' if ok and all(s['status']=='complete' for s in ledger['stages'].values()) and len(ledger['stages'])>=len(steps) else 'partial'
    (output/'experiment_ledger.json').write_text(json.dumps(ledger,indent=2))
    print('FINAL REVIEWER PROTOCOL:',ledger['overall_status'],'OUTPUT:',output,flush=True)
    return 0 if ledger['overall_status']=='complete' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--test-dir',type=Path,required=True)
    p.add_argument('--development-dir',type=Path,required=True)
    p.add_argument('--watermark-dir',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/manuscript_frozen.json')
    p.add_argument('--video-corpus',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,default=300)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--quick',action='store_true')
    p.add_argument('--image-only',action='store_true')
    p.add_argument('--video-only',action='store_true')
    p.add_argument('--continue-on-error',action='store_true')
    args=p.parse_args()
    if args.video_only and args.image_only:p.error('Choose at most one track-only option')
    sys.exit(main(args))
