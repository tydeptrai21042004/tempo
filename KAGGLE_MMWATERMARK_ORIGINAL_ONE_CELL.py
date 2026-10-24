# Paste this entire file as ONE Kaggle cell after pushing the patch files to GitHub.
# Settings: CPU / Accelerator None; Internet ON.
# Reuses a previous genuine held-out split/config when available; otherwise
# checks the repository's byte-verified frozen 2026-10-21 split and parameters.
# To force a NEW full TEMPO study (including real video), set environment
# variable TEMPO_MMW_RUN_NATIVE_FULL=1 before executing this cell.
# No fabricated data, no mock watermarking models, no local substitution.

import os, sys, subprocess, shutil, json, time, zipfile, signal, hashlib
from pathlib import Path
from datetime import datetime, timezone

GITHUB = 'https://github.com/tydeptrai21042004/tempo.git'
WORK = Path('/kaggle/working')
assert WORK.is_dir(), 'Run this cell on Kaggle.'
RUN = WORK / ('TEMPO_MMW_ORIGINAL_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
REPO = RUN / 'tempo'
OUT = RUN / 'extra' / 'external_official_mmwatermark'
META = RUN / 'run_metadata'
META.mkdir(parents=True, exist_ok=True)
ZIP = WORK / 'TEMPO_ORIGINAL_MMWATERMARK_RESULTS.zip'
RUN_MAIN_IF_NO_PREVIOUS_RESULTS = os.environ.get("TEMPO_MMW_RUN_NATIVE_FULL", "0") == "1"
MAX_HOURS = 11.4
DEADLINE = time.monotonic() + MAX_HOURS * 3600
status = {'status':'STARTING', 'native_pipeline_rerun':False, 'source_data_type':None}

def run_stage(label, cmd, *, cwd=None, env=None, reserve=420):
    remaining = DEADLINE - time.monotonic() - reserve
    if remaining < 60:
        raise TimeoutError('Kaggle time budget insufficient for ' + label)
    print('\n==', label, '==\n', ' '.join(map(str,cmd)), flush=True)
    with (META/(label+'.log')).open('w') as f:
        p = subprocess.Popen(list(map(str,cmd)), cwd=cwd, env=env,
                             stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = p.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            try: p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            raise TimeoutError(label + ' exceeded the deadline')
    print(label,'exit:',rc,flush=True)
    if rc:
        lines = (META/(label+'.log')).read_text(errors='replace').splitlines()
        print('\n'.join(lines[-40:]),flush=True)
        raise RuntimeError(label + ' failed; inspect the archived log')


def eligible(root):
    """Only reuse a complete, correctly hashed 3/9 held-out native dataset."""
    report=root/'reports'
    config=report/'configs/proposal_selected_all_watermarks.json'
    split=report/'holdout/split_manifest.json'
    marks=root/'selected_data/watermark'
    try:
        if not (config.is_file() and split.is_file() and marks.is_dir()):
            return False
        parameters=json.loads(config.read_text())
        if not isinstance(parameters,dict) or not isinstance(parameters.get('parameters'),dict):
            return False
        entries=json.loads(split.read_text())
        if len(entries.get('development',[])) != 3 or len(entries.get('test',[])) != 9:
            return False
        for group in ('development','test'):
            names=set()
            for row in entries[group]:
                name=row['name']
                file=report/'holdout'/group/name
                if name in names or any(x in name.lower() for x in ('lena','lenna')):
                    return False
                names.add(name)
                if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=row['sha256']:
                    return False
        if {p.name for p in marks.glob('*.png')} != {
                'watermark_001_Cc.logo.circle.png','watermark_002_wm.png'}:
            return False
        return True
    except (ValueError, KeyError, TypeError, OSError):
        return False

try:
    run_stage('00_clone_tempo', ['git','clone','--depth','1','--branch','main',GITHUB,REPO],cwd=WORK)
    status['tempo_git_commit'] = subprocess.check_output(
        ['git','-C',str(REPO),'rev-parse','HEAD'], text=True).strip()
    required = ['scripts/run_mmwatermark_in_official_repo.py',
                'scripts/run_official_mmwatermark_tempo.py',
                'scripts/prepare_mmwatermark_inputs.py',
                'scripts/baseline_preflight.py',
                'scripts/kaggle_pipeline.sh']
    missing=[x for x in required if not (REPO/x).is_file()]
    if missing:
        raise RuntimeError('Push the patch to GitHub main first. Missing: '+repr(missing))
    env=os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES':'-1','PYTHONPATH':str(REPO/'src')+os.pathsep+str(REPO),
                'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
                'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1',
                'TEMPO_OFFICIAL_MMWATERMARK':'0','KAGGLE_RESUME_FROM_VIDEO':'0',
                'KAGGLE_PYTHON':sys.executable,
                'IMAGE_ATTACK_PRESET':'unified','HP_ATTACK_PRESET':'standard',
                'HOST_LIMIT':'0','ABLATION_HOST_LIMIT':'0','HP_HOSTS':'3',
                'WM_SIZE':'64','SEED':'2026'})
    run_stage('01_setup',['bash',REPO/'scripts/kaggle_setup.sh'],cwd=REPO,env=env)
    selected_python_file = REPO/'.kaggle_python'
    selected_python = selected_python_file.read_text().strip() if selected_python_file.is_file() else sys.executable
    status['python_interpreter'] = selected_python
    all_results=sorted((x for x in WORK.glob('TEMPO_FRESH_*') if eligible(x)),
                       key=lambda x:x.stat().st_mtime, reverse=True)
    if all_results:
        DATA=all_results[0]
        status['source_data_type']='previous_fresh_pipeline_outputs'
        print('Reusing real frozen proposal config and held-out images:',DATA)
    else:
        DATA=RUN
        if RUN_MAIN_IF_NO_PREVIOUS_RESULTS:
            status['native_pipeline_rerun']=True
            status['source_data_type']='newly_optimized_full_native_pipeline'
            env['KAGGLE_REPORT_ROOT']=str(DATA/'reports')
            env['KAGGLE_SELECTED_DATA_DIR']=str(DATA/'selected_data')
            env['KAGGLE_MASTER_ZIP']=str(DATA/'TEMPO_PIPELINE_RESULTS.zip')
            run_stage('02_full_native', ['bash', REPO/'scripts/kaggle_pipeline.sh','full'],
                      cwd=REPO,env=env, reserve=1600)
        else:
            # This is the MMWatermark-only workflow. No needless 2.6-hour
            # native video/baseline run, and no tuning on final held-out hosts.
            status['source_data_type']='repository_frozen_20261021_not_newly_tuned'
            run_stage('02_prepare_verified_inputs', [selected_python,
                      REPO/'scripts/prepare_mmwatermark_inputs.py',
                      '--repo',REPO,'--run',RUN],cwd=REPO,env=env)
        if not eligible(DATA):
            raise RuntimeError('Inputs do not have a verified held-out split and frozen proposal')
    status['source_data_root']=str(DATA)
    status['config_file_sha256'] = __import__(
        'hashlib').sha256((DATA/'reports/configs/proposal_selected_all_watermarks.json').read_bytes()).hexdigest()
    # Check executable baseline implementations separately. These are NOT
    # claimed to be full copies of the original authors' end-to-end pipelines.
    run_stage('03_baseline_preflight', [selected_python,REPO/'scripts/baseline_preflight.py',
              '--host-dir',DATA/'reports/holdout/test',
              '--watermark-dir',DATA/'selected_data/watermark',
              '--out',RUN/'extra/baseline_preflight.json'],cwd=REPO,env=env)
    run_stage('04_original_mmwatermark', [
        selected_python, REPO/'scripts/run_mmwatermark_in_official_repo.py',
        '--official-repo',RUN/'MMWatermark-Robustness',
        '--companion-repo',RUN/'MM_Robustness',
        '--host-dir',DATA/'reports/holdout/test',
        '--watermark-dir',DATA/'selected_data/watermark',
        '--config',DATA/'reports/configs/proposal_selected_all_watermarks.json',
        '--out-dir',OUT, '--severities',1,2,3,4,5,
    ],cwd=REPO,env=env,reserve=420)
    summary=json.loads((OUT/'official_mmwatermark_manifest.json').read_text())
    status['full_coverage']=summary['full_coverage']
    status['completed_rows']=summary['completed_rows']
    status['expected_rows']=summary['expected_full_requested_rows']
    if not summary['full_coverage']:
        raise RuntimeError('MMWatermark manifest reports partial or failed coverage')
    status['status']='COMPLETED_FULL_OFFICIAL'
except Exception as exc:
    status.update(status='FAILED_OR_INCOMPLETE', error=repr(exc))
    print('Run did not fully complete:',repr(exc),flush=True)
finally:
    (META/'status.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf8')
    # Original Git checkouts are intentionally excluded: the provenance file
    # records their commits; preserve all experimental outputs and error logs.
    zip_tmp=ZIP.with_suffix('.zip.part')
    with zipfile.ZipFile(zip_tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for directory in (OUT,META,RUN/'extra'):
            if directory.is_dir():
                for file in directory.rglob('*'):
                    if file.is_file() and not file.name.endswith(('.part','.tmp')):
                        name=str(file.relative_to(RUN))
                        if name not in z.namelist(): z.write(file,arcname=name)
        if 'DATA' in globals() and eligible(DATA):
            for file in (DATA/'reports/configs/proposal_selected_all_watermarks.json',
                         DATA/'reports/holdout/split_manifest.json'):
                z.write(file,arcname='frozen_inputs/'+file.name)
    zip_tmp.replace(ZIP)
    print('Final ZIP:',ZIP, 'size MB:',round(ZIP.stat().st_size/1e6,2))
    print(json.dumps(status,indent=2))
    if status['status']!='COMPLETED_FULL_OFFICIAL':
        print('Inspect 04_original_mmwatermark.log, preflight JSON and status.json in ZIP.')
