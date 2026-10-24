# Paste this entire file as ONE Kaggle cell after pushing the patch files to GitHub.
# Settings: CPU / Accelerator None; Internet ON.
# Finds previous complete split/config in /kaggle/working, otherwise starts a
# genuine full native TEMPO pipeline before the original upstream benchmark.
# No fabricated data, no mock watermarking models, no local substitution.

import os, sys, subprocess, shutil, json, time, zipfile, signal
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
RUN_MAIN_IF_NO_PREVIOUS_RESULTS = True
MAX_HOURS = 11.4
DEADLINE = time.monotonic() + MAX_HOURS * 3600
status = {'status':'STARTING', 'native_pipeline_rerun':False}

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
    report = root / 'reports'
    return (report/'configs/proposal_selected_all_watermarks.json').is_file() and \
           (report/'holdout/test').is_dir() and \
           (root/'selected_data/watermark').is_dir() and \
           (report/'holdout/split_manifest.json').is_file()

try:
    run_stage('00_clone_tempo', ['git','clone','--depth','1','--branch','main',GITHUB,REPO],cwd=WORK)
    status['tempo_git_commit'] = subprocess.check_output(
        ['git','-C',str(REPO),'rev-parse','HEAD'], text=True).strip()
    required = ['scripts/run_mmwatermark_in_official_repo.py',
                'scripts/run_official_mmwatermark_tempo.py',
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
        print('Reusing real frozen proposal config and held-out images:',DATA)
    else:
        if not RUN_MAIN_IF_NO_PREVIOUS_RESULTS:
            raise RuntimeError('No previous split/config. Enable full native rerun or attach previous results.')
        DATA=RUN
        status['native_pipeline_rerun']=True
        env['KAGGLE_REPORT_ROOT']=str(DATA/'reports')
        env['KAGGLE_SELECTED_DATA_DIR']=str(DATA/'selected_data')
        env['KAGGLE_MASTER_ZIP']=str(DATA/'TEMPO_PIPELINE_RESULTS.zip')
        run_stage('02_full_native', ['bash', REPO/'scripts/kaggle_pipeline.sh','full'],
                  cwd=REPO,env=env, reserve=1600)
        if not eligible(DATA):
            raise RuntimeError('Full TEMPO run did not produce a valid frozen split/config')
    status['source_data_root']=str(DATA)
    run_stage('03_original_mmwatermark', [
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
        for directory in (OUT,META):
            if directory.is_dir():
                for file in directory.rglob('*'):
                    if file.is_file() and not file.name.endswith(('.part','.tmp')):
                        z.write(file,arcname=str(file.relative_to(RUN)))
        if 'DATA' in globals() and eligible(DATA):
            for file in (DATA/'reports/configs/proposal_selected_all_watermarks.json',
                         DATA/'reports/holdout/split_manifest.json'):
                z.write(file,arcname='frozen_inputs/'+file.name)
    zip_tmp.replace(ZIP)
    print('Final ZIP:',ZIP, 'size MB:',round(ZIP.stat().st_size/1e6,2))
    print(json.dumps(status,indent=2))
    if status['status']!='COMPLETED_FULL_OFFICIAL':
        print('Inspect 03_original_mmwatermark.log, preflight JSON and status.json in ZIP.')
