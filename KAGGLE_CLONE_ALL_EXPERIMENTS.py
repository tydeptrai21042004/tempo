# TEMPO: paste this WHOLE file into ONE Kaggle Python cell.
# Accelerator=None (CPU); Internet=ON. Upload the PATCH to GitHub *before* running.
# This clones GitHub and runs measured experiments; it does NOT upload an old ZIP.
import os, sys, json, time, signal, hashlib, zipfile, subprocess, traceback
from pathlib import Path
from datetime import datetime, timezone

REPO_URL = 'https://github.com/tydeptrai21042004/tempo.git'
GIT_REF = 'main'                  # branch containing your manually uploaded patch
PROFILE = 'all'                  # all | internal | official_kodak | smoke
RUN_PAPER_NATIVE = True         # additional native-paper protocols, separate evidence
RUN_INTERNAL_KODAK_109 = False  # per your request: Kodak uses official MMWatermark only
RUN_INDEPENDENT_HAAR_QIM = False # NOT a published paper reproduction
TIME_LIMIT_HOURS = 11.3
CHECKPOINT_EVERY_MINUTES = 60

W = Path('/kaggle/working'); assert W.is_dir(), 'Use a Kaggle notebook (CPU, Internet ON)'
RID = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
ROOT = W / ('TEMPO_'+RID); ROOT.mkdir(parents=True,exist_ok=True)
REPO = ROOT/'tempo'; REPORT = ROOT/'reports'; REPORT.mkdir()
META = ROOT/'run_metadata'; META.mkdir()
SELECTED = ROOT/'selected_data'; KODAK = ROOT/'kodak_source'; KODAK_BMP = ROOT/'kodak_bmp'
OFFICIAL = ROOT/'MMWatermark-upstream'; COMPANION = ROOT/'MM-Robustness-upstream'
FINAL = W / ('TEMPO_EVIDENCE_'+RID+'.zip')
DEADLINE = time.monotonic() + TIME_LIMIT_HOURS*3600
state = {'run_id':RID,'repo_url':REPO_URL,'ref':GIT_REF,'profile':PROFILE,
         'scientific_acceptance_gate':False,'stages':{},'complete':False}
last_checkpoint = [0.0]

# Package real artifacts even after an experiment fails or the time budget expires.
def package(force=False):
    if not force and time.monotonic()-last_checkpoint[0] < CHECKPOINT_EVERY_MINUTES*60:return
    (META/'stage_status.json').write_text(json.dumps(state,indent=2,default=str))
    temp=Path(str(FINAL)+'.partial')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for directory,rootname in ((REPORT,'reports'),(META,'run_metadata')):
            for file in sorted(directory.rglob('*')):
                if not file.is_file():continue
                if file.suffix.lower() in ('.mp4','.avi','.mkv','.mov','.zip','.lock','.partial'):continue
                try:z.write(file, str(Path(rootname)/file.relative_to(directory)))
                except OSError:continue
    temp.replace(FINAL)
    last_checkpoint[0]=time.monotonic()
    print('CHECKPOINT:',FINAL,'bytes:',FINAL.stat().st_size,flush=True)

def run(label,args,env=None):
    args=list(map(str,args)); start=time.monotonic()
    log=META/(label+'.log')
    print('\n==',label,'==\n',' '.join(args),flush=True)
    rec={'status':'running','command':args,'log':str(log)};state['stages'][label]=rec
    try:
        with log.open('w',encoding='utf8') as f:
            p=subprocess.Popen(args,cwd=str(REPO if REPO.exists() else W),env=env,
                               stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            while p.poll() is None:
                if time.monotonic() > DEADLINE-180:
                    try:os.killpg(p.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    try:p.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(p.pid,signal.SIGKILL);p.wait()
                    break
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:package()
        rc=p.returncode
        rec.update(status='complete' if rc==0 else 'failed',exit_code=rc)
        if rc:
            print('FAILED:',label,'\n'+'\n'.join(log.read_text(errors='replace').splitlines()[-15:]),flush=True)
    except Exception as e:
        rec.update(status='failed',error=str(e));print('FAILED:',label,str(e),flush=True)
    rec['elapsed_s']=round(time.monotonic()-start,2)
    package(force=True)
    return rec['status']=='complete'

env=os.environ.copy()
env.update({'CUDA_VISIBLE_DEVICES':'-1','PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1',
            'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
            'NUMEXPR_NUM_THREADS':'1','NUMBA_NUM_THREADS':'4','JILP_NUM_THREADS':'4',
            'PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1','SEED':'2026','WM_SIZE':'64',
            'TEMPO_CONFIG_MODE':'manuscript','TEMPO_MATH_REFINEMENT':'0',
            'TEMPO_OFFICIAL_MMWATERMARK':'0',
            'TEMPO_RUN_PAPER_NATIVE_SEPARATELY':'0',
            'TEMPO_RUN_INDEPENDENT_HAAR_QIM_CONTROL':'1' if RUN_INDEPENDENT_HAAR_QIM else '0',
            'KAGGLE_REPORT_ROOT':str(REPORT),'KAGGLE_SELECTED_DATA_DIR':str(SELECTED),
            'KAGGLE_MASTER_ZIP':str(ROOT/'main_pipeline_results.zip')})
try:
    if not run('00_clone',['git','clone','--depth','1','--branch',GIT_REF,REPO_URL,REPO],env):
        raise RuntimeError('Cannot clone the selected GitHub branch')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    state['git_commit']=commit
    required=['README.md','configs/manuscript_frozen.json','configs/paper_2024_reference.json',
              'scripts/run_recent_svd_paper_structure.py','scripts/generate_manuscript_figures.py',
              'scripts/kaggle_pipeline.sh','scripts/run_mmwatermark_in_official_repo.py']
    missing=[x for x in required if not (REPO/x).is_file()]
    if missing:raise RuntimeError('GitHub branch is missing PATCH files: '+repr(missing))
    env['PYTHONPATH']=str(REPO/'src')+os.pathsep+str(REPO)
    print('GIT COMMIT:',commit,flush=True)
    if not run('01_setup',['bash','scripts/kaggle_setup.sh'],env):
        raise RuntimeError('Python dependencies/setup failed')
    if PROFILE in ('all','internal','smoke'):
        # The repo pipeline runs BOTH logos, held-out image attacks, matched PSNR,
        # DOI 2024 informed reproduction, reviewer statistics/figures and video.
        run('02_main_'+('quick' if PROFILE=='smoke' else 'full'),
            ['bash','scripts/kaggle_pipeline.sh','quick' if PROFILE=='smoke' else 'full'],env)
    if RUN_PAPER_NATIVE and PROFILE in ('all','internal'):
        test=REPORT/'holdout'/'test'
        if test.is_dir():
            run('03_paper_native_secondary',[sys.executable,'scripts/run_paper_native_benchmarks.py',
                 '--data-dir',test,'--out',REPORT/'paper_native_secondary'],env)
        else:state['stages']['03_paper_native_secondary']={'status':'not_run','reason':'held-out split unavailable'}
    if PROFILE in ('all','official_kodak'):
        wm=SELECTED/'watermark'; cfg=REPORT/'configs'/'proposal_selected_all_watermarks.json'
        if not wm.is_dir() or not cfg.is_file():
            # An independent Kodak-only rerun still needs deterministic source/split setup.
            if run('04_select_inputs',[sys.executable,'scripts/select_kaggle_dataset.py',
                    '--source-data','data/user_dataset','--output',SELECTED,'--manifest-dir',REPORT],env):
                cfg=REPO/'configs'/'manuscript_frozen.json'
            else:raise RuntimeError('Cannot prepare Kodak input watermark dataset')
        if run('05_download_kodak',[sys.executable,'scripts/download_full_kodak.py','--out-dir',KODAK],env):
            if run('06_prepare_kodak',[sys.executable,'scripts/prepare_kodak_official_512.py',
                    '--dataset-dir',KODAK,'--out-dir',KODAK_BMP,'--offline'],env):
                official_out=REPORT/'external_official_mmwatermark_kodak24'
                if run('07_official_mmwatermark_kodak24',[sys.executable,
                    'scripts/run_mmwatermark_in_official_repo.py',
                    '--official-repo',OFFICIAL,'--companion-repo',COMPANION,
                    '--host-dir',KODAK_BMP,'--watermark-dir',wm,'--config',cfg,
                    '--out-dir',official_out,'--severities','1','2','3','4','5','--allow-partial'],env):
                    raw=official_out/'official_mmwatermark_raw.csv';man=official_out/'official_mmwatermark_manifest.json'
                    if raw.is_file() and man.is_file():
                        run('08_official_summary',[sys.executable,'scripts/summarize_official_kodak.py',
                             '--manifest',man,'--raw',raw,'--out-dir',official_out/'summaries'],env)
    if RUN_INTERNAL_KODAK_109 and KODAK.is_dir():
        cfg=REPORT/'configs'/'proposal_selected_all_watermarks.json'
        if not cfg.is_file():cfg=REPO/'configs'/'manuscript_frozen.json'
        run('09_separate_kodak109',[sys.executable,'scripts/run_full_kodak_external.py',
            '--dataset-dir',KODAK,'--watermark-dir',SELECTED/'watermark','--config',cfg,
            '--out-dir',REPORT/'external_kodak109','--attack-profile','unified109','--offline'],env)
    # PDF figures should only reflect actual generated experimental evidence.
    cfg=REPORT/'configs'/'proposal_selected_all_watermarks.json'
    if cfg.is_file() and (SELECTED/'host').is_dir():
        run('10_result_pdf_figures',[sys.executable,'scripts/generate_manuscript_figures.py',
             '--results-root',REPORT,'--host-dir',SELECTED/'host',
             '--watermark-dir',SELECTED/'watermark','--config',cfg,
             '--out-dir',REPORT/'manuscript_figures'],env)
    run('11_completeness_report',[sys.executable,'scripts/reviewer_completeness.py',
         '--results-dir',REPORT,'--mode','quick' if PROFILE=='smoke' else 'full'],env)
finally:
    state['finished_utc']=datetime.now(timezone.utc).isoformat()
    state['complete']=all(v.get('status')=='complete' for v in state['stages'].values())
    package(force=True)
    print('\nPER-STAGE STATUS:',json.dumps({k:v.get('status') for k,v in state['stages'].items()}),flush=True)
    print('DOWNLOAD RESULTS:',FINAL,flush=True)
