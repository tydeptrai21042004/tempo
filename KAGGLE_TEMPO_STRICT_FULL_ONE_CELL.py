# STRICT scientific validation: clone CURRENT GitHub main, run a completely fresh TEMPO
# native benchmark, extra reviewer experiments, FPS variants, and the original
# MMWatermark-Robustness code with the TEMPO proposal. Internet must be ON.
# CPU-only; genuine image/video data; never substitutes synthetic observations.
import os, sys, time, json, signal, hashlib, zipfile, threading, subprocess, traceback
from pathlib import Path
from datetime import datetime, timezone

# -------- Options --------
GITHUB = 'https://github.com/tydeptrai21042004/tempo.git'
MAX_HOURS = 11.40
CHECKPOINT_MINUTES = 30
RUN_REVIEWER_EXPANSION = True
RUN_FPS_VARIANTS = True
RUN_ORIGINAL_MMWATERMARK = True
RUN_BASELINE_PREFLIGHT = True
# Only supply these when real independently obtained evidence exists.
EXTERNAL_HOST_DIR = None
LEARNED_RESULTS_CSV = None
LEARNED_PROVENANCE_JSON = None
EXTERNAL_BASELINES_CSV = None
INCLUDE_VIDEO_FILES_IN_ZIP = False  # Original videos stay in the run directory.

WORK = Path('/kaggle/working')
if not WORK.is_dir():
    raise RuntimeError('Run this cell in Kaggle with Internet ON and Accelerator None (CPU).')
RID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
ROOT = WORK / ('TEMPO_COMPLETE_FRESH_' + RID)
REPO = ROOT / 'tempo'
REPORT = ROOT / 'reports'
DATA = ROOT / 'selected_data'
EXTRA = ROOT / 'extra'
META = ROOT / 'run_metadata'
for p in (ROOT, EXTRA, META):
    p.mkdir(parents=True, exist_ok=True)
FINAL_ZIP = WORK / ('TEMPO_ALL_RESULTS_FRESH_' + RID + '.zip')
CHECKPOINT_ZIP = WORK / ('TEMPO_ALL_RESULTS_CHECKPOINT_' + RID + '.zip')
NATIVE_ZIP = ROOT / 'TEMPO_PIPELINE_RESULTS.zip'
START = time.monotonic()
DEADLINE = START + MAX_HOURS * 3600
stages = {}
commit = None
zip_lock = threading.RLock()
checkpoint_stop = threading.Event()


def status_write():
    payload = dict(run_id=RID, git_url=GITHUB, git_commit=commit,
                   run_mode='FRESH_NO_RESUME', seed=2026,
                   attack_preset='unified_109', stages=stages,
                   status=stages,
                   generated_utc=datetime.now(timezone.utc).isoformat())
    with zip_lock:
        tmp = META / 'status.json.tmp'
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
        tmp.replace(META / 'status.json')


def archive(target):
    """Evidence snapshot. Archive is atomic; excludes raw videos and nested ZIPs."""
    with zip_lock:
        temp = Path(str(target) + '.part')
        try:
            with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=1, allowZip64=True) as z:
                for folder, prefix in ((REPORT, 'reports'), (DATA, 'selected_data'),
                                       (EXTRA, 'extra'), (META, 'run_metadata')):
                    if not folder.exists():
                        continue
                    for f in sorted(folder.rglob('*')):
                        if not f.is_file() or f.is_symlink():
                            continue
                        if f.suffix.lower() in ('.zip', '.part', '.tmp', '.lock'):
                            continue
                        if not INCLUDE_VIDEO_FILES_IN_ZIP and f.suffix.lower() in ('.mp4', '.avi', '.mkv', '.mov'):
                            continue
                        try:
                            z.write(f, str(Path(prefix) / f.relative_to(folder)))
                        except (OSError, FileNotFoundError):
                            # Some experiment files can be atomically replaced during snapshots.
                            pass
            temp.replace(target)
            print('EVIDENCE ZIP:', target, '%.2f MB' % (target.stat().st_size / 1e6), flush=True)
        except Exception as exc:
            print('Archive warning:', repr(exc), flush=True)
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass


def checkpoint_worker():
    while not checkpoint_stop.wait(CHECKPOINT_MINUTES * 60):
        status_write()
        archive(CHECKPOINT_ZIP)


def skipped(label, reason):
    stages[label] = {'status': 'NOT_RUN', 'reason': reason}
    print(label, 'NOT_RUN:', reason, flush=True)
    status_write()


def launch(label, command, env=None, reserve=600):
    """Launch isolated subprocess with deadline, log and actual exit code."""
    budget = DEADLINE - time.monotonic() - reserve
    if budget < 60:
        skipped(label, 'KAGGLE_TIME_BUDGET')
        return False
    args = [str(x) for x in command]
    log = META / (label + '.log')
    stages[label] = {'status': 'RUNNING', 'command': args, 'log': str(log)}
    status_write()
    print('\n' + '='*68 + '\n' + label + '\n' + ' '.join(args), flush=True)
    begin = time.monotonic()
    rc = -101
    try:
        with log.open('w', encoding='utf-8') as output:
            proc = subprocess.Popen(args, cwd=REPO if REPO.exists() else WORK,
                                    env=env, stdout=output, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            last = ''
            while True:
                remaining = DEADLINE - time.monotonic() - reserve
                try:
                    rc = proc.wait(timeout=min(60, max(1, remaining)))
                    break
                except subprocess.TimeoutExpired:
                    try:
                        tail = '\n'.join(log.read_text(errors='replace').splitlines()[-2:])
                        if tail and tail != last:
                            print('[' + label + ']', tail[-350:], flush=True)
                            last = tail
                    except OSError:
                        pass
                    if remaining <= 0:
                        try:
                            os.killpg(proc.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        try:
                            proc.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            try:
                                os.killpg(proc.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            proc.wait()
                        rc = -100
                        break
    except Exception as exc:
        log.write_text(log.read_text(errors='replace') + '\nWRAPPER ERROR: ' + repr(exc), encoding='utf8')
    outcome = ('COMPLETED' if rc == 0 else
               'STOPPED_TIME_LIMIT' if rc == -100 else 'FAILED_EXIT_' + str(rc))
    stages[label] = {'status': outcome, 'exit_code': rc,
                     'elapsed_seconds': round(time.monotonic()-begin,1),
                     'command': args, 'log': str(log)}
    status_write()
    print(label, outcome, flush=True)
    if rc != 0:
        print('\n'.join(log.read_text(errors='replace').splitlines()[-35:]), flush=True)
    archive(CHECKPOINT_ZIP)
    return rc == 0


def verify_fresh_inputs():
    manifest = REPORT / 'holdout/split_manifest.json'
    selected = REPORT / 'configs/proposal_selected_all_watermarks.json'
    mark_dir = DATA / 'watermark'
    if not (manifest.is_file() and selected.is_file() and mark_dir.is_dir()):
        raise RuntimeError('Full run did not create its selected config/split/watermarks')
    cfg = json.loads(selected.read_text())
    if not isinstance(cfg, dict) or not isinstance(cfg.get('parameters'), dict):
        raise RuntimeError('Selected proposal configuration is not a frozen parameter record')
    split = json.loads(manifest.read_text())
    if len(split.get('development',[])) != 3 or len(split.get('test',[])) != 9:
        raise RuntimeError('Expected exactly 3 development + 9 held-out hosts')
    seen = set()
    for part in ('development', 'test'):
        for row in split[part]:
            name = row['name']
            path = REPORT / 'holdout' / part / name
            if name in seen or any(t in name.lower() for t in ('lena','lenna')):
                raise RuntimeError('Duplicate or excluded host: ' + name)
            seen.add(name)
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise RuntimeError('Split image checksum mismatch: ' + name)
    expected = {'watermark_001_Cc.logo.circle.png', 'watermark_002_wm.png'}
    if {p.name for p in mark_dir.glob('*.png')} != expected:
        raise RuntimeError('Both 64x64 watermarks required, with their selected names')
    return selected, manifest


def actual_video():
    manifest = REPORT / 'video_sources/corpus.json'
    if manifest.exists():
        for row in json.loads(manifest.read_text()).get('videos',[]):
            p = Path(row.get('prepared_path') or '')
            if p.is_file() and p.suffix.lower() in ('.mp4', '.avi', '.mkv'):
                return p
    return None


threading.Thread(target=checkpoint_worker, daemon=True).start()
try:
    if not launch('00_clone_tempo', ['git','clone','--depth','1','--branch','main',GITHUB,REPO],reserve=1200):
        raise RuntimeError('GitHub clone failed')
    commit = subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    (META/'git_commit.txt').write_text(commit+'\n')
    print('FRESH CLONED COMMIT:',commit,flush=True)
    required = ['scripts/kaggle_pipeline.sh','scripts/kaggle_setup.sh',
                'scripts/run_mmwatermark_in_official_repo.py',
                'scripts/run_official_mmwatermark_tempo.py',
                'scripts/baseline_preflight.py', 'scripts/run_reviewer_expansion.py',
                'scripts/proposal_fps_ablation.py', 'scripts/audit_kaggle_run.py',
                'scripts/benchmark_frozen_720p.py', 'scripts/strict_scientific_gate.py']
    missing = [x for x in required if not (REPO/x).is_file()]
    if missing:
        raise RuntimeError('Latest GitHub main is missing required files: '+repr(missing))
    env = os.environ.copy()
    env.update({
        'KAGGLE_PYTHON': sys.executable, 'PYTHON': sys.executable,
        'PYTHONPATH': str(REPO/'src') + os.pathsep + str(REPO),
        'PYTHONDONTWRITEBYTECODE': '1','PYTHONUNBUFFERED':'1',
        'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1','PYTHONHASHSEED':'0',
        'CUDA_VISIBLE_DEVICES':'-1', 'OMP_NUM_THREADS':'1',
        'JILP_NUM_THREADS':'4',
        'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
        'NUMEXPR_NUM_THREADS':'1','NUMBA_NUM_THREADS':'4','OMP_DYNAMIC':'FALSE',
        'KAGGLE_REPORT_ROOT':str(REPORT),
        'KAGGLE_SELECTED_DATA_DIR':str(DATA),
        'KAGGLE_MASTER_ZIP':str(NATIVE_ZIP),
        'KAGGLE_RESUME_FROM_VIDEO':'0',
        'TEMPO_MATH_REFINEMENT':'0',
        'TEMPO_OFFICIAL_MMWATERMARK':'0',  # run official separately after full export
        'FAIL_ON_INCOMPLETE':'0',  # preserve missing-evidence ledger
        'IMAGE_ATTACK_PRESET':'unified','HP_ATTACK_PRESET':'standard',
        'HOST_LIMIT':'0','ABLATION_HOST_LIMIT':'0','HP_HOSTS':'3',
        'SEED':'2026','WM_SIZE':'64',
        'VIDEO_FRAMES':'300','STREAM_FRAMES':'300',
        'STREAM_RESOLUTIONS':'512x512,1280x720',
        'VIDEO_STREAM_CHANNELS':'clean,jpeg90,jpeg70,jpeg40,gaussian5,blur3,drop10',
    })
    if not launch('01_setup',['bash',REPO/'scripts/kaggle_setup.sh'],env,reserve=1100):
        raise RuntimeError('Kaggle dependency setup failed')
    pyfile = REPO / '.kaggle_python'
    py = pyfile.read_text().strip() if pyfile.exists() else sys.executable
    stages['python_interpreter'] = {'status':'INFO','value':py}
    status_write()
    # The native pipeline already runs its complete pytest suite and all
    # required paper-native, image, video, ablation and matched-PSNR baselines.
    native_ok = launch('02_full_native', ['bash',REPO/'scripts/kaggle_pipeline.sh','full'],
                       env,reserve=1800)
    if not native_ok:
        raise RuntimeError('Full native pipeline failed: inspect 02_full_native.log')
    selected, split = verify_fresh_inputs()
    (META/'frozen_config_sha256.txt').write_text(hashlib.sha256(selected.read_bytes()).hexdigest()+'\n')
    status_write()
    if RUN_BASELINE_PREFLIGHT:
        launch('03_baseline_preflight', [py,REPO/'scripts/baseline_preflight.py',
               '--host-dir',REPORT/'holdout/test', '--watermark-dir',DATA/'watermark',
               '--out', EXTRA/'baseline_preflight.json'],env,reserve=1200)
    else:
        skipped('03_baseline_preflight','DISABLED')

    if RUN_REVIEWER_EXPANSION:
        names=['paired','pareto','sync','integer','latency','attack']
        args=[py,REPO/'scripts/run_reviewer_expansion.py',
              '--output',EXTRA/'reviewer_expansion', '--only',','.join(names),
              '--host-dir',REPORT/'holdout/test',
              '--development-dir',REPORT/'holdout/development',
              '--watermark-dir',DATA/'watermark', '--config',selected,
              '--seed',2026,'--no-zip']
        video=actual_video()
        if video is not None:
            names.append('temporal')
            args[args.index('--only')+1]=','.join(names)
            args += ['--video',video]
        if EXTERNAL_HOST_DIR is not None and Path(EXTERNAL_HOST_DIR).is_dir():
            names.append('external')
            args[args.index('--only')+1]=','.join(names)
            args += ['--external-host-dir',EXTERNAL_HOST_DIR]
        if all(x is not None and Path(x).is_file() for x in (LEARNED_RESULTS_CSV,LEARNED_PROVENANCE_JSON)):
            names.append('learned')
            args[args.index('--only')+1]=','.join(names)
            args += ['--learned-csv',LEARNED_RESULTS_CSV,
                     '--learned-provenance',LEARNED_PROVENANCE_JSON]
        ok=launch('04_reviewer_expansion',args,env,reserve=1000)
        ledger=EXTRA/'reviewer_expansion/expansion_ledger.json'
        if ok and ledger.exists():
            x=json.loads(ledger.read_text())
            if x.get('status') != 'complete':
                stages['04_reviewer_expansion']['status']='PARTIAL_CHECK_EXPANSION_LEDGER'
                stages['04_reviewer_expansion']['incomplete_stages']=x.get('incomplete_stages',[])
                status_write()
    else:
        skipped('04_reviewer_expansion','DISABLED')

    if RUN_FPS_VARIANTS:
        launch('05_fps_variants',[py,REPO/'scripts/proposal_fps_ablation.py',
            '--config',selected,
            '--host-dir',REPO/'data/user_dataset/host',
            '--watermark-dir',REPO/'data/user_dataset/watermark',
            '--split','development','--split-manifest',split,
            '--group','all','--attack-preset','stress',
            '--attacks','clean,jpeg70,rotate1,translate4_4',
            '--warmups',1,'--repeats',3,
            '--out-dir',EXTRA/'fps_variants'],env,reserve=950)
    else:
        skipped('05_fps_variants','DISABLED')

    if RUN_ORIGINAL_MMWATERMARK:
        ok=launch('06_original_mmwatermark', [
            py,REPO/'scripts/run_mmwatermark_in_official_repo.py',
            '--official-repo',ROOT/'MMWatermark-Robustness',
            '--companion-repo',ROOT/'MM_Robustness',
            '--host-dir',REPORT/'holdout/test',
            '--watermark-dir',DATA/'watermark',
            '--config',selected,
            '--out-dir',EXTRA/'external_official_mmwatermark',
            '--severities',1,2,3,4,5,
            # No --allow-partial: full upstream coverage is REQUIRED.
        ],env,reserve=650)
        manifest=EXTRA/'external_official_mmwatermark/official_mmwatermark_manifest.json'
        if manifest.is_file():
            m=json.loads(manifest.read_text())
            record=stages['06_original_mmwatermark']
            record['full_coverage']=bool(m.get('full_coverage'))
            record['completed_rows']=m.get('completed_rows')
            record['expected_rows']=m.get('expected_full_requested_rows')
            if ok and not m.get('full_coverage'):
                record['status']='PARTIAL_OFFICIAL_COVERAGE'
            status_write()
        elif ok:
            stages['06_original_mmwatermark']['status']='FAILED_NO_MANIFEST'
            status_write()
    else:
        skipped('06_original_mmwatermark','DISABLED')

    # NEW: true 720p/JPEG70 measurement of the frozen image proposal.
    # Video samples are real and predecoded; no 45-dB step recalibration.
    if not RUN_ORIGINAL_MMWATERMARK:
        raise RuntimeError('Complete evidence requires genuine original MMWatermark')
    video_manifest=REPORT/'video_sources/corpus.json'
    selected_video=None
    if video_manifest.is_file():
        corpus=json.loads(video_manifest.read_text())
        for item in corpus.get('videos',[]):
            # Only the *original* verified native 720p download qualifies.
            # The prepared_path is letterboxed/downsampled to 512x512 and
            # MUST NEVER support a native 720p timing claim.
            path=Path(item.get('original_path') or '')
            if (item.get('is_real_source_video') is True
                and item.get('native_720p') is True and path.is_file()
                and path.suffix.lower() in ('.mp4','.avi','.mkv','.mov')
                and hashlib.sha256(path.read_bytes()).hexdigest()==item.get('original_sha256')):
                selected_video=path
                break
    if selected_video is None:
        stages['09_frozen_720p_fps']={'status':'FAILED_MISSING_REAL_VIDEO'}
        status_write()
    else:
        launch('09_frozen_720p_fps', [py,REPO/'scripts/benchmark_frozen_720p.py',
            '--video',selected_video, '--watermark-dir',DATA/'watermark',
            '--config',selected,'--out-dir',EXTRA/'frozen_720p_30fps',
            '--frames',300,'--fps',32],env,reserve=540)

    if EXTERNAL_BASELINES_CSV is not None and Path(EXTERNAL_BASELINES_CSV).is_file():
        launch('07_external_baseline_import', [py,REPO/'scripts/import_external_baseline_results.py',
            '--input',EXTERNAL_BASELINES_CSV,'--split-manifest',split,
            '--watermark-dir',DATA/'watermark','--out-dir',EXTRA/'validated_external_methods',
            '--payload-bits',4096],env,reserve=600)
    else:
        skipped('07_external_baseline_import','NO_GENUINE_EXTERNAL_RESULTS_CSV')

    if (REPO/'scripts/audit_kaggle_run.py').is_file():
        launch('08_audit', [py,REPO/'scripts/audit_kaggle_run.py',
            '--run-root',ROOT,'--out',META/'diagnostic.json',
            '--target-psnr',50,'--target-fps',30],env,reserve=420)
    strict_ok=launch('10_strict_scientific_gate',[
        py,REPO/'scripts/strict_scientific_gate.py',
        '--run-root',ROOT,'--out',META/'scientific_gate.json',
        '--target-fps',30,'--target-psnr',50],env,reserve=220)
    if not strict_ok:
        print('SCIENTIFIC TARGETS FAILED: inspect run_metadata/scientific_gate.json',flush=True)
        stages['SCIENTIFIC_FINAL_VERDICT']={'status':'FAILED_TARGETS','gate':str(META/'scientific_gate.json')}
    else:
        print('SCIENTIFIC TARGETS PASSED; see verifiable evidence ledger',flush=True)
        stages['SCIENTIFIC_FINAL_VERDICT']={'status':'PASS','gate':str(META/'scientific_gate.json')}
    status_write()
except BaseException as exc:
    print('UNEXPECTED / BLOCKING ERROR:',repr(exc),flush=True)
    (META/'wrapper_error.txt').write_text(traceback.format_exc(),encoding='utf8')
finally:
    checkpoint_stop.set()
    status_write()
    archive(FINAL_ZIP)
    print('\n'+'='*68+'\nFINAL RUN SUMMARY',flush=True)
    print('Git commit:',commit,flush=True)
    print('Run directory:',ROOT,flush=True)
    print('Evidence archive:',FINAL_ZIP,flush=True)
    print('Checkpoint archive:',CHECKPOINT_ZIP,flush=True)
    print('Native evidence ZIP:',NATIVE_ZIP if NATIVE_ZIP.is_file() else 'not produced',flush=True)
    print(json.dumps({k:v.get('status') for k,v in stages.items()},indent=2),flush=True)
    print('SCIENTIFIC VERDICT:',stages.get('SCIENTIFIC_FINAL_VERDICT',{}).get('status','NOT_VERIFIED'),flush=True)
    print('Check scientific_gate.json: zero failed scientific checks required for PASS.',flush=True)
