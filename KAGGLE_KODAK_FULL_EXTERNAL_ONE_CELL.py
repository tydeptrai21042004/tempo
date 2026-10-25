# TEMPO - Kaggle CPU, Internet ON, full Kodak dataset (24/24), full 4096-bit watermark.
# Upload the patch contents to GitHub main before running this cell.
# This cell itself requires NO external data upload and NO external CSV.
import os, subprocess, sys, zipfile, json
from pathlib import Path

WORK=Path('/kaggle/working')
assert WORK.is_dir(), 'Use Kaggle (CPU; Internet ON).'
REPO=WORK/'tempo_full_kodak_src'
OUT=WORK/'TEMPO_KODAK_FULL_EXTERNAL'
DATA=WORK/'kodak_full_native_24'
OFFICIAL_HOSTS=WORK/'kodak_official_512_crops_24'
OUT.mkdir(parents=True,exist_ok=True)

# 'basic' => all 24 x 2 x 3 = 144 native-resolution rows.
# 'unified109' => all 24 x 2 x 110 = 5280 native-resolution rows (can be slow).
PROFILE='basic'
# The official MMWatermark source benchmarks all 24 center crops, both watermarks,
# all upstream attacks/severities; 24 x 2 x 116 = 5568 trials if 23 attacks are
# available. Set true for that additional independent upstream suite.
RUN_OFFICIAL_FULL_24=True

os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['NUMEXPR_NUM_THREADS']='1'
os.environ['PYTHONHASHSEED']='0'

def run(args, env=None):
    print('RUN:', ' '.join(str(x) for x in args), flush=True)
    return subprocess.run([str(x) for x in args],check=True,env=env)

if not (REPO/'.git').is_dir():
    run(['git','clone','--depth','1','--branch','main',
         'https://github.com/tydeptrai21042004/tempo.git',REPO])
# Never silently run an old GitHub repository missing the full external bench.
required=['scripts/download_full_kodak.py','scripts/run_full_kodak_external.py',
          'scripts/prepare_kodak_official_512.py']
missing=[x for x in required if not (REPO/x).is_file()]
if missing:raise RuntimeError('Push the full Kodak patch to GitHub first: '+str(missing))
env={**os.environ,'PYTHONPATH':f'{REPO}/src:{REPO}'}
run(['bash',REPO/'scripts/kaggle_setup.sh'],env=env)

try:
    run([sys.executable,REPO/'scripts/run_full_kodak_external.py',
         '--dataset-dir',DATA,'--watermark-dir',REPO/'data/user_dataset/watermark',
         '--config',REPO/'configs/frozen_selected_20261021.json',
         '--out-dir',OUT/'kodak_native_proposal','--attack-profile',PROFILE],env=env)
    if RUN_OFFICIAL_FULL_24:
        run([sys.executable,REPO/'scripts/prepare_kodak_official_512.py',
             '--dataset-dir',DATA,'--out-dir',OFFICIAL_HOSTS,'--offline'],env=env)
        run([sys.executable,REPO/'scripts/run_mmwatermark_in_official_repo.py',
             '--official-repo',WORK/'TEMPO_KODAK_MMWatermark-Robustness',
             '--companion-repo',WORK/'TEMPO_KODAK_MM_Robustness',
             '--host-dir',OFFICIAL_HOSTS,
             '--watermark-dir',REPO/'data/user_dataset/watermark',
             '--config',REPO/'configs/frozen_selected_20261021.json',
             '--out-dir',OUT/'official_mmwatermark_full_24',
             '--severities','1','2','3','4','5'],env=env)
finally:
    # Preserve partial trials and diagnostics even if a stage fails or Kaggle
    # times out. Dataset images are intentionally not copied into the ZIP.
    archive=WORK/'TEMPO_KODAK_FULL_EXTERNAL_RESULTS.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(OUT))
        for path in (DATA/'kodak_full_manifest.json',OFFICIAL_HOSTS/'kodak_official_crops_manifest.json'):
            if path.is_file():z.write(path,'provenance/'+path.name)
    print('Evidence ZIP:',archive,'size',round(archive.stat().st_size/1e6,2),'MB',flush=True)

manifest=json.loads((OUT/'kodak_native_proposal/kodak_external_full_manifest.json').read_text())
assert manifest['status']=='COMPLETE' and manifest['host_count']==24 and manifest['watermark_count']==2
if RUN_OFFICIAL_FULL_24:
    official=json.loads((OUT/'official_mmwatermark_full_24/official_mmwatermark_manifest.json').read_text())
    assert official['full_coverage'] and len(official['hosts'])==24 and len(official['watermarks'])==2
print('COMPLETE: Native Kodak 24/24 + official MMW 24/24' if RUN_OFFICIAL_FULL_24 else 'COMPLETE: Native Kodak 24/24')
