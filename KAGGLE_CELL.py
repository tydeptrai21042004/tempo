# One Kaggle notebook cell. Upload TEMPO_23_VARIANT_FPS_ABLATION_PATCH.zip as a Kaggle dataset
# or manually set PATCH_ZIP below. This cell clones the public repository, patches it,
# validates tests, then evaluates all 23 variants on the original 3 development hosts.
import glob, os, pathlib, subprocess, sys, zipfile
repo = pathlib.Path('/kaggle/working/tempo')
PATCH_ZIP = next((pathlib.Path(p) for p in glob.glob('/kaggle/input/**/*TEMPO_23_VARIANT_FPS_ABLATION_PATCH.zip',recursive=True)),None)
if PATCH_ZIP is None:
    raise FileNotFoundError('Upload TEMPO_23_VARIANT_FPS_ABLATION_PATCH.zip to Kaggle inputs first')
if not repo.exists():
    subprocess.run(['git','clone','--depth','1','https://github.com/tydeptrai21042004/tempo.git',str(repo)],check=True)
with zipfile.ZipFile(PATCH_ZIP) as archive:
    archive.extractall(repo)
env={**os.environ,'PYTHONPATH':f'{repo}/src:{repo}',
     'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'4','JILP_NUM_THREADS':'4'}
subprocess.run([sys.executable,'-m','pytest','-q','tests/test_fps_method_ablation.py'],cwd=repo,env=env,check=True)
hosts=repo/'data/user_dataset/host'
marks=repo/'data/user_dataset/watermark'
if not (hosts.is_dir() and marks.is_dir()):
    raise FileNotFoundError('Point hosts/marks at your actual 12-BMP and two-PNG dataset directories')
subprocess.run([sys.executable,'scripts/proposal_fps_ablation.py',
    '--config','configs/frozen_selected_20261021.json',
    '--host-dir',str(hosts),'--watermark-dir',str(marks),
    '--split','development','--group','all','--attack-preset','stress',
    '--attacks','','--repeats','5','--warmups','1',
    '--out-dir','/kaggle/working/TEMPO_FPS_ABLATION_DEVELOPMENT'],
    cwd=repo,env=env,check=True)
print('Saved development results: /kaggle/working/TEMPO_FPS_ABLATION_DEVELOPMENT')
# Freeze the chosen finalist configs before any --split holdout or 720p stream testing.
