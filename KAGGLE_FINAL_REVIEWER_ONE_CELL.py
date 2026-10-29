# Kaggle CPU notebook: Internet ON. Run this entire cell after uploading the
# patch to the GitHub repository. Uses real videos; no pretrained neural models.
from pathlib import Path
from IPython.display import display,FileLink
import os,subprocess,sys
REPO_URL='https://github.com/tydeptrai21042004/tempo.git'
BRANCH='main'
ROOT=Path('/kaggle/working/tempo_final_reviewer_repo')
if not ROOT.is_dir():
    subprocess.run(['git','clone','--depth','1','-b',BRANCH,REPO_URL,str(ROOT)],check=True)
print('Source SHA:',subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
env=os.environ.copy()
env.update(PYTHON=sys.executable,PYTHONPATH=str(ROOT/'src')+os.pathsep+str(ROOT),
    TEMPO_MATH_REFINEMENT='0', TEMPO_RUN_INDEPENDENT_HAAR_QIM_CONTROL='0',
    TEMPO_OFFICIAL_MMWATERMARK='0')
subprocess.run(['bash','scripts/kaggle_setup.sh'],cwd=ROOT,env=env,check=True)
subprocess.run(['bash','scripts/kaggle_final_reviewer_pipeline.sh','full'],cwd=ROOT,env=env,check=True)
out=Path('/kaggle/working/TEMPO_FINAL_REVIEWER_EVIDENCE.zip')
assert out.is_file() and out.stat().st_size>0
print('Evidence package:',out)
display(FileLink(str(out)))
