#!/usr/bin/env python3
"""Record source files, hashes and environment, without a result-quality gate."""
import argparse,hashlib,json,platform,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.repo).resolve();out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
files=[x for folder in ('src','scripts','configs') for x in (root/folder).rglob('*')
       if x.is_file() and x.suffix in ('.py','.json','.sh')]
sha={str(x.relative_to(root)):hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(files)}
try:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
except Exception:commit=None
manifest={'git_commit':commit,'python':platform.python_version(),'platform':platform.platform(),
          'code_sha256':sha,'file_count':len(sha),'source_status':'recorded'}
out.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Recorded',len(sha),'source files', 'commit',commit)
