#!/usr/bin/env python3
"""Export ONLY compact numeric evidence and publication-ready PDF figures.

Deliberately excludes attacked/watermarked/extracted bitmap sequences, video,
training data, model weights, and upstream source checkouts. No G01--G24 audit.
"""
from __future__ import annotations
import argparse
import json
import zipfile
from pathlib import Path

ALLOWED={'.csv','.tsv','.json','.jsonl','.tex','.pdf','.md','.txt','.parquet','.gz'}
DENIED_DIRS={'data','dataset','datasets','media','frames','frame_cache','upstream','models','weights',
              'checkpoints','__pycache__','.git','image_cache','attacked_images','extracted_images',
              'watermarked_images','visualization_cache','videos','video_cache','tmp','temp'}
DENIED_SUFFIX={'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.mp4','.avi','.mov','.webp','.npy','.npz','.pt','.pth','.onnx','.log','.zip'}

def package(root:Path, output:Path, max_file_mb:float=25.0):
    root=root.resolve();output=output.resolve()
    if not root.is_dir():raise FileNotFoundError(root)
    output.parent.mkdir(parents=True,exist_ok=True)
    files=[];excluded=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.resolve()==output:continue
        relative=p.relative_to(root)
        if any(part.lower() in DENIED_DIRS for part in relative.parts[:-1]):
            excluded.append(str(relative));continue
        if p.suffix.lower() in DENIED_SUFFIX or p.suffix.lower() not in ALLOWED or p.stat().st_size>max_file_mb*1024*1024:
            excluded.append(str(relative));continue
        if p.suffix.lower()=='.pdf' and not any(token in str(relative).lower() for token in ('figure','plot','gallery','visual','selected')):
            excluded.append(str(relative));continue
        files.append(p)
    index={'packaging':'TEMPO compact paper evidence v2','saved_files':len(files),
           'excluded_files':len(excluded),'excluded_examples':excluded[:30],
           'explicit_exclusions':'No raw extracted/attacked/watermarked images; no video frames; no model weights or log files.',
           'file_manifest':[{'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size} for p in files]}
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zipf:
        zipf.writestr('EVIDENCE_ARCHIVE_INDEX.json',json.dumps(index,indent=2))
        for p in files:zipf.write(p,p.relative_to(root).as_posix())
    return index

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results-dir',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--max-file-mb',type=float,default=25.)
    a=ap.parse_args()
    print(json.dumps(package(a.results_dir,a.output,a.max_file_mb),indent=2))
