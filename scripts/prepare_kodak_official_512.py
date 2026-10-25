#!/usr/bin/env python3
"""Make 24 explicitly documented 512x512 Kodak central crops for the
legacy 512x512 official MMWatermark perturbations. The full native Kodak
proposal experiment is a separate, uncropped track.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import sys
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parent))
from download_full_kodak import download,NAMES
from reviewer_expansion_common import write_json


def prepare(dataset_dir: Path, out_dir: Path, offline=False):
    source=Path(dataset_dir).resolve(); out=Path(out_dir).resolve()
    download(source,offline=offline)
    out.mkdir(parents=True, exist_ok=True)
    rows=[]
    for name in NAMES:
        p=source/name
        with Image.open(p) as im:
            im=im.convert('RGB')
            w,h=im.size
            if (w,h) not in ((768,512),(512,768)): raise ValueError(f'Unsupported original {name}: {w}x{h}')
            x0=(w-512)//2; y0=(h-512)//2
            patch=im.crop((x0,y0,x0+512,y0+512))
            dest=out/name
            patch.save(dest,format='PNG',optimize=False)
        rows.append({'name':name,'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
            'crop_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
            'source_resolution':[w,h],'crop_xy':[x0,y0], 'crop_resolution':[512,512]})
    manifest={'status':'COMPLETE_24_OF_24','dataset':'Kodak PCD0992',
      'purpose':'Official MMWatermark compatibility; all 24 source image identities preserved',
      'transform':'deterministic central 512x512 crop; no scaling or resampling',
      'not_native_resolution':True,'expected':24,'created':len(rows),'files':rows}
    write_json(out/'kodak_official_crops_manifest.json',manifest)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--offline',action='store_true')
    a=p.parse_args();prepare(a.dataset_dir,a.out_dir,offline=a.offline)
if __name__=='__main__':main()
