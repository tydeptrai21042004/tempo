#!/usr/bin/env python3
"""Prepare all 24 lossless Kodak center crops as 512x512 RGB BMP inputs
for the official MMWatermark-Robustness adapter (requires *.bmp). No image
rescaling, no native 109-attack Kodak experiment and no external CSV.
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
            dest=out/(Path(name).stem+'.bmp')
            patch.save(dest,format='BMP')
            with Image.open(dest) as check:
                if check.format!='BMP' or check.size!=(512,512) or check.mode!='RGB' or check.tobytes()!=patch.tobytes():
                    raise RuntimeError(f'Lossless BMP roundtrip check failed: {dest}')
        rows.append({'name':name,'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
            'crop_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
            'source_filename':name,'bmp_filename':dest.name,
            'crop_pixel_sha256':hashlib.sha256(patch.tobytes()).hexdigest(),
            'source_resolution':[w,h],'crop_xy':[x0,y0], 'crop_resolution':[512,512]})
    bmps=sorted(out.glob('*.bmp'))
    if len(rows)!=24 or len(bmps)!=24 or {x['bmp_filename'] for x in rows}!={x.name for x in bmps}:
        raise RuntimeError('Expected exactly 24 lossless Kodak BMP crops, got '+str(len(bmps)))
    manifest={'status':'COMPLETE_24_OF_24','dataset':'Kodak PCD0992',
      'official_host_count':24,'format':'RGB BMP 24-bit',
      'purpose':'Official MMWatermark compatibility; all 24 source image identities preserved',
      'transform':'deterministic central 512x512 crop; no scaling or resampling',
      'official_runner_glob':'*.bmp',
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
