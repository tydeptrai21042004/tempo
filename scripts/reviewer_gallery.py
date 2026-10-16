#!/usr/bin/env python3
"""Auditable extracted-watermark gallery; proposal only, no invented baselines."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from realtime_watermark.core import load_binary_watermark,bit_accuracy,nc_binary
from realtime_watermark.proposals.jilp_qim.config import JILPQIMConfig
from realtime_watermark.proposals.jilp_qim.method import embed_proposal,extract_jilp_qim
from reviewer_experiments import warp,jpeg,attenuation,csv_write


def tile(bit_image,scale=4):
    z=(np.asarray(bit_image).astype(np.uint8)*255)
    return Image.fromarray(z,'L').convert('RGB').resize((z.shape[1]*scale,z.shape[0]*scale),Image.Resampling.NEAREST)


def run(a):
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    cfg=JILPQIMConfig.load(a.config)
    hosts=sorted(Path(a.host_dir).glob('*.bmp'))[:a.max_hosts]
    wm_files=sorted(Path(a.watermark_dir).glob('*.png'))[:a.max_watermarks]
    rows=[]
    for wp in wm_files:
        watermark=load_binary_watermark(wp,size=64)
        for hp in hosts:
            host=np.asarray(Image.open(hp).convert('RGB'))
            encoded,key=embed_proposal(host,watermark,seed=a.seed,method_params=cfg.to_dict())
            eta=float(key.params['eta'])
            conditions=[('clean',encoded),('jpeg30',jpeg(encoded,30)),('rotate2p5',warp(encoded,angle=2.5)),
                        ('payload_zero',attenuation(encoded,eta,'payload',1.0)),
                        ('pilot_zero',attenuation(encoded,eta,'pilot',1.0))]
            w=300*len(conditions)+300;h=350
            canvas=Image.new('RGB',(w,h),'white');draw=ImageDraw.Draw(canvas)
            canvas.paste(tile(watermark),(10,45));draw.text((10,12),'Original 4096-bit watermark',fill='black')
            for ci,(label,received) in enumerate(conditions):
                extracted=extract_jilp_qim(received,key)
                score=bit_accuracy(watermark,extracted)
                nc=nc_binary(watermark,extracted)
                x=300*(ci+1)+10
                canvas.paste(tile(extracted),(x,45))
                draw.text((x,12),f'{label}: acc={score:.3f}',fill='black')
                rows.append({'host':hp.name,'watermark':wp.name,'attack':label,
                             'bit_accuracy':float(score),'nc':float(nc),'payload_bits':4096})
            name=f'{hp.stem}_{wp.stem}_extraction_gallery.png'
            canvas.save(out/name)
    csv_write(out/'gallery_data.csv',rows)
    (out/'gallery_manifest.json').write_text(json.dumps({
      'gallery_count':len(hosts)*len(wm_files),'observations':len(rows),
      'scope':'proposal-only extracted watermark gallery; modern baseline gallery requires source implementations'},indent=2))
    print(f'Created {len(hosts)*len(wm_files)} proposal galleries')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--host-dir',required=True);p.add_argument('--watermark-dir',required=True)
    p.add_argument('--config',required=True);p.add_argument('--out-dir',required=True)
    p.add_argument('--max-hosts',type=int,default=2);p.add_argument('--max-watermarks',type=int,default=2)
    p.add_argument('--seed',type=int,default=2026)
    run(p.parse_args())
