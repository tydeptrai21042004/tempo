#!/usr/bin/env python3
"""Measure target and NON-target orthogonal-carrier displacements after nonlinear steps.

This is a controlled basis-level diagnostic, NOT a universal recovery guarantee.
The same geometric interpolation is applied to the unmodified control block.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import cv2
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from realtime_watermark.core import dct_basis, save_csv, save_json


def main(a):
    hosts=sorted(p for p in Path(a.host_dir).glob('*.bmp') if p.is_file() and 'lena' not in p.name.lower())
    if not hosts: raise FileNotFoundError('No test RGB BMP hosts')
    phi01=dct_basis(8,0,1).astype(float)
    phi10=dct_basis(8,1,0).astype(float)
    basis={'payload':(phi01-phi10)/2, 'pilot':(phi01+phi10)/2}
    eta=float(a.eta); g=np.array([.299+eta,.587-eta/2,.114-eta/2])
    rng=np.random.default_rng(a.seed); rows=[]
    for path in hosts[:a.max_hosts if a.max_hosts else None]:
        img=np.asarray(Image.open(path).convert('RGB'),dtype=np.uint8)
        h,w=img.shape[:2]; blocks=img[:h-h%8,:w-w%8].reshape(h//8,8,w//8,8,3).transpose(0,2,1,3,4).reshape(-1,8,8,3)
        choices=rng.choice(len(blocks),size=min(a.blocks_per_host,len(blocks)),replace=False)
        for index in choices:
            orig=blocks[index].astype(float)
            for target in ('payload','pilot'):
                vec=basis[target]; other=basis['pilot' if target=='payload' else 'payload']
                delta=2*a.shift*vec[:,:,None]*g[None,None,:]/float(g@g)
                continuous=orig+delta
                rounded=np.clip(np.rint(continuous),0,255)
                for transformation in ('continuous','uint8','rotate_2p5','rotate_7p5','subpixel_half','shear_0p1'):
                    if transformation=='continuous':
                        modified,control=continuous,orig
                    elif transformation=='uint8':
                        modified,control=rounded,orig
                    else:
                        if transformation.startswith('rotate'):
                            angle=2.5 if transformation=='rotate_2p5' else 7.5
                            warp=cv2.getRotationMatrix2D((3.5,3.5),angle,1.0)
                        elif transformation=='subpixel_half':
                            warp=np.array([[1,0,0.5],[0,1,0.5]],dtype=float)
                        else:
                            warp=np.array([[1,0.1,0],[0,1,0]],dtype=float)
                        modified=cv2.warpAffine(rounded,warp,(8,8),flags=cv2.INTER_LINEAR,
                                                   borderMode=cv2.BORDER_REFLECT_101).astype(float)
                        control=cv2.warpAffine(orig,warp,(8,8),flags=cv2.INTER_LINEAR,
                                                  borderMode=cv2.BORDER_REFLECT_101).astype(float)
                    field=(modified-control)@g
                    gain=float(np.sum(field*vec));leak=float(np.sum(field*other))
                    rows.append({'host':path.name,'block':int(index),'target':target,
                        'transformation':transformation,'requested_displacement':float(a.shift),
                        'realized_target_displacement':gain,'non_target_carrier_leakage':leak,
                        'absolute_cross_interference':abs(leak),
                        'relative_cross_interference':abs(leak)/max(abs(gain),1e-12),
                        'target_displacement_error':gain-float(a.shift)})
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    save_csv(out/'cross_interference_per_block.csv',rows)
    summaries=[]
    for target in ('payload','pilot'):
        for transformation in sorted({r['transformation'] for r in rows}):
            data=[r for r in rows if r['target']==target and r['transformation']==transformation]
            v=np.array([r['absolute_cross_interference'] for r in data])
            summaries.append({'target':target,'transformation':transformation,'samples':len(data),
                'mean_abs_cross_interference':float(np.mean(v)),
                'p95_abs_cross_interference':float(np.percentile(v,95)),
                'max_abs_cross_interference':float(np.max(v))})
    save_csv(out/'cross_interference_summary.csv',summaries)
    save_json(out/'manifest.json',{'status':'complete','simulation':'isolated DCT carrier shifts on actual RGB blocks',
        'geometry_note':'local 8x8 resampling control; not full image alignment or pilot decoding',
        'rounding':'np.rint then clip [0,255]','seed':a.seed,'samples':len(rows)})
    print('Cross-interference observations:',len(rows))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host-dir',required=True)
    p.add_argument('--out-dir',required=True)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--shift',type=float,default=8.0)
    p.add_argument('--eta',type=float,default=.07)
    p.add_argument('--blocks-per-host',type=int,default=40)
    p.add_argument('--max-hosts',type=int,default=0)
    main(p.parse_args())
