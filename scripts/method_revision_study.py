"""Paired, CPU-only methodology audit. No code paths replaced by synthetic metrics.

Run: PYTHONPATH=src python scripts/method_revision_study.py --split holdout --out results/method_revision
The development split was used to choose the trial step; report holdout separately.
"""
from __future__ import annotations
import argparse, csv, json, os, time
from pathlib import Path
import numpy as np
import cv2
from realtime_watermark.core import load_binary_watermark, load_rgb, psnr, nc_binary
from realtime_watermark.registry import embed
from realtime_watermark.proposals.jilp_qim.config import DEFAULT_CONFIG
from realtime_watermark.proposals.jilp_qim.method import extract_jilp_qim
from realtime_watermark.methods import warmup_numba

DEVELOPMENT = ('lenna.bmp', 'baboon.bmp', 'pepper.bmp', 'sailboat.bmp')
WATERMARKS = ('wm.png', 'Cc.logo.circle.png')
ATTACKS = ('clean', 'jpeg70', 'jpeg40', 'rotation1.25', 'rotation2.25',
           'rotation3.25', 'rotation15', 'translate0.5', 'scale1.02', 'shear0.02', 'gaussian5', 'dct_zero')


def attack_rgb(im, kind):
    h, w = im.shape[:2]
    if kind == 'clean': return im
    if kind.startswith('jpeg'):
        qual = int(kind[4:])
        ok, jpg = cv2.imencode('.jpg', cv2.cvtColor(im, cv2.COLOR_RGB2BGR),
                                [int(cv2.IMWRITE_JPEG_QUALITY), qual])
        assert ok
        return cv2.cvtColor(cv2.imdecode(jpg, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if kind.startswith('rotation'):
        mat = cv2.getRotationMatrix2D((w/2, h/2), float(kind[8:]), 1.0)
    elif kind == 'translate0.5': mat = np.float32([[1, 0, 0.5], [0, 1, 0.5]])
    elif kind == 'scale1.02': mat = cv2.getRotationMatrix2D((w/2, h/2), 0, 1.02)
    elif kind == 'shear0.02': mat = np.float32([[1, .02, -.01*h], [0, 1, 0]])
    elif kind == 'gaussian5':
        return np.uint8(np.clip(im.astype(np.float32)
                                + np.random.default_rng(3101).normal(0, 5, im.shape), 0, 255))
    elif kind == 'dct_zero':
        # Public (0,1)/(1,0) erasure intentionally destroys the public carrier;
        # no claim that phase search or tuning can reverse it.
        bgr = im.astype(np.float32).copy()
        for row in range(0,h-h%8,8):
            for col in range(0,w-w%8,8):
                for ch in range(3):
                    patch=cv2.dct(bgr[row:row+8,col:col+8,ch]); patch[0,1]=patch[1,0]=0
                    bgr[row:row+8,col:col+8,ch]=cv2.idct(patch)
        return np.uint8(np.clip(np.rint(bgr),0,255))
    else: raise ValueError(kind)
    return cv2.warpAffine(im, mat, (w,h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))


def params_for(name):
    data = DEFAULT_CONFIG.to_dict()
    if name in ('lower_step','candidate'):
        data['step'] = 17.0
    if name == 'candidate':
        data['phase_refinement'] = 'selective'
        data['certified_phase_pruning'] = True
    return data


def run(split, out, max_hosts=0, attacks=ATTACKS):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1)
    root=Path(__file__).resolve().parents[1]/'data/user_dataset'
    hosts=sorted(f.name for f in (root/'host').glob('*.bmp'))
    if split=='development': hosts=[h for h in hosts if h in DEVELOPMENT]
    elif split=='holdout': hosts=[h for h in hosts if h not in DEVELOPMENT]
    if max_hosts: hosts=hosts[:max_hosts]
    warmup_numba()
    rows=[]
    for hn in hosts:
        im=load_rgb(root/'host'/hn,mode='full')
        for wn in WATERMARKS:
            wm=load_binary_watermark(root/'watermark'/wn,size=64)
            for name in ('baseline','lower_step','candidate'):
                t0=time.perf_counter_ns()
                marked,key=embed('proposal',im,wm,seed=2026,method_params=params_for(name))
                emb_ms=(time.perf_counter_ns()-t0)/1e6
                quality=psnr(im,marked)
                for attack in attacks:
                    altered=attack_rgb(marked,attack)
                    t1=time.perf_counter_ns()
                    pred,meta=extract_jilp_qim(altered,key,sync=True,return_meta=True)
                    decode_ms=(time.perf_counter_ns()-t1)/1e6
                    row=dict(split=split,host=hn,watermark=wn,variant=name,attack=attack,
                             psnr=quality,nc=nc_binary(wm,pred),embed_ms=emb_ms,
                             decode_ms=decode_ms,phase_score=meta.get('phase_score',''),
                             candidate=meta.get('candidate',''),
                             pilot_corrections=key.params.get('integer_pilot_unit_updates',-1),
                             payload_corrections=key.params.get('integer_lattice_unit_updates',-1))
                    rows.append(row)
                print(split,hn,wn,name,'psnr',round(quality,4),flush=True)
    path=out/f'{split}_paired.csv'
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=[]
    for atk in attacks:
        for name in ('baseline','lower_step','candidate'):
            sub=[r for r in rows if r['attack']==atk and r['variant']==name]
            summary.append(dict(attack=atk,variant=name,n=len(sub),psnr_mean=float(np.mean([r['psnr'] for r in sub])),
                                nc_mean=float(np.mean([r['nc'] for r in sub])),
                                median_extract_ms=float(np.median([r['decode_ms'] for r in sub])),
                                median_embed_ms=float(np.median([r['embed_ms'] for r in sub]))))
    with (out/f'{split}_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    (out/'protocol.json').write_text(json.dumps(dict(split=split,hosts=hosts,watermarks=WATERMARKS,
                  attacks=list(attacks),seed=2026,variants={n:params_for(n) for n in ('baseline','lower_step','candidate')},
                  cpu_threads=os.environ.get('NUMBA_NUM_THREADS'),opencv_threads=1),indent=2))
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['development','holdout','all'],default='holdout')
    ap.add_argument('--out',default='results/method_revision');ap.add_argument('--max-hosts',type=int,default=0)
    ap.add_argument('--quick',action='store_true');args=ap.parse_args()
    arr=('clean','jpeg70','jpeg40','rotation1.25','rotation2.25','rotation3.25','translate0.5','scale1.02') if args.quick else ATTACKS
    res=run(args.split,args.out,args.max_hosts,arr)
    for r in res: print(r)
