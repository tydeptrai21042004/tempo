"""Paired original TEMPO vs sparse-48 vs frequency-preserving sparse-48.

Run from repository root with NUMBA_NUM_THREADS=2, JILP_NUM_THREADS=2 and
PYTHONPATH=src.  Numerical results are saved under --output.
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
from realtime_watermark.core import (load_rgb, load_binary_watermark, psnr, ssim_gray, nc_binary, ber,
                                     attack_jpeg, attack_blur, attack_box_blur)
from realtime_watermark.proposals.jilp_qim.config import DEFAULT_CONFIG
from realtime_watermark.proposals.jilp_qim import method

@dataclass(frozen=True)
class Candidate:
    label: str
    mix: float
    support: int
    mode: str = 'sparse'

VARIANTS = (Candidate('original',0.0,0),
            Candidate('sparse48',0.6,48),
            Candidate('frequency_preserving48',0.6,48,'frequency_preserving'))
HOLDOUT = ('Girl','airplane','athens','butfly','house','manhatan','milkdrop','safari','tiffany')
TIMING_HOSTS = ('lenna','baboon','pepper','sailboat')

def save_csv(path: Path, rows: list[dict[str,Any]]) -> None:
    if not rows: return
    with path.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def run_one(host: np.ndarray, wm: np.ndarray, variant: Candidate):
    c=DEFAULT_CONFIG
    opts=dict(seed=2026, step=c.step, rho_frac=c.rho_frac,
              pilot_count=c.pilot_count,pilot_step=c.pilot_step,
              pilot_rho_frac=c.pilot_rho_frac,eta=c.eta,
              hartley_mix=variant.mix,hartley_support=variant.support,
              hartley_carrier_mode=variant.mode)
    st=time.perf_counter()
    marked,params=method.embed_jilp_qim(host,wm,**opts)
    em=(time.perf_counter()-st)*1000
    key={'method_id':'proposal','host_shape':host.shape,'watermark_shape':wm.shape,
         'seed':opts['seed'],'params':params}
    st=time.perf_counter()
    rec=method.extract_jilp_qim(marked,key,sync=False)
    ex=(time.perf_counter()-st)*1000
    return marked,key,params,em,ex,rec

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,default=Path('data/user_dataset'))
    ap.add_argument('--output',type=Path,default=Path('results/hartley_frequency_compare'))
    ap.add_argument('--quick',action='store_true',help='Three pairs and 3 timing iterations for smoke testing')
    ap.add_argument('--timing-runs',type=int,default=8)
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    host_dir=args.data/'host'
    wm_paths=sorted((args.data/'watermark').glob('*.png'))
    if len(wm_paths)<2:
        raise RuntimeError('At least two reference watermarks are required')
    watermarks=[(p.name,load_binary_watermark(p,size=64)) for p in wm_paths[:2]]
    method.warmup_jilp_qim()
    rows=[]
    hosts=HOLDOUT[:2] if args.quick else HOLDOUT
    for hname in hosts:
        host=load_rgb(host_dir/(hname+'.bmp'))
        wms=watermarks[:1] if args.quick else watermarks
        for wmname,wm in wms:
            for variant in VARIANTS:
                marked,key,params,em,ex,rec=run_one(host,wm,variant)
                if not np.array_equal(rec,wm):
                    raise AssertionError(f'clean decoding failed: {hname} {variant.label}')
                row=dict(host=hname,watermark=wmname,variant=variant.label,
                         psnr=float(psnr(host,marked)),ssim=float(ssim_gray(host,marked)),
                         nc_clean=float(nc_binary(wm,rec)),ber_clean=float(ber(wm,rec)),
                         unresolved=int(params['integer_lattice_unresolved']))
                attacked={
                    'jpeg70':attack_jpeg(marked,70),
                    'jpeg40':attack_jpeg(marked,40),
                    'blur1':attack_blur(marked,1.0),
                    'blur15':attack_blur(marked,1.5),
                    'box1':attack_box_blur(marked,1.0),
                }
                for attack_name,img in attacked.items():
                    recovered=method.extract_jilp_qim(img,key,sync=False)
                    row['nc_'+attack_name]=float(nc_binary(wm,recovered))
                    row['ber_'+attack_name]=float(ber(wm,recovered))
                rows.append(row)
            print('Quality',hname,wmname,flush=True)
    save_csv(args.output/'quality_pairs.csv',rows)
    summary={}
    for var in VARIANTS:
        group=[r for r in rows if r['variant']==var.label]
        summary[var.label]={'pairs':len(group)}
        for field in ['psnr','ssim','nc_clean','ber_clean','nc_jpeg70','nc_jpeg40','nc_blur1','nc_blur15','nc_box1','unresolved']:
            summary[var.label][field]=float(np.mean([r[field] for r in group]))
    (args.output/'quality_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Quality summary:',json.dumps(summary,indent=2),flush=True)
    time_rows=[]
    # Alternating order counters scheduling drift, first two passes excluded.
    for hname in TIMING_HOSTS[:2 if args.quick else 4]:
        original=load_rgb(host_dir/(hname+'.bmp'))
        for width,height in ((512,512),) if args.quick else ((512,512),(1280,720)):
            host=original if (width,height)==(512,512) else np.asarray(Image.fromarray(original).resize((width,height),Image.Resampling.BICUBIC),dtype=np.uint8)
            wm=watermarks[0][1]
            for var in VARIANTS:
                run_one(host,wm,var)
            repeats=3 if args.quick else max(4,args.timing_runs)
            for repeat in range(repeats):
                order=VARIANTS if repeat%2==0 else tuple(reversed(VARIANTS))
                for variant in order:
                    _,_,_,em,ex,rec=run_one(host,wm,variant)
                    if repeat >= 2:
                        time_rows.append(dict(host=hname,width=width,height=height,repeat=repeat,
                                            variant=variant.label,embed_ms=em,extract_ms=ex,
                                            total_ms=em+ex,nc_clean=float(nc_binary(wm,rec))))
            print('Timing',hname,width,height,flush=True)
    save_csv(args.output/'timings.csv',time_rows)
    speed={}
    for width,height in ((512,512),) if args.quick else ((512,512),(1280,720)):
        speed[f'{width}x{height}']={}
        for var in VARIANTS:
            group=[r for r in time_rows if r['width']==width and r['variant']==var.label]
            times=np.array([r['total_ms'] for r in group])
            speed[f'{width}x{height}'][var.label]=dict(n=int(len(times)),median_ms=float(np.median(times)),
                                                      algorithm_fps=float(1000/np.median(times)),
                                                      p95_ms=float(np.percentile(times,95)))
    (args.output/'timing_summary.json').write_text(json.dumps(speed,indent=2)+'\n')
    print('Timing summary:',json.dumps(speed,indent=2),flush=True)

if __name__=='__main__':main()
