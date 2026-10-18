"""A/B RCME versus original pilot-free TEMPO; local images, no network.

Example:
 PYTHONPATH=src python scripts/rcme_compare.py --out evidence/rcme --max-hosts 13
All metrics are measured; warm-up and service time exclude JIT compilation.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from statistics import mean
import numpy as np
from PIL import Image, ImageFilter
from realtime_watermark.core import load_binary_watermark, load_rgb, psnr, nc_binary, bit_accuracy
from realtime_watermark.registry import embed,extract
from realtime_watermark.proposals.jilp_qim.config import DEFAULT_CONFIG

IMAGE_EXT={'.bmp','.png','.jpg','.jpeg','.tif','.tiff'}


def attack(img:np.ndarray,kind:str)->np.ndarray:
    if kind=='clean':
        return img
    if kind.startswith('jpeg'):
        q=int(kind[4:]); b=io.BytesIO();Image.fromarray(img).save(b,format='JPEG',quality=q)
        b.seek(0);return np.asarray(Image.open(b).convert('RGB'),dtype=np.uint8)
    if kind=='blur3':
        return np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=.8)),dtype=np.uint8)
    if kind=='noise3':
        rng=np.random.default_rng(3101)
        return np.clip(np.rint(img.astype(np.float32)+rng.normal(0,3,img.shape)),0,255).astype(np.uint8)
    raise ValueError(kind)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--hosts',type=Path,default=Path('data/user_dataset/host'))
    ap.add_argument('--watermarks',type=Path,default=Path('data/user_dataset/watermark'))
    ap.add_argument('--out',type=Path,default=Path('evidence/rcme'))
    ap.add_argument('--max-hosts',type=int,default=13)
    ap.add_argument('--timing720',action='store_true',help='measure sequential JPEG70 service FPS on two resized 720p hosts')
    ap.add_argument('--include-original',action='store_true',help='include original TEMPO with pilots/recovery for a slower full-reference A/B')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    hosts=sorted(p for p in args.hosts.iterdir() if p.suffix.lower() in IMAGE_EXT)[:args.max_hosts]
    wms=sorted(p for p in args.watermarks.iterdir() if p.suffix.lower() in IMAGE_EXT)
    if not hosts or not wms: raise FileNotFoundError('Host or watermark files not found')
    cfg=replace(DEFAULT_CONFIG,enable_pilots=False,enable_sync_search=False,
                projection_mode='minimum_distortion',use_integer_lattice_closure=True)
    # Both methods have identical QIM step, block carrier, seed and host/watermark.
    cases={'pilot_free_original':('proposal',cfg.to_dict()),'rcme_qim':('rcme_qim',{
        'step':cfg.step,'eta':cfg.eta,'reference_margin_fraction':cfg.rho_frac})}
    if args.include_original:
        cases['original_TEMPO_with_pilots'] = ('proposal',DEFAULT_CONFIG.to_dict())
    # Initial warmup to remove JIT compilation from timed observations.
    warm_host=load_rgb(hosts[0]);warm_wm=load_binary_watermark(wms[0],64)
    for method,params in cases.values():
        m,k=embed(method,warm_host,warm_wm,seed=2026,method_params=params)
        _=extract(m,k)
    rows=[]; t720=[]
    for path in hosts:
        host=load_rgb(path)
        for wp in wms:
            wm=load_binary_watermark(wp,64)
            for label,(method,params) in cases.items():
                t0=time.perf_counter(); marked,key=embed(method,host,wm,seed=2026,method_params=params);t1=time.perf_counter()
                decoded=extract(marked,key);t2=time.perf_counter()
                for kind in ('clean','jpeg70','jpeg40','blur3','noise3'):
                    attacked=attack(marked,kind)
                    recovered=extract(attacked,key)
                    rows.append({'variant':label,'host':path.name,'watermark':wp.name,'attack':kind,
                        'psnr':round(psnr(host,marked),6),'nc':round(float(nc_binary(wm,recovered)),8),
                        'bit_accuracy':round(float(bit_accuracy(wm,recovered)),8),
                        'embed_ms':round(1e3*(t1-t0),4),'extract_clean_ms':round(1e3*(t2-t1),4),
                        'unit_corrections':key.params.get('integer_lattice_unit_updates',0),
                        'guard_failures':key.params.get('integer_lattice_final_guard_failures',0)})
    with (args.out/'pairs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    stats=[]
    for name in cases:
        r=[x for x in rows if x['variant']==name]
        clean=[x for x in r if x['attack']=='clean']
        other=[x for x in r if x['attack']!='clean']
        stats.append({'method':name,'pairs':len(clean),'psnr_mean':mean(x['psnr'] for x in clean),
            'exact_clean_pairs':sum(x['bit_accuracy']==1.0 for x in clean),
            'nongeometric_nc':mean(x['nc'] for x in other),
            'jpeg70_nc':mean(x['nc'] for x in r if x['attack']=='jpeg70'),
            'jpeg40_nc':mean(x['nc'] for x in r if x['attack']=='jpeg40'),
            'embed_ms_median':float(np.median([x['embed_ms'] for x in clean])),
            'extract_ms_median':float(np.median([x['extract_clean_ms'] for x in clean]))})
    if args.timing720:
        from realtime_watermark.core import make_demo_watermark
        wm=load_binary_watermark(wms[0],64)
        for path in hosts[:2]:
            img=Image.fromarray(load_rgb(path)).resize((1280,720),Image.Resampling.BILINEAR)
            host=np.asarray(img,dtype=np.uint8)
            for label,(method,params) in cases.items():
                for trial in range(12):
                    t0=time.perf_counter();marked,key=embed(method,host,wm,seed=2026,method_params=params)
                    jpeg=attack(marked,'jpeg70');_ = extract(jpeg,key);t1=time.perf_counter()
                    t720.append({'method':label,'host':path.name,'trial':trial,
                        'service_ms':(t1-t0)*1000.0,'fps_inverse_service':1/max(t1-t0,1e-12)})
        with (args.out/'timing720.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(t720[0]));w.writeheader();w.writerows(t720)
        for row in stats:
            t=[x['service_ms'] for x in t720 if x['method']==row['method']]
            row['service_fps_720_jpeg70']=1000/mean(t)
            row['service_p95_ms_720_jpeg70']=float(np.percentile(t,95))
    (args.out/'summary.json').write_text(json.dumps({'protocol':{
        'dataset':'user-supplied 13 host images and 2 64x64 watermarks',
        'attacks':['clean','jpeg70','jpeg40','blur3','noise3'],
        'fully_blind':True,'same_QIM_step':True,
        'service_fps_not_live_effective_fps':True,'numba_num_threads':os.environ.get('NUMBA_NUM_THREADS')},
        'summary':stats},indent=2))
    for row in stats:print(json.dumps(row,indent=2))
    print('Saved',args.out)

if __name__=='__main__': main()
