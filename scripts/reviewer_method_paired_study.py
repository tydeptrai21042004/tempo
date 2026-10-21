#!/usr/bin/env python3
"""Paired original-versus-refined configurations, without modifying either method.

The original/revised configuration JSON files are read verbatim; this is an
operating-point comparison, NOT matched-PSNR or a causal component attribution.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from reviewer_expansion_common import *
from reviewer_experiments import jpeg, warp

ATTACKS = {
    'clean': lambda im: im,
    'jpeg70': lambda im: jpeg(im, 70),
    'jpeg40': lambda im: jpeg(im, 40),
    'rotation2p5': lambda im: warp(im, angle=2.5),
    'subpixel0p5': lambda im: warp(im, dx=.5, dy=.5),
    'rotation3p25': lambda im: warp(im, angle=3.25),
}


def run(a):
    init_cpu()
    out=Path(a.out_dir)
    hosts, marks=dataset(a)
    configs={'original':config(a.original_config), 'refined':config(a.refined_config)}
    rows=[]
    for hp in hosts:
        host=load_host(hp)
        for wi,wp in enumerate(marks):
            wm=mark(wp)
            # Warm BOTH unchanged configurations before any measured case.
            # First-call JIT compilation must not appear as an FPS benefit.
            for warm_cfg in configs.values():
                warm_image,warm_key,_=encode(host,wm,warm_cfg,case_seed(a.seed,wi))
                decode(warm_image,warm_key)
            for variant,cfg in configs.items():
                marked,key,em=encode(host,wm,cfg,case_seed(a.seed,wi))
                attacks=['clean','jpeg70'] if a.quick else list(ATTACKS)
                for attack in attacks:
                    received=ATTACKS[attack](marked)
                    bits,meta,dt=decode(received,key)
                    rows.append({'host':hp.name, 'watermark':wp.name, 'variant':variant,
                        'attack':attack, 'config_sha256':cfg.sha256(), 'seed':case_seed(a.seed,wi),
                        'embed_ms':em,'extract_ms':dt,
                        'extract_path':meta.get('path',''), 'sync_warp_count':meta.get('warp_count',''),
                        **quality(host,marked,wm,bits)})
    write_rows(out/'method_rows.csv',rows)
    ix={(r['host'],r['watermark'],r['attack'],r['variant']):r for r in rows}
    paired=[]
    for r in rows:
        if r['variant'] != 'refined':continue
        original=ix[(r['host'],r['watermark'],r['attack'],'original')]
        paired.append({'host':r['host'],'watermark':r['watermark'],'attack':r['attack'],
            **{f'delta_{col}':r[col]-original[col] for col in ('psnr_db','nc','ber','embed_ms','extract_ms')}})
    write_rows(out/'paired_effects.csv',paired)
    stats=[]
    for attack in sorted({p['attack'] for p in paired}):
        pp=[p for p in paired if p['attack']==attack]
        for col in ('delta_psnr_db','delta_nc','delta_ber','delta_embed_ms','delta_extract_ms'):
            stats.append({'attack':attack,'metric':col,**host_bootstrap(pp,col,seed=a.seed)})
    write_rows(out/'paired_host_ci.csv',stats)
    write_json(out/'manifest.json',{'status':'complete','configs':{k:v.sha256() for k,v in configs.items()},
        'hosts':{p.name:md5_file(p) for p in hosts},'watermarks':{p.name:md5_file(p) for p in marks},
        'rows':len(rows),'pairs':len(paired),'attacks':attacks,
        'caveat':'Not matched PSNR; configs differ in parameters/phase policy, not a single-factor causal test',**info()})


def parser():
    p=base_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument('--original-config',default=str(ROOT/'configs/proposal_frozen.json'))
    p.add_argument('--refined-config',default=str(ROOT/'configs/proposal_minimal_refined.json'))
    return p
if __name__=='__main__':run(parser().parse_args())
