#!/usr/bin/env python3
"""Empirical PSNR–NC–FPS frontier across unchanged proposal parameter points.

Only scan on development data. Never select a configuration using held-out test
hosts; use frozen saved operating points for the subsequent independent test.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from reviewer_expansion_common import *
from reviewer_experiments import jpeg


def nondominated(rows):
    """Three maximized criteria; NaN/infeasible records are never winners."""
    result=[]
    for i,r in enumerate(rows):
        values=np.array([r['mean_psnr_db'],r['mean_nc'],r['effective_fps']],float)
        if not np.all(np.isfinite(values)):
            continue
        dominated=False
        for j,q in enumerate(rows):
            if i==j:continue
            alt=np.array([q['mean_psnr_db'],q['mean_nc'],q['effective_fps']],float)
            if np.all(np.isfinite(alt)) and np.all(alt>=values) and np.any(alt>values):
                dominated=True;break
        if not dominated:result.append(r)
    return result


def run(a):
    init_cpu()
    if a.data_role != 'development':
        raise ValueError('Pareto operating-point sweep must use development data, not held-out test data')
    out=Path(a.out_dir)
    hosts,marks=dataset(a)
    base=config(a.config)
    # One-dimensional parameter sweep, not a new embedding algorithm.
    step_values=[14.,17.,20.] if a.quick else [12.,14.,16.,18.,20.,22.,24.]
    rows=[]; per=[]
    # Warm the actual embedding and attacked extraction kernels, including JPEG.
    first_host=load_host(hosts[0]);first_wm=mark(marks[0])
    warm_image,warm_key,_=encode(first_host,first_wm,base,case_seed(a.seed,0))
    decode(jpeg(warm_image,40),warm_key)
    for step in step_values:
        cfg=base.with_step(step)
        ems=[];dts=[]; qualities=[]; ncs=[]
        for hp in hosts:
            im=load_host(hp)
            for wi,wp in enumerate(marks):
                wm=mark(wp)
                marked,key,em=encode(im,wm,cfg,case_seed(a.seed,wi))
                attacked=jpeg(marked,40)
                bits,meta,dt=decode(attacked,key)
                ems.append(em);dts.append(dt)
                qualities.append(psnr(im,marked));ncs.append(nc_binary(wm,bits))
                per.append({'step':step,'host':hp.name,'watermark':wp.name,
                    'psnr_db':float(qualities[-1]),'nc_jpeg40':float(ncs[-1]),
                    'embed_ms':em,'extract_ms':dt,'path':meta.get('path','')})
        mean_service_ms=float(np.mean(np.asarray(ems)+np.asarray(dts)))
        rows.append({'step':step,'config_hash':cfg.sha256(),
            'mean_psnr_db':float(np.mean(qualities)), 'mean_nc':float(np.mean(ncs)),
            'effective_fps':1000/mean_service_ms,'mean_service_ms':mean_service_ms,
            'n_cases':len(qualities)})
    frontier=nondominated(rows)
    write_rows(out/'pareto_points.csv',rows)
    write_rows(out/'pareto_frontier.csv',frontier)
    write_rows(out/'pareto_per_case.csv',per)
    write_json(out/'manifest.json',{'status':'complete','data_role':a.data_role,
        'scan_is_development_only':True,'operating_points':len(rows),'nondominated':len(frontier),
        'hosts':{p.name:md5_file(p) for p in hosts},
        'metric_definition':'FPS=1000/mean(embed_ms + JPEG40 attacked extract_ms); JPEG attack time excluded',
        'caveats':['Experimental sequential rate, not live-stream deadline FPS','This sweep does not establish matched-PSNR superiority'],**info()})


def parser():
    p=base_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument('--data-role',choices=['development','heldout'],default='development')
    return p
if __name__=='__main__':run(parser().parse_args())
