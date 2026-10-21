#!/usr/bin/env python3
"""Repeatable image-JPEG-extraction timing and optional diagnostic call profiling.

No instrumenting/wrapping/patching of the proposal's timing-critical code.
Recorded stage timings are directly observed; cProfile breakdown is separate
and explicitly NOT used as end-to-end performance evidence.
"""
from __future__ import annotations
import argparse
import cProfile
import csv
import pstats
from pathlib import Path
import numpy as np
from reviewer_expansion_common import *
from reviewer_experiments import jpeg


def run(a):
    init_cpu()
    out=Path(a.out_dir)
    hosts,marks=dataset(a)
    configs={'original':config(a.original_config),'refined':config(a.refined_config)}
    rows=[];profiles=[]
    for hp in hosts:
        image=load_host(hp, resolution=(1280,720))
        for wi,wp in enumerate(marks):
            wm=mark(wp)
            # Warm both paths; warmup does not contribute measured time.
            for cfg in configs.values():
                im,key,_=encode(image,wm,cfg,case_seed(a.seed,wi))
                decode(jpeg(im,70),key)
            for i in range(a.repeats):
                order=('original','refined') if i%2==0 else ('refined','original')
                for name in order:
                    cfg=configs[name]
                    start=time.perf_counter_ns()
                    im,key,_=encode(image,wm,cfg,case_seed(a.seed,wi))
                    te=time.perf_counter_ns()
                    attacked=jpeg(im,70)
                    tc=time.perf_counter_ns()
                    bits,meta,_=decode(attacked,key)
                    td=time.perf_counter_ns()
                    em=(te-start)/1e6;codec=(tc-te)/1e6;dt=(td-tc)/1e6
                    rows.append({'host':hp.name,'watermark':wp.name,'iteration':i,
                        'variant':name,'embed_ms':em,'codec_ms':codec,'extract_ms':dt,
                        'total_ms':(td-start)/1e6,'on_time_30fps':(td-start)/1e6<=1000/30,
                        'nc':float(nc_binary(wm,bits)), 'psnr_db':float(psnr(image,im)),
                        'path':meta.get('path',''),'warp_count':meta.get('warp_count',''),
                        'configuration_sha256':cfg.sha256()})
                    if a.profile and i==0:
                        profiler=cProfile.Profile();profiler.enable()
                        extract_jilp_qim(attacked,key,sync=True)
                        profiler.disable()
                        st=pstats.Stats(profiler)
                        for (file,line,fn),value in st.stats.items():
                            calls,prim,tottime,cumulative,callers=value
                            if 'jilp_qim' in file or 'phase_registration' in file:
                                profiles.append({'host':hp.name,'watermark':wp.name,'variant':name,
                                  'function':fn,'source':Path(file).name,'calls':prim,
                                  'profile_cpu_self_ms':tottime*1000,'profile_cpu_cumulative_ms':cumulative*1000})
    write_rows(out/'latency_trials.csv',rows)
    if profiles:write_rows(out/'diagnostic_profile_not_wallclock.csv',profiles)
    summary=[]
    for name in configs:
        rr=[r for r in rows if r['variant']==name]
        duration=np.array([r['total_ms'] for r in rr])
        summary.append({'variant':name,'trials':len(rr),
            'median_total_ms':float(np.median(duration)),
            'p95_total_ms':float(np.percentile(duration,95)),
            'p99_total_ms':float(np.percentile(duration,99)),
            'mean_effective_fps':float(1000/np.mean(duration)),
            'on_time_30fps_fraction':float(np.mean([r['on_time_30fps'] for r in rr])),
            **{f'mean_{name}_ms':float(np.mean([r[f'{name}_ms'] for r in rr])) for name in ('embed','codec','extract')},
            'mean_nc':float(np.mean([r['nc'] for r in rr])),
            'mean_psnr_db':float(np.mean([r['psnr_db'] for r in rr]))})
    write_rows(out/'latency_summary.csv',summary)
    write_json(out/'manifest.json',{'status':'complete','resolution':'1280x720',
        'channel':'Pillow JPEG quality 70, subsampling=0','trials':len(rows),
        'hardware':info(),'profile_used_for_fps':False,
        'caveat':'Sequential CPU service rate, not paced stream FPS. Use live_stream_benchmark for deadline evidence.',
        'original_hash':configs['original'].sha256(),'refined_hash':configs['refined'].sha256()})


def parser():
    p=base_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument('--original-config',default=str(ROOT/'configs/proposal_frozen.json'))
    p.add_argument('--refined-config',default=str(ROOT/'configs/proposal_minimal_refined.json'))
    p.add_argument('--repeats',type=int,default=8)
    p.add_argument('--profile',action='store_true')
    return p
if __name__=='__main__':run(parser().parse_args())
