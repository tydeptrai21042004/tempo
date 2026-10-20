#!/usr/bin/env python3
"""Paired joint/legacy Hartley-48 comparison, fixed input, payload and attacks.

Reports the raw observations. Timings are steady-state image timings, NOT
effective stream FPS or a claim that a new mathematical method is faster.
"""
import argparse
import json
import statistics
import time
from pathlib import Path
import numpy as np
from realtime_watermark.core import load_rgb,load_binary_watermark,psnr,attack_jpeg
from realtime_watermark.registry import embed,extract
from realtime_watermark.proposals.jilp_qim.config import JILPQIMConfig

ROOT=Path(__file__).resolve().parents[1]

def nc(bits, truth):
    return float(1-2*np.mean(np.asarray(bits)!=np.asarray(truth)))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('joint_dct_hartley_results.json'))
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--limit-hosts',type=int,default=0)
    args=p.parse_args()
    base=JILPQIMConfig.load(ROOT/'configs/hartley_frequency_preserving48_candidate.json')
    joint=JILPQIMConfig.load(ROOT/'configs/joint_dct_hartley48_candidate.json')
    hosts=sorted(h for h in (ROOT/'data/user_dataset/host').glob('*.bmp') if 'lenna' not in h.name.lower())
    if args.limit_hosts: hosts=hosts[:args.limit_hosts]
    watermarks=sorted((ROOT/'data/user_dataset/watermark').glob('*.png'))
    rows=[]
    for hostpath in hosts:
        host=load_rgb(hostpath)
        for wmpath in watermarks:
            wm=load_binary_watermark(wmpath)
            for name,cfg in (('sequential',base),('orthogonal_joint',joint)):
                params=cfg.to_dict()
                marked,key=embed('proposal',host,wm,method_params=params)
                clean=extract(marked,key)
                if not np.array_equal(clean,wm):
                    raise AssertionError(f'failed clean decoding: {hostpath.name} {wmpath.name} {name}')
                times_e=[];times_x=[]
                for _ in range(args.repeats):
                    t=time.perf_counter_ns();out,k=embed('proposal',host,wm,method_params=params)
                    times_e.append((time.perf_counter_ns()-t)/1e6)
                    t=time.perf_counter_ns();extract(out,k)
                    times_x.append((time.perf_counter_ns()-t)/1e6)
                attacked=attack_jpeg(marked,70)
                rec=extract(attacked,key)
                row={'host':hostpath.name,'watermark':wmpath.name,'mode':name,
                     'psnr_db':psnr(host,marked),'clean_nc':nc(clean,wm),
                     'jpeg70_nc':nc(rec,wm), 'embed_median_ms':statistics.median(times_e),
                     'extract_median_ms':statistics.median(times_x),
                     'joint_overlap_blocks':key.params.get('joint_overlap_blocks',0)}
                rows.append(row)
                print(hostpath.name,wmpath.name,name,'PSNR',round(row['psnr_db'],4),
                      'JPEG70 NC',round(row['jpeg70_nc'],5),'Embed ms',round(row['embed_median_ms'],3),flush=True)
    paired={}
    for mode in ('sequential','orthogonal_joint'):
        data=[r for r in rows if r['mode']==mode]
        paired[mode]={metric:float(np.mean([r[metric] for r in data])) for metric in ('psnr_db','clean_nc','jpeg70_nc','embed_median_ms','extract_median_ms')}
    result={'protocol':{'hosts':len(hosts),'watermarks':len(watermarks),'bits':4096,
                       'attacks':['clean','jpeg70'],'num_repeats':args.repeats},
            'summary':paired,'rows':rows}
    args.output.parent.mkdir(exist_ok=True,parents=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT',json.dumps(paired,indent=2),flush=True)

if __name__=='__main__':main()
