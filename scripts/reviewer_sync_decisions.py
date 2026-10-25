#!/usr/bin/env python3
"""Observed synchronization decisions with raw and forced-search references.

Forced search uses the identical detector with quick_accept > 1 and changes
only an evaluation argument, not the implementation or the published method.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from reviewer_expansion_common import *
from reviewer_experiments import jpeg, warp

from realtime_watermark.attack_manifest import reviewer_attacks, reviewer_manifest_summary
ATTACKS = dict(reviewer_attacks())


def run(a):
    init_cpu()
    out=Path(a.out_dir)
    hosts,marks=dataset(a)
    cfg=config(a.config)
    choices=('clean','jpeg70','rotate_2p5','subpixel_translate_0p5_0p5') if a.quick else ATTACKS
    rows=[]
    for hp in hosts:
        im=load_host(hp)
        for wi,wp in enumerate(marks):
            wm=mark(wp)
            marked,key,_=encode(im,wm,cfg,case_seed(a.seed,wi))
            for label in choices:
                received=ATTACKS[label](marked)
                raw,_,raw_ms=decode(received,key,sync=False)
                ordinary,meta,sync_ms=decode(received,key,sync=True)
                start=time.perf_counter_ns()
                forced,mf=extract_jilp_qim(received,key,sync=True,
                                          quick_accept=1.1,return_meta=True)
                force_ms=(time.perf_counter_ns()-start)/1e6
                nr=float(nc_binary(wm,raw)); ns=float(nc_binary(wm,ordinary)); nf=float(nc_binary(wm,forced))
                triggered=not str(meta.get('path','')).startswith('raw_fast')
                helpful=nf>nr+a.min_gain
                rows.append({'host':hp.name,'watermark':wp.name,'attack':label,
                    'raw_nc':nr,'normal_nc':ns,'forced_search_nc':nf,
                    'gain_normal':ns-nr,'gain_forced':nf-nr,
                    'search_triggered':triggered,'forced_search_helpful':helpful,
                    'unnecessary_search':triggered and not helpful,
                    'missed_helpful_search':not triggered and helpful,
                    'recovery_harmed_nc':ns<nr-a.min_gain,
                    'raw_ms':raw_ms,'normal_ms':sync_ms,'forced_ms':force_ms,
                    'normal_path':meta.get('path',''),'forced_path':mf.get('path',''),
                    'normal_candidate':str(meta.get('candidate','')),
                    'forced_candidate':str(mf.get('candidate',''))})
    write_rows(out/'sync_decision_cases.csv',rows)
    summary=[]
    for label in choices:
        group=[r for r in rows if r['attack']==label]
        summary.append({'attack':label,'n_cases':len(group),
            'search_trigger_fraction':float(np.mean([r['search_triggered'] for r in group])),
            'unnecessary_search_fraction':float(np.mean([r['unnecessary_search'] for r in group])),
            'missed_helpful_search_fraction':float(np.mean([r['missed_helpful_search'] for r in group])),
            'sync_harm_fraction':float(np.mean([r['recovery_harmed_nc'] for r in group])),
            'mean_gain_nc':float(np.mean([r['gain_normal'] for r in group]))})
    write_rows(out/'sync_decision_summary.csv',summary)
    write_json(out/'manifest.json',{'status':'complete','cases':len(rows),
      'helpful_definition':f'forced search NC exceeds raw by >{a.min_gain}',
      'caveat':'Forced search is a diagnostic counterfactual and not the deployed default; no general geometric invariance claim',
      'hosts':{p.name:md5_file(p) for p in hosts},'attack_manifest_sha256':reviewer_manifest_summary()['sha256'],**info()})


def parser():
    p=base_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument('--min-gain',type=float,default=.01)
    return p
if __name__=='__main__':run(parser().parse_args())
