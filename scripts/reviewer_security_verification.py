#!/usr/bin/env python3
"""Security verification: dev-calibrated known-message threshold + held-out attacks.

Separate raw and synchronization-aware protocols; every threshold is calibrated
using only its own development negatives. Never infer cryptographic authenticity.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import numpy as np
from reviewer_expansion_common import *
from reviewer_experiments import jpeg, warp
from realtime_watermark.proposals.jilp_qim.method import extract_jilp_qim
try:
    from scipy.stats import beta
except ImportError:
    beta=None


def wilson_exact(k,n,confidence=.95):
    if not n:return [None,None]
    alpha=1-confidence
    if beta is None:return [None,None]
    return [float(beta.ppf(alpha/2,k,n-k+1)) if k else 0.0,
            float(beta.ppf(1-alpha/2,k+1,n-k)) if k<n else 1.0]


def roc_auc(rows):
    pos=np.array([x['score'] for x in rows if x['is_positive']],float)
    neg=np.array([x['score'] for x in rows if not x['is_positive']],float)
    if not pos.size or not neg.size:return None
    return float(np.mean((pos[:,None]>neg[None,:])+0.5*(pos[:,None]==neg[None,:])))


def group(folder, marks, cfg, seed, negative_keys, *, sync, role, quick):
    hosts=paths(folder,{'.bmp'},1 if quick else 0)
    host_images=[load_host(p) for p in hosts]
    output=[]
    for wi,wp in enumerate(marks):
        wm=mark(wp)
        for hi,hp in enumerate(hosts):
            original=host_images[hi]
            marked,key,_=encode(original,wm,cfg,case_seed(seed,wi))
            params={'params':dict(key.params),'host_shape':tuple(original.shape),
                    'watermark_shape':tuple(wm.shape),'seed':int(key.seed),
                    'method_id':key.method_id,'step':key.step,
                    'arnold_iter':key.arnold_iter,'arnold_period':key.arnold_period}
            trials=[('marked_valid',True,marked,wm,params),
                    ('unmarked',False,original,wm,params)]
            if not quick and len(hosts)>1:
                other=host_images[(hi+1)%len(hosts)]
                if other.shape==original.shape:
                    trials.append(('unrelated_host',False,other,wm,params))
            for j in range(negative_keys):
                other_seed=int(params['seed'])+7919+1009*j
                wrong={**params,'seed':other_seed,'params':{**params['params'],'seed':other_seed}}
                trials.append(('marked_wrong_key',False,marked,wm,wrong))
                stable=int.from_bytes(hashlib.sha256(f'{seed}|{hp.name}|{wp.name}|{j}'.encode()).digest()[:8],'big')
                rng=np.random.default_rng(stable)
                bogus=(rng.integers(0,2,wm.shape,dtype=np.uint8)*255).astype(np.uint8)
                trials.append(('marked_wrong_message',False,marked,bogus,params))
            if role=='heldout':
                trials.extend([('marked_jpeg40',True,jpeg(marked,40),wm,params),
                               ('marked_rotation2p5',True,warp(marked,angle=2.5),wm,params),
                               ('unmarked_jpeg40',False,jpeg(original,40),wm,params)])
            for name,valid,received,claim,k in trials:
                decoded,meta,ms=decode(received,k,sync=sync)
                output.append({'split':role,'host':hp.name,'watermark':wp.name,
                               'condition':name,'is_positive':valid,'score':float(bit_accuracy(claim,decoded)),
                               'ber_against_claim':float(1-bit_accuracy(claim,decoded)),
                               'nc_against_claim':float(nc_binary(claim,decoded)),
                               'sync':sync,'detector_path':str(meta.get('path','')),
                               'extract_ms':ms,'config_sha256':cfg.sha256()})
    return output


def main(a):
    init_cpu();out=Path(a.out_dir);cfg=config(a.config)
    marks=paths(a.watermark_dir,{'.png'},1 if a.quick else 0)
    development=paths(a.development_dir,{'.bmp'},0)
    heldout=paths(a.test_dir,{'.bmp'},0)
    if {md5_file(p) for p in development}&{md5_file(p) for p in heldout}:
        raise ValueError('Development and held-out sets contain duplicate bytes')
    modes=(False,True) if a.both_modes else (a.sync,)
    summaries=[]
    for sync in modes:
        dev=group(a.development_dir,marks,cfg,a.seed,2 if a.quick else a.negative_keys,
                  sync=sync,role='development',quick=a.quick)
        test=group(a.test_dir,marks,cfg,a.seed,2 if a.quick else a.negative_keys,
                   sync=sync,role='heldout',quick=a.quick)
        dev_neg=np.array([r['score'] for r in dev if not r['is_positive']],float)
        if not dev_neg.size:raise ValueError('No development negative controls')
        threshold=float(np.nextafter(np.quantile(dev_neg,1-a.target_fpr,method='higher'),np.inf))
        for row in dev+test:row.update({'threshold':threshold,'accepted':bool(row['score']>=threshold)})
        prefix='sync' if sync else 'direct'
        write_rows(out/f'{prefix}_development_scores.csv',dev)
        write_rows(out/f'{prefix}_heldout_scores.csv',test)
        for condition in sorted(set(r['condition'] for r in test)):
            sample=[r for r in test if r['condition']==condition]
            successes=sum(r['accepted'] for r in sample)
            summaries.append({'mode':prefix,'condition':condition,'positive_truth':sample[0]['is_positive'],
                'n':len(sample),'accepted':successes,'acceptance_rate':successes/len(sample),
                'exact_binomial_95_ci_conditional_on_independence':wilson_exact(successes,len(sample)),
                'calibrated_threshold':threshold,'auc_all_heldout_conditions':roc_auc(test)})
    write_rows(out/'verification_by_condition.csv',summaries)
    write_json(out/'manifest.json',{'status':'complete','config_sha256':cfg.sha256(),
       'development_sha256':{p.name:md5_file(p) for p in development},
       'heldout_sha256':{p.name:md5_file(p) for p in heldout},
       'modes':['sync' if x else 'direct' for x in modes],
       'target_development_fpr':a.target_fpr,
       'claim':'known-message verification only; not cryptographic proof of content authenticity',
       'statistical_caveats':['Repeated keys and attacks on the same host are correlated: row-level exact-binomial intervals are descriptive, not independent population guarantees.',
          'Threshold is calibrated on development negative controls only, separately for each detector mode.',
          'A positive label refers to the genuinely marked host even when an attack destroys recoverability.',
          'Detector search may increase false alarms; sync and direct scores must not share thresholds.'],**info()})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--development-dir',required=True,type=Path)
    p.add_argument('--test-dir',required=True,type=Path)
    p.add_argument('--watermark-dir',required=True,type=Path)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--out-dir',required=True,type=Path)
    p.add_argument('--target-fpr',type=float,default=.05)
    p.add_argument('--negative-keys',type=int,default=16)
    p.add_argument('--sync',action='store_true')
    p.add_argument('--both-modes',action='store_true',help='More expensive: calibrate direct AND sync modes independently')
    p.add_argument('--quick',action='store_true')
    p.add_argument('--seed',type=int,default=2026)
    main(p.parse_args())
