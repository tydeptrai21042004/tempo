#!/usr/bin/env python3
"""Claim-conditioned watermark verification with development-only threshold calibration.

This is a presence verifier for a KNOWN 4096-bit watermark; extraction without
knowing what to verify is not a binary presence detector. It is not an
independent cryptographic authentication primitive.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from realtime_watermark.core import load_binary_watermark, bit_accuracy
from realtime_watermark.proposals.jilp_qim.config import JILPQIMConfig
from realtime_watermark.proposals.jilp_qim.method import embed_proposal, extract_jilp_qim


def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r)) if rows else ['status']
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f, fieldnames=fields);w.writeheader();w.writerows(rows)


def observed_score(received, watermark, key, sync):
    bits=extract_jilp_qim(received,key,sync=sync)
    return float(bit_accuracy(watermark,bits))


def single_group(host_dir,wm_dir,cfg,seed,neg_per_pair,sync,group):
    result=[]
    hosts=sorted(Path(host_dir).glob('*.bmp'))
    watermarks=sorted(Path(wm_dir).glob('*.png'))
    for widx, wp in enumerate(watermarks):
        wm=load_binary_watermark(wp,size=64)
        if wm.size != 4096: raise ValueError('The main verifier requires a 4096-bit payload')
        for hp in hosts:
            host=np.asarray(Image.open(hp).convert('RGB'))
            encoded,key=embed_proposal(host,wm,seed=seed+104729*widx,method_params=cfg.to_dict())
            params={**key.params, 'host_shape':tuple(host.shape), 'watermark_shape':tuple(wm.shape)}
            checks=[('marked_correct_key',True,encoded,params),
                    ('unmarked_correct_key',False,host,params)]
            for j in range(neg_per_pair):
                wrong=dict(params);wrong['seed']=int(params['seed'])+7919+1009*j
                checks.append((f'wrong_key_{j}',False,encoded,wrong))
            for label,positive,received,k in checks:
                score=observed_score(received,wm,k,sync)
                result.append({'group':group,'host':hp.name,'watermark':wp.name,
                               'trial':label,'is_positive':positive,'score_bit_accuracy':score,
                               'payload_bits':wm.size,'sync_used':sync,'status':'ok'})
    return result


def roc_from_test(rows):
    positives=np.asarray([r['score_bit_accuracy'] for r in rows if r['is_positive']],float)
    negatives=np.asarray([r['score_bit_accuracy'] for r in rows if not r['is_positive']],float)
    if not len(positives) or not len(negatives): raise ValueError('Both positive and negative samples required')
    thresholds=np.r_[np.inf,np.linspace(1.,0.,1001),-np.inf]
    roc=[{'threshold':float(t),'tpr':float(np.mean(positives>=t)),
          'fpr':float(np.mean(negatives>=t))} for t in thresholds]
    auc=float(np.mean((positives[:,None]>negatives[None,:])+0.5*(positives[:,None]==negatives[None,:])))
    return roc,auc


def calibrate(negative_scores,target_fpr=0.05):
    # False alarms at threshold are counted using >=, so use nextafter for strictness.
    z=np.asarray(negative_scores,float)
    if not z.size: raise ValueError('No development negatives')
    upper=float(np.quantile(z,1-target_fpr,method='higher'))
    return float(np.nextafter(upper,np.inf))


def run(opt):
    out=Path(opt.out_dir);out.mkdir(parents=True,exist_ok=True)
    cfg=JILPQIMConfig.load(opt.config)
    dev=single_group(opt.development_dir,opt.watermark_dir,cfg,opt.seed,opt.negative_keys,opt.sync,'development')
    test=single_group(opt.test_dir,opt.watermark_dir,cfg,opt.seed,opt.negative_keys,opt.sync,'heldout_test')
    dev_neg=[r['score_bit_accuracy'] for r in dev if not r['is_positive']]
    threshold=calibrate(dev_neg,opt.target_fpr)
    roc,auc=roc_from_test(test)
    positives=[r for r in test if r['is_positive']];negatives=[r for r in test if not r['is_positive']]
    tpr=float(np.mean([r['score_bit_accuracy']>=threshold for r in positives]))
    fpr=float(np.mean([r['score_bit_accuracy']>=threshold for r in negatives]))
    write_csv(out/'development_scores.csv',dev)
    write_csv(out/'heldout_scores.csv',test)
    write_csv(out/'heldout_roc.csv',roc)
    report={'configuration_sha256':cfg.sha256(),'seed':opt.seed,'decision_rule':'known-message bit accuracy >= threshold',
            'threshold_chosen_using':'development negatives only','target_development_fpr':opt.target_fpr,
            'calibrated_threshold':threshold,'heldout_tpr':tpr,'heldout_fpr':fpr,'heldout_auc_exploratory':auc,
            'positive_samples':len(positives),'negative_samples':len(negatives),
            'development_host_count':len(set(r['host'] for r in dev)),
            'heldout_host_count':len(set(r['host'] for r in test)),
            'limitations':['known-message verification, not arbitrary watermark discovery',
                           'trial observations on repeated hosts and keys are not independent population samples',
                           'the finite-sample FPR cannot certify a very low operating false-positive probability',
                           'wrong-key trials alter the PRNG seed; they do not prove secrecy or key entropy'],
            'status':'complete'}
    (out/'detection_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({'heldout_auc':auc,'tpr':tpr,'fpr':fpr,'test_rows':len(test)},indent=2))
    return report


def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--development-dir',required=True)
    p.add_argument('--test-dir',required=True)
    p.add_argument('--watermark-dir',required=True)
    p.add_argument('--config',required=True)
    p.add_argument('--out-dir',required=True)
    p.add_argument('--negative-keys',type=int,default=5)
    p.add_argument('--target-fpr',type=float,default=.05)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--sync',action='store_true',help='Use synchronization when evaluating detection; costly for negatives')
    return p

if __name__=='__main__':run(parser().parse_args())
