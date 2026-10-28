#!/usr/bin/env python3
"""Exploratory TEMPO ownership/removal/overwrite security experiments.

No cryptographic authentication claim is made. All attacks affect *copies* of
marked images, never the embedding method. Verification threshold calibration
is performed by reviewer_detection_protocol.py on disjoint development hosts.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from reviewer_expansion_common import *
from reviewer_experiments import attenuation, jpeg, warp
from realtime_watermark.proposals.jilp_qim.method import extract_jilp_qim
from skimage.metrics import structural_similarity as color_ssim


def evaluate(received, original, target_wm, key, *, sync):
    recovered, meta, elapsed = decode(received, key, sync=sync)
    return {'bit_accuracy':float(bit_accuracy(target_wm,recovered)),
            'nc':float(nc_binary(target_wm,recovered)),
            'ber':float(1-bit_accuracy(target_wm,recovered)),
            'psnr_vs_marked_db':float(psnr(original,received)),
            'ssim_vs_marked':float(color_ssim(original,received,channel_axis=-1,data_range=255)),
            'extract_ms':elapsed,'detector_path':str(meta.get('path',''))}


def main(a):
    init_cpu()
    hosts, marks=dataset(a)
    cfg=config(a.config)
    out=Path(a.out_dir)
    rng=np.random.default_rng(a.seed)
    rows=[]
    for hp in hosts:
        host=load_host(hp)
        for wi,wp in enumerate(marks):
            wm=mark(wp)
            marked,key,embed_ms=encode(host,wm,cfg,case_seed(a.seed,wi))
            params={'params':dict(key.params),'host_shape':tuple(host.shape),
                    'watermark_shape':tuple(wm.shape),'seed':int(key.seed),
                    'method_id':key.method_id,'step':key.step,
                    'arnold_iter':key.arnold_iter,'arnold_period':key.arnold_period}
            def add(kind, altered, claim, key_to_use, *, sync=False, attack_strength='', note=''):
                metrics=evaluate(altered,marked,claim,key_to_use,sync=sync)
                rows.append({'host':hp.name,'watermark':wp.name,'experiment':kind,
                    'strength':attack_strength,'sync':bool(sync),'note':note,
                    'clean_psnr_db':float(psnr(host,marked)),
                    'attack_psnr_vs_host_db':float(psnr(host,altered)),
                    'embed_ms':embed_ms,'config_sha256':cfg.sha256(),**metrics})
            add('marked_correct_claim',marked,wm,params,sync=True)
            add('unmarked_correct_claim',host,wm,params)
            for ni in range(2 if a.quick else a.negative_keys):
                wrong={**params,'params':dict(params['params'])}
                wrong['seed']=int(params['seed']) + 7919 + 1009*ni
                wrong['params']['seed']=wrong['seed']
                add('wrong_key',marked,wm,wrong,attack_strength=ni)
                # Different random expected payloads: not equivalent to changing the key.
                fake=np.where(rng.integers(0,2,wm.shape,dtype=np.uint8)>0,255,0).astype(np.uint8)
                add('wrong_message',marked,fake,params,attack_strength=ni)
            if len(hosts)>1:
                other=hosts[(hosts.index(hp)+1)%len(hosts)]
                other_host=load_host(other)
                if other_host.shape==host.shape:
                    add('unrelated_host',other_host,wm,params,note=other.name)
                    transplanted=np.clip(other_host.astype(np.int16)+marked.astype(np.int16)-host.astype(np.int16),0,255).astype(np.uint8)
                    add('residual_transplant',transplanted,wm,params,note=f'transplanted into {other.name}')
            if len(marks)>1:
                other_wm=mark(marks[(wi+1)%len(marks)])
                # Deliberately re-embed another claim; this is not a blind passive attack.
                overwrite,new_key,_=encode(marked,other_wm,cfg,case_seed(a.seed,wi+1))
                add('overwrite_original_claim',overwrite,wm,params,sync=True)
                other_params={'params':dict(new_key.params),'host_shape':tuple(host.shape),
                    'watermark_shape':tuple(other_wm.shape),'seed':int(new_key.seed),
                    'method_id':new_key.method_id,'step':new_key.step,
                    'arnold_iter':new_key.arnold_iter,'arnold_period':new_key.arnold_period}
                add('overwrite_new_claim',overwrite,other_wm,other_params,sync=True)
                separately_marked,_,_=encode(host,other_wm,cfg,case_seed(a.seed,wi+1))
                colluded=np.clip(np.rint((marked.astype(np.float64)+separately_marked)/2),0,255).astype(np.uint8)
                add('two_copy_collusion',colluded,wm,params,sync=True,
                    note='Average of two separately marked copies, not a many-user collusion bound')
            strengths=(0,.5,1) if a.quick else (0,.1,.25,.5,.75,1)
            for target in ('payload','pilot','both'):
                for strength in strengths:
                    altered=marked.copy()
                    if target in ('payload','both'):
                        altered=attenuation(altered,float(key.params['eta']),'payload',strength)
                    if target in ('pilot','both'):
                        altered=attenuation(altered,float(key.params['eta']),'pilot',strength)
                    add('public_carrier_suppression_'+target,altered,wm,params,sync=True,
                        attack_strength=strength,note='Key-independent public subspace attenuation')
            for quality in ((40,) if a.quick else (90,70,50,30)):
                add('jpeg_removal',jpeg(marked,quality),wm,params,sync=True,attack_strength=quality)
            for angle in ((3.,) if a.quick else (1.,3.,5.,10.)):
                add('geometric_rotation',warp(marked,angle=angle),wm,params,sync=True,attack_strength=angle)
    write_rows(out/'security_cases.csv',rows)
    summaries=[]
    for name in sorted(set(r['experiment'] for r in rows)):
        sample=[r for r in rows if r['experiment']==name]
        summaries.append({'experiment':name,'observations':len(sample),
            'mean_nc':float(np.mean([r['nc'] for r in sample])),
            'mean_ber':float(np.mean([r['ber'] for r in sample])),
            'mean_attack_psnr_db':float(np.mean([r['psnr_vs_marked_db'] for r in sample if np.isfinite(r['psnr_vs_marked_db'])])) if any(np.isfinite(r['psnr_vs_marked_db']) for r in sample) else 'infinite for identity'})
    write_rows(out/'security_summary.csv',summaries)
    write_json(out/'manifest.json',{'status':'complete','rows':len(rows),'config_sha256':cfg.sha256(),
        'hosts':{p.name:md5_file(p) for p in hosts},'watermarks':{p.name:md5_file(p) for p in marks},
        'notes':['Exploratory score distributions; no calibrated verification FPR is implied.',
                 'The dedicated detection stage calibrates a threshold on disjoint development images.',
                 'PSNR is against the actual marked image. Identity cases have infinite PSNR.',
                 'The watermark design offers no cryptographic authentication guarantee.',
                 'The two-copy collusion case does not establish resistance to general collusion.',
                 'Carrier attenuation assumes the public transform family, not knowledge of secret keyed positions.'],**info()})
    print(f'Wrote {len(rows)} security case rows')

if __name__=='__main__':
    p=base_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument('--negative-keys',type=int,default=16)
    main(p.parse_args())
