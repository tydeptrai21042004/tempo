#!/usr/bin/env python3
"""Temporal majority voting: real decoded-frame outcomes and iid reference curve.

No ECC, pilot, or proposal-method change. The iid curve is illustrative ONLY;
attacked video-frame bit errors can be correlated.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np
from PIL import Image
from realtime_watermark.core import load_binary_watermark,nc_binary,bit_accuracy
from realtime_watermark.proposals.jilp_qim.config import JILPQIMConfig
from realtime_watermark.proposals.jilp_qim.method import embed_proposal,extract_jilp_qim
from reviewer_experiments import jpeg,warp,csv_write


def iid_majority_ber(p,t):
    # T odd; independent Bernoulli(p) reference, not an empirical guarantee.
    return sum(math.comb(t,k)*(p**k)*((1-p)**(t-k)) for k in range(t//2+1,t+1))


def run(a):
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    cfg=JILPQIMConfig.load(a.config)
    hosts=sorted(Path(a.host_dir).glob('*.bmp'))[:a.max_hosts]
    wms=sorted(Path(a.watermark_dir).glob('*.png'))[:a.max_watermarks]
    lengths=(1,3,5,9,15,31)
    rows=[];frame_rows=[]
    for wp in wms:
        wm=load_binary_watermark(wp,size=64)
        for hp in hosts:
            host=np.asarray(Image.open(hp).convert('RGB'))
            marked,key=embed_proposal(host,wm,seed=a.seed,method_params=cfg.to_dict())
            for channel in ('jpeg','correlated_translation'):
                for rep in range(a.repetitions):
                    rng=np.random.default_rng(a.seed+rep*991+int.from_bytes(hp.name.encode()[:3].ljust(3,b'0'),'big'))
                    decoded=[]
                    for t in range(max(lengths)):
                        if channel=='jpeg':
                            # Quality fluctuates by frame; naturally correlated content remains.
                            received=jpeg(marked,int(rng.integers(30,81)))
                        else:
                            # Fixed whole-sequence displacement: deliberately correlated errors.
                            received=warp(marked,dy=2,dx=1)
                        bits=extract_jilp_qim(received,key)
                        decoded.append(np.asarray(bits,dtype=np.uint8).reshape(-1))
                        frame_rows.append({'host':hp.name,'watermark':wp.name,'channel':channel,
                                           'repeat':rep,'frame':t+1,'ber':1-bit_accuracy(wm,bits)})
                    matrix=np.stack(decoded)
                    single_p=float(np.mean(matrix!=wm.reshape(1,-1)))
                    for n in lengths:
                        voted=(np.sum(matrix[:n],axis=0)*2 >= n).astype(np.uint8).reshape(wm.shape)
                        rows.append({'host':hp.name,'watermark':wp.name,'channel':channel,'repeat':rep,
                                     'frame_count':n,'empirical_ber':float(1-bit_accuracy(wm,voted)),
                                     'empirical_nc':float(nc_binary(wm,voted)),
                                     'mean_single_frame_ber':single_p,
                                     'iid_reference_ber':float(iid_majority_ber(single_p,n))})
    csv_write(out/'temporal_vote_results.csv',rows)
    csv_write(out/'per_frame_errors.csv',frame_rows)
    (out/'temporal_manifest.json').write_text(json.dumps({'results':len(rows),'frame_rows':len(frame_rows),
        'tested_lengths':list(lengths),'config_hash':cfg.sha256(),
        'note':'iid reference is not a bound under correlated video-frame errors; same watermark repeated on every frame'},indent=2))
    print(f'Wrote {len(rows)} temporal observations')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--host-dir',required=True);p.add_argument('--watermark-dir',required=True)
    p.add_argument('--config',required=True);p.add_argument('--out-dir',required=True)
    p.add_argument('--max-hosts',type=int,default=1);p.add_argument('--max-watermarks',type=int,default=2)
    p.add_argument('--repetitions',type=int,default=3);p.add_argument('--seed',type=int,default=2026)
    run(p.parse_args())
