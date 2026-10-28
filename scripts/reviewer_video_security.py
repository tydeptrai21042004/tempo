#!/usr/bin/env python3
"""Real-video temporal manipulation tests without synthetic substitute frames.

Attacks include codec+rotation, frame deletion/duplication/reordering and
correlated errors. Fixed framewise payload has no cryptographic time binding:
reordering alone must not be represented as an authentication failure detector.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from reviewer_expansion_common import *
from reviewer_real_temporal import load_video
from reviewer_experiments import jpeg,warp


def vote(frames, wm, window):
    z=(np.asarray(frames[:window])>127).astype(np.uint8)
    votes=(np.sum(z,axis=0)*2>=window).astype(np.uint8)*255
    return votes.reshape(wm.shape)


def main(a):
    init_cpu(); out=Path(a.out_dir)
    frames=load_video(a.video,a.frames)
    marks=paths(a.watermark_dir,{'.png','.bmp'},a.max_watermarks)
    cfg=config(a.config);rng=np.random.default_rng(a.seed)
    frame_rows=[]; vote_rows=[]
    for wi,wp in enumerate(marks):
        wm=mark(wp)
        decoded_by_attack={}
        for attack_name in ('clean','jpeg40','jpeg40_rotate3','persistent_offset'):
            rec=[]
            for fi,host in enumerate(frames):
                marked,key,ems=encode(host,wm,cfg,case_seed(a.seed,wi))
                if attack_name=='clean': altered=marked
                elif attack_name=='jpeg40':altered=jpeg(marked,40)
                elif attack_name=='jpeg40_rotate3':altered=warp(jpeg(marked,40),angle=3)
                else:altered=warp(marked,dx=2,dy=1)
                bits,meta,dms=decode(altered,key,sync=True)
                rec.append(np.asarray(bits,dtype=np.uint8))
                frame_rows.append({'video':Path(a.video).name,'watermark':wp.name,'attack':attack_name,
                    'frame':fi,'nc':float(nc_binary(wm,bits)),
                    'ber':float(1-bit_accuracy(wm,bits)),
                    'psnr_vs_marked_db':float(psnr(marked,altered)),
                    'embedding_ms':ems,'extraction_ms':dms,'sync_path':meta.get('path','')})
            decoded_by_attack[attack_name]=rec
        count=len(frames)
        # All index manipulations operate on actual decoded frames and decoded payloads.
        permuted=np.arange(count)
        rng.shuffle(permuted)
        keep_random=np.where(rng.random(count)>=.2)[0]
        if not len(keep_random):keep_random=np.array([0])
        burst_len=max(1,int(.2*count))
        start=int(rng.integers(0,count-burst_len+1))
        keep_burst=np.r_[np.arange(start),np.arange(start+burst_len,count)]
        if not len(keep_burst):keep_burst=np.array([0])
        index_policies={'original':np.arange(count),
                        'random_20pct_drop':keep_random,
                        'burst_20pct_drop':keep_burst,
                        'reverse_order':np.arange(count-1,-1,-1),
                        'random_reorder':permuted,
                        'duplicate_every_third':np.array([j for i in range(count) for j in ([i,i] if i%3==0 else [i])])}
        for name,decoded in decoded_by_attack.items():
            for policy,indices in index_policies.items():
                bits=[decoded[int(i)] for i in indices]
                for window in ((1,3) if a.quick else (1,3,5,9,15,31)):
                    if window>len(bits):continue
                    recovered=vote(bits,wm,window)
                    vote_rows.append({'video':Path(a.video).name,'watermark':wp.name,'attack':name,
                        'temporal_policy':policy,'source_frames':count,'available_frames':len(bits),
                        'window':window,'decision_delay_arrivals':window-1,
                        'voted_nc':float(nc_binary(wm,recovered)),
                        'voted_ber':float(1-bit_accuracy(wm,recovered)),
                        'mean_single_ber':float(np.mean([1-bit_accuracy(wm,b) for b in bits[:window]])),
                        'notice':'No temporal authenticity claim; identical payload across frames cannot prove frame order'})
    write_rows(out/'video_frame_security.csv',frame_rows)
    write_rows(out/'temporal_attack_voting.csv',vote_rows)
    write_json(out/'manifest.json',{'status':'complete','config_sha256':cfg.sha256(),
        'video_sha256':md5_file(a.video),'frames_decoded':len(frames),
        'frame_rows':len(frame_rows),'voting_rows':len(vote_rows),
        'limitations':['All frames are real video decoded by OpenCV and resized to 512x512.',
            'Frames carry an identical claim: this is robustness to delivery edits, not cryptographic temporal integrity.',
            'Arrival-index voting latency is not wall-clock end-to-end streaming latency.',
            'No learned attacks or forged labels are implied.'],**info()})

if __name__=='__main__':
    p=base_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument('--video',required=True)
    p.add_argument('--frames',type=int,default=31)
    main(p.parse_args())
