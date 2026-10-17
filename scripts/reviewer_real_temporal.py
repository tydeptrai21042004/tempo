#!/usr/bin/env python3
"""Majority voting on genuinely varying VIDEO frames; NO synthetic fallback.

Every source frame is watermarked separately with the fixed 4096-bit message.
Independent JPEG errors and persistent geometric offsets are explicitly separated.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from reviewer_expansion_common import *
from reviewer_experiments import jpeg,warp


def load_video(path, count, size=512):
    p=Path(path)
    if not p.is_file():raise FileNotFoundError(p)
    cap=cv2.VideoCapture(str(p))
    if not cap.isOpened():raise ValueError(f'Unable to read video: {p}')
    frames=[]
    try:
        while len(frames)<count:
            ok, bgr=cap.read()
            if not ok:break
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            if rgb.shape[:2]!=(size,size):
                rgb=cv2.resize(rgb,(size,size),interpolation=cv2.INTER_AREA)
            frames.append(np.ascontiguousarray(rgb,dtype=np.uint8))
    finally:cap.release()
    if len(frames)<3:raise ValueError('Require >=3 actual decoded frames; cannot use synthetic placeholders')
    return frames


def run(a):
    init_cpu()
    out=Path(a.out_dir)
    frames=load_video(a.video, a.frames)
    marks=paths(a.watermark_dir,{'.png','.bmp'},a.max_watermarks)
    cfg=config(a.config)
    windows=(1,3) if a.quick else (1,3,5,9,15,31)
    windows=tuple(n for n in windows if n<=len(frames))
    rows=[]; votes=[]
    for wi,wp in enumerate(marks):
        wm=mark(wp)
        for attack in ('jpeg40','persistent_translation'):
            errors=[]; durations=[]
            for fi,host in enumerate(frames):
                marked,key,em=encode(host,wm,cfg,case_seed(a.seed,wi))
                altered=(jpeg(marked,40) if attack=='jpeg40' else warp(marked,dx=2,dy=1))
                decoded,meta,dt=decode(altered,key)
                current=(np.asarray(decoded).reshape(-1)!=wm.reshape(-1)).astype(np.uint8)
                errors.append(current)
                durations.append(em+dt)
                drift=(float(np.mean(np.abs(host.astype(float)-frames[fi-1].astype(float))))
                       if fi>0 else 0.0)
                rows.append({'video':Path(a.video).name,'watermark':wp.name,'frame':fi,
                    'attack':attack,'single_ber':float(current.mean()),
                    'single_nc':float(nc_binary(wm,decoded)),
                    'scene_change_mean_abs_rgb':drift,
                    'embed_plus_extract_ms':em+dt,'path':meta.get('path','')})
            e=np.stack(errors)
            # Error co-occurrence and adjacent-frame bit-error correlation are
            # measured directly; iid majority theory is not claimed as a bound.
            adjacent=float(np.mean((e[1:]==1)&(e[:-1]==1)))
            for n in windows:
                block=e[:n]
                voted=(np.mean(block,axis=0)>.5).astype(np.uint8)
                recovered=(np.bitwise_xor((wm.reshape(-1)>127).astype(np.uint8),voted)*255).reshape(wm.shape)
                votes.append({'video':Path(a.video).name,'watermark':wp.name,'attack':attack,
                    'voting_frames':n,'voted_ber':float(np.mean(voted)),
                    'voted_nc':float(nc_binary(wm,recovered)),
                    'mean_single_ber':float(np.mean(e[:n])),
                    'adjacent_error_cooccurrence_all_frames':adjacent,
                    'mean_frame_service_ms':float(np.mean(durations[:n])),
                    'decision_delay_frames':n-1})
    write_rows(out/'real_video_frame_errors.csv',rows)
    write_rows(out/'real_video_voting.csv',votes)
    write_json(out/'manifest.json',{'status':'complete','video_sha256':md5_file(a.video),
       'decoded_real_frames':len(frames),'windows':list(windows),
       'frame_resampling':'RGB resized to 512x512 for a valid 4096-bit payload',
       'limitations':['Temporal window starts at frame 0; no bitrate or frame-acquisition deadline guarantee',
          'Errors across neighboring video frames may be dependent','Voting requires one fixed watermark claim across frames'],
       'config_sha256':cfg.sha256(),**info()})


def parser():
    p=base_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument('--video',required=True,help='Real video .mp4 decoded by OpenCV; no synthetic fallback')
    p.add_argument('--frames',type=int,default=31)
    return p
if __name__=='__main__':run(parser().parse_args())
