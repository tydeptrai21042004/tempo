#!/usr/bin/env python3
"""Time frozen TEMPO embed+blind extraction over 300 REAL native-720p JPEG70 frames.

JPEG70 is actually applied but codec, decoder, quality metrics and disk I/O
are EXCLUDED from the algorithm-only timer. This is NOT end-to-end FPS.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,time
from pathlib import Path
import cv2,numpy as np
from realtime_watermark.registry import embed
from realtime_watermark.core import load_binary_watermark
from realtime_watermark.proposals.jilp_qim.config import JILPQIMConfig
from realtime_watermark.proposals.jilp_qim.method import extract_jilp_qim
from realtime_watermark.stream_pipeline import apply_stream_channel


def benchmark(video:Path,mark_dir:Path,config_path:Path,out_dir:Path,frames:int=300):
    if frames!=300:
        raise ValueError('The comparable benchmark always evaluates exactly 300 frames')
    config=JILPQIMConfig.load(config_path);params=config.to_dict()
    wm_files=sorted(mark_dir.glob('*.png'))
    if len(wm_files)!=2:raise RuntimeError('Two watermark files are required')
    out_dir.mkdir(parents=True,exist_ok=True)
    manifest={'scope':'algorithm_only_embed_plus_blind_extraction',
        'excluded_from_algorithm_timer':['JPEG70 codec','video decode','PSNR','NC','BER','queue','file I/O'],
        'included_in_algorithm_timer':['per-frame embedding and scheduling','blind extraction including pilot recovery'],
        'not_a_causal_stream_throughput_claim':True,
        'source_video':str(video),'source_video_sha256':hashlib.sha256(video.read_bytes()).hexdigest(),
        'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'payload_bits':4096,'frames_per_watermark':300,'channel':'jpeg70','cases':[], 'status':'INCOMPLETE'}
    for wp in wm_files:
        wm=load_binary_watermark(wp,size=64)
        if wm.shape!=(64,64) or wm.size!=4096:
            raise ValueError(f'Expected 64x64 watermark: {wp}')
        cap=cv2.VideoCapture(str(video))
        if not cap.isOpened():raise RuntimeError('Cannot read real video '+str(video))
        src_size=[int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        if src_size[0]<1280 or src_size[1]<720:
            cap.release();raise RuntimeError('720p source required')
        trace=[]
        try:
            for i in range(frames+2):
                ok,frame=cap.read()
                if not ok:break
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                if rgb.shape[:2]!=(720,1280):
                    rgb=cv2.resize(rgb,(1280,720),interpolation=cv2.INTER_AREA)
                t0=time.perf_counter_ns()
                marked,key=embed('proposal',rgb,wm,seed=2026,repeat='auto',
                                 step=float(config.step),method_params=params)
                t1=time.perf_counter_ns()
                attacked=apply_stream_channel(marked,'jpeg70',frame_id=i,seed=2026)
                t2=time.perf_counter_ns()
                recovered,meta=extract_jilp_qim(attacked,key,sync=True,return_meta=True)
                t3=time.perf_counter_ns()
                if i<2:continue
                if recovered.shape!=wm.shape:raise RuntimeError('Detector output shape mismatch')
                errors=int(np.count_nonzero((recovered>127)!=(wm>127)))
                embed_ms=(t1-t0)*1e-6;codec_ms=(t2-t1)*1e-6;extract_ms=(t3-t2)*1e-6
                trace.append({'frame':i-2,'embed_ms':embed_ms,'jpeg70_codec_ms_excluded':codec_ms,
                    'blind_extract_ms':extract_ms,'algorithm_ms':embed_ms+extract_ms,
                    'ber':errors/4096,'extraction_path':meta.get('path','')})
        finally:
            cap.release()
        if len(trace)!=300:
            raise RuntimeError(f'{wp.name}: only {len(trace)}/300 genuine frames; no extrapolation')
        path=out_dir/('algorithm_only_'+wp.stem.replace('.','_')+'.csv')
        with path.open('w',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,list(trace[0]));writer.writeheader();writer.writerows(trace)
        a=np.array([r['algorithm_ms'] for r in trace]);e=np.array([r['embed_ms'] for r in trace]);x=np.array([r['blind_extract_ms'] for r in trace]);c=np.array([r['jpeg70_codec_ms_excluded'] for r in trace])
        manifest['cases'].append({'watermark':wp.name,'frames':len(trace),'resolution':'1280x720',
            'mean_embed_ms':float(e.mean()),'mean_extract_ms':float(x.mean()),
            'mean_algorithm_ms':float(a.mean()),'p95_algorithm_ms':float(np.percentile(a,95)),
            'algorithm_equivalent_fps':float(1000/a.mean()),
            'jpeg70_codec_ms_mean_excluded':float(c.mean()),
            'mean_ber':float(np.mean([r['ber'] for r in trace]))})
        print('ALGORITHM-ONLY',manifest['cases'][-1],flush=True)
    manifest['status']='COMPLETE' if len(manifest['cases'])==2 else 'INCOMPLETE'
    (out_dir/'algorithm_only_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if manifest['status']!='COMPLETE':raise RuntimeError('Incomplete algorithm-only video evidence')
    return manifest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--video',type=Path,required=True)
    ap.add_argument('--watermark-dir',type=Path,required=True)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--frames',type=int,default=300)
    a=ap.parse_args();benchmark(a.video,a.watermark_dir,a.config,a.out_dir,a.frames)
if __name__=='__main__':main()
