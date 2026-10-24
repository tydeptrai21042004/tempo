#!/usr/bin/env python3
"""Independent CPU 720p/JPEG70 FPS check, with the frozen selected TEMPO proposal.

Important: real video frames are decoded/preloaded before the timed producer.
The timed scope is causal watermark embed + JPEG70 attack + blind extraction.
No recalibration or change to selected embedding step is permitted.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from realtime_watermark.core import load_binary_watermark, save_csv
from realtime_watermark.proposals.jilp_qim.config import JILPQIMConfig
from realtime_watermark.registry import embed, extract
from realtime_watermark.stream_pipeline import _run_realtime, _process_packet, StreamPacket, summarize_stream_run, load_stream_frames


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def benchmark(*,video, watermark_dir, config_file, out_dir, frames=300, fps=30.0,
              queue_size=4, seed=2026):
    video=Path(video).resolve(); watermark_dir=Path(watermark_dir).resolve()
    config_file=Path(config_file).resolve();out_dir=Path(out_dir).resolve()
    if not video.is_file() or video.suffix.lower() not in ('.mp4','.avi','.mkv','.mov'):
        raise FileNotFoundError('Real-video input required. Synthetic-frame fallback is forbidden.')
    import cv2
    cap=cv2.VideoCapture(str(video))
    try:
        native_w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        native_h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if native_w<1280 or native_h<720:
        raise RuntimeError(f'Video native resolution {native_w}x{native_h} is not 720p')
    config=JILPQIMConfig.load(config_file)
    params=config.to_dict()
    config_sha=sha(config_file)
    marks=sorted(watermark_dir.glob('*.png'))
    if len(marks)!=2:
        raise RuntimeError(f'Exactly 2 genuine watermarks expected, found {len(marks)}')
    frames=int(frames)
    if frames<300:raise ValueError('Require at least 300 frames for the scientific FPS claim')
    if fps<=0:raise ValueError('Target fps must be positive')
    out_dir.mkdir(parents=True,exist_ok=True)
    real_frames=load_stream_frames(video,width=1280,height=720,max_frames=frames)
    if len(real_frames)<frames:
        raise RuntimeError(f'Real video only contains {len(real_frames)} readable frames (required {frames})')
    results=[]
    for watermark_path in marks:
        wm=load_binary_watermark(watermark_path,size=64)
        if wm.size!=4096:raise RuntimeError('Both watermarks must be 4096 bits')
        # Compile original embed/extract kernels BEFORE the timed run. Same frozen
        # step, seed, method, image shape, and 64x64 watermark; no calibration.
        marked,key=embed('proposal',real_frames[0],wm,seed=seed,repeat='auto',
                         step=float(config.step),method_params=params)
        _=extract(marked,key)
        # Pre-warm the ACTUAL JPEG70 route, including any conditional
        # synchronization JIT path not exercised by clean-image warm-up.
        warm = _process_packet(StreamPacket(0,0.0,real_frames[0]),
            method_id='proposal',watermark=wm,channel='jpeg70',
            seed=int(seed),step=float(config.step),method_params=params)
        if warm.get('recovered') is None:
            raise RuntimeError('JPEG70 warm-up did not produce a watermark')
        del warm, marked, key
        traces,drops,seconds=_run_realtime(
            real_frames,method_id='proposal',watermark=wm,channel='jpeg70',
            target_fps=float(fps),queue_size=int(queue_size),seed=int(seed),
            step=float(config.step),method_params=params)
        # Compact diagnostic trace; no 720p image arrays retained/exported.
        save_csv(out_dir/f"trace_{watermark_path.stem.replace('.', '_')}.csv", [
            {k:v for k,v in row.items() if k not in ('marked','attacked','recovered')}
            for row in traces])
        summary=summarize_stream_run(
            traces,method_id='proposal',channel='jpeg70',watermark=wm,
            target_fps=float(fps),resolution=(1280,720),pace_mode='realtime',
            queue_size=int(queue_size),queue_drops=drops,total_duration_s=seconds,
            calibrated_step=float(config.step),calibration_psnr=None)
        summary['watermark']=watermark_path.name
        summary['real_video']=True
        summary['frozen_proposal']=True
        summary['proposal_config_sha256']=config_sha
        summary['watermark_sha256']=sha(watermark_path)
        results.append(summary)
        print(watermark_path.name, 'FPS=%.3f'%summary['effective_output_fps'],
              'service_p95_ms=%.3f'%summary['service_ms_p95'],
              'on_time=%.3f'%summary['on_time_completion_ratio'],
              'drops=',drops,flush=True)
    manifest={'status':'MEASURED', 'video':str(video),'video_sha256':sha(video), 'source_native_resolution':[native_w,native_h],
       'scope':'predecoded real-video frames: embed + JPEG70 encode/decode + blind extraction; excludes video input decoding and disk output',
       'config':str(config_file),'config_sha256':config_sha,
       'frozen_config_step':float(config.step), 'fps_target':float(fps),
       'frames_per_watermark':frames,'cases':results,'cpu_threads':os.environ.get('JILP_NUM_THREADS','default')}
    path=out_dir/'frozen_fps_manifest.json'
    path.write_text(json.dumps(manifest,indent=2,default=str)+'\n',encoding='utf8')
    save_csv(out_dir/'frozen_fps_cases.csv',results)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',required=True,type=Path)
    p.add_argument('--watermark-dir',required=True,type=Path)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--out-dir',required=True,type=Path)
    p.add_argument('--frames',type=int,default=300)
    p.add_argument('--fps',type=float,default=30.0)
    p.add_argument('--queue-size',type=int,default=4)
    p.add_argument('--seed',type=int,default=2026)
    a=p.parse_args()
    benchmark(video=a.video,watermark_dir=a.watermark_dir,config_file=a.config,
       out_dir=a.out_dir,frames=a.frames,fps=a.fps,queue_size=a.queue_size,seed=a.seed)

if __name__=='__main__':main()
