#!/usr/bin/env python3
"""Real-video 720p wall-clock follow-up for frozen FPS ablation finalists.

Requires a genuine input video. Each variant/watermark is a separate causal
stream experiment using the repository's existing scheduler and JPEG channel.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,help='Real (non-synthetic) video file; preferably native 1280x720')
    p.add_argument('--watermark-dir',required=True)
    p.add_argument('--configs-dir',required=True,help='configs folder from proposal_fps_ablation output')
    p.add_argument('--variants',default='baseline,gate_080,gate_065,joint_center,gate065_pilot064')
    p.add_argument('--out-dir',default='reports/fps_ablation_video')
    p.add_argument('--frames',type=int,default=180)
    p.add_argument('--fps',default='30,60')
    p.add_argument('--resolution',default='1280x720')
    args=p.parse_args()
    video=Path(args.input)
    if not video.is_file():raise FileNotFoundError(f'Video required (no synthetic fallback): {video}')
    marks=sorted(Path(args.watermark_dir).glob('*.png'))
    if len(marks)!=2:raise RuntimeError('Exactly two PNG watermark files required')
    variants=[v.strip() for v in args.variants.split(',') if v.strip()]
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    commands=[]
    for variant in variants:
        config=Path(args.configs_dir)/f'{variant}.json'
        if not config.is_file():raise FileNotFoundError(f'Missing frozen variant config: {config}')
        for watermark in marks:
            label=watermark.stem.replace('.','_')
            dest=out/variant/label
            cmd=[sys.executable,str(ROOT/'scripts/live_stream_benchmark.py'),
                '--input',str(video),'--watermark',str(watermark),'--workdir',str(dest),
                '--methods','proposal','--channels','clean,jpeg70',
                '--resolutions',args.resolution,'--target-fps',args.fps,
                '--frames',str(args.frames),'--wm-size','64','--queue-size','2',
                '--pace-mode','realtime','--proposal-config',str(config)]
            commands.append({'variant':variant,'watermark':watermark.name,'config':str(config),
                'output':str(dest),'command':cmd})
    (out/'runs.json').write_text(json.dumps(commands,indent=2),encoding='utf-8')
    for task in commands:
        print('Running',task['variant'],task['watermark'],flush=True)
        subprocess.run(task['command'],cwd=ROOT,check=True)
    print('All live-stream runs saved to:',out.resolve(),flush=True)

if __name__=='__main__':main()
