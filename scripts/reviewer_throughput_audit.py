#!/usr/bin/env python3
"""Recompute every 4096-bit wall-clock stream throughput from observed FPS.

Virtual-rate, processor service-rate, effective-output FPS, and target FPS are
never treated as interchangeable. Synthetic streams are explicitly labeled.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path


def read_csv(path):
    with path.open(newline='', encoding='utf8') as f:
        return list(csv.DictReader(f))


def write_csv(path, values):
    if not values:
        raise ValueError('No wall-clock stream observations available')
    with path.open('w', newline='', encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)


def audit(root:Path, output:Path, require_720p=False):
    stream_files=sorted(root.glob('*/ALL_STREAM_run_manifest.json'))
    if not stream_files:
        raise FileNotFoundError(f'No stream manifests under {root}')
    observations=[]
    issues=[]
    sources=set()
    for mp in stream_files:
        manifest=json.loads(mp.read_text())
        name=mp.parent.name
        source=manifest.get('source_video','unknown')
        sources.add(source)
        summary_path=mp.parent/'ALL_STREAM_realtime_summary.csv'
        if not summary_path.is_file():
            issues.append(f'{name}: stream summary missing');continue
        for r in read_csv(summary_path):
            try:
                fps=float(r['effective_output_fps'])
                payload=int(r['payload_bits'])
                reported=float(r['payload_throughput_bps'])
                deadline=float(r['service_ms_p95'])
                target=float(r['target_fps'])
            except (KeyError,ValueError,TypeError) as exc:
                issues.append(f'{name}: missing/malformed throughput data: {exc}');continue
            recalculated=fps*payload
            abs_error=abs(reported-recalculated)
            on_time=float(r['on_time_completion_ratio'])
            observations.append({
                'watermark':name,'source_kind':'real_video' if source!='deterministic_synthetic_stream' else 'synthetic_moving_frames',
                'source_video':source,'resolution':r['resolution'],'channel':r['channel'],
                'pace_mode':r['pace_mode'],'payload_bits':payload,'target_fps':target,
                'effective_output_fps':fps,'reported_payload_throughput_bps':reported,
                'recomputed_payload_throughput_bps':recalculated,
                'throughput_absolute_error_bps':abs_error,
                'payload_throughput_formula':'payload_bits * effective_output_fps',
                'service_ms_p95':deadline,'on_time_completion_ratio':on_time,
                'frames_generated':int(r['frames_generated']),
                'frames_processed':int(r['frames_processed']),
                'queue_drops':int(r['frames_dropped_queue']),
                'channel_drops':int(r['frames_dropped_channel']),
                'meets_target_fps':fps>=target,
                'all_frames_on_time':on_time>=1-1e-10,
                'meets_p95_deadline':deadline<=1000/target,
            })
            if not all(math.isfinite(x) for x in (fps,reported,deadline,target,on_time)):
                issues.append(f'{name}/{r["resolution"]}/{r["channel"]}: non-finite timing or throughput')
            if fps<0 or target<=0 or deadline<0 or not 0<=on_time<=1:
                issues.append(f'{name}/{r["resolution"]}/{r["channel"]}: invalid measured value')
            if int(r['frames_processed'])>int(r['frames_generated']) or int(r['frames_generated'])<=0:
                issues.append(f'{name}/{r["resolution"]}/{r["channel"]}: inconsistent frame counts')
            if not abs_error<=max(1e-6, 1e-8*abs(recalculated)):
                issues.append(f'{name}/{r["resolution"]}/{r["channel"]}: throughput mismatch {abs_error}')
            if r['pace_mode']!='realtime':
                issues.append(f'{name}: non-wall-clock pace mode')
            if payload!=4096:
                issues.append(f'{name}: payload was {payload}, expected 4096')
    output.mkdir(parents=True,exist_ok=True)
    write_csv(output/'wallclock_throughput_audit.csv',observations)
    tested={(x['resolution'],x['channel']) for x in observations}
    if require_720p and not {('1280x720','clean'),('1280x720','jpeg70')}.issubset(tested):
        issues.append('Full mode requires 1280x720 clean and jpeg70 conditions')
    manifest={'status':'complete' if not issues else 'failed','observations':len(observations),
        'source_kinds':sorted({x['source_kind'] for x in observations}),
        'watermark_streams':len(stream_files),
        'has_real_video':any(x['source_kind']=='real_video' for x in observations),
        'throughput_formula':'4096 * actual effective output FPS (not nominal target FPS)',
        'issues':issues,
        'limitations':['A synthetic moving stream is not a real-video performance claim.',
            'Soft-real-time criteria are measured separately: effective FPS, on-time completion, p95 service latency.',
            'A run below the 30 FPS deadline is a negative result, not omitted.']}
    (output/'wallclock_throughput_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    if issues:
        raise RuntimeError('Wall-clock throughput audit failed: '+'; '.join(issues[:5]))
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stream-dir',required=True)
    p.add_argument('--out-dir',required=True)
    p.add_argument('--require-720p',action='store_true')
    a=p.parse_args()
    print(json.dumps(audit(Path(a.stream_dir),Path(a.out_dir),a.require_720p),indent=2))
