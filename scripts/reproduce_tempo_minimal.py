#!/usr/bin/env python3
"""Paired verification: unchanged embedding, one universal phase search.

Run with PYTHONPATH=src (or pip install -e .).  No auto-generated result is
reported as experimental evidence without writing its individual observations.
The optional 720p loop reuses *resized still images*; it is not real-video proof.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from realtime_watermark.core import load_binary_watermark
from realtime_watermark.proposals.jilp_qim.config import JILPQIMConfig
from realtime_watermark.proposals.jilp_qim.method import embed_proposal, extract_jilp_qim

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows):
    if not rows:
        raise RuntimeError('No observations generated')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def jpeg70(rgb):
    ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        raise RuntimeError('JPEG encoder failed')
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError('JPEG decoder failed')
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def rotate(rgb, degrees):
    height, width = rgb.shape[:2]
    mat = cv2.getRotationMatrix2D((width/2, height/2), degrees, 1.0)
    return cv2.warpAffine(rgb, mat, (width, height), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)


def nc_ber(reference, prediction):
    err = float(np.mean((reference > 127) != (prediction > 127)))
    return 1.0 - 2.0*err, err


def read_host(path, size=None):
    arr = np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)
    if size is not None:
        arr = cv2.resize(arr, size, interpolation=cv2.INTER_LINEAR)
    return arr


def paired_image(hosts, watermarks, configs, output):
    rows = []
    conditions = (('clean', lambda x: x), ('jpeg70', jpeg70),
                  ('rot1.25', lambda x: rotate(x, 1.25)),
                  ('rot2.50', lambda x: rotate(x, 2.5)),
                  ('rot3.25', lambda x: rotate(x, 3.25)))
    for hostpath in hosts:
        host = read_host(hostpath)
        for wm_name, wm in watermarks.items():
            embeds = {}
            for label, cfg in configs.items():
                start = time.perf_counter_ns()
                marked, key = embed_proposal(host, wm, method_params=cfg.to_dict())
                ems = (time.perf_counter_ns()-start)/1e6
                embeds[label] = (marked, key, ems)
            if not np.array_equal(embeds['control'][0], embeds['refined'][0]):
                raise RuntimeError('Modulation changed despite a decoder-only refinement')
            for attack_name, attack in conditions:
                attacked = attack(embeds['control'][0])
                for label, (marked, key, embed_ms) in embeds.items():
                    t = time.perf_counter_ns()
                    rec, meta = extract_jilp_qim(attacked, key, return_meta=True)
                    ext_ms = (time.perf_counter_ns()-t)/1e6
                    nc, ber = nc_ber(wm, rec)
                    rows.append(dict(host=hostpath.name, watermark=wm_name,
                                     condition=attack_name, variant=label,
                                     config_sha256=configs[label].sha256(),
                                     payload_bits=wm.size, psnr_db=float(cv2.PSNR(host, marked)),
                                     nc_signed=nc, ber=ber, embed_ms=embed_ms,
                                     extract_ms=ext_ms, path=meta.get('path', ''),
                                     candidate=str(meta.get('candidate', '')),
                                     warp_count=meta.get('warp_count', 0)))
    write_csv(output/'paired_image.csv', rows)
    return rows


def paced_stills(hosts, watermarks, cfg, output, frames, hz):
    images = [read_host(p, (1280, 720)) for p in hosts]
    period_s = 1.0/float(hz)
    rows = []
    # Warm up image dimensions, JPEG codec and both watermarks; do not time JIT.
    for watermark in watermarks.values():
        out, key = embed_proposal(images[0], watermark, method_params=cfg.to_dict())
        extract_jilp_qim(jpeg70(out), key)
    for wm_name, wm in watermarks.items():
        stream_start = time.perf_counter()
        for i in range(frames):
            scheduled = stream_start + i*period_s
            to_sleep = scheduled - time.perf_counter()
            if to_sleep > 0:
                time.sleep(to_sleep)
            start = time.perf_counter_ns()
            marked, key = embed_proposal(images[i % len(images)], wm, method_params=cfg.to_dict())
            embedded_at = time.perf_counter_ns()
            attacked = jpeg70(marked)
            codec_at = time.perf_counter_ns()
            rec, meta = extract_jilp_qim(attacked, key, return_meta=True)
            finished_at = time.perf_counter_ns()
            nc, ber = nc_ber(wm, rec)
            end_at = finished_at/1e9
            rows.append(dict(watermark=wm_name, frame=i, host=hosts[i % len(images)].name,
                             scheduled_arrival_s=scheduled, actual_start_s=start/1e9,
                             completed_s=end_at,
                             embed_ms=(embedded_at-start)/1e6,
                             codec_ms=(codec_at-embedded_at)/1e6,
                             extract_ms=(finished_at-codec_at)/1e6,
                             service_ms=(finished_at-start)/1e6,
                             latency_ms=(end_at-scheduled)*1000,
                             on_time=int(end_at <= scheduled+period_s),
                             nc_signed=nc, ber=ber, path=meta.get('path',''),
                             candidate=str(meta.get('candidate',''))))
        elapsed = time.perf_counter()-stream_start
        these = [r for r in rows if r['watermark'] == wm_name]
        print(json.dumps(dict(watermark=wm_name, frames=frames, hz=hz,
             effective_output_fps=frames/elapsed,
             p95_service_ms=float(np.percentile([x['service_ms'] for x in these],95)),
             p95_latency_ms=float(np.percentile([x['latency_ms'] for x in these],95)),
             on_time_ratio=float(np.mean([x['on_time'] for x in these])),
             mean_nc_signed=float(np.mean([x['nc_signed'] for x in these])))))
    write_csv(output/'paced_720p_still_trace.csv', rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--max-hosts', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--paced-frames', type=int, default=24)
    parser.add_argument('--hz', type=int, default=60)
    parser.add_argument('--skip-paced', action='store_true')
    args = parser.parse_args()
    if args.max_hosts < 1 or args.paced_frames < 1 or args.hz < 1:
        parser.error('max-hosts, paced-frames and hz must be positive')
    os.environ['JILP_NUM_THREADS'] = str(max(1, args.threads))
    cv2.setNumThreads(1)
    hosts = sorted(p for p in (ROOT/'data/user_dataset/host').glob('*.bmp')
                   if 'lenna' not in p.name.lower())[:args.max_hosts]
    watermarks = {name: load_binary_watermark(ROOT/'data/user_dataset/watermark'/name, 64)
                  for name in ('wm.png', 'Cc.logo.circle.png')}
    configs = {name: JILPQIMConfig.load(ROOT/'configs'/f'proposal_{file}.json')
               for name, file in [('control','frozen'),('refined','minimal_refined')]}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paired_image(hosts, watermarks, configs, args.out_dir)
    if not args.skip_paced:
        paced_stills(hosts, watermarks, configs['refined'], args.out_dir,
                     args.paced_frames, args.hz)
    (args.out_dir/'protocol.json').write_text(json.dumps(dict(
        purpose='paired, exploratory CPU evaluation',
        source_hashes={name: cfg.sha256() for name,cfg in configs.items()},
        watermark_names=list(watermarks), host_names=[h.name for h in hosts],
        input_type='720p resized still-host loop; NOT independently sourced video',
        codec='OpenCV BGR JPEG70 -> BGR decode -> RGB',
        mode='sequential paced arrivals; no independent camera producer or queue',
        clean_claim='Clean NC=1 is checked only on evaluated cases',
        metric='NC_signed=1-2*BER, NOT binary cosine NC',
        environment=dict(python=platform.python_version(), numpy=np.__version__,
                         opencv=cv2.__version__, cpu=platform.processor(),
                         threads=args.threads),
        ), indent=2)+'\n')


if __name__ == '__main__':
    main()
