#!/usr/bin/env python3
"""Full Kodak proposal-only benchmark: 24 original images x 2 full 4096-bit watermarks.

Built-in basic attack profile: clean, genuine JPEG70, rotation 2.5 degrees.
Optional unified profile: all 1+109 TEMPO local attack conditions, NOT the
separate official MMWatermark attack catalog. Both profiles evaluate all images.
No external CSV, pretrained detector or comparison method is required.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from download_full_kodak import NAMES, download
from reviewer_expansion_common import (config, decode, encode, info, init_cpu, load_host,
    mark, md5_file, paths, write_json, write_rows)
from realtime_watermark.core import psnr, ssim_gray, bit_accuracy, nc_binary
from realtime_watermark.attack_manifest import unified_attacks, attack_manifest_sha256
from reviewer_experiments import jpeg, warp


def attacks(profile):
    if profile == 'basic':
        return [('clean', lambda x: x), ('jpeg70', lambda x: jpeg(x, 70)),
                ('rotation2p5', lambda x: warp(x, angle=2.5))]
    if profile == 'unified109':
        return unified_attacks()
    raise ValueError(profile)


def _read_completed(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def _write_atomic_csv(out: Path, rows: list[dict]):
    if not rows:
        return
    path = out/'kodak_all_cases.csv'
    temp = out/'kodak_all_cases.csv.tmp'
    write_rows(temp, rows)
    os.replace(temp, path)


def _summaries(rows: list[dict], total_hosts: int):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['watermark'], row['attack'])].append(row)
    out = []
    for (watermark, attack), group in sorted(groups.items()):
        metric = lambda field: [float(x[field]) for x in group]
        out.append({'watermark': watermark, 'attack': attack, 'host_count': len(group),
            'expected_host_count': total_hosts,
            'mean_psnr_db': statistics.mean(metric('psnr_db')),
            'mean_ssim': statistics.mean(metric('ssim')),
            'mean_nc': statistics.mean(metric('nc')),
            'mean_ber': statistics.mean(metric('ber')),
            'mean_embed_ms': statistics.mean(metric('embed_ms')),
            'mean_extract_ms': statistics.mean(metric('extract_ms')),
            'mean_algorithm_ms': statistics.mean(metric('algorithm_ms')),
            'algorithm_equivalent_fps': 1000.0/statistics.mean(metric('algorithm_ms'))})
    return out


def run(a):
    output = Path(a.out_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(a.dataset_dir).resolve()
    download(dataset_root, offline=a.offline)
    host_paths = [dataset_root/name for name in NAMES]
    if any(not p.is_file() for p in host_paths):
        raise AssertionError('Missing Kodak image after downloader returned complete')
    watermark_paths = paths(a.watermark_dir, {'.png', '.bmp'}, 0)
    if len(watermark_paths) != 2:
        raise ValueError(f'Expected exactly two 64x64 watermarks, found {len(watermark_paths)}')
    marks = [(p, mark(p)) for p in watermark_paths]
    if any(w.shape != (64, 64) or w.size != 4096 for _,w in marks):
        raise AssertionError('Both watermarks must be full 4096-bit')
    cfg = config(a.config)
    attack_cases = attacks(a.attack_profile)
    attack_ids = [k for k,_ in attack_cases]
    if len(attack_ids) != len(set(attack_ids)) or 'clean' not in attack_ids:
        raise RuntimeError('Attack manifest invalid')
    expected_keys = {(hp.name,wp.name,name) for hp in host_paths for wp,_ in marks for name in attack_ids}
    dataset_sha = {hp.name:md5_file(hp) for hp in host_paths}
    watermark_sha = {wp.name:md5_file(wp) for wp,_ in marks}
    run_identity = {'dataset_sha256': dataset_sha, 'watermark_sha256': watermark_sha,
        'config_sha256':cfg.sha256(),'attack_profile':a.attack_profile,
        'attack_manifest_sha256':attack_manifest_sha256() if a.attack_profile=='unified109' else None,
        'seed': a.seed, 'payload_bits':4096, 'expected_rows':len(expected_keys)}
    identity_path = output/'kodak_run_identity.json'
    if identity_path.is_file():
        previous = json.loads(identity_path.read_text())
        if previous != run_identity:
            raise RuntimeError('Cannot resume with different dataset, watermark, config, attack set or seed; use a fresh output directory')
    else:
        write_json(identity_path, run_identity)
    rows = _read_completed(output/'kodak_all_cases.csv')
    found = [(r.get('host'),r.get('watermark'),r.get('attack')) for r in rows]
    if len(found) != len(set(found)) or not set(found).issubset(expected_keys):
        raise RuntimeError('Existing CSV contains duplicate, unexpected or invalid trials')
    done = set(found)
    init_cpu()
    # One untimed same-size, full-4096-bit warmup. The 64x64 Numba warmup
    # does not specialize every full-resolution code path. Keep this honest.
    first_image = load_host(host_paths[0])
    first_wp, first_wm = marks[0]
    warm_marked, warm_key, _ = encode(first_image, first_wm, cfg, a.seed)
    decode(warm_marked, warm_key)
    failures=[]
    started=time.monotonic()
    # One embedding per host/watermark; reuse it across all attacks. Warmup and
    # quality metrics excluded from algorithm-only timings.
    for index,hp in enumerate(host_paths,1):
        host = load_host(hp)  # crucial: NATIVE Kodak resolution, no crop or resize
        if (host.shape[1], host.shape[0]) not in ((768,512),(512,768)):
            raise AssertionError(f'Unexpected decoded resolution: {hp}: {host.shape}')
        for wi,(wp,wm) in enumerate(marks):
            pending=[(label,fn) for label,fn in attack_cases if (hp.name,wp.name,label) not in done]
            if not pending:
                continue
            try:
                # Seed policy identical across all hosts for each watermark; reproducible.
                marked,key,embed_ms=encode(host,wm,cfg,a.seed+wi*1000003)
                embedding_psnr = float(psnr(host,marked))
                embedding_ssim = float(ssim_gray(host,marked))
            except Exception as ex:
                failures.append({'host':hp.name,'watermark':wp.name,'stage':'embed','error':repr(ex)})
                _write_atomic_csv(output,rows)
                write_json(output/'kodak_errors.json',failures)
                raise RuntimeError(f'Embedding failed on {hp.name}, {wp.name}') from ex
            for attack_name, attack_fn in pending:
                try:
                    attacked=attack_fn(marked)
                    if attacked.shape != host.shape or attacked.dtype != host.dtype:
                        raise RuntimeError('Attack output changed dimensions or dtype')
                    decoded,meta,extract_ms=decode(attacked,key)
                    if decoded.shape != (64,64):
                        raise RuntimeError('Wrong watermark recovery size')
                    nc = float(nc_binary(wm,decoded))
                    ber = 1.-float(bit_accuracy(wm,decoded))
                    row = {'dataset':'Kodak_PCD0992_full_24','host':hp.name,'watermark':wp.name,
                        'attack':attack_name,'resolution':f'{host.shape[1]}x{host.shape[0]}',
                        'payload_bits':4096, 'method_id':'proposal','source_sha256':dataset_sha[hp.name],
                        'watermark_sha256':watermark_sha[wp.name], 'config_sha256':cfg.sha256(),
                        'psnr_db':embedding_psnr,'ssim':embedding_ssim, 'nc':nc,'ber':ber,
                        'clean_exact': int(ber < 1e-12) if attack_name=='clean' else '',
                        'embed_ms':embed_ms,'extract_ms':extract_ms,'algorithm_ms':embed_ms+extract_ms,
                        'recovery_path':meta.get('path',''), 'status':'PASS_EXECUTION'}
                    rows.append(row);done.add((hp.name,wp.name,attack_name))
                    # Frequent durable checkpoints, particularly important for
                    # 5280-trial unified109 runs on time-limited Kaggle CPUs.
                    if len(rows) % 10 == 0:
                        _write_atomic_csv(output,rows)
                        write_json(output/'kodak_progress.json',
                            {'status':'RUNNING_PARTIAL','completed_rows':len(done),
                             'expected_rows':len(expected_keys),'last_host':hp.name,
                             'last_watermark':wp.name,'last_attack':attack_name})
                except Exception as ex:
                    failures.append({'host':hp.name,'watermark':wp.name,'attack':attack_name,'error':repr(ex)})
                    _write_atomic_csv(output,rows)
                    write_json(output/'kodak_errors.json',failures)
                    raise RuntimeError(f'External Kodak attack failed on {hp.name}/{wp.name}/{attack_name}') from ex
            _write_atomic_csv(output,rows)
            print(f'[Kodak] {index:02d}/24 {hp.name} {wp.name} complete; total={len(done)}/{len(expected_keys)}',flush=True)
    exact_keys={(r['host'],r['watermark'],r['attack']) for r in rows}
    missing=expected_keys-exact_keys
    extra=exact_keys-expected_keys
    complete=not missing and not extra and len(rows)==len(expected_keys) and not failures
    clean_cases=[r for r in rows if r['attack']=='clean']
    not_exact=[{'host':r['host'],'watermark':r['watermark'],'ber':r['ber']}
               for r in clean_cases if abs(float(r['ber']))>1e-12]
    summary = _summaries(rows, len(host_paths))
    if summary: write_rows(output/'kodak_summary_by_watermark_attack.csv',summary)
    manifest={'status':'COMPLETE' if complete else 'FAILED_COVERAGE',
       'dataset':'Kodak_PCD0992_full_24','method_id':'proposal',
       'dataset_manifest':str(dataset_root/'kodak_full_manifest.json'),
       'host_count':24,'watermark_count':2,'payload_bits':4096,
       'native_resolution':True,'resize_or_crop':'none',
       'attacks':attack_ids,'attacks_per_case':len(attack_ids),
       'attack_provider':'TEMPO unified local 109' if a.attack_profile=='unified109' else 'TEMPO basic (clean/JPEG70/rotation2p5)',
       'expected_rows':len(expected_keys), 'completed_rows':len(rows),
       'missing_trials':sorted(list(missing)), 'unexpected_trials':sorted(list(extra)),
       'failures':failures, 'clean_cases':len(clean_cases),
       'clean_exact_cases':len(clean_cases)-len(not_exact),
       'clean_recovery_all_exact':len(clean_cases)==48 and not not_exact,
       'clean_recovery_not_exact':not_exact,
       'execution_seconds':time.monotonic()-started,
       'quality_calculated_after_timing':True,
       'timing_scope':'Algorithm-only embed + blind extraction; excludes JPEG/attacks, PSNR/SSIM/NC/BER, file I/O, download, queue',
       'scientific_note':'Kodak is an unseen external test suite. Do not tune parameters on Kodak and then report it as independent test.',
       **run_identity, **info()}
    # info() contains payload_bits, timing metadata but not status/method.
    write_json(output/'kodak_external_full_manifest.json',manifest)
    write_json(output/'kodak_progress.json',{
        'status':manifest['status'], 'completed_rows':len(rows), 'expected_rows':len(expected_keys)})
    if not complete:
        raise RuntimeError(f'Kodak full benchmark incomplete: {len(rows)}/{len(expected_keys)} rows. See manifest.')
    print(f'KODAK FULL SUITE PASSED: {len(host_paths)} images x 2 watermarks x {len(attack_cases)} attacks = {len(rows)} rows.', flush=True)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir',type=Path,required=True)
    p.add_argument('--watermark-dir',type=Path,default=ROOT/'data/user_dataset/watermark')
    p.add_argument('--config',type=Path,default=ROOT/'configs/frozen_selected_20261021.json')
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--attack-profile',choices=['basic','unified109'],default='basic')
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--offline',action='store_true')
    a=p.parse_args()
    run(a)


if __name__=='__main__':
    main()
