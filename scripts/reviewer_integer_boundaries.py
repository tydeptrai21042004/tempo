#!/usr/bin/env python3
"""Unmodified embedder's feasibility under actual/synthetic uint8 boundary images.

All attempted embeddings are counted, including exceptions. Rejections are outcomes,
not ignored samples. Missing diagnostics from a rejected embedding are NA, never zero.
These are controlled stress fixtures, not estimates of natural-image prevalence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image

from realtime_watermark.core import bit_accuracy, load_binary_watermark, psnr
from realtime_watermark.proposals.jilp_qim.config import JILPQIMConfig
from realtime_watermark.proposals.jilp_qim.method import embed_proposal, extract_jilp_qim


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot export an empty boundary experiment: {path}")
    cols = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def fixtures(shape: tuple[int, int, int], quick: bool) -> list[tuple[str, np.ndarray]]:
    h, w, ch = shape
    if ch != 3:
        raise ValueError('Require RGB fixture shape')
    xs = np.arange(w, dtype=np.uint16)[None, :]
    ys = np.arange(h, dtype=np.uint16)[:, None]
    def repeat(g):
        return np.repeat(np.asarray(g, dtype=np.uint8)[:, :, None], 3, axis=2)
    patterns = [
        ('black_0', np.zeros(shape, dtype=np.uint8)),
        ('white_255', np.full(shape, 255, dtype=np.uint8)),
        ('shadow_0to3', repeat(np.broadcast_to((xs + ys) % 4, (h, w)))),
        ('highlight_252to255', repeat(252 + np.broadcast_to((xs + ys) % 4, (h, w)))),
        ('gray_128', np.full(shape, 128, dtype=np.uint8)),
    ]
    contrast = np.empty(shape, dtype=np.uint8)
    contrast[:, :, 0] = 0
    contrast[:, :, 1] = 255
    contrast[:, :, 2] = np.broadcast_to(((xs + ys) % 2) * 255, (h, w))
    patterns.append(('rgb_channel_saturation', contrast))
    return patterns[:2] if quick else patterns


def classify_error(exc: Exception) -> str:
    if isinstance(exc, RuntimeError) and ('infeasible' in str(exc).lower() or 'parity' in str(exc).lower()):
        return 'rejected_by_integer_constraints'
    return 'unexpected_error'


def wilson(k: int, n: int, z: float = 1.95996398454) -> tuple[float, float]:
    """Descriptive per-attempt binomial interval; stress trials are not iid field samples."""
    if n <= 0:
        return float('nan'), float('nan')
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    spread = z * np.sqrt((p*(1-p) + z*z/(4*n))/n) / denom
    return float(max(0, center-spread)), float(min(1, center+spread))


def run(args: argparse.Namespace) -> dict:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = JILPQIMConfig.load(args.config)
    hosts = sorted(Path(args.host_dir).glob('*.bmp'))
    watermarks = sorted(Path(args.watermark_dir).glob('*.png'))
    if args.quick:
        hosts = hosts[:1]
        watermarks = watermarks[:1]
    if not hosts or not watermarks:
        raise RuntimeError('Boundary validation requires hosts and watermark files')
    rows: list[dict] = []
    image_digest: dict[str, str] = {}
    unexpected: list[str] = []
    for wi, wp in enumerate(watermarks):
        wm = load_binary_watermark(wp, size=64)
        if wm.size != 4096:
            raise RuntimeError('Boundary validation uses the unchanged 4096-bit payload')
        # Run realistic original hosts, plus deterministic synthetic intensity boundaries.
        for hi, hp in enumerate(hosts):
            real = np.asarray(Image.open(hp).convert('RGB'))
            if real.shape[0] < 512 or real.shape[1] < 512:
                raise ValueError(f'Host too small for 4096-bit test: {hp}')
            cases = [('original_host', real)]
            # Avoid treating every synthetic fixture on each host as an independent sample.
            if hi == 0:
                cases += fixtures(real.shape, args.quick)
            for scenario, image in cases:
                before = time.perf_counter()
                record = {
                    'host': hp.name, 'watermark': wp.name, 'scenario': scenario,
                    'payload_bits': int(wm.size), 'attempted': 1,
                    'pixels_at_zero_fraction': float(np.mean(image == 0)),
                    'pixels_at_255_fraction': float(np.mean(image == 255)),
                    'host_min': int(image.min()), 'host_max': int(image.max()),
                    'config_sha256': config.sha256(),
                }
                if hi == 0:
                    image_digest[scenario] = hashlib.sha256(image.tobytes()).hexdigest()
                try:
                    marked, key = embed_proposal(image, wm,
                        seed=args.seed+1000003*wi, method_params=config.to_dict())
                    record.update({
                        'status': 'accepted', 'rejection': 0,
                        'elapsed_embedding_ms': (time.perf_counter()-before)*1000,
                        'psnr': float(psnr(image, marked)),
                        'clipped_output_fraction': float(np.mean((marked == 0) | (marked == 255))),
                        'integer_payload_initial_parity_failures': int(key.params['integer_lattice_initial_parity_failures']),
                        'integer_payload_initial_guard_failures': int(key.params['integer_lattice_initial_guard_failures']),
                        'integer_payload_corrected_blocks': int(key.params['integer_lattice_corrected_blocks']),
                        'integer_payload_unit_updates': int(key.params['integer_lattice_unit_updates']),
                        'integer_payload_unresolved': int(key.params['integer_lattice_unresolved']),
                        'integer_pilot_corrected_blocks': int(key.params['integer_pilot_corrected_blocks']),
                        'integer_pilot_unit_updates': int(key.params['integer_pilot_unit_updates']),
                        'integer_pilot_unresolved': int(key.params['integer_pilot_unresolved']),
                    })
                    decoded = extract_jilp_qim(marked, key, sync=False)
                    record['raw_clean_bit_accuracy'] = float(bit_accuracy(wm, decoded))
                except Exception as exc:
                    category = classify_error(exc)
                    record.update({'status':category, 'rejection': 1 if category.startswith('rejected_') else 0,
                        'elapsed_embedding_ms':(time.perf_counter()-before)*1000,
                        'error_type':type(exc).__name__, 'error_message':str(exc)[:300],
                        'raw_clean_bit_accuracy':'', 'psnr':'',
                        'integer_payload_unresolved':'', 'integer_pilot_unresolved':''})
                    if category == 'unexpected_error':
                        unexpected.append(f'{hp.name}/{wp.name}/{scenario}: {type(exc).__name__}: {exc}')
                rows.append(record)
    write_csv(out/'embedding_attempts.csv', rows)
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row['scenario'], []).append(row)
    summaries = []
    for scenario, records in sorted(groups.items()):
        n = len(records)
        k = sum(r['rejection'] for r in records)
        lo, hi = wilson(k, n)
        summaries.append({'scenario':scenario, 'attempts':n, 'accepted':sum(r['status']=='accepted' for r in records),
            'rejections':k, 'unexpected_errors':sum(r['status']=='unexpected_error' for r in records),
            'rejection_rate':k/n, 'wilson_95_lower_descriptive':lo,
            'wilson_95_upper_descriptive':hi,
            'interpretation':'controlled stress attempts; no natural-image population inference'})
    write_csv(out/'rejection_summary.csv', summaries)
    total = len(rows)
    manifest = {'status':'failed' if unexpected else 'complete', 'payload_bits':4096,
        'config_sha256':config.sha256(), 'seed':args.seed, 'attempts':total,
        'accepted':sum(r['status']=='accepted' for r in rows),
        'rejected':sum(r['rejection'] for r in rows), 'unexpected_errors':len(unexpected),
        'scenarios': sorted(groups), 'watermarks': [p.name for p in watermarks],
        'hosts':[p.name for p in hosts], 'fixture_sha256':image_digest,
        'python':platform.python_version(),
        'limitations':['If embedding rejects, internal correction counters are not available from the public API and are recorded as missing.',
                       'A completed embedding is not proof of globally minimum integer energy.',
                       'Stress fixtures are deterministic and not an estimate of real-world rejection prevalence.',
                       'Binomial intervals are descriptive only because host/watermark/scenario trials are correlated.'],
        'errors':unexpected}
    (out/'boundary_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(json.dumps({'attempts':total,'rejected':manifest['rejected'],'unexpected':len(unexpected)},indent=2))
    if unexpected:
        raise RuntimeError(f'{len(unexpected)} unexpected embedding errors, see embedding_attempts.csv')
    return manifest


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host-dir',required=True);p.add_argument('--watermark-dir',required=True)
    p.add_argument('--config',required=True);p.add_argument('--out-dir',required=True)
    p.add_argument('--seed',type=int,default=2026);p.add_argument('--quick',action='store_true')
    run(p.parse_args())
