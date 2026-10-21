"""Shared read-only evaluation utilities for the supplementary reviewer experiments.

Never changes the watermark method, its configuration, or its side information.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from realtime_watermark.core import bit_accuracy, nc_binary, psnr, load_binary_watermark
from realtime_watermark.methods import warmup_numba
from realtime_watermark.proposals.jilp_qim.config import JILPQIMConfig
from realtime_watermark.proposals.jilp_qim.method import embed_proposal, extract_jilp_qim

ROOT = Path(__file__).resolve().parents[1]


def paths(folder, extensions, limit=0):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    out = sorted(x for x in folder.iterdir() if x.is_file() and x.suffix.lower() in extensions)
    if limit:
        out = out[:limit]
    if not out:
        raise ValueError(f"No matching data under {folder}")
    return out


def dataset(a):
    hosts = paths(a.host_dir, {'.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff'}, a.max_hosts)
    marks = paths(a.watermark_dir, {'.png', '.bmp'}, a.max_watermarks)
    return hosts, marks


def load_host(p, resolution=None):
    im = np.asarray(Image.open(p).convert('RGB'), dtype=np.uint8)
    if resolution:
        w, h = resolution
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA if w < im.shape[1] else cv2.INTER_LINEAR)
    return im


def mark(p):
    return load_binary_watermark(p, size=64)


def md5_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_rows(p, rows):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f'No rows for {p}; never count empty results as complete')
    cols = list(dict.fromkeys(key for row in rows for key in row))
    with p.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def read_rows(p):
    with Path(p).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_json(p, data):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False, default=str), encoding='utf-8')


def config(p):
    return JILPQIMConfig.load(p)


def init_cpu():
    cv2.setNumThreads(1)
    warmup_numba()


def encode(im, wm, cfg, seed):
    start = time.perf_counter_ns()
    out, key = embed_proposal(im, wm, seed=seed, method_params=cfg.to_dict())
    return out, key, (time.perf_counter_ns() - start)/1e6


def decode(received, key, sync=True):
    start = time.perf_counter_ns()
    bits, meta = extract_jilp_qim(received, key, sync=sync, return_meta=True)
    return bits, meta, (time.perf_counter_ns()-start)/1e6


def quality(host, im, wm, bits):
    return {'psnr_db': float(psnr(host, im)), 'nc': float(nc_binary(wm, bits)),
            'ber': float(1.0-bit_accuracy(wm, bits))}


def case_seed(seed, wi):
    return int(seed) + 1000003 * wi


def info():
    return {'python': platform.python_version(), 'numpy': np.__version__,
            'opencv': cv2.__version__, 'cpu_threads': os.environ.get('NUMBA_NUM_THREADS', 'default'),
            'timing': 'perf_counter_ns wall clock; warmup separate; single OpenCV thread',
            'payload_bits': 4096, 'method_modified': False}


def host_bootstrap(pairs, metric, *, seed=2026, samples=1000):
    """Paired host-cluster uncertainty; both watermarks remain in each host draw."""
    per_host = {}
    for r in pairs:
        per_host.setdefault(r['host'], []).append(float(r[metric]))
    vals = np.array([np.mean(v) for _, v in sorted(per_host.items())], float)
    if not len(vals):
        raise ValueError('No paired host observations')
    ci = [None, None]
    if len(vals) > 1:
        rng = np.random.default_rng(seed)
        draws = rng.choice(vals, size=(samples, len(vals)), replace=True).mean(axis=1)
        ci = [float(q) for q in np.quantile(draws, [.025, .975])]
    return {'mean_host_delta': float(vals.mean()), 'ci95': ci,
            'host_clusters': len(vals), 'uncertainty_unit': 'host; intervals unavailable if n<2'}


def base_args(parser):
    parser.add_argument('--host-dir', default=str(ROOT/'data/user_dataset/host'))
    parser.add_argument('--watermark-dir', default=str(ROOT/'data/user_dataset/watermark'))
    parser.add_argument('--config', default=str(ROOT/'configs/proposal_minimal_refined.json'))
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--max-hosts', type=int, default=0)
    parser.add_argument('--max-watermarks', type=int, default=0)
    parser.add_argument('--quick', action='store_true')
    return parser
