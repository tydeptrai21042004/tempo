#!/usr/bin/env python3
"""Create a disjoint paper-specific dev/test split without changing host bytes.

The main hyperparameter search and any new synchronization threshold calibration
must use development hosts ONLY. Test hosts are not available to calibration.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_hosts(host_dir: Path, output: Path, development_count: int = 3) -> dict:
    hosts = sorted(host_dir.glob('*.bmp'))
    if not 1 <= development_count < len(hosts):
        raise ValueError('Need at least one development and one held-out BMP host')
    development, test = hosts[:development_count], hosts[development_count:]
    hashes = {h: sha256(h) for h in hosts}
    if {hashes[h] for h in development} & {hashes[h] for h in test}:
        raise RuntimeError('Duplicate image bytes across development and held-out tests')
    output.mkdir(parents=True, exist_ok=True)
    for group, paths in (('development', development), ('test', test)):
        target = output / group
        target.mkdir(parents=True, exist_ok=True)
        for path in paths:
            shutil.copy2(path, target / path.name)
    result = {
        'protocol': 'sorted selected BMP: first N for development, remaining held out',
        'calibration_policy': 'no final-test hosts or attacks used for parameter selection',
        'development': [{'name': p.name, 'sha256': hashes[p]} for p in development],
        'test': [{'name': p.name, 'sha256': hashes[p]} for p in test],
        'split_sizes': {'development': len(development), 'test': len(test)},
    }
    (output / 'split_manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--host-dir', required=True)
    a.add_argument('--out-dir', required=True)
    a.add_argument('--development-count', type=int, default=3)
    opt = a.parse_args()
    print(json.dumps(split_hosts(Path(opt.host_dir), Path(opt.out_dir), opt.development_count)['split_sizes']))
