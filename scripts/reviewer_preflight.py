#!/usr/bin/env python3
"""Fail early on invalid primary evidence before costly hyperparameter/video jobs.

This only validates dataset inputs; recovered watermarks are allowed to be constant.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from realtime_watermark.core import load_binary_watermark, watermark_statistics


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(host_dir: Path, watermark_dir: Path, wm_size: int = 64) -> dict:
    hosts=sorted(host_dir.glob('*.bmp'))
    watermarks=sorted(watermark_dir.glob('*.png'))
    errors=[]
    if len(hosts)!=12:
        errors.append(f'Expected 12 image hosts (all BMP except Lena), found {len(hosts)}')
    if len(watermarks)!=2:
        errors.append(f'Expected exactly two original watermark PNGs, found {len(watermarks)}')
    host_rows=[]
    for path in hosts:
        try:
            with Image.open(path) as im:
                size=im.size
                im.verify()
            ok=size[0]>=512 and size[1]>=512
            if not ok: errors.append(f'Host too small for 4096 bits: {path.name} {size}')
            host_rows.append({'name':path.name,'sha256':checksum(path), 'width':size[0], 'height':size[1], 'valid':ok})
        except (OSError,ValueError) as exc:
            errors.append(f'Unreadable host: {path.name}: {exc}')
    wm_rows=[]
    for path in watermarks:
        try:
            bits=load_binary_watermark(path,size=wm_size)
            stats=watermark_statistics(bits)
            if bits.size!=4096:
                errors.append(f'Unexpected original watermark payload size {bits.size}: {path.name}')
            wm_rows.append({'name':path.name,'sha256':checksum(path), **stats})
        except (OSError,ValueError) as exc:
            errors.append(f'Invalid original watermark {path.name}: {exc}')
    return {'status':'passed' if not errors else 'failed',
            'payload_bits':wm_size*wm_size, 'host_count':len(hosts),'watermark_count':len(watermarks),
            'hosts':host_rows, 'watermarks':wm_rows, 'python':sys.version, 'platform':platform.platform(),
            'numpy':np.__version__, 'errors':errors,
            'scope':'input preflight; not evidence that expensive experiments passed'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host-dir',type=Path,required=True)
    p.add_argument('--watermark-dir',type=Path,required=True)
    p.add_argument('--wm-size',type=int,default=64)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    report=inspect(a.host_dir,a.watermark_dir,a.wm_size)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2))
    print(f'Dataset preflight: {report["status"]}, {report["host_count"]} hosts, {report["watermark_count"]} watermark files')
    if report['errors']:
        raise SystemExit('Preflight failed: '+'; '.join(report['errors']))


if __name__=='__main__': main()
