#!/usr/bin/env python3
"""Runtime smoke test for the registered original baseline *implementations*.

Confirms deterministic embedding, extraction, capacity, decoder access class,
and the declared common adaptation. This does NOT certify verbatim reproduction
of the complete papers; paper-native metrics are a separate TEMPO protocol.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from realtime_watermark.core import load_rgb, load_binary_watermark, bit_accuracy, psnr, save_json
from realtime_watermark.registry import (embed, extract, COMMON_IMAGE_METHODS,
                                        STREAM_BENCHMARK_METHODS)
from realtime_watermark.paper_fidelity import fidelity_status


def preflight(host_path: Path, watermark_path: Path, *, seed: int = 2026,
              methods: list[str] | None = None) -> dict:
    host = load_rgb(host_path)
    mark = load_binary_watermark(watermark_path, size=64)
    if host.shape != (512,512,3) or mark.shape != (64,64):
        raise ValueError('Preflight requires a real RGB 512 host and 64x64 binary watermark')
    if mark.size != 4096: raise ValueError('Incorrect common-protocol bit capacity')
    methods = list(COMMON_IMAGE_METHODS) if methods is None else list(methods)
    if len(set(methods)) != len(methods):
        raise ValueError('Duplicate method IDs')
    records=[]
    original_hash=hashlib.sha256(host.tobytes()).hexdigest()
    for method in methods:
        disclosure = fidelity_status(method)
        if method == 'proposal':
            continue  # Separate proposal config is supplied to the MMW benchmark.
        if method not in COMMON_IMAGE_METHODS:
            raise ValueError(f'Not a registered 4096-bit image baseline: {method}')
        output,key = embed(method,host,mark,seed=seed,repeat=1)
        repeat_output,repeat_key = embed(method,host,mark,seed=seed,repeat=1)
        if not np.array_equal(output,repeat_output) or key.method_id != repeat_key.method_id:
            raise AssertionError(f'{method}: embedding is not deterministic under fixed seed')
        if output.shape != host.shape or output.dtype != np.uint8:
            raise AssertionError(f'{method}: invalid RGB output')
        if hashlib.sha256(host.tobytes()).hexdigest() != original_hash:
            raise AssertionError(f'{method}: mutated the host')
        if key.method_id != method or tuple(key.watermark_shape) != mark.shape:
            raise AssertionError(f'{method}: mismatched method key/payload')
        if key.fully_blind != (disclosure['blindness_tier']=='blind'):
            raise AssertionError(f'{method}: detector-side-information class is wrong')
        rec=extract(output,key)
        if rec.shape != mark.shape or not np.isin(rec,[0,255]).all():
            raise AssertionError(f'{method}: detector did not return 4096 bits')
        p=psnr(host,output)
        if not np.isfinite(p) or p <= 0:
            raise AssertionError(f'{method}: invalid watermarked image fidelity')
        records.append({'method_id':method,'payload_bits':4096,
                        'clean_bit_accuracy':float(bit_accuracy(mark,rec)),
                        'embedding_psnr':float(p),
                        'detector_class':disclosure['blindness_tier'],
                        'paper_reproduction_eligible':disclosure['paper_reproduction_eligible'],
                        'common_primary_eligible':disclosure['common_primary_eligible'],
                        'deterministic_same_seed': True,
                        'status':'EXECUTED_VALID_OUTPUT'})
    if not records: raise RuntimeError('No common-image baselines were evaluated')
    return {'schema_version':'1.0', 'host':host_path.name,
            'host_sha256':hashlib.sha256(host_path.read_bytes()).hexdigest(),
            'watermark':watermark_path.name,
            'watermark_sha256':hashlib.sha256(watermark_path.read_bytes()).hexdigest(),
            'common_protocol_bits':4096,'results':records,
            'warning':'Successful execution and formula tests are not evidence of full paper reproduction'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host-dir',type=Path,required=True)
    p.add_argument('--watermark-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    hosts=sorted(a.host_dir.glob('*.bmp'))
    watermarks=sorted(a.watermark_dir.glob('*.png'))
    if not hosts or len(watermarks)!=2: raise RuntimeError('Need held-out hosts and exactly 2 marks')
    data=[]
    for wm in watermarks:
        data.append(preflight(hosts[0],wm))
    # DEW is a genuine sixth baseline, but its two-block-per-bit carrier
    # cannot encode 4096 distinct bits into a 512x512 8x8-block image.
    # Test it separately at 256 bits rather than silently dropping it or
    # pretending the paper supports the impossible common capacity.
    host=load_rgb(hosts[0])
    dew=[]
    for wm in watermarks:
        payload=load_binary_watermark(wm,size=16)
        output,key=embed('dew_langelaar2001',host,payload,seed=2026,repeat=1)
        recovered=extract(output,key)
        if recovered.shape != payload.shape or not key.fully_blind:
            raise AssertionError('DEW low-capacity baseline detector failed')
        dew.append({'host':hosts[0].name,'watermark':wm.name,'payload_bits':256,
                    'clean_bit_accuracy':float(bit_accuracy(payload,recovered)),
                    'embedding_psnr':float(psnr(host,output)),
                    'status':'EXECUTED_VALID_OUTPUT_CAPACITY_256'})
    save_json(a.out,{'watermarks':data,'baseline_methods':list(COMMON_IMAGE_METHODS[:-1]),
                     'dew_capacity_256':dew,
                     'unsupported_4096_baseline':'dew_langelaar2001 (requires >=2 blocks/bit)'})
    print('Baseline preflight passed:',len(data),'watermarks,',len(data[0]['results']),'4096-bit common methods and DEW 256-bit')


if __name__=='__main__':main()
