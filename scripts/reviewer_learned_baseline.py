#!/usr/bin/env python3
"""Fail-closed provenance validator for independently executed pretrained baselines.

Does not fake a trained network, extrapolate 32-bit results to 4096 bits, or
silently equate native-payload quality to matched-payload/PSNR experiments.
Supply evidence exported from the baseline's ORIGINAL implementation.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from reviewer_expansion_common import read_rows,write_rows,write_json,md5_file

REQUIRED={'host','watermark','attack','payload_bits','psnr_db','nc','ber','embed_ms','extract_ms'}

def run(a):
    doc=json.loads(Path(a.provenance).read_text(encoding='utf8'))
    for key in ('model_name','paper_url','source_commit','weights_sha256','native_payload_bits','detector_assumptions','evaluation_protocol'):
        if not doc.get(key):raise ValueError(f'Missing independent pretrained baseline provenance: {key}')
    if len(str(doc['weights_sha256']))!=64 or len(str(doc['source_commit']))<7:
        raise ValueError('No full weights sha256 or valid code commit reference')
    records=read_rows(a.observations)
    if not records:raise ValueError('No independently computed inference rows')
    for row in records:
        missing=REQUIRED-set(row)
        if missing:raise ValueError(f'Missing baseline metrics {sorted(missing)}')
        for field in ('payload_bits','psnr_db','nc','ber','embed_ms','extract_ms'):
            if not math.isfinite(float(row[field])):raise ValueError(f'Nonfinite baseline {field}')
        if int(row['payload_bits'])!=int(doc['native_payload_bits']):
            raise ValueError('Observed payload bits disagree with original model native payload')
        if not (0<=float(row['ber'])<=1):raise ValueError('Invalid BER')
        row['model_name']=doc['model_name'];row['comparison_track']='external_native_payload_not_matched'
    dest=Path(a.out_dir)
    write_rows(dest/'baseline_native_results.csv',records)
    write_json(dest/'manifest.json',{'status':'complete_native_only',
        'source_observations_sha256':md5_file(a.observations),
        'source_provenance_sha256':md5_file(a.provenance),
        'model_name':doc['model_name'],'native_payload_bits':int(doc['native_payload_bits']),
        'rows':len(records),'weights_sha256':doc['weights_sha256'],
        'claims_prohibited':['4096-bit matched payload unless native 4096-bit model was actually evaluated',
            'matched PSNR comparison without paired calibrated heldout trials',
            'identical detector assumptions without explicit audit']})

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--provenance',required=True)
    p.add_argument('--observations',required=True)
    p.add_argument('--out-dir',required=True)
    return p
if __name__=='__main__':run(parser().parse_args())
