#!/usr/bin/env python3
"""Validate, NEVER synthesize, externally produced recent-method benchmark outputs.

Modern blind video/DCT competitors require independently sourced faithful code.
This adapter only accepts actual per-observation records with evidence metadata.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path

REQUIRED={'method_id','reference_doi','implementation_url','implementation_commit',
          'detector_class','host','watermark','attack','payload_bits','psnr',
          'nc','ber','embed_ms','extract_ms','source_result_sha256'}


def validate(rows, selected_hosts, selected_watermarks, payload=4096):
    if not rows:raise ValueError('Empty external baseline result set')
    valid_detectors={'blind','semi-blind','informed'}
    keys=set();report=[]
    for i,r in enumerate(rows,2):
        missing=REQUIRED-set(r)
        if missing:raise ValueError(f'CSV missing columns: {sorted(missing)}')
        if any(not str(r[k]).strip() for k in REQUIRED):raise ValueError(f'Row {i}: empty field')
        if r['detector_class'] not in valid_detectors:raise ValueError(f'Row {i}: detector class')
        if r['host'] not in selected_hosts or r['watermark'] not in selected_watermarks:
            raise ValueError(f'Row {i}: not a member of frozen held-out protocol')
        if int(r['payload_bits'])!=payload:raise ValueError(f'Row {i}: not {payload} bits')
        for k in ('psnr','nc','ber','embed_ms','extract_ms'):
            if not math.isfinite(float(r[k])):raise ValueError(f'Row {i}: nonfinite {k}')
        if not (-1<=float(r['nc'])<=1) or not (0<=float(r['ber'])<=1):
            raise ValueError(f'Row {i}: invalid metric range')
        if float(r['embed_ms'])<0 or float(r['extract_ms'])<0:raise ValueError(f'Row {i}: negative runtime')
        key=(r['method_id'],r['host'],r['watermark'],r['attack'])
        if key in keys:raise ValueError(f'Duplicate trial {key}')
        keys.add(key);report.append(dict(r))
    return report


def main(a):
    evidence=json.loads(Path(a.split_manifest).read_text())
    selected_hosts={item['name'] for item in evidence['test']}
    wms={x.name for x in Path(a.watermark_dir).glob('*.png')}
    with Path(a.input).open(newline='',encoding='utf8') as fd: rows=list(csv.DictReader(fd))
    checked=validate(rows,selected_hosts,wms,a.payload_bits)
    target=Path(a.out_dir);target.mkdir(parents=True,exist_ok=True)
    out=target/'validated_external_methods.csv'
    with out.open('w',newline='',encoding='utf8') as fd:
        writer=csv.DictWriter(fd,fieldnames=list(checked[0]));writer.writeheader();writer.writerows(checked)
    (target/'external_import_manifest.json').write_text(json.dumps({
        'validated_rows':len(checked),'method_ids':sorted({x['method_id'] for x in checked}),
        'data_provenance':'provided by external source; method execution not verified by this importer',
        'important':'this CSV must NOT be combined with matched-PSNR primary ranking unless PSNR eligibility and per-method attack pairing pass separate checks'},indent=2))
    print('Validated',len(checked),'external baseline observations')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True);p.add_argument('--split-manifest',required=True)
    p.add_argument('--watermark-dir',required=True);p.add_argument('--out-dir',required=True)
    p.add_argument('--payload-bits',type=int,default=4096)
    main(p.parse_args())
