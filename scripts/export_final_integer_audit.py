#!/usr/bin/env python3
"""Aggregate observed integer counters from unique clean embeddings (not 45x attacks)."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path

FIELDS=('integer_lattice_initial_parity_failures','integer_lattice_initial_guard_failures',
        'integer_lattice_corrected_blocks','integer_lattice_unit_updates','integer_lattice_unresolved',
        'integer_lattice_payload_blocks')

def main(source,out):
    with Path(source).open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    unique={};errors=[]
    for r in rows:
        if r.get('method_id')!='proposal' or r.get('attack')!='clean':continue
        key=(r.get('host'),r.get('watermark_name'))
        if key in unique: errors.append(f'duplicate clean case {key}')
        unique[key]=r
    if errors or not unique: raise ValueError(';'.join(errors) or 'No clean proposal embeddings found')
    totals={}
    for f in FIELDS:
        values=[r.get(f) for r in unique.values()]
        if any(v is None or v=='' for v in values):
            totals[f]=None; continue
        totals[f]=sum(int(v) for v in values)
    if totals['integer_lattice_payload_blocks'] is None:
        raise ValueError('Missing integer payload counters: cannot claim correction rate')
    total=totals['integer_lattice_payload_blocks']
    summary={'unique_clean_host_watermark_embeddings':len(unique),'counters':totals,
        'initial_parity_failure_rate':totals['integer_lattice_initial_parity_failures']/total,
        'initial_guard_failure_rate':totals['integer_lattice_initial_guard_failures']/total,
        'corrected_block_rate':totals['integer_lattice_corrected_blocks']/total,
        'unresolved_count':totals['integer_lattice_unresolved'],
        'caveat':'Successes counted only for accepted clean embeddings; rejected attempts require rejection-log counters and cannot be inferred from successful keys.'}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'integer_clean_embedding_summary.json').write_text(json.dumps(summary,indent=2))
    print('Integer clean audit:',len(unique),'unique embeddings',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--out-dir',required=True)
    a=p.parse_args();main(a.input,a.out_dir)
