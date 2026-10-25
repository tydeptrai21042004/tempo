#!/usr/bin/env python3
"""Summarize verified full-coverage Kodak upstream perturbation results.
No thresholds, data substitution or scientific-performance acceptance gate.
"""
from __future__ import annotations
import argparse,csv,json,math,statistics
from pathlib import Path
from collections import defaultdict

def summarize(manifest_path:Path,raw_path:Path,out:Path):
    m=json.loads(Path(manifest_path).read_text())
    if not (m.get('full_coverage') and len(m.get('hosts',[]))==24 and
            len(m.get('watermarks',[]))==2 and m.get('methods')==['proposal']):
        raise RuntimeError('Full 24x2 proposal-only MMWatermark coverage is required')
    with Path(raw_path).open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    expected=m.get('expected_full_requested_rows')
    if len(rows)!=expected or m.get('completed_rows')!=expected:
        raise RuntimeError(f'Expected {expected} genuine external rows; got {len(rows)}')
    seen=set();g=defaultdict(list);hosts=defaultdict(list);payloads=set()
    for r in rows:
        if r.get('method_id')!='proposal':raise RuntimeError('Non-proposal row found')
        if r.get('host') not in m['hosts'] or r.get('watermark') not in m['watermarks']:
            raise RuntimeError('Unexpected host/watermark')
        payload=int(r['payload_bits']);payloads.add(payload)
        key=(r['host'],r['watermark'],r['method_id'],r['attack'])
        if key in seen:raise RuntimeError('Duplicate attack trial: '+str(key))
        seen.add(key);g[(r['watermark'],r['attack'])].append(r)
        hosts[(r['watermark'],r['host'])].append(r)
    if payloads!={4096}:raise RuntimeError(f'Inconsistent payloads: {payloads}')
    expected_cond=1+len(m['selected_attack_functions'])*len(m['severities'])
    if len(seen)!=24*2*expected_cond:raise RuntimeError('Missing upstream conditions')
    for k,rs in hosts.items():
        if len(rs)!=expected_cond:raise RuntimeError('Incomplete host-watermark: '+str(k))
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    def stats(rs):
        doc={'trials':len(rs)}
        for k in ('clean_embedding_psnr','nc_bipolar','ber','embed_ms','extract_ms'):
            v=[float(r[k]) for r in rs]
            if not all(math.isfinite(x) for x in v):raise RuntimeError('Nonfinite '+k)
            doc['mean_'+k]=statistics.fmean(v)
            doc['std_'+k]=statistics.stdev(v) if len(v)>1 else 0.0
            doc['min_'+k]=min(v);doc['max_'+k]=max(v)
        return doc
    outputs=[('by_watermark_attack.csv', [{'watermark':k[0], 'attack':k[1],**stats(rs)} for k,rs in sorted(g.items())]),
             ('by_watermark_host.csv',[{'watermark':k[0], 'host':k[1],**stats(rs)} for k,rs in sorted(hosts.items())]),
             ('by_watermark_overall.csv',[{'watermark':wm, **stats([r for r in rows if r['watermark']==wm])} for wm in m['watermarks']])]
    for filename,data in outputs:
        with (out/filename).open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    result={'status':'COMPLETE','provider':'official_MMWatermark_Robustness',
            'dataset':'Kodak 24 lossless center crops 512x512 BMP',
            'methods':['proposal'],'payload_bits':4096,'hosts':24,'watermarks':2,
            'conditions_per_pair':expected_cond,'trials':len(rows),
            'mean_metrics_are_measured_not_thresholds':True,
            'by_watermark_attack_groups':len(g),'by_watermark_host_groups':len(hosts),
            'output_csv':[p[0] for p in outputs]}
    (out/'summary_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--raw',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,required=True)
    a=ap.parse_args();print(json.dumps(summarize(a.manifest,a.raw,a.out_dir),indent=2))
if __name__=='__main__':main()
