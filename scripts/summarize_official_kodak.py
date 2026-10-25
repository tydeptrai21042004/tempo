#!/usr/bin/env python3
"""Summarize real official Kodak MMWatermark rows without inventing unavailable attacks.

complete_supported_catalog=True means all preflight-supported trials exist.
full_original_catalog=True means *no* upstream conditions were excluded.
These are different claims and MUST NOT be conflated.
"""
from __future__ import annotations
import argparse,csv,json,math,statistics
from pathlib import Path
from collections import defaultdict


def summarize(manifest_path:Path,raw_path:Path,out:Path):
    m=json.loads(Path(manifest_path).read_text(encoding='utf8'))
    if not (m.get('complete') and len(m.get('hosts',[]))==24 and
            len(set(m['hosts']))==24 and len(m.get('watermarks',[]))==2 and
            len(set(m['watermarks']))==2 and m.get('methods')==['proposal']):
        raise RuntimeError('All 24 Kodak hosts, both watermarks and completed proposal trials required')
    unsupported=m.get('preflight_unsupported_conditions',[])
    if not isinstance(unsupported,list):raise RuntimeError('Malformed preflight exclusions')
    requested={(a,int(sev)) for a in m['selected_attack_functions'] for sev in m['severities']}
    if len(requested)!=len(m['selected_attack_functions'])*len(m['severities']):
        raise RuntimeError('Duplicate requested attack/severity')
    omitted={(x['attack'],int(x['severity'])) for x in unsupported}
    if len(omitted)!=len(unsupported) or not omitted<=requested:
        raise RuntimeError('Duplicated or unexpected unsupported conditions')
    supported=requested-omitted
    labels={'clean'}|{f'mmw::{a}::severity{sev}' for a,sev in supported}
    expected_per_pair=len(labels)
    expected=24*2*expected_per_pair
    expected_full=24*2*(1+len(requested))
    if (m.get('expected_compatible_rows')!=expected or
        m.get('completed_rows')!=expected or
        m.get('expected_full_requested_rows')!=expected_full or
        bool(m.get('full_coverage')) != (not omitted and m.get('complete') and expected==expected_full) or
        m.get('failures')):
        raise RuntimeError('Manifest counts/coverage claims contradict genuine condition inventory')
    with Path(raw_path).open(newline='',encoding='utf8') as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=expected:
        raise RuntimeError(f'Expected {expected} completed supported trials, got {len(rows)}')
    seen=set(); per_att=defaultdict(list); per_host=defaultdict(list); by_mark=defaultdict(list)
    for row in rows:
        if row['method_id']!='proposal' or row['host'] not in m['hosts'] or row['watermark'] not in m['watermarks'] or row['attack'] not in labels:
            raise RuntimeError('Unexpected method, Kodak host, watermark or attack label')
        if int(row['payload_bits'])!=4096:
            raise RuntimeError('External comparison requires genuine 4096-bit payload')
        key=(row['host'],row['watermark'],row['method_id'],row['attack'])
        if key in seen:raise RuntimeError('Duplicate external benchmark row '+str(key))
        seen.add(key)
        per_att[(row['watermark'],row['attack'])].append(row)
        per_host[(row['watermark'],row['host'])].append(row)
        by_mark[row['watermark']].append(row)
    for mark in m['watermarks']:
        for host in m['hosts']:
            actual={r['attack'] for r in per_host[(mark,host)]}
            if actual!=labels:
                raise RuntimeError(f'Missing supported attacks for {mark}/{host}: {sorted(labels-actual)}')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    def calc(rs):
        stats={'trials':len(rs)}
        for metric in ('clean_embedding_psnr','nc_bipolar','ber','embed_ms','extract_ms'):
            values=[float(row[metric]) for row in rs]
            if not all(math.isfinite(v) for v in values):raise RuntimeError('Nonfinite '+metric)
            stats['mean_'+metric]=statistics.fmean(values)
            stats['std_'+metric]=statistics.stdev(values) if len(values)>1 else 0.0
            stats['min_'+metric]=min(values)
            stats['max_'+metric]=max(values)
        return stats
    outputs={
        'by_watermark_attack.csv':[{'watermark':k[0],'attack':k[1],**calc(rs)} for k,rs in sorted(per_att.items())],
        'by_watermark_host.csv':[{'watermark':k[0],'host':k[1],**calc(rs)} for k,rs in sorted(per_host.items())],
        'by_watermark_overall.csv':[{'watermark':k,**calc(rs)} for k,rs in sorted(by_mark.items())],
    }
    for filename,data in outputs.items():
        with (out/filename).open('w',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    with (out/'upstream_unsupported_conditions.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=['attack','severity','reason']);writer.writeheader()
        for x in unsupported:writer.writerow({k:x.get(k,'') for k in ('attack','severity','reason')})
    result={
        'status':'COMPLETE' if not omitted else 'COMPLETE_SUPPORTED_CATALOG_ONLY',
        'provider':'official_MMWatermark_Robustness',
        'dataset':'Kodak 24 lossless 512x512 center-cropped RGB BMPs',
        'methods':['proposal'],'payload_bits':4096,'hosts':24,'watermarks':2,
        'supported_conditions_per_pair':expected_per_pair,
        'original_requested_conditions_per_pair':1+len(requested),
        'trials':len(rows),'completed_supported_rows':len(rows),'expected_supported_rows':expected,
        'unavailable_upstream_conditions':len(unsupported),
        'missing_supported_trials':0,'full_original_catalog':not bool(unsupported),
        'all_supported_trials_complete':True,
        'warning':('Unsupported upstream attack/severity conditions have no metrics; see '
                   'upstream_unsupported_conditions.csv. Do not claim full requested coverage.' if unsupported else ''),
        'outputs':list(outputs)+['upstream_unsupported_conditions.csv'],
    }
    (out/'summary_manifest.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--raw',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,required=True)
    a=ap.parse_args()
    print(json.dumps(summarize(a.manifest,a.raw,a.out_dir),indent=2))

if __name__=='__main__':main()
