#!/usr/bin/env python3
"""Fail-closed common-suite completeness check without making scientific claims.

The main method comparison must contain every 46-condition attack for every
recorded host/watermark/method. Matched-PSNR must contain the same 46 cases
before eligibility filtering; missing pairs remain a failure, not zeros.
"""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from realtime_watermark.attack_manifest import reviewer_manifest_summary

def read_rows(p:Path):
    if not p.is_file(): raise FileNotFoundError(p)
    with p.open(newline='',encoding='utf8') as handle: return list(csv.DictReader(handle))

def audit(root:Path):
    spec=reviewer_manifest_summary()
    ids=set(spec['attack_ids']);out={'protocol':spec['protocol_id'],'sha256':spec['sha256'],'conditions_per_pair':len(ids),'stages':{}}
    main=read_rows(root/'all_methods_all_metrics'/'all_watermarks_all_methods_all_metrics.csv')
    group={}
    for row in main:
        key=(row['host'],row['watermark_name'],row['method_id'])
        group.setdefault(key,[]).append(row['attack'])
        if int(float(row['payload_bits']))!=4096:
            raise RuntimeError(f'Primary payload mismatch: {key}: {row["payload_bits"]}')
    failures=[]
    for key,observed in group.items():
        if len(observed)!=len(ids) or set(observed)!=ids:
            failures.append({'pair':key,'missing':sorted(ids-set(observed)),'extra':sorted(set(observed)-ids),'rows':len(observed)})
    out['stages']['all_methods']={'pairs':len(group),'observations':len(main),'expected':len(group)*len(ids),'failed_pairs':failures[:25]}
    matched=read_rows(root/'matched_psnr'/'matched_psnr_per_attack.csv')
    group2={};invalid=[]
    for row in matched:
        key=(row['host'],row['watermark'],row['method_id'])
        group2.setdefault(key,[]).append(row['phase'])
        if 'payload_bits' in row and row['payload_bits'] and int(float(row['payload_bits']))!=4096:
            invalid.append(key)
    failures2=[]
    for key,observed in group2.items():
        if set(observed)!=ids or len(observed)!=len(ids):
            failures2.append({'pair':key,'missing':sorted(ids-set(observed)), 'extra':sorted(set(observed)-ids),'rows':len(observed)})
    out['stages']['matched_psnr']={'pairs':len(group2),'observations':len(matched),'failed_pairs':failures2[:25],'wrong_payload_pairs':invalid[:25],
      'note':'Primary eligible rows are an independent subset; do not count rejected methods as successful comparisons'}
    main_methods={k[2] for k in group};matched_methods={k[2] for k in group2}
    out['stages']['method_alignment']={'main_methods':sorted(main_methods),'matched_methods':sorted(matched_methods),
                                      'missing_matched_methods':sorted(matched_methods-main_methods)}
    out['status']='COMPLETE' if group and group2 and not(failures or failures2 or invalid or (matched_methods-main_methods)) else 'INCOMPLETE'
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--reports',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    report=audit(args.reports);args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print('46-condition internal coverage:', report['status'])
    if report['status']!='COMPLETE':raise SystemExit(1)
