#!/usr/bin/env python3
"""Fail-closed canonical internal attack audit, reproducible fingerprint."""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from realtime_watermark.attack_manifest import (load_manifest,load_reviewer_manifest,
    reviewer_manifest_summary,REVIEWER_MANIFEST)

def run(out: Path, source_copy: Path|None=None):
    internal=load_reviewer_manifest(); original=load_manifest()
    a={x['id']:x for x in original['attacks']}
    for e in internal['attacks']:
        if e != a.get(e['id']):
            raise RuntimeError(f"Altered operator/parameters for {e['id']}: not the original attack")
    summary=reviewer_manifest_summary()
    assert summary['total_conditions']<50
    out.parent.mkdir(parents=True,exist_ok=True)
    if source_copy:
        source_copy.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(REVIEWER_MANIFEST,source_copy)
    summary.update({'status':'PASS','source_registry':'unified_109.json',
        'external_exception':'Kodak full native remains unified109 and upstream official MMWatermark retains its own catalog',
        'attack_definitions_unchanged':True,
        'frozen_attack_manifest':str(REVIEWER_MANIFEST.resolve())})
    out.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source-copy',type=Path)
    x=p.parse_args();print(json.dumps(run(x.out,x.source_copy),indent=2))
