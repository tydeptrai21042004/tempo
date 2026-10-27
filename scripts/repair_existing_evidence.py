#!/usr/bin/env python3
"""Repair postprocessing/evidence from an earlier TEMPO Kaggle evidence ZIP.

No watermark benchmarks, codecs, official Kodak attacks, or baseline experiments
are rerun. Only measured Baboon JPEG30 replay and result-figure generation execute.
This creates a NEW derived evidence directory; input ZIP is never changed.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

REPO=Path(__file__).resolve().parents[1]


def execute(*args:object) -> None:
    cmd=[sys.executable,*map(str,args)]
    print('+',' '.join(cmd),flush=True)
    subprocess.run(cmd,check=True,cwd=REPO)


def main()->None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evidence-zip',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--source-data',type=Path,default=REPO/'data/user_dataset')
    a=ap.parse_args()
    root=a.out_dir.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'Use an empty directory to avoid mixing revisions: {root}')
    root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.evidence_zip) as archive:
        for member in archive.infolist():
            name=Path(member.filename)
            if (name.is_absolute() or '..' in name.parts
                    or (name.parts and name.parts[0] not in {'reports','run_metadata'})):
                if name.parts == ('README_EVIDENCE.txt',):
                    continue
                raise ValueError(f'Unexpected ZIP entry: {name}')
            target=(root/name).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f'Invalid ZIP path: {name}')
            if member.is_dir():
                target.mkdir(parents=True,exist_ok=True)
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(member) as read, target.open('wb') as write:
                    while True:
                        block=read.read(1024*1024)
                        if not block:break
                        write.write(block)
    reports=root/'reports'
    selected=root/'selected_data'
    execute(REPO/'scripts/select_kaggle_dataset.py',
        '--source-data',a.source_data.resolve(),'--output',selected,
        '--manifest-dir',root/'selection_metadata')
    frozen=reports/'configs/proposal_selected_all_watermarks.json'
    matched=reports/'matched_psnr'
    gallery=reports/'result_figure_sources/baboon'
    execute(REPO/'scripts/export_result_gallery.py',
        '--matched-dir',matched,'--host-dir',selected/'host',
        '--watermark-dir',selected/'watermark','--proposal-config',frozen,
        '--out-dir',gallery)
    execute(REPO/'scripts/export_common_video_baselines.py',
        '--video-dir',reports/'video', '--corpus',reports/'video_sources/corpus.json',
        '--out-dir',reports/'video_common_baselines','--mode','full')
    execute(REPO/'scripts/generate_manuscript_figures.py',
        '--results-root',reports,'--host-dir',selected/'host',
        '--watermark-dir',selected/'watermark','--config',frozen,
        '--out-dir',reports/'manuscript_figures','--include-nonresult-figures')
    execute(REPO/'scripts/reviewer_completeness.py',
        '--results-dir',reports,'--mode','full')
    checks=json.loads((reports/'reviewer_completeness.json').read_text())
    final={'kind':'postprocessing_repair_from_existing_execution',
        'new_experimental_trials':0,'official_kodak_reexecution':False,
        'original_evidence_zip':a.evidence_zip.name,
        'result_figure_evidence':checks['checks'].get('result_figure_evidence'),
        'same_video_multiclass_baseline_control':checks['checks'].get('same_video_multiclass_baseline_control'),
        'remaining_failed_checks':checks['failed_checks'],
        'limitations':'A derived evidence repair is not a new independent benchmark or proof of publication readiness.'}
    (root/'REPAIR_SUMMARY.json').write_text(json.dumps(final,indent=2)+'\n')
    print(json.dumps(final,indent=2))

if __name__=='__main__':main()
