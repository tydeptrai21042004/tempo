#!/usr/bin/env python3
"""Stage real repository images and a documented pre-existing frozen proposal.

This is intentionally NOT a replacement for the full TEMPO optimisation pipeline.
No parameter tuning takes place on the held-out 9-host test set. The committed
2026-10-21 split and frozen proposal are used only when the notebook lacks the
actual native pipeline outputs; this distinction is saved in a manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from realtime_watermark.proposals.jilp_qim.config import JILPQIMConfig
from scripts.create_reviewer_holdout import split_hosts


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(repo: Path, run: Path, *, config: Path | None = None,
            reference_split: Path | None = None) -> dict:
    repo = Path(repo).resolve()
    run = Path(run).resolve()
    input_root = repo / 'data/user_dataset'
    host_dir = input_root / 'host'
    wm_dir = input_root / 'watermark'
    if not host_dir.is_dir() or not wm_dir.is_dir():
        raise FileNotFoundError('Real repository BMP/PNG dataset unavailable')
    hosts = sorted(x for x in host_dir.glob('*.bmp')
                   if not any(n in x.name.casefold() for n in ('lena','lenna')))
    watermarks = sorted(wm_dir.glob('*.png'))
    if len(hosts) != 12 or len(watermarks) != 2:
        raise RuntimeError(f'Expected 12 real hosts and 2 PNG marks; got {len(hosts)}/{len(watermarks)}')
    reference_split = Path(reference_split or repo / 'configs/holdout_split_manifest_20261021.json')
    reference = json.loads(reference_split.read_text(encoding='utf-8'))
    # Compare the byte-level input dataset, including the held-out allocation.
    for group, selected in [('development', hosts[:3]), ('test', hosts[3:])]:
        saved = reference[group]
        if len(saved) != len(selected):
            raise RuntimeError(f'{group} allocation differs from the committed split')
        for item, source in zip(saved, selected):
            if item['name'].split('_', 2)[-1] != source.name or item['sha256'] != sha256(source):
                raise RuntimeError(f'{group} hash/split mismatch for {source.name}')
    frozen = Path(config or repo / 'configs/frozen_selected_20261021.json').resolve()
    obj = JILPQIMConfig.load(frozen)
    report = run / 'reports'
    select = run / 'selected_data'
    target_hosts = select / 'host'
    target_marks = select / 'watermark'
    target_hosts.mkdir(parents=True, exist_ok=True)
    target_marks.mkdir(parents=True, exist_ok=True)
    for idx, host in enumerate(hosts, start=1):
        shutil.copy2(host, target_hosts / f'host_{idx:03d}_{host.name}')
    for idx, wm in enumerate(watermarks, start=1):
        shutil.copy2(wm, target_marks / f'watermark_{idx:03d}_{wm.name}')
    split = split_hosts(target_hosts, report / 'holdout', development_count=3)
    actual_split = json.loads((report / 'holdout/split_manifest.json').read_text())
    for group in ('development','test'):
        if [(x['name'],x['sha256']) for x in actual_split[group]] != [
                (x['name'],x['sha256']) for x in reference[group]]:
            raise RuntimeError(f'Prepared split diverged from committed reference: {group}')
    target_config = report / 'configs/proposal_selected_all_watermarks.json'
    target_config.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(frozen, target_config)
    if JILPQIMConfig.load(target_config).sha256() != obj.sha256():
        raise RuntimeError('Frozen proposal parameters changed during staging')
    metadata = {
        'source': 'repository_frozen_20261021_not_newly_tuned',
        'warning': 'External MMWatermark run uses committed frozen config, NOT current Kaggle optimisation output',
        'config_sha256_file': sha256(target_config),
        'config_parameters_sha256': obj.sha256(),
        'reference_split_sha256': sha256(reference_split),
        'host_count': len(hosts), 'heldout_count': len(split['test']),
        'watermark_count': len(watermarks),
        'watermarks': [{'name': x.name, 'sha256': sha256(x)} for x in watermarks],
    }
    manifest = report / 'configs/mmwatermark_input_provenance.json'
    manifest.write_text(json.dumps(metadata, indent=2)+'\n',encoding='utf-8')
    return metadata


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--config',type=Path)
    args=p.parse_args()
    print(json.dumps(prepare(args.repo,args.run,config=args.config),indent=2))


if __name__=='__main__':
    main()
