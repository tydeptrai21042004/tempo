#!/usr/bin/env python3
"""Clone the REAL upstream repositories and run TEMPO proposal *from inside* the
original MMWatermark-Robustness checkout, without modifying tracked upstream code.

Produces an auditable untracked integration entry at
<official>/perturbations/tempo_proposal_runner.py, then invokes that entry from
<official>/perturbations. This entry loads genuine official attacks and embeds/
extracts via the selected TEMPO proposal implementation in the TEMPO source tree.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys
from PIL import Image
from pathlib import Path

import run_official_mmwatermark_tempo as official


def install_bridge(official_repo: Path, tempo_source: Path) -> tuple[Path,str]:
    """Require pristine upstream Git BEFORE installing a single untracked entry."""
    official.prepare_repo(official_repo, clone=True)
    bridge = official_repo/'perturbations'/'tempo_proposal_runner.py'
    if bridge.exists():
        raise RuntimeError(f'Bridge already exists; use a fresh official checkout: {bridge}')
    if not (bridge.parent/'image_perturb.py').is_file():
        raise RuntimeError('Original upstream image_perturb.py is missing')
    shutil.copyfile(tempo_source, bridge)
    digest = hashlib.sha256(bridge.read_bytes()).hexdigest()
    return bridge, digest


def build_command(args: argparse.Namespace, bridge: Path, upstream_sha: str) -> list[str]:
    cmd = [sys.executable, str(bridge),
           '--no-clone','--official-repo',str(args.official_repo),
           '--expected-sha',upstream_sha,
           '--companion-repo',str(args.companion_repo),
           '--host-dir',str(args.host_dir),
           '--watermark-dir',str(args.watermark_dir),
           '--config',str(args.config),
           '--out-dir',str(args.out_dir),
           '--methods','proposal', '--seed',str(args.seed),
           '--protocol',str(args.protocol),
           '--severities',*[str(x) for x in args.severities]]
    if args.attacks:
        cmd.extend(['--attacks',args.attacks])
    if args.host_limit:
        cmd.extend(['--host-limit',str(args.host_limit)])
    if args.watermark_limit:
        cmd.extend(['--watermark-limit',str(args.watermark_limit)])
    if not args.allow_partial:
        cmd.append('--strict-official-coverage')
    return cmd


def main(args: argparse.Namespace) -> None:
    tempo_root = Path(__file__).resolve().parents[1]
    entry = tempo_root/'scripts'/'run_official_mmwatermark_tempo.py'
    for name in ('official_repo','companion_repo','host_dir','watermark_dir','config','out_dir'):
        setattr(args,name,getattr(args,name).expanduser().resolve())
    if args.official_repo==args.companion_repo or args.companion_repo.is_relative_to(args.official_repo):
        raise ValueError('Companion checkout must be separate from the original Git checkout')
    if args.out_dir.is_relative_to(args.official_repo) or args.out_dir.is_relative_to(args.companion_repo):
        raise ValueError('Evidence directory must be outside upstream Git checkouts')
    # Fail before cloning two upstream Git repositories. The integration entry
    # only enumerates BMP hosts; PNG center crops made the prior Kaggle job fail.
    host_files=sorted(p for p in args.host_dir.glob('*.bmp') if p.stem.lower() not in ('lena','lenna'))
    if not host_files:
        raise RuntimeError('Official MMWatermark requires BMP hosts (*.bmp), but none were found in '+str(args.host_dir))
    wm_files=sorted(args.watermark_dir.glob('*.png'))
    if not wm_files:
        raise RuntimeError('Official MMWatermark requires PNG watermark files in '+str(args.watermark_dir))
    for host in host_files:
        with Image.open(host) as im:
            if im.format!='BMP' or im.size!=(512,512) or im.mode!='RGB':
                raise RuntimeError(f'Invalid 512x512 RGB BMP host: {host} (format={im.format}, size={im.size}, mode={im.mode})')
    for mark in wm_files:
        with Image.open(mark) as im:
            if im.size!=(64,64):
                raise RuntimeError(f'Expected full 64x64 (4096-bit) watermark: {mark}: {im.size}')
    if not args.config.is_file():
        raise RuntimeError('Frozen proposal config not found: '+str(args.config))
    # Clone/verify both originals before generating any integration source.
    upstream_sha = official.prepare_repo(args.official_repo)
    companion_sha = official.prepare_companion(args.companion_repo)
    bridge, digest = install_bridge(args.official_repo, entry)
    environment=os.environ.copy()
    environment['TEMPO_OFFICIAL_BRIDGE_SHA256']=digest
    environment['PYTHONDONTWRITEBYTECODE']='1'
    previous=environment.get('PYTHONPATH','')
    environment['PYTHONPATH']=os.pathsep.join(filter(None,(str(tempo_root/'src'),str(tempo_root),previous)))
    command=build_command(args,bridge,upstream_sha)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    evidence={'official_git_url':official.OFFICIAL_URL,'official_git_commit':upstream_sha,
              'companion_git_url':official.COMPANION_URL,'companion_git_commit':companion_sha,
              'integration_bridge':str(bridge),'bridge_sha256':digest,
              'cwd':str(bridge.parent),'tempo_source':str(entry),
              'tempo_source_sha256':hashlib.sha256(entry.read_bytes()).hexdigest(),
              'command':command,'strict_coverage':not args.allow_partial}
    (args.out_dir/'official_integration_provenance.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print('OFFICIAL checkout:',args.official_repo,upstream_sha,flush=True)
    print('PROPOSAL ENTRY inside original repository:',bridge,flush=True)
    print('COMPANION checkout:',args.companion_repo,companion_sha,flush=True)
    print('EXECUTING from:',bridge.parent,flush=True)
    # The TEMPO proposal's embed/extract implementation runs through the bridge;
    # the perturbation functions are imported from upstream source, unmodified.
    subprocess.run(command,cwd=bridge.parent,env=environment,check=True)
    manifest=args.out_dir/'official_mmwatermark_manifest.json'
    if not manifest.is_file():
        raise RuntimeError('Upstream integration ended without writing the result manifest')
    state=json.loads(manifest.read_text())
    print('Full requested coverage:',state.get('full_coverage'),
          'Completed/expected:',state.get('completed_rows'),
          state.get('expected_full_requested_rows'),flush=True)
    if not args.allow_partial and not state.get('full_coverage',False):
        raise RuntimeError('Original repository run did not pass full official coverage')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--official-repo',type=Path,required=True,
                   help='Empty destination, cloned directly from official Jielin-Qiu upstream')
    p.add_argument('--companion-repo',type=Path,required=True,
                   help='Empty separate destination, cloned from official MM_Robustness upstream')
    p.add_argument('--host-dir',type=Path,required=True)
    p.add_argument('--watermark-dir',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--severities',type=int,nargs='+',default=[1,2,3,4,5])
    p.add_argument('--protocol',choices=['paper100','source_all','custom'],default='paper100',
                   help='Default: exactly the original paper’s 20 families and five severities')
    p.add_argument('--attacks',default='',help='Only for custom; paper100 resolves the official 20 function aliases')
    p.add_argument('--host-limit',type=int,default=0)
    p.add_argument('--watermark-limit',type=int,default=0)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--allow-partial',action='store_true',
                   help='By default, ALL upstream attack/severity conditions must pass.')
    main(p.parse_args())
