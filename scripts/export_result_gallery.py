#!/usr/bin/env python3
"""Create actual, auditable Baboon/JPEG30 panel assets from a matched-PSNR run.

Reads frozen steps and verifies redecoded per-case NC against the corresponding
archived CSV row before writing any source PNG. No invented/hand-drawn recovery.
"""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
from realtime_watermark.core import load_binary_watermark, load_rgb, nc_binary, attack_jpeg
from realtime_watermark.registry import embed,extract
from realtime_watermark.proposals.jilp_qim.config import JILPQIMConfig


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--matched-dir',required=True);p.add_argument('--host-dir',required=True)
 p.add_argument('--watermark-dir',required=True);p.add_argument('--proposal-config',required=True)
 p.add_argument('--out-dir',required=True);p.add_argument('--host-substring',default='baboon')
 p.add_argument('--attack',default='jpeg30',choices=['jpeg30'])
 args=p.parse_args(); root=Path(args.matched_dir);out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
 proto=json.loads((root/'matched_psnr_protocol.json').read_text())
 frozen=proto.get('frozen_steps_by_watermark',{})
 source=root/'matched_psnr_per_attack.csv'
 with source.open(newline='',encoding='utf8') as f:rows=list(csv.DictReader(f))
 host=next((x for x in sorted(Path(args.host_dir).glob('*.bmp')) if args.host_substring in x.stem.lower()),None)
 if host is None: raise FileNotFoundError('Missing named held-out host '+args.host_substring)
 config=JILPQIMConfig.load(args.proposal_config)
 manifest={'protocol_file_sha256':digest(root/'matched_psnr_protocol.json'),
           'matched_rows_sha256':digest(source),'configuration_sha256':config.sha256(),
           'host':host.name,'attack':args.attack,'source_kind':'actual matched-step replay verified against archived per-case NC',
           'panels':[], 'status':'incomplete'}
 host_rgb=load_rgb(host,mode='512'); Image.fromarray(host_rgb).save(out/'host.png')
 for wm_path in sorted(Path(args.watermark_dir).glob('*.png')):
  if wm_path.name not in frozen:continue
  wm=load_binary_watermark(wm_path,size=64)
  wtag=wm_path.stem
  wm_out=out/wtag;wm_out.mkdir(exist_ok=True)
  Image.fromarray(wm).save(wm_out/'original.png')
  matching=[r for r in rows if r.get('host')==host.name and r.get('watermark')==wm_path.name and r.get('phase')==args.attack]
  if not matching: continue
  for r in matching:
   method=r.get('method_id')
   step=float(frozen[wm_path.name][method])
   method_params={'proposal':config.to_dict()}.get(method)
   marked,key=embed(method,host_rgb,wm,seed=int(r['evaluation_seed']),repeat='auto',step=step,method_params=method_params)
   attacked=attack_jpeg(marked,30)
   decoded=extract(attacked,key)
   score=float(nc_binary(wm,decoded)); expected=float(r['nc_bipolar'])
   # Exported gallery can be used as a manuscript comparison only when it
   # exactly reproduces that CSV's NC (allow source CSV roundoff only).
   if abs(score-expected)>1e-5:
    raise RuntimeError(f'Gallery/benchmark NC mismatch: {method} {wm_path.name}: {score} vs {expected}')
   Image.fromarray(marked).save(wm_out/(method+'_marked.png'))
   Image.fromarray(attacked).save(wm_out/(method+'_attacked.png'))
   Image.fromarray(np.asarray(decoded,dtype=np.uint8)).save(wm_out/(method+'_decoded.png'))
   manifest['panels'].append({'watermark':wm_path.name,'method':method,'nc':score,
      'step':step,'seed':int(r['evaluation_seed']), 'source_case':{'host':host.name,'watermark':wm_path.name,'phase':args.attack},
      'decoded_png':str(Path(wtag)/(method+'_decoded.png'))})
 if not manifest['panels']: raise RuntimeError('No matching archived benchmark cases to visualize')
 manifest['status']='complete'
 (out/'gallery_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
 print('Verified and exported',len(manifest['panels']),'matched-condition panels')

if __name__=='__main__':main()
