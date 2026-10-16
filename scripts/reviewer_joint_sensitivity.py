#!/usr/bin/env python3
"""Small factorial parameter-coupling experiment on DEVELOPMENT HOSTS only.

No parameter is selected here, and no held-out image is loaded. These data
support a joint sensitivity figure, not a final best-configuration claim.
"""
from __future__ import annotations
import argparse,itertools,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from realtime_watermark.core import load_binary_watermark,psnr,nc_binary,bit_accuracy
from realtime_watermark.proposals.jilp_qim.config import JILPQIMConfig
from realtime_watermark.proposals.jilp_qim.method import embed_proposal,extract_jilp_qim
from reviewer_experiments import jpeg,warp,csv_write


def run(args):
    cfg=JILPQIMConfig.load(args.config)
    hosts=sorted(Path(args.development_dir).glob('*.bmp'))[:args.max_hosts]
    wms=sorted(Path(args.watermark_dir).glob('*.png'))[:args.max_watermarks]
    if not hosts or not wms:raise ValueError('Development images and watermark required')
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    factors=[('step',(cfg.step*.85,cfg.step*1.15)),
             ('pilot_step',(cfg.pilot_step*.85,cfg.pilot_step*1.15)),
             ('pilot_count',(min(cfg.pilot_count,64),max(cfg.pilot_count,96)))]
    rows=[]
    for hp in hosts:
        host=np.asarray(Image.open(hp).convert('RGB'))
        for wp in wms:
            wm=load_binary_watermark(wp,size=64)
            for vals in itertools.product(*[f[1] for f in factors]):
                params={**cfg.to_dict(),**dict(zip([x[0] for x in factors],vals))}
                candidate=JILPQIMConfig.from_mapping(params)
                t=time.perf_counter()
                y,k=embed_proposal(host,wm,seed=args.seed,method_params=candidate.to_dict())
                emb_ms=1000*(time.perf_counter()-t)
                for attack,received in [('clean',y),('jpeg40',jpeg(y,40)),('rotation2p5',warp(y,angle=2.5))]:
                    t=time.perf_counter()
                    dec=extract_jilp_qim(received,k)
                    ext_ms=1000*(time.perf_counter()-t)
                    rows.append({'host':hp.name,'watermark':wp.name,'payload_bits':wm.size,
                                 'attack':attack,'step':candidate.step,'pilot_step':candidate.pilot_step,
                                 'pilot_count':candidate.pilot_count,'psnr':float(psnr(host,y)),
                                 'bit_accuracy':float(bit_accuracy(wm,dec)),
                                 'nc':float(nc_binary(wm,dec)),
                                 'embed_ms':emb_ms,'extract_ms':ext_ms,
                                 'config_sha256':candidate.sha256()})
    csv_write(out/'joint_parameter_coupling.csv',rows)
    (out/'joint_sensitivity_manifest.json').write_text(json.dumps({
      'host_scope':'development only','factorial_cells':8,
      'factors':{k:list(map(float,v)) for k,v in factors},
      'hosts':[x.name for x in hosts], 'watermarks':[x.name for x in wms],
      'rows':len(rows),'no_test_host_access':True,
      'limitations':'2-level local factorial scans are not a global hyperparameter optimum'},indent=2))
    print(f'Wrote {len(rows)} joint sensitivity observations')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--development-dir',required=True);p.add_argument('--watermark-dir',required=True)
    p.add_argument('--config',required=True);p.add_argument('--out-dir',required=True)
    p.add_argument('--max-hosts',type=int,default=2);p.add_argument('--max-watermarks',type=int,default=2)
    p.add_argument('--seed',type=int,default=2026)
    run(p.parse_args())
