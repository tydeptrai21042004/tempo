#!/usr/bin/env python3
"""Numerical verification of continuous TWO-CARRIER minimum energy and uint8 effects.

This numerical experiment is not a substitute for the theorem's mathematical
proof and never claims discrete global optimality.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from reviewer_expansion_common import *
from realtime_watermark.core import dct_basis


def main(a):
    hosts,_=dataset(a); cfg=config(a.config);rng=np.random.default_rng(a.seed)
    phi01=dct_basis(8,0,1).astype(float);phi10=dct_basis(8,1,0).astype(float)
    ap=(phi01-phi10)/2; as_=(phi01+phi10)/2
    eta=float(cfg.eta)
    g=np.array([.299+eta,.587-eta/2,.114-eta/2],float)
    g2=float(g@g)
    wp=ap[:,:,None]*g;ws=as_[:,:,None]*g
    # The explicit two-constraint solution: x* = 2(dp ap+ds as)g / ||g||^2.
    rows=[]
    for hp in hosts:
        im=load_host(hp)
        h,w=im.shape[:2]; blocks=im[:h-h%8,:w-w%8].reshape((h//8),8,(w//8),8,3).transpose(0,2,1,3,4).reshape(-1,8,8,3)
        choose=rng.choice(len(blocks),size=min(len(blocks),4 if a.quick else 80),replace=False)
        for idx in choose:
            original=blocks[idx].astype(float)
            dp=float(rng.uniform(-12,12));ds=float(rng.uniform(-12,12))
            delta=2*(dp*wp+ds*ws)/g2
            continuous=original+delta
            integer=np.clip(np.rint(continuous),0,255)
            ec=continuous-original;eq=integer-original
            predicted=float(2*(dp*dp+ds*ds)/g2)
            measured=float(np.sum(ec*ec))
            rows.append({'host':hp.name,'block':int(idx),'dp':dp,'ds':ds,
                'continuous_dp':float(np.sum(ec*wp)), 'continuous_ds':float(np.sum(ec*ws)),
                'uint8_dp':float(np.sum(eq*wp)),'uint8_ds':float(np.sum(eq*ws)),
                'continuous_energy':measured,'predicted_minimum_energy':predicted,
                'quantized_payload_displacement':float(np.sum(eq*wp)),
                'quantized_pilot_displacement':float(np.sum(eq*ws)),
                'round_clip_payload_error':float(np.sum((integer-continuous)*wp)),
                'round_clip_pilot_error':float(np.sum((integer-continuous)*ws)),
                'energy_abs_error':abs(measured-predicted),
                'max_rounding_carrier_error':float(max(abs(np.sum((integer-continuous)*wp)),abs(np.sum((integer-continuous)*ws)))),
                'clipped_channels':int(np.count_nonzero((continuous<0)|(continuous>255))),
                'integer_energy':float(np.sum(eq*eq))})
    out=Path(a.out_dir)
    write_rows(out/'joint_carrier_numerical.csv',rows)
    write_json(out/'manifest.json',{'status':'complete','config_sha256':cfg.sha256(),
        'samples':len(rows),'basis_inner_product':float(np.sum(ap*as_)),
        'max_continuous_energy_error':float(max(r['energy_abs_error'] for r in rows)),
        'theoretical_min_energy':'2*(dp**2+ds**2)/||g||^2',
        'caveat':'Integer projection is clipped and rounded; its energy need not be globally minimal. This tests the continuous theorem numerically only.',**info()})

if __name__=='__main__':main(base_args(argparse.ArgumentParser(description=__doc__)).parse_args())
