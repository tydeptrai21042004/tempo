#!/usr/bin/env python3
"""Small exact RESTRICTED local integer feasibility reference on real payload blocks.

This exhausts three selected 8-bit coordinates within +/-radius of the actual
encoded block, retaining the remaining 189 coordinates fixed. It is NOT a global
192-variable integer optimum and does not establish full algorithm optimality.
"""
from __future__ import annotations
import argparse
import itertools
from pathlib import Path
import numpy as np
from reviewer_expansion_common import *
from realtime_watermark.core import dct_basis
from realtime_watermark.proposals.jilp_qim.method import _schedules_from_key


def carrier(eta):
    spatial=(dct_basis(8,0,1)-dct_basis(8,1,0))/2.
    g=np.array([.299+eta,.587-.5*eta,.114-.5*eta])
    return spatial[:,:,None]*g[None,None,:]


def search_local(host_block, encoded_block, weighted, step, rho_frac, radius=2):
    x=encoded_block.astype(np.int16)
    h=host_block.astype(np.int16)
    orig=float(np.sum((x.astype(float)-h)**2))
    # The encoded block itself supplies the desired QIM parity/guard coset.
    v=float(np.sum(x*weighted));q=int(np.rint(v/step));parity=q%2
    coords=np.argsort(np.abs(weighted.reshape(-1)))[-3:]
    old=x.reshape(-1)[coords]
    host_values=h.reshape(-1)[coords]
    w=weighted.reshape(-1)[coords]
    fixed=v-float(old@w)
    # Exact enumeration over all 3D integer lattice points in the stated box.
    original_feasible=abs(v/step-np.rint(v/step))<=0.5-rho_frac+1.e-6
    best=orig if original_feasible else float('inf')
    best_vector=None;feasible=0;evaluated=0
    for deviations in itertools.product(range(-radius,radius+1),repeat=3):
        trial=old+np.asarray(deviations,dtype=np.int16)
        if np.any(trial<0) or np.any(trial>255):continue
        evaluated+=1
        projection=fixed+float(trial@w)
        rounded=np.rint(projection/step)
        if int(rounded)%2 != parity:continue
        if abs(projection/step-rounded)>0.5-rho_frac+1.e-6:continue
        feasible+=1
        energy=orig-float(np.sum((old-host_values)**2))+float(np.sum((trial-host_values)**2))
        if energy<best-1e-9:
            best=energy;best_vector=trial.tolist()
    return {'original_energy':orig,
        'best_restricted_energy':float(best) if np.isfinite(best) else None,
        'certified_local_saving':max(0.,orig-best) if original_feasible and np.isfinite(best) else None,
        'fraction_local_saving':max(0.,(orig-best)/orig) if orig and original_feasible and np.isfinite(best) else None,
        'total_enumerated':evaluated,'feasible_candidates':feasible,
        'strict_restricted_improvement':bool(original_feasible and best<orig-1e-9),
        'original_feasible_for_requested_guard':bool(original_feasible),
        'restricted_neighborhood_has_feasible_candidate':bool(feasible),
        'tested_coordinates_flat':coords.tolist(),'best_values':best_vector,
        'original_guard':float(0.5-abs(v/step-np.rint(v/step))),
        'original_passed_requested_margin':bool(original_feasible)}


def run(a):
    init_cpu()
    out=Path(a.out_dir)
    hosts,marks=dataset(a)
    cfg=config(a.config)
    rows=[]
    for hp in hosts:
        im=load_host(hp)
        for wi,wp in enumerate(marks):
            wm=mark(wp)
            marked,key,_=encode(im,wm,cfg,case_seed(a.seed,wi))
            payload,pilots,_,_,_,wc=_schedules_from_key({**key.params, 'host_shape': tuple(im.shape), 'watermark_shape': tuple(wm.shape)})
            width_blocks=wc//8
            pilot_set=set(int(i) for i in pilots)
            chosen=[int(i) for i in payload if int(i) not in pilot_set][:a.blocks]
            if len(chosen)<a.blocks:
                raise ValueError('Not enough nonoverlapping payload blocks for restricted integer study')
            weight=carrier(float(key.params['eta']))
            step=float(key.params['step_carrier1'])
            rho=float(key.params['rho_frac'])
            for index in chosen:
                r=(index//width_blocks)*8;c=(index%width_blocks)*8
                result=search_local(im[r:r+8,c:c+8],marked[r:r+8,c:c+8],weight,step,rho,a.radius)
                rows.append({'host':hp.name,'watermark':wp.name,'block_index':index,
                    'radius':a.radius,**result})
    write_rows(out/'restricted_integer_optimality.csv',rows)
    write_json(out/'manifest.json',{'status':'complete','block_cases':len(rows),'coordinate_dimension':3,
      'method_not_changed':True, 'limitation':'Exact only in the declared 3D integer neighborhood with other 189 coordinates frozen; cannot call it the global 192D optimum',
      'host_sha256':{p.name:md5_file(p) for p in hosts},'config_sha256':cfg.sha256(),**info()})


def parser():
    p=base_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument('--blocks',type=int,default=10)
    p.add_argument('--radius',type=int,default=2)
    return p
if __name__=='__main__':run(parser().parse_args())
