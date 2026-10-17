#!/usr/bin/env python3
"""Distortion-calibrated public DCT-carrier and pilot attenuation attack study.

Attacks modify a COPY of the watermarked image. Never alter proposal code.
Each negative quality cost is measured against the actual marked image.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from reviewer_expansion_common import *
from reviewer_experiments import attenuation


def run(a):
    init_cpu()
    out=Path(a.out_dir)
    hosts,marks=dataset(a)
    cfg=config(a.config)
    strengths=[0.,.5,1.] if a.quick else [0.,.1,.25,.5,.75,1.]
    rows=[]
    for hp in hosts:
        im=load_host(hp)
        for wi,wp in enumerate(marks):
            wm=mark(wp)
            marked,key,_=encode(im,wm,cfg,case_seed(a.seed,wi))
            for which in ('payload','pilot'):
                for alpha in strengths:
                    attacked=marked if alpha==0 else attenuation(marked,float(key.params['eta']),which,alpha)
                    bits,meta,ms=decode(attacked,key)
                    braw,_,_=decode(attacked,key,sync=False)
                    rows.append({'host':hp.name,'watermark':wp.name,'target':which,
                      'attenuation_fraction':alpha,
                      'attack_psnr_vs_marked_db':float(psnr(marked,attacked)),
                      'marked_psnr_vs_host_db':float(psnr(im,marked)),
                      'attacked_psnr_vs_host_db':float(psnr(im,attacked)),
                      'nc_sync':float(nc_binary(wm,bits)),
                      'nc_raw':float(nc_binary(wm,braw)),
                      'ber_sync':float(1-bit_accuracy(wm,bits)),
                      'extract_ms':ms,'sync_path':meta.get('path','')})
    write_rows(out/'attack_budget_cases.csv',rows)
    write_json(out/'manifest.json',{'status':'complete','cases':len(rows),
       'attack_definition':'Key-independent attenuation of public low-frequency carrier direction',
       'limitations':['PSNR measures distortion cost; this is not an optimized perceptual attack','No key entropy or cryptographic security claim'],
       'config_sha256':cfg.sha256(),**info()})


def parser():return base_args(argparse.ArgumentParser(description=__doc__))
if __name__=='__main__':run(parser().parse_args())
