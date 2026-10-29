#!/usr/bin/env python3
"""Payload capacity versus image quality and the complete frozen 45-attack protocol.

A watermark of side n carries n*n binary bits; 2048 is not square and is not
claimed here. No parameter tuning takes place on held-out hosts.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from reviewer_expansion_common import *
from realtime_watermark.attack_manifest import reviewer_attacks, reviewer_manifest_summary
from realtime_watermark.core import load_binary_watermark
from skimage.metrics import structural_similarity as color_ssim


def main(a):
    init_cpu();out=Path(a.out_dir)
    hosts,marks=dataset(a);cfg=config(a.config)
    attacks=reviewer_attacks() if not a.quick else [(n,f) for n,f in reviewer_attacks() if n in ('clean','jpeg40','rotate_2p5')]
    sizes=(16,32) if a.quick else (16,32,48,64)
    rows=[]
    for n in sizes:
        for hp in hosts:
            im=load_host(hp)
            capacity=(im.shape[0]//8)*(im.shape[1]//8)
            if n*n>capacity:
                raise ValueError(f'Payload {n*n} bits exceeds {capacity} blocks for {hp}')
            for wi,wp in enumerate(marks):
                wm=load_binary_watermark(wp,size=n)
                marked,key,ems=encode(im,wm,cfg,case_seed(a.seed,wi))
                for attack_id,attack_fn in attacks:
                    attacked=attack_fn(marked)
                    recovered,meta,exms=decode(attacked,key,sync=True)
                    rows.append({'host':hp.name,'watermark':wp.name,'payload_side':n,'payload_bits':n*n,
                        'bits_per_host_pixel':float(n*n/(im.shape[0]*im.shape[1])),
                        'block_occupancy':float(n*n/capacity),'attack':attack_id,
                        'clean_psnr_db':float(psnr(im,marked)),
                        'clean_ssim':float(color_ssim(im,marked,channel_axis=-1,data_range=255)),
                        'nc':float(nc_binary(wm,recovered)),
                        'ber':float(1-bit_accuracy(wm,recovered)),
                        'embedding_ms':ems,'extraction_ms':exms,'sync_path':meta.get('path','')})
    write_rows(out/'payload_capacity_per_attack.csv',rows)
    summary=[]
    for n in sizes:
        subset=[r for r in rows if r['payload_side']==n]
        clean=[r for r in subset if r['attack']=='clean']
        attacked=[r for r in subset if r['attack']!='clean']
        summary.append({'payload_bits':n*n,'hosts':len(hosts),'watermarks':len(marks),
            'attacks_per_pair':len(attacks)-1,'trials':len(subset),
            'mean_clean_psnr_db':float(np.mean([r['clean_psnr_db'] for r in clean])),
            'mean_clean_nc':float(np.mean([r['nc'] for r in clean])),
            'mean_attacked_nc':float(np.mean([r['nc'] for r in attacked])),
            'mean_attacked_ber':float(np.mean([r['ber'] for r in attacked])),
            'embed_ms_median':float(np.median([r['embedding_ms'] for r in clean])),
            'embed_ms_p95':float(np.percentile([r['embedding_ms'] for r in clean],95)),
            'aligned_extract_ms_median':float(np.median([r['extraction_ms'] for r in clean])),
            'aligned_extract_ms_p95':float(np.percentile([r['extraction_ms'] for r in clean],95)),
            'attacked_extract_ms_p95':float(np.percentile([r['extraction_ms'] for r in attacked],95)),
            'timing_scope':'instrumented per-case times; not independently warmed microbenchmarks'})
    write_rows(out/'payload_capacity_summary.csv',summary)
    write_json(out/'manifest.json',{'status':'complete','config_sha256':cfg.sha256(),
        'protocol':reviewer_manifest_summary(),'evaluated_attacks':[x[0] for x in attacks],
        'payload_sizes':[n*n for n in sizes], 'rows':len(rows),
        'limitations':['No error correction and no automatic strength retuning between payload sizes.',
                      '4096 bits occupies every 8x8 block at resolution 512x512.'],**info()})

if __name__=='__main__':main(base_args(argparse.ArgumentParser(description=__doc__)).parse_args())
