#!/usr/bin/env python3
"""Independent unseen-host evaluation with hash-enforced non-overlap.

Requires caller-provided external images. Refuses to relabel repository images
as external or to evaluate training/development hosts as a new test set.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from reviewer_expansion_common import *
from reviewer_experiments import jpeg,warp


def run(a):
    init_cpu()
    out=Path(a.out_dir)
    ext=paths(a.external_host_dir,{'.bmp','.png','.jpg','.jpeg','.tif','.tiff'},a.max_hosts)
    official=paths(a.development_host_dir,{'.bmp','.png','.jpg','.jpeg','.tif','.tiff'},0)
    repository_hosts=paths(ROOT/'data/user_dataset/host',{'.bmp','.png','.jpg','.jpeg','.tif','.tiff'},0)
    hashes={md5_file(p) for p in official + repository_hosts}
    test_hash={p.name:md5_file(p) for p in ext}
    if len(set(test_hash.values()))!=len(test_hash) or hashes.intersection(test_hash.values()):
        raise RuntimeError('External hosts duplicate each other or overlap the development split')
    marks=paths(a.watermark_dir,{'.png','.bmp'},a.max_watermarks)
    cfg=config(a.config)
    rows=[]
    for hp in ext:
        host=load_host(hp)
        if host.shape[0]<512 or host.shape[1]<512:
            raise ValueError(f'External host must have >=512 pixels each dimension: {hp}')
        for wi,wp in enumerate(marks):
            wm=mark(wp)
            marked,key,em=encode(host,wm,cfg,case_seed(a.seed,wi))
            attacks={'clean':marked,'jpeg70':jpeg(marked,70),
                'rotation2p5':warp(marked,angle=2.5)}
            for name in (('clean','jpeg70') if a.quick else attacks):
                decoded,meta,ms=decode(attacks[name],key)
                rows.append({'host':hp.name,'watermark':wp.name,'attack':name,
                    'host_sha256':test_hash[hp.name],'embed_ms':em,'extract_ms':ms,
                    'path':meta.get('path',''),**quality(host,marked,wm,decoded)})
    write_rows(out/'external_cases.csv',rows)
    write_json(out/'manifest.json',{'status':'complete','external_hosts':test_hash,
      'development_and_repository_hosts_sha256':sorted(hashes),'rows':len(rows),'config_sha256':cfg.sha256(),
      'split_guarantee':'Exact file SHA256 non-overlap only; provenance/content near-duplicates require separate audit',**info()})


def parser():
    p=base_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument('--external-host-dir',required=True)
    p.add_argument('--development-host-dir',required=True)
    return p
if __name__=='__main__':run(parser().parse_args())
