#!/usr/bin/env python3
"""Host-clustered paired performance uncertainty from the common-protocol CSV."""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np


def save(path, records):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not records:
        path.write_text('',encoding='utf8'); return
    with path.open('w',newline='',encoding='utf8') as fd:
        w=csv.DictWriter(fd,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)


def compute(rows, seed=2026, nboot=2000):
    rng=np.random.default_rng(seed)
    # Keep rows as paired host + watermark + attack observations.
    data={}
    for r in rows:
        if r['attack']=='clean': continue
        k=(r['host'],r['watermark_name'],r['attack'],r['method_id'])
        if k in data:
            raise ValueError(f'Duplicate experimental key: {k}')
        data[k]=float(r['nc_bipolar'])
    methods=sorted({k[3] for k in data})
    host_names=sorted({k[0] for k in data})
    effects=[]; summaries=[]
    for method in methods:
        host_score=[]
        for host in host_names:
            v=[val for (h,wm,a,m),val in data.items() if h==host and m==method]
            if v: host_score.append(float(np.mean(v)))
        if not host_score: continue
        x=np.asarray(host_score)
        res=rng.choice(x,size=(nboot,len(x)),replace=True).mean(axis=1)
        summaries.append({'method_id':method,'n_hosts':len(x),
                          'mean_host_averaged_nc':float(x.mean()),
                          'std_host_averaged_nc':float(x.std(ddof=1)) if x.size>1 else float('nan'),
                          'ci95_lower':float(np.quantile(res,.025)), 'ci95_upper':float(np.quantile(res,.975))})
        if method=='proposal': continue
        host_diff=[]; matched_rows=0
        for host in host_names:
            paired=[]
            for h,wm,a,m in data:
                if h==host and m==method and (h,wm,a,'proposal') in data:
                    paired.append(data[(h,wm,a,'proposal')]-data[(h,wm,a,m)])
            if paired:
                matched_rows += len(paired)
                host_diff.append(float(np.mean(paired)))
        if host_diff:
            x=np.asarray(host_diff)
            res=rng.choice(x,size=(nboot,len(x)),replace=True).mean(axis=1)
            effects.append({'baseline':method,'n_paired_hosts':len(x),'n_paired_conditions':matched_rows,
                            'mean_delta_nc_proposal_minus_baseline':float(x.mean()),
                            'ci95_lower':float(np.quantile(res,.025)),
                            'ci95_upper':float(np.quantile(res,.975)),
                            'design':'host cluster, paired same host/watermark/attack; not matched distortion'})
    return summaries,effects


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True)
    p.add_argument('--out-dir',required=True)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--bootstrap',type=int,default=2000)
    a=p.parse_args()
    with open(a.input,newline='',encoding='utf8') as fd: rows=list(csv.DictReader(fd))
    s,e=compute(rows,seed=a.seed,nboot=a.bootstrap)
    save(Path(a.out_dir)/'method_host_cluster_ci.csv',s)
    save(Path(a.out_dir)/'paired_proposal_effects.csv',e)
    print({'methods':len(s),'paired_baselines':len(e)})
