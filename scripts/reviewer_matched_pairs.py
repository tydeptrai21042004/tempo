#!/usr/bin/env python3
"""Only compare eligible proposal/baseline observations at the SAME held-out case.

No calibration or watermark method is changed. Missing/nonmatched cases are
reported, not imputed. Bootstrap resamples hosts (not correlated attack rows).
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def read(path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline='',encoding='utf8') as f:
        return list(csv.DictReader(f))


def save(path, rows):
    if not rows:
        path.write_text('status\nno_eligible_pairs\n',encoding='utf8')
        return
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)


def is_eligible(row):
    return str(row.get('primary_rank_eligible','')).strip().lower() in ('true','1')


def matched_effects(summary, attacks, seed=2026, nboot=2000):
    # (host, watermark, method) eligibility must be true at the pair level.
    eligibility={}
    methods=set()
    for r in summary:
        key=(r['host'],r['watermark'],r['method_id'])
        if key in eligibility:
            raise ValueError(f'Duplicate matched-PSNR summary key {key}')
        eligibility[key]=is_eligible(r)
        methods.add(r['method_id'])
    if 'proposal' not in methods:
        raise ValueError('Matched-PSNR CSV lacks the proposed method')
    observations={}
    for r in attacks:
        if r.get('phase')=='clean':
            continue
        key=(r['host'],r['watermark'],r['phase'],r['method_id'])
        if key in observations:
            raise ValueError(f'Duplicate paired observation {key}')
        observations[key]=r
    out=[]; report=[]
    rng=np.random.default_rng(seed)
    for baseline in sorted(methods-{'proposal'}):
        matched=[]
        eligible_pairs=set()
        discarded=[]
        keys=sorted({(h,w) for h,w,m in eligibility if m in ('proposal',baseline)})
        for host,wm in keys:
            if not (eligibility.get((host,wm,'proposal'),False) and eligibility.get((host,wm,baseline),False)):
                discarded.append({'host':host,'watermark':wm,'baseline':baseline,'reason':'one or both PSNR/clean-accuracy ineligible'})
                continue
            eligible_pairs.add((host,wm))
            cases=sorted(a for h,w,a,m in observations if h==host and w==wm and m=='proposal')
            for attack in cases:
                x=observations.get((host,wm,attack,'proposal'))
                y=observations.get((host,wm,attack,baseline))
                if x is None or y is None:
                    discarded.append({'host':host,'watermark':wm,'baseline':baseline,'reason':f'missing paired attack {attack}'})
                    continue
                if not (is_eligible(x) and is_eligible(y)):
                    discarded.append({'host':host,'watermark':wm,'baseline':baseline,'reason':f'ineligible attack {attack}'})
                    continue
                pval=float(x['nc_bipolar']); bval=float(y['nc_bipolar'])
                if not (np.isfinite(pval) and np.isfinite(bval)):
                    discarded.append({'host':host,'watermark':wm,'baseline':baseline,'reason':f'nonfinite NC {attack}'})
                    continue
                matched.append({'baseline':baseline,'host':host,'watermark':wm,'attack':attack,
                    'proposal_nc':pval,'baseline_nc':bval,'delta_nc':pval-bval,
                    'proposal_psnr_eligible':True,'baseline_psnr_eligible':True,
                    'matching_unit':'same held-out host, watermark, attack'})
        out.extend(matched)
        by_host=defaultdict(list)
        for r in matched: by_host[r['host']].append(r['delta_nc'])
        hmean=np.array([np.mean(a) for _,a in sorted(by_host.items())],float)
        ci=(float('nan'),float('nan'))
        if len(hmean)>1:
            boot=rng.choice(hmean,size=(nboot,len(hmean)),replace=True).mean(axis=1)
            ci=tuple(float(q) for q in np.quantile(boot,[.025,.975]))
        report.append({'baseline':baseline,'eligible_common_host_watermark_pairs':len({(r['host'],r['watermark']) for r in matched}),
            'n_paired_attack_rows':len(matched),'n_paired_hosts':len(hmean),
            'mean_proposal_minus_baseline_host_averaged_nc':float(np.mean(hmean)) if len(hmean) else float('nan'),
            'ci95_host_cluster_lo':ci[0],'ci95_host_cluster_hi':ci[1],
            'ineligible_or_missing_case_count':len(discarded),
            'uncertainty_unit':'host cluster; descriptive if fewer than two hosts',
            'comparison_population':'pairwise complete cases; populations may differ by baseline'})
    return out,report


def run(args):
    src=Path(args.matched_dir);out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    summaries=read(src/'matched_psnr_by_host_watermark.csv')
    attacks=read(src/'matched_psnr_per_attack.csv')
    observations,effects=matched_effects(summaries,attacks,args.seed,args.bootstrap)
    save(out/'matched_psnr_strict_paired_attacks.csv',observations)
    save(out/'matched_psnr_strict_paired_effects.csv',effects)
    manifest={'status':'complete' if effects and any(r['n_paired_attack_rows']>0 for r in effects) else 'no_eligible_pairs',
              'seed':args.seed,'bootstrap_iterations':args.bootstrap,
              'baselines':len(effects),'total_paired_attack_rows':len(observations),
              'eligible_baselines':sum(r['n_paired_attack_rows']>0 for r in effects),
              'method':'host-cluster bootstrap on same host, watermark and attack after BOTH methods satisfy eligibility',
              'limitations':['Pairwise available population differs among baselines.',
                             'At least two independent hosts are needed for nontrivial confidence intervals.',
                             'The comparison is conditional on each method meeting the distortion and fidelity constraints.',
                             'No unreported case is imputed or replaced with another host.']}
    (out/'matched_psnr_strict_paired_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(json.dumps({'paired':len(observations),'eligible_baselines':manifest['eligible_baselines']},indent=2))
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matched-dir',required=True);p.add_argument('--out-dir',required=True)
    p.add_argument('--seed',type=int,default=2026);p.add_argument('--bootstrap',type=int,default=2000)
    run(p.parse_args())
