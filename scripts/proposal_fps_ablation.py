#!/usr/bin/env python3
"""CPU-only, fixed-payload FPS and mathematical-proposal ablations.

No performance outcome is assumed. All failed variants are recorded and never
silently omitted. Timing is operation-level, not video effective FPS.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT/'src',ROOT):
    if str(path) not in sys.path:sys.path.insert(0,str(path))
import numpy as np
from realtime_watermark.core import (default_attacks,load_binary_watermark,load_rgb,
    psnr,ssim_gray,nc_binary,bit_accuracy,make_demo_host,make_demo_watermark)
from realtime_watermark.proposals.jilp_qim.config import JILPQIMConfig
from realtime_watermark.proposals.jilp_qim.method import embed_proposal,extract_jilp_qim


# All variant keys are explicit so they can be selected individually and kept
# fixed before evaluating a held-out partition. No attack-specific tuning.
VARIANTS={
    'baseline':('reference operating point',{}),
    'gate_080':('phase threshold control',{'phase_fast_threshold':0.80}),
    'gate_070':('phase threshold 0.70',{'phase_fast_threshold':0.70}),
    'gate_065':('phase threshold 0.65',{'phase_fast_threshold':0.65}),
    'gate_055':('aggressive phase threshold (risk study)',{'phase_fast_threshold':0.55}),
    'no_pruning':('without bounded phase pruning',{'certified_phase_pruning':False,'phase_fast_threshold':0.80}),
    'no_refinement':('without local phase refinement',{'phase_refinement':'off','phase_fast_threshold':0.80}),
    'no_phase_search_acceleration':('neither pruning nor refinement',{'phase_refinement':'off','certified_phase_pruning':False,'phase_fast_threshold':0.80}),
    'pilot_064':('64 pilots',{'pilot_count':64}),
    'pilot_096':('96 pilots',{'pilot_count':96}),
    'pilot_192':('192 pilots',{'pilot_count':192}),
    'pilot_random':('random instead of spatially stratified pilots',{'pilot_schedule':'random'}),
    'pilot_no_search':('disable geometric registration (negative control)',{'enable_sync_search':False}),
    'pilot_disabled':('remove pilots and registration (negative control)',{'pilot_count':0,'enable_pilots':False,'enable_sync_search':False}),
    'joint_center':('experimental joint two-constraint DCT center-QIM',{'joint_embedding':'orthogonal_joint_center'}),
    'joint_center_gate065':('joint center QIM + phase threshold 0.65',{'joint_embedding':'orthogonal_joint_center','phase_fast_threshold':0.65}),
    'projection_guarded':('guarded minimum-distortion QIM vs center',{'projection_mode':'minimum_distortion'}),
    'closure_disabled':('remove integer closure (clean BER negative control)',{'use_integer_lattice_closure':False}),
    'field_opponent':('opponent color projection',{'field_mode':'opponent'}),
    'field_luminance':('luminance projection',{'field_mode':'luminance'}),
    'gate065_pilot096':('interaction: gate 0.65 + 96 pilots',{'phase_fast_threshold':0.65,'pilot_count':96}),
    'gate065_pilot064':('interaction: gate 0.65 + 64 pilots',{'phase_fast_threshold':0.65,'pilot_count':64}),
    'joint_pilot096':('interaction: joint DCT + 96 pilots',{'joint_embedding':'orthogonal_joint_center','pilot_count':96}),
    # Convolution-derived, frequency-regularized, exactly orthogonal sparse
    # payload carrier + joint center-QIM. These change the embedding method.
    'sparse_center24':('24-tap sparse Hartley joint center-QIM',{'hartley_mix':0.25,'hartley_support':24,'hartley_carrier_mode':'frequency_preserving','joint_embedding':'orthogonal_joint_sparse_center'}),
    'sparse_center32':('32-tap sparse Hartley joint center-QIM',{'hartley_mix':0.25,'hartley_support':32,'hartley_carrier_mode':'frequency_preserving','joint_embedding':'orthogonal_joint_sparse_center'}),
    'sparse_center48':('48-tap sparse Hartley joint center-QIM',{'hartley_mix':0.25,'hartley_support':48,'hartley_carrier_mode':'frequency_preserving','joint_embedding':'orthogonal_joint_sparse_center'}),
    'sparse_center56':('56-tap sparse Hartley joint center-QIM',{'hartley_mix':0.25,'hartley_support':56,'hartley_carrier_mode':'frequency_preserving','joint_embedding':'orthogonal_joint_sparse_center'}),
}
GROUPS={
    'phase':('baseline','gate_080','gate_070','gate_065','gate_055','no_pruning','no_refinement','no_phase_search_acceleration'),
    'pilot':('baseline','pilot_064','pilot_096','pilot_192','pilot_random','pilot_no_search','pilot_disabled'),
    'embedding':('baseline','joint_center','projection_guarded','closure_disabled','field_opponent','field_luminance'),
    'interaction':('baseline','joint_center_gate065','gate065_pilot096','gate065_pilot064','joint_pilot096'),
}
GROUPS['method_fps']=('baseline','joint_center_gate065','sparse_center24','sparse_center32','sparse_center48','sparse_center56')
GROUPS['all']=tuple(VARIANTS)
GROUPS['quick']=('baseline','gate_080','gate_065','joint_center','joint_center_gate065','pilot_096')


def variants_for(base:JILPQIMConfig,names:list[str]):
    out=[]
    for name in names:
        if name not in VARIANTS:raise ValueError(f'Unknown variant: {name}')
        description,changes=VARIANTS[name]
        cfg=replace(base,**changes).validated()
        out.append((name,description,cfg))
    return out


def get_hosts(host_dir:str|None,limit:int,split:str,split_manifest:str|None)->list[tuple[str,np.ndarray,str]]:
    if host_dir is None:
        if split=='holdout':raise ValueError('Holdout requires actual host BMP directory')
        return [('demo_512',make_demo_host(512,512),'synthetic')]
    base=Path(host_dir)
    all_files={path.name:path for path in base.glob('*.bmp') if 'lena' not in path.name.lower()}
    if not all_files:raise RuntimeError('No BMP hosts found (Lena excluded)')
    if split_manifest:
        doc=json.loads(Path(split_manifest).read_text(encoding='utf-8'))
        groups=['development','test'] if split=='all' else (['test'] if split=='holdout' else ['development'])
        selected=[]
        for group in groups:
            for item in doc[group]:
                # Official split prepends host_NNN_ to each actual BMP basename.
                filename=item['name'].split('_',2)[2]
                path=all_files.get(filename)
                if path is None:raise FileNotFoundError(f'Expected split host not found: {filename}')
                digest=hashlib.sha256(path.read_bytes()).hexdigest()
                if digest!=item['sha256']:
                    raise RuntimeError(f'Host checksum differs from frozen split: {filename}')
                selected.append((path.name,load_rgb(path,mode='full'),digest))
                if limit and len(selected)>=limit:return selected
        return selected
    if split!='all':
        raise ValueError('For development/holdout runs, pass a verified --split-manifest')
    selected=[]
    for path in sorted(all_files.values()):
        selected.append((path.name,load_rgb(path,mode='full'),hashlib.sha256(path.read_bytes()).hexdigest()))
        if limit and len(selected)>=limit:break
    return selected


def get_marks(watermark_dir:str|None):
    if watermark_dir is None:return [('demo',make_demo_watermark(64),'synthetic')]
    paths=sorted(Path(watermark_dir).glob('*.png'))
    if len(paths)!=2:raise RuntimeError('Expected exactly 2 source PNG watermarks')
    return [(p.name,load_binary_watermark(p,size=64),hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]


def write_csv(path,rows):
    if not rows:return
    cols=list(dict.fromkeys(k for row in rows for k in row))
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)


def summarise(rows):
    answer=[]
    for variant in sorted({r['variant'] for r in rows}):
        subset=[r for r in rows if r['variant']==variant]
        clean=[r for r in subset if r['attack']=='clean' and r['status']=='ok']
        attacked=[r for r in subset if r['attack']!='clean' and r['status']=='ok']
        failures=[r for r in subset if r['status']!='ok']
        numeric=lambda rr,col: [float(r[col]) for r in rr if r.get(col) not in (None,'')]
        mean=lambda rr,col: float(np.mean(numeric(rr,col))) if numeric(rr,col) else float('nan')
        med=lambda rr,col: float(np.median(numeric(rr,col))) if numeric(rr,col) else float('nan')
        p95=lambda rr,col: float(np.percentile(numeric(rr,col),95)) if numeric(rr,col) else float('nan')
        answer.append(dict(variant=variant,description=VARIANTS[variant][0],n_cases=len(subset),failures=len(failures),
            mean_psnr_db=mean(clean,'psnr_db'),mean_ssim=mean(clean,'ssim'),
            clean_nc=mean(clean,'nc'),clean_ber=mean(clean,'ber'),
            attack_nc=mean(attacked,'nc'),attack_ber=mean(attacked,'ber'),
            embed_ms_median=med(clean,'embed_ms'),embed_ms_p95=p95(clean,'embed_ms'),
            clean_extract_ms_median=med(clean,'extract_ms'),
            attack_extract_ms_median=med(attacked,'extract_ms'),attack_extract_ms_p95=p95(attacked,'extract_ms'),
            recovery_fraction=mean(attacked,'recovery'),jpeg70_pipeline_ms_median=med([r for r in subset if r['attack']=='jpeg70' and r['status']=='ok'],'pipeline_ms'),
            jpeg70_stage_fps_estimate=(1000./med([r for r in subset if r['attack']=='jpeg70' and r['status']=='ok'],'pipeline_ms') if med([r for r in subset if r['attack']=='jpeg70' and r['status']=='ok'],'pipeline_ms') > 0 else float('nan'))))
    return answer


def paired_deltas(rows,bootstrap_samples=250,seed=2026):
    """Compare each candidate against reference on identical host/wm/attack keys.

    Draw bootstrap units at the host level to avoid falsely treating attacks
    on the same host as independent samples.
    """
    base={(r['host'],r['watermark'],r['attack']):r for r in rows
          if r['variant']=='baseline' and r['status']=='ok'}
    deltas=[]
    for row in rows:
        if row['variant']=='baseline' or row['status']!='ok':continue
        key=(row['host'],row['watermark'],row['attack'])
        original=base.get(key)
        if original is None:continue
        deltas.append(dict(host=row['host'],watermark=row['watermark'],attack=row['attack'],variant=row['variant'],
            delta_nc=float(row['nc'])-float(original['nc']),
            delta_ber=float(row['ber'])-float(original['ber']),
            delta_psnr_db=float(row['psnr_db'])-float(original['psnr_db']),
            delta_embed_ms=float(row['embed_ms'])-float(original['embed_ms']),
            delta_extract_ms=float(row['extract_ms'])-float(original['extract_ms']),
            delta_pipeline_ms=float(row['pipeline_ms'])-float(original['pipeline_ms'])))
    ci=[];rng=np.random.default_rng(seed)
    for variant in sorted({r['variant'] for r in deltas}):
        rr=[r for r in deltas if r['variant']==variant]
        for category,subset in [('all',rr),('attacked',[r for r in rr if r['attack']!='clean']),
                                 ('jpeg70',[r for r in rr if r['attack']=='jpeg70'])]:
            if not subset:continue
            hosts=sorted({r['host'] for r in subset})
            # One draw repeats ALL watermarks and attacks of one sampled host.
            by_host={h:[r for r in subset if r['host']==h] for h in hosts}
            for metric in ('delta_nc','delta_ber','delta_psnr_db','delta_embed_ms','delta_extract_ms','delta_pipeline_ms'):
                observed=float(np.mean([r[metric] for r in subset]))
                simulated=[]
                for _ in range(bootstrap_samples):
                    sampled=rng.choice(hosts,size=len(hosts),replace=True)
                    values=[row[metric] for h in sampled for row in by_host[h]]
                    simulated.append(float(np.mean(values)))
                lower,upper=np.percentile(simulated,[2.5,97.5])
                ci.append(dict(variant=variant,attack_scope=category,metric=metric,n_hosts=len(hosts),
                    n_pairs=len(subset),mean_delta=observed,ci95_low=float(lower),ci95_high=float(upper)))
    return deltas,ci

def run(args):
    base=JILPQIMConfig.load(args.config)
    names=(list(GROUPS[args.group]) if not args.variants else [x.strip() for x in args.variants.split(',') if x.strip()])
    if args.repeats < 1 or args.warmups < 1:raise ValueError("At least one warmup and timed repeat required")
    selected=variants_for(base,names)
    hosts=get_hosts(args.host_dir,args.max_hosts,args.split,args.split_manifest)
    marks=get_marks(args.watermark_dir)
    attacks=default_attacks(args.attack_preset)
    if args.attacks:
        choices=set(args.attacks.split(','))
        attacks=[a for a in attacks if a[0] in choices]
        if not attacks or len(attacks)!=len(choices):raise ValueError('Requested attack not found in selected preset')
    if not any(n=='clean' for n,_ in attacks):attacks=[('clean',lambda x:x),*attacks]
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for label,_desc,cfg in selected:
        cfg.save(out/'configs'/f'{label}.json')
    # Pre-warm each distinct branch without recording compilation latency.
    # A failed warmup is not silently skipped: its later embed yields an error row.
    from realtime_watermark.methods import warmup_numba
    warmup_numba()
    for host_number,(host_name,host,host_digest) in enumerate(hosts):
        if host.shape[0]//8 * (host.shape[1]//8) < 4096:raise RuntimeError(f'Insufficient 8x8 capacity: {host_name}')
        for mark_number,(mark_name,mark,mark_digest) in enumerate(marks):
            seed=int(args.seed)
            # Rotate execution order to avoid always measuring baseline first.
            offset=(host_number*len(marks)+mark_number)%len(selected)
            run_order=selected[offset:]+selected[:offset]
            for label,description,cfg in run_order:
                # Warm-up is performed for each configuration (JIT and reuse caches).
                err=None
                try:
                    for _ in range(args.warmups):
                        warm,key=embed_proposal(host,mark,seed=seed,method_params=cfg.to_dict())
                        extract_jilp_qim(warm,key,sync=True)
                        # Compile the recovery path before starting the timer.
                        for attack_name,attack in attacks:
                            if attack_name != 'clean':
                                extract_jilp_qim(attack(warm),key,sync=True)
                    embed_times=[]
                    for _ in range(args.repeats):
                        t0=time.perf_counter_ns()
                        marked,key=embed_proposal(host,mark,seed=seed,method_params=cfg.to_dict())
                        embed_times.append((time.perf_counter_ns()-t0)*1e-6)
                    quality=psnr(host,marked)
                    similarity=ssim_gray(host,marked)
                    outputs=[]
                    for attack_name,attack in attacks:
                        channel_times=[]
                        for _ in range(args.repeats):
                            t_chan=time.perf_counter_ns()
                            attacked=marked if attack_name=='clean' else attack(marked)
                            channel_times.append((time.perf_counter_ns()-t_chan)*1e-6)
                        extract_times=[]
                        for _ in range(args.repeats):
                            t0=time.perf_counter_ns()
                            recovered,meta=extract_jilp_qim(attacked,key,sync=True,return_meta=True)
                            extract_times.append((time.perf_counter_ns()-t0)*1e-6)
                        decoded=(recovered>127).astype(np.uint8)
                        truth=(mark>127).astype(np.uint8)
                        outputs.append(dict(attack=attack_name,status='ok',
                            nc=nc_binary(mark,recovered),ber=1.-bit_accuracy(mark,recovered),
                            psnr_db=quality,ssim=similarity,
                            embed_ms=float(np.median(embed_times)),extract_ms=float(np.median(extract_times)),
                            attack_channel_ms=float(np.median(channel_times)),
                            pipeline_ms=float(np.median(embed_times)+np.median(channel_times)+np.median(extract_times)),
                            extract_ms_p95=float(np.percentile(extract_times,95)),
                            recovery=int(meta.get('path')=='wide_sync_jilp'),
                            path=meta.get('path',''),phase_score=meta.get('raw_phase_score',None),
                            bitmap_sha256=hashlib.sha256(decoded.tobytes()).hexdigest(),
                            marked_sha256=hashlib.sha256(marked.tobytes()).hexdigest(),
                            joint_overlap_blocks=key.params.get('joint_overlap_blocks',0),
                            pilot_integer_updates=key.params.get('integer_pilot_unit_updates',0),
                            payload_integer_updates=key.params.get('integer_lattice_unit_updates',0)))
                except Exception as ex:
                    err=f'{type(ex).__name__}: {ex}'
                    outputs=[dict(attack=a[0],status='failed',error=err) for a in attacks]
                    if args.strict:raise
                for result in outputs:
                    rows.append(dict(host=host_name,host_sha256=host_digest,watermark=mark_name,
                        watermark_sha256=mark_digest,seed=seed,payload_bits=int(mark.size),
                        variant=label,changed=description,config_sha256=cfg.sha256(),**result))
                summary=('FAILED '+err if err else f'PSNR={quality:.3f} clean NC={outputs[0]["nc"]:.4f} '
                           f'embed={np.median(embed_times):.3f} ms')
                print(host_name,mark_name,label,summary,flush=True)
    write_csv(out/'fps_ablation_per_attack.csv',rows)
    write_csv(out/'fps_ablation_summary.csv',summarise(rows))
    if 'baseline' in names:
        paired,intervals=paired_deltas(rows,seed=args.seed)
        write_csv(out/'fps_ablation_paired_deltas.csv',paired)
        write_csv(out/'fps_ablation_host_bootstrap_ci.csv',intervals)
    cfgs={n:{'description':d,'parameters':c.to_dict(),'sha256':c.sha256()} for n,d,c in selected}
    (out/'fps_ablation_manifest.json').write_text(json.dumps(dict(
        base_config_sha256=base.sha256(),base_config=base.to_dict(),variants=cfgs,
        seed=int(args.seed),repeats=args.repeats,warmups=args.warmups,
        split=args.split,split_manifest=args.split_manifest,
        attack_preset=args.attack_preset,attacks=[n for n,_ in attacks],
        hosts={name:digest for name,_,digest in hosts},
        watermarks={name:digest for name,_,digest in marks},
        comparisons_are_matched_distortion=False,
        bootstrap='host-cluster bootstrap; descriptive only until sufficient independent hosts',
        benchmarking_note='CPU operation-level timing, NOT wall-clock video effective FPS; no silent variant failures. Holdout must not be used to choose variants.',
    ),indent=2),encoding='utf-8')
    print('Saved',out, 'rows',len(rows),flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,help='Previously selected, frozen configuration JSON')
    p.add_argument('--host-dir',default=None)
    p.add_argument('--watermark-dir',default=None)
    p.add_argument('--out-dir',default='reports/proposal_fps_ablation')
    p.add_argument('--group',choices=sorted(GROUPS),default='quick')
    p.add_argument('--variants',default=None,help='Comma-separated explicit variant names; overrides --group')
    p.add_argument('--attack-preset',choices=['smoke','standard','extended','stress'],default='standard')
    p.add_argument('--attacks',default='clean,jpeg70,rotate1,translate4_4',help='Comma-separated attack names; pass an empty string to use every attack in the preset')
    p.add_argument('--max-hosts',type=int,default=0)
    p.add_argument('--split',choices=['development','holdout','all'],default='development')
    p.add_argument('--split-manifest',default=str(ROOT/'configs/holdout_split_manifest_20261021.json'))
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--warmups',type=int,default=1)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--strict',action='store_true')
    return p

if __name__=='__main__':run(parser().parse_args())
