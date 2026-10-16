#!/usr/bin/env python3
"""Independent, source-backed reviewer evidence for the unchanged QIM carriers.

Provides joint integer/pilot diagnostics, off-grid geometry, intentional carrier
attenuation, unwatermarked/wrong-key scores, capacity sensitivity and statistics.
Does not tune any model or assert that experimental attacks are certified defenses.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import platform
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

from realtime_watermark.core import (
    load_binary_watermark, nc_binary, psnr, bit_accuracy, dct_basis
)
from realtime_watermark.proposals.jilp_qim.config import JILPQIMConfig
from realtime_watermark.proposals.jilp_qim.method import (
    embed_proposal, extract_jilp_qim, _pilot_score_fast, _schedules_from_key
)
from realtime_watermark.proposals.jilp_qim import engine


def csv_write(filename: Path, rows: list[dict]) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        filename.write_text('', encoding='utf8')
        return
    headers = list(dict.fromkeys(k for row in rows for k in row))
    with filename.open('w', newline='', encoding='utf8') as out:
        writer = csv.DictWriter(out, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def warp(img: np.ndarray, *, angle=0.0, dx=0.0, dy=0.0, scale=1.0, shear=0.0) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[0, 1] += shear
    m[0, 2] += dx - (shear * h / 2)
    m[1, 2] += dy
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    sink = io.BytesIO()
    Image.fromarray(img).save(sink, 'JPEG', quality=quality, subsampling=0)
    sink.seek(0)
    return np.asarray(Image.open(sink).convert('RGB')).copy()


def projected_coordinates(image: np.ndarray, indices: np.ndarray, eta: float, which: str) -> np.ndarray:
    # Fixed DCT geometry matches the original orthogonal mathematical model.
    h, w = image.shape[:2]
    h -= h % 8; w -= w % 8
    blocks = image[:h, :w].reshape(h//8, 8, w//8, 8, 3).transpose(0, 2, 1, 3, 4)
    g = np.array([.299+eta, .587-eta/2, .114-eta/2])
    spatial = np.tensordot(blocks, g, axes=([4], [0]))
    a01 = dct_basis(8,0,1)
    a10 = dct_basis(8,1,0)
    carrier = (a01 + a10 if which == 'pilot' else a01 - a10) / 2
    return np.einsum('bcij,ij->bc', spatial, carrier).reshape(-1)[indices]


def attenuation(image: np.ndarray, eta: float, which: str, fraction: float) -> np.ndarray:
    """Key-independent attenuation of the public fixed carrier subspace."""
    h, w = image.shape[:2]
    hc, wc = h-h%8, w-w%8
    out = image.copy()
    g = np.array([.299+eta, .587-eta/2, .114-eta/2], dtype=np.float64)
    basis = (dct_basis(8,0,1) + (1 if which=='pilot' else -1)*dct_basis(8,1,0))/2
    blocks = image[:hc, :wc].reshape(hc//8,8,wc//8,8,3).transpose(0,2,1,3,4).astype(np.float64)
    f = np.tensordot(blocks, g, axes=([4],[0]))
    v = np.einsum('bcij,ij->bc',f,basis)
    delta = -fraction * 2*v[:,:,None,None,None]*basis[None,None,:,:,None]*g[None,None,None,None,:]/float(g@g)
    modified = np.clip(np.rint(blocks+delta),0,255).astype(np.uint8)
    out[:hc,:wc]=modified.transpose(0,2,1,3,4).reshape(hc,wc,3)
    return out


def gaussian_variance(image: np.ndarray, variance: float) -> np.ndarray:
    # Variance is defined in normalized [0,1] RGB intensity units.
    rng=np.random.default_rng(202611)
    x=image.astype(np.float64)/255.
    return np.clip(np.rint((x+rng.normal(0,np.sqrt(variance),x.shape))*255),0,255).astype(np.uint8)


def speckle_variance(image: np.ndarray, variance: float) -> np.ndarray:
    rng=np.random.default_rng(202612)
    x=image.astype(np.float64)/255.
    return np.clip(np.rint((x*(1+rng.normal(0,np.sqrt(variance),x.shape)))*255),0,255).astype(np.uint8)


def blur(image: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(image,(0,0),sigmaX=sigma)


def attacks(quick: bool):
    seq = [('rot_2.5',lambda y: warp(y,angle=2.5)),
           ('translate_1',lambda y: warp(y,dy=1)),
           ('subpixel_0.5',lambda y: warp(y,dy=.5,dx=.5)),
           ('jpeg_40',lambda y: jpeg(y,40)),
           ('gaussian_var_0p003',lambda y: gaussian_variance(y,.003)),
           ('speckle_var_0p001',lambda y: speckle_variance(y,.001)),
           ('payload_zero',lambda y,eta: attenuation(y,eta,'payload',1.0)),
           ('pilot_zero',lambda y,eta: attenuation(y,eta,'pilot',1.0))]
    if not quick:
        seq += [('rot_0.5',lambda y:warp(y,angle=.5)),
                ('rot_7.5',lambda y:warp(y,angle=7.5)),
                ('rot_15',lambda y:warp(y,angle=15)),
                ('scale_0.9',lambda y:warp(y,scale=.9)),
                ('scale_1.1',lambda y:warp(y,scale=1.1)),
                ('shear_0.05',lambda y:warp(y,shear=.05)),
                ('translate_8',lambda y:warp(y,dy=8)),
                ('translate_12',lambda y:warp(y,dy=12)),
                ('jpeg_30',lambda y:jpeg(y,30)),
                ('gaussian_var_0p05',lambda y:gaussian_variance(y,.05)),
                ('speckle_var_0p03',lambda y:speckle_variance(y,.03)),
                ('blur_sigma_1p5',lambda y:blur(y,1.5)),
                ('payload_half',lambda y,eta:attenuation(y,eta,'payload',.5)),
                ('pilot_half',lambda y,eta:attenuation(y,eta,'pilot',.5))]
    return seq


def aggregate_uncertainty(rows: list[dict], *, seed: int, boot: int=500) -> list[dict]:
    """Host-cluster bootstrap; do not treat attacks as independent hosts."""
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for r in rows:
        if r.get('status') == 'ok':
            groups[(r['watermark'],r['attack'],r['host'])].append(r)
    summaries=[]
    by_pair=defaultdict(dict)
    for (wm,attack,host),rs in groups.items():
        by_pair[(wm,attack)][host]=np.mean([float(x['nc']) for x in rs])
    for (wm,attack),values in by_pair.items():
        x=np.asarray(list(values.values()),float)
        if x.size==0: continue
        samples=rng.choice(x,size=(boot,x.size),replace=True).mean(axis=1)
        summaries.append({'watermark':wm,'attack':attack,'hosts':x.size,
                          'mean_nc':float(x.mean()), 'sd_nc':float(x.std(ddof=1)) if x.size>1 else float('nan'),
                          'ci95_lo':float(np.quantile(samples,.025)),
                          'ci95_hi':float(np.quantile(samples,.975)),
                          'uncertainty_unit':'host cluster; exploratory if n is small'})
    return summaries


def run(args):
    from importlib.metadata import version
    output=Path(args.out_dir)
    output.mkdir(parents=True,exist_ok=True)
    config=JILPQIMConfig.load(args.config)
    host_paths=sorted(Path(args.host_dir).glob('*.bmp'))
    wm_paths=sorted(Path(args.watermark_dir).glob('*.png'))
    if args.max_hosts: host_paths=host_paths[:args.max_hosts]
    if args.max_watermarks: wm_paths=wm_paths[:args.max_watermarks]
    if not host_paths or not wm_paths: raise RuntimeError('Missing BMP hosts or PNG watermarks')
    rows=[]; security=[]; ortho=[]; capacity=[]
    for wm_i,wm_path in enumerate(wm_paths):
        wm=load_binary_watermark(wm_path,size=64)
        for host_path in host_paths:
            host=np.asarray(Image.open(host_path).convert('RGB'))
            if host.shape[0] < 512 or host.shape[1] < 512:
                raise ValueError('Primary 4,096-bit tests require at least 512x512 host images')
            # Full image extraction preserves the actual geometry of the host.
            marked,key=embed_proposal(host,wm,seed=args.seed+wm_i*1000003,
                                      method_params=config.to_dict())
            key_params={**key.params, "host_shape":tuple(host.shape), "watermark_shape":tuple(wm.shape)}
            _,pilot_indices,pilot_bits,_,_,wc=_schedules_from_key(key_params)
            eta=float(key_params['eta'])
            pilot_before,_=_pilot_score_fast(marked,key_params)
            ortho.append({'host':host_path.name,'watermark':wm_path.name,
                          'payload_length':int(wm.size),'continuous_inner_product':float(np.sum(((dct_basis(8,0,1)-dct_basis(8,1,0))/2)*((dct_basis(8,0,1)+dct_basis(8,1,0))/2))),
                          'pilot_accuracy_uint8':pilot_before,
                          'pilot_lattice_unit_updates':key_params.get('pilot_lattice_unit_updates',0),
                          'pilot_lattice_initial_parity_failures':key_params.get('pilot_lattice_initial_parity_failures',0),
                          'pilot_lattice_unresolved':key_params.get('pilot_lattice_unresolved',0),
                          'payload_lattice_unresolved':key_params.get('integer_lattice_unresolved',0),
                          'pilot_overlap_unresolved':key_params.get('pilot_overlap_unresolved',0),
                          'psnr':float(psnr(host,marked))})
            # Compare final image to pilot-only stage: includes payload and
            # final pilot integer closure, not a pure continuous cross term.
            intermediate=host.copy()
            engine._embed_one_carrier_md(host,intermediate,pilot_indices,pilot_bits,
                                        engine.PILOT_BA,engine.PILOT_BB,float(key_params['step_pilot']),
                                        float(key_params['step_pilot'])*float(key_params['pilot_rho_frac']),
                                        int(wc),8,eta)
            leakage=np.abs(projected_coordinates(marked,pilot_indices,eta,'pilot')-
                           projected_coordinates(intermediate,pilot_indices,eta,'pilot'))
            ortho[-1].update({'pilot_coordinate_difference_final_vs_pilot_stage_mean':float(leakage.mean()),
                              'pilot_coordinate_difference_final_vs_pilot_stage_p95':float(np.percentile(leakage,95)),
                              'pilot_coordinate_difference_final_vs_pilot_stage_max':float(leakage.max())})
            def evaluate(label,received,kind):
                t=time.perf_counter()
                decoded,meta=extract_jilp_qim(received,key,return_meta=True)
                elapsed=(time.perf_counter()-t)*1000
                raw=extract_jilp_qim(received,key,sync=False)
                rows.append({'host':host_path.name,'watermark':wm_path.name,
                             'attack':label,'kind':kind,'payload_bits':int(wm.size),
                             'bit_accuracy':float(bit_accuracy(wm,decoded)),
                             'nc':float(nc_binary(wm,decoded)),
                             'raw_nc':float(nc_binary(wm,raw)),
                             'attack_psnr_vs_marked':float(psnr(marked,received)),
                             'embed_psnr':float(psnr(host,marked)),
                             'extract_ms':elapsed,
                             'pilot_score':float(meta.get('score',float('nan'))),
                             'path':meta.get('path',''),
                             'synchronization_confident':meta.get('synchronization_confident',False),
                             'status':'ok'})
            evaluate('clean',marked,'clean')
            for item in attacks(args.mode=='quick'):
                label,fun=item[0],item[1]
                try:
                    received=fun(marked,eta) if label.startswith(('payload_','pilot_')) else fun(marked)
                    evaluate(label,received,'attack')
                except Exception as e:
                    rows.append({'host':host_path.name,'watermark':wm_path.name,'attack':label,
                                 'status':'failed','error':str(e)})
            # Detect claimed payload, not merely decode an arbitrary bitstring.
            for label,received,kparams in [
                ('marked_correct_key',marked,key_params),
                ('unwatermarked_correct_key',host,key_params),
            ]:
                dec=extract_jilp_qim(received,kparams,sync=False)
                security.append({'host':host_path.name,'watermark':wm_path.name,
                                 'trial':label,'positive':label=='marked_correct_key',
                                 'nc':float(nc_binary(wm,dec)),
                                 'bit_accuracy':float(bit_accuracy(wm,dec))})
            for j in range(5 if args.mode=='full' else 2):
                bad=dict(key_params)
                bad['seed']=int(key_params['seed'])+100003+j*173
                dec=extract_jilp_qim(marked,bad,sync=False)
                security.append({'host':host_path.name,'watermark':wm_path.name,
                                 'trial':f'wrong_key_{j}','positive':False,
                                 'nc':float(nc_binary(wm,dec)),
                                 'bit_accuracy':float(bit_accuracy(wm,dec))})
        # capacity is a separate supplementary study, not a redefinition of 4,096-bit primary payload
        if host_paths:
            host=np.asarray(Image.open(host_paths[0]).convert('RGB'))
            for size in (16,32,48,64):
                smaller=load_binary_watermark(wm_path,size=size)
                start=time.perf_counter()
                y,k=embed_proposal(host,smaller,seed=args.seed+wm_i*1000003,method_params=config.to_dict())
                duration=(time.perf_counter()-start)*1000
                rec=extract_jilp_qim(y,k)
                capacity.append({'host':host_paths[0].name,'watermark':wm_path.name,
                                 'payload_bits':size*size,'psnr':float(psnr(host,y)),
                                 'clean_bit_accuracy':float(bit_accuracy(smaller,rec)),
                                 'embed_ms':duration})
    csv_write(output/'geometry_and_targeted_attacks.csv',rows)
    csv_write(output/'orthogonality_and_integer.csv',ortho)
    csv_write(output/'claimed_watermark_detection.csv',security)
    csv_write(output/'capacity_sensitivity.csv',capacity)
    csv_write(output/'host_cluster_ci.csv',aggregate_uncertainty(rows,seed=args.seed,boot=args.bootstrap))
    thresholds=np.linspace(-1,1,201)
    positive=np.asarray([x['nc'] for x in security if x['positive']],float)
    negative=np.asarray([x['nc'] for x in security if not x['positive']],float)
    roc=[{'threshold':float(t),'tpr':float(np.mean(positive>=t)),
          'fpr':float(np.mean(negative>=t))} for t in thresholds] if positive.size and negative.size else []
    csv_write(output/'roc.csv',roc)
    manifest={'seed':args.seed,'mode':args.mode,'config_sha256':config.sha256(),
              'host_paths':[str(p) for p in host_paths],
              'watermark_paths':[str(p) for p in wm_paths],
              'attack_cases':len(attacks(args.mode=='quick'))+1,
              'expected_attack_rows':len(host_paths)*len(wm_paths)*(len(attacks(args.mode=='quick'))+1),
              'actual_attack_rows':len(rows),'failed_attack_rows':sum(x.get('status')!='ok' for x in rows),
              'python':platform.python_version(), 'numpy':np.__version__,
              'opencv':cv2.__version__, 'scope':'supplementary; not a substitute for main matched-PSNR/video benchmark',
              'attack_variance_definition':'noise variances expressed on 0-1 normalized RGB intensity; independent fixed RNG streams',
              'roc_limitations':'claim-conditioned extraction; small finite negative set; not calibrated on independent development data',
              'statistics':'host-cluster confidence intervals, not independent-attack bootstrap'}
    (output/'reviewer_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(json.dumps({'rows':len(rows),'failures':manifest['failed_attack_rows'],'output':str(output)},indent=2))
    if manifest['failed_attack_rows']: raise RuntimeError('Reviewer experiment has failed attack rows; see CSV')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--host-dir',required=True)
    p.add_argument('--watermark-dir',required=True)
    p.add_argument('--config',required=True)
    p.add_argument('--out-dir',required=True)
    p.add_argument('--mode',choices=['quick','full'],default='quick')
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--max-hosts',type=int,default=0)
    p.add_argument('--max-watermarks',type=int,default=0)
    p.add_argument('--bootstrap',type=int,default=500)
    run(p.parse_args())
