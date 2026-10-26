#!/usr/bin/env python3
"""Independent EQUATION-LEVEL reproduction of Sudrajat--Ernawan--Amrullah (2024).

DOI 10.52866/ijcsm.2024.05.03.043, page 700, Eqs. (10)--(22).
Original paper method: R-level Haar LL -> Hessenberg -> SVD -> S + alpha*S_W;
inverse SVD -> inverse Hessenberg -> inverse Haar; extraction subtracts
ORIGINAL HOST SPECTRUM, then inverse SVD. The extracted logo has 64x64
pixels but DOES NOT contain 4096 independent blind symbols. Detector is
informed. Not an author-code reproduction, and the paper leaves some
implementation choices unspecified or algebraically inconsistent.

Explicitly documented choices made here:
* R=3 for 512x512 host / 64x64 logo, to match LL to logo dimensions.
* Standard orthonormal 2D Haar implementation; grayscale host (paper-native).
* Mathematically valid inverse of LL=P H P^T uses P H' P^T, whereas the
  paper's displayed Eq.(16) writes P H' (omitting the transpose).
* Eq.(22) uses singular vectors of the RECEIVED Hessenberg matrix.
* Optional paper-2023 watermark singular-vector reconstruction is SEPARATE.
* Scaling alpha is selected on development hosts only, not tuned on test hosts.

This script runs real attacks and writes all trial data; it never fabricates
PSNR, NC, BER, or completion status. It is not in the blind 4096-bit ranking.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.linalg import hessenberg
from realtime_watermark.core import psnr, load_binary_watermark
from realtime_watermark.attack_manifest import reviewer_attacks, reviewer_manifest_summary

REF = '10.52866/ijcsm.2024.05.03.043'
METHOD = 'sudrajat_ernawan_amrullah_2024_lwt_hd_svd_equations'


def _gray(image):
    a=np.asarray(image)
    if a.ndim==2: return a.astype(np.float64)
    if a.ndim==3 and a.shape[-1]==3:
        # The paper uses grayscale cover images. For direct calls with RGB we
        # explicitly convert to a deterministic BT.601 luminance projection.
        return np.einsum('hwc,c->hw',a.astype(np.float64),np.array([.299,.587,.114]))
    raise ValueError('Expected grayscale or HxWx3 image')


def haar_forward(image, levels=3):
    """Orthonormal separable Haar DWT; return final LL and all detail triplets."""
    ll=_gray(image); details=[]
    if levels < 1:raise ValueError('levels must be >= 1')
    for _ in range(levels):
        h,w=ll.shape
        if h%2 or w%2:raise ValueError('Haar levels require dimensions divisible by 2^levels')
        a,b,c,d=ll[::2,::2],ll[::2,1::2],ll[1::2,::2],ll[1::2,1::2]
        nxt=(a+b+c+d)*.5
        lh=(a-b+c-d)*.5
        hl=(a+b-c-d)*.5
        hh=(a-b-c+d)*.5
        details.append((lh,hl,hh))
        ll=nxt
    return ll,details


def haar_inverse(ll, details):
    curr=np.asarray(ll,dtype=np.float64)
    for lh,hl,hh in reversed(details):
        h,w=curr.shape
        if (lh.shape,hl.shape,hh.shape)!=(curr.shape,)*3:
            raise ValueError('Detail dimensions do not match LL')
        r=np.empty((h*2,w*2),dtype=np.float64)
        r[::2,::2]=(curr+lh+hl+hh)*.5
        r[::2,1::2]=(curr-lh+hl-hh)*.5
        r[1::2,::2]=(curr+lh-hl-hh)*.5
        r[1::2,1::2]=(curr-lh-hl+hh)*.5
        curr=r
    return curr


def paper_spectrum(ll):
    H,P=hessenberg(ll,calc_q=True,check_finite=False)
    U,S,Vt=np.linalg.svd(H,full_matrices=True)
    return P,H,U,S,Vt


def embed(host,watermark,alpha,levels=3):
    if not np.isfinite(alpha) or alpha<=0:raise ValueError('alpha must be positive finite')
    gray=_gray(host); logo=_gray(watermark)
    if gray.shape[0]!=gray.shape[1] or logo.shape[0]!=logo.shape[1]:
        raise ValueError('Paper uses square host and watermark')
    ll,details=haar_forward(gray,levels)
    if ll.shape!=logo.shape:
        raise ValueError(f'Paper Eq.(22) requires matching dimensions: LL {ll.shape}, logo {logo.shape}; adjust levels')
    P,H,U,S,Vt=paper_spectrum(ll)
    Uw,Sw,Vtw=np.linalg.svd(logo,full_matrices=True)
    markedS=S+alpha*Sw
    Hmarked=(U*markedS[None,:])@Vt
    # Mathematically correct inverse of LL=P H P^T (not the typo P H in Eq.16)
    LLmarked=P@Hmarked@P.T
    marked=haar_inverse(LLmarked,details)
    marked=np.clip(np.rint(marked),0,255).astype(np.uint8)
    return marked,dict(host_sv=S.copy(),wm_u=Uw,wm_vt=Vtw,alpha=float(alpha),
        levels=levels,shape=logo.shape,detector_class='informed',
        side_information='original_host_singular_values',
        reference_doi=REF, paper_equation_reproduction=True,
        authors_code_reproduced=False)


def decode(image,key,reconstruction='paper2024_eq22'):
    if reconstruction not in ('paper2024_eq22','paper2023_watermark_vectors'):
        raise ValueError('Unknown reconstruction rule')
    ll,_=haar_forward(_gray(image),int(key['levels']))
    _P,_H,U,S,Vt=paper_spectrum(ll)
    recovered=(S-np.asarray(key['host_sv']))/float(key['alpha'])
    if reconstruction=='paper2024_eq22':
        # The 2024 paper's displayed Eq. (22): received HU, HS, HV
        out=(U*recovered[None,:])@Vt
    else:
        # Different rule explicitly described in Ernawan et al. 2023:
        # ORIGINAL watermark singular vectors are used at detection.
        out=(key['wm_u']*recovered[None,:])@key['wm_vt']
    if out.shape!=tuple(key['shape']):raise ValueError('Incorrect reconstruction shape')
    return np.clip(np.rint(out),0,255).astype(np.uint8)


def find_bmp(path):
    return sorted(x for x in Path(path).glob('*.bmp') if x.name.lower() not in ('lena.bmp','lenna.bmp'))


def _save(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:raise ValueError('Do not publish empty experimental results')
    with path.open('w',newline='',encoding='utf8') as f:
        cols=list(dict.fromkeys(k for d in rows for k in d));w=csv.DictWriter(f,fieldnames=cols)
        w.writeheader();w.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-dir',required=True)
    parser.add_argument('--test-dir',required=True)
    parser.add_argument('--watermark-dir',required=True)
    parser.add_argument('--out-dir',required=True)
    parser.add_argument('--target-psnr',type=float,default=39.5640,help='2024 paper average is an optional calibration reference')
    parser.add_argument('--alpha-min',type=float,default=.00001)
    parser.add_argument('--alpha-max',type=float,default=0.1)
    parser.add_argument('--calibration-steps',type=int,default=7)
    parser.add_argument('--max-hosts',type=int,default=0)
    parser.add_argument('--max-watermarks',type=int,default=0)
    parser.add_argument('--levels',type=int,default=3)
    parser.add_argument('--reconstruction',choices=['paper2024_eq22','paper2023_watermark_vectors'],default='paper2024_eq22')
    parser.add_argument('--quick',action='store_true')
    args=parser.parse_args()
    dev=find_bmp(args.development_dir);hosts=find_bmp(args.test_dir)
    wms=sorted(Path(args.watermark_dir).glob('*.png'))
    if args.max_hosts:hosts=hosts[:args.max_hosts]
    if args.max_watermarks:wms=wms[:args.max_watermarks]
    if not dev or not hosts or not wms:raise ValueError('Need dev/test hosts and watermark images')
    if {hashlib.sha256(p.read_bytes()).hexdigest() for p in dev}&{hashlib.sha256(p.read_bytes()).hexdigest() for p in hosts}:
        raise ValueError('Development/test overlap')
    attacks=dict(reviewer_attacks())
    attack_names=['clean','jpeg70','jpeg40','rotate_2p5'] if args.quick else list(attacks)
    data=[];cals=[]
    for wp in wms:
        wm=load_binary_watermark(wp,size=64)
        if wm.size!=4096:raise ValueError('Logo must be 64x64')
        devgray=[np.asarray(Image.open(p).convert('L')) for p in dev]
        calib=[]
        for alpha in np.geomspace(args.alpha_min,args.alpha_max,max(3,args.calibration_steps)):
            q=[]
            for x in devgray:
                marked,_=embed(x,wm,float(alpha),args.levels)
                q.append(float(psnr(x,marked)))
            median=float(np.median(q))
            calib.append((abs(median-args.target_psnr),float(alpha),median))
        _,alpha,median=min(calib,key=lambda z:z[0])
        cals.append(dict(watermark=wp.name,alpha=alpha,development_median_psnr=median,
                         target_psnr=args.target_psnr,calibration_count=len(dev),source='development_only'))
        for path in hosts:
            gray=np.asarray(Image.open(path).convert('L'))
            start=time.perf_counter();marked,key=embed(gray,wm,alpha,args.levels);end=time.perf_counter()
            quality=float(psnr(gray,marked))
            for attack in attack_names:
                atk=attacks[attack](np.repeat(marked[...,None],3,axis=2))
                recv=np.asarray(Image.fromarray(atk).convert('L'))
                t0=time.perf_counter();rec=decode(recv,key,args.reconstruction);t1=time.perf_counter()
                truth=wm>127;guess=rec>127
                ber=float(np.mean(truth!=guess))
                # Explicitly record the original paper's normalized cross correlation too.
                x=wm.astype(float).ravel();y=rec.astype(float).ravel()
                nc_paper=float(np.dot(x,y)/max(np.linalg.norm(x)*np.linalg.norm(y),1e-12))
                data.append(dict(method_id=METHOD,reference_doi=REF,host=path.name,
                    watermark=wp.name,attack=attack,detector_class='informed',
                    reconstructed_logo_pixels=4096,native_independent_payload_bits='NOT_4096',
                    reconstruction_rule=args.reconstruction,grayscale_host=True,
                    paper_equation_reproduction=True,authors_code_reproduced=False,
                    psnr=quality,ber=ber,nc=1-2*ber,nc_paper=nc_paper,
                    embed_ms=1000*(end-start),extract_ms=1000*(t1-t0),
                    alpha=alpha,status='measured'))
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    _save(out/'paper_2024_equation_trials.csv',data)
    _save(out/'paper_2024_dev_calibration.csv',cals)
    meta=dict(doi=REF,article='Sudrajat, Ernawan, Amrullah (2024), Iraqi Journal for Computer Science and Mathematics 5(3), 695-706',
        reproduction_level='independent equation-level implementation; author code unavailable',
        paper_equations='(10)-(22), p.700',haar_levels=args.levels,
        hessenberg_inverse='P @ Hmarked @ P.T (algebraically correct inverse; paper Eq.16 omits P.T)',
        reconstruction=args.reconstruction,
        detector='informed: original host singular-value vector',
        logo_not_4096_blind_bits=True,
        grayscale_native=True,
        known_unfixed_paper_choices=['the 2024 paper does not pin a unique R-level and alpha for each trial',
           'Eq.(16) prints P@H, not the orthogonal similarity inverse',
           'Eq.(22) uses attacked host singular vectors rather than watermark singular vectors'],
        attack_manifest_sha256=reviewer_manifest_summary()['sha256'],
        development_hosts=[p.name for p in dev],evaluation_hosts=[p.name for p in hosts],
        watermarks=[p.name for p in wms],rows=len(data),attacks=attack_names,
        completed=True)
    (out/'paper_2024_fidelity.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    print('Executed',len(data),'measured 2024 paper-equation trials (informed, native grayscale)')

if __name__=='__main__':main()
