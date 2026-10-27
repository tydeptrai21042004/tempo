#!/usr/bin/env python3
"""Reproduce two manuscript-style *RESULT* PDF layouts from the frozen 45-attack protocol.

The reference PDFs define layout only, not method names or measurements. The
5-column comparison uses actual implemented 4096-bit blind methods with frozen
PSNR calibration. The Tiffany gallery visualizes proposed-method recovery for
EVERY one of the 45 non-clean canonical attacks plus clean (three PDF pages).

Each NC is recomputed from the extracted 64x64 watermark and must match the
archived matched-PSNR per-case CSV; disagreements fail the figure export.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from realtime_watermark.attack_manifest import reviewer_attacks, reviewer_manifest_summary
from realtime_watermark.core import load_binary_watermark, load_rgb, nc_binary, psnr
from realtime_watermark.registry import embed, extract
from realtime_watermark.proposals.jilp_qim.config import JILPQIMConfig

plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,
    'pdf.fonttype':42, 'ps.fonttype':42, 'savefig.facecolor':'white'})

COMPARISON_METHODS = (
    ('proposal', 'Proposed'),
    ('dct_dm_qim_chen2001', 'DCT DM-QIM'),
    ('dct_stdm_qim_chen2001', 'STDM-QIM'),
    ('dct_iss_malvar2003', 'ISS'),
)
# Only conditions in frozen 45; none added to the evaluation set.
COMPARISON_ATTACKS = ('clean','jpeg30','gaussian_variance_0p05',
                      'speckle_variance_0p03','rotate_15p0')
LABELS = {
    'clean': 'No attack', 'jpeg30': 'JPEG quality 30',
    'gaussian_variance_0p05': 'Gaussian var. 0.05',
    'speckle_variance_0p03': 'Speckle var. 0.03',
    'rotate_15p0': 'Rotation 15°',
}


def sha256(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readcsv(path:Path):
    if not path.is_file():raise FileNotFoundError(f'Measured matched-PSNR rows missing: {path}')
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def select_named(paths, substring):
    matching=[p for p in sorted(paths) if substring.casefold() in p.name.casefold()]
    if len(matching)!=1:
        raise ValueError(f'Expected one file containing {substring!r}; found {[p.name for p in matching]}')
    return matching[0]


def binmask(arr):
    x=np.asarray(arr)
    if x.ndim!=2 or x.shape!=(64,64):raise ValueError(f'Invalid decoded watermark shape {x.shape}')
    return np.where(x>127,255,0).astype(np.uint8) if np.max(x)>1 else (x.astype(np.uint8)*255)


def read_benchmark(matched_dir):
    proto_file=matched_dir/'matched_psnr_protocol.json'
    proto=json.loads(proto_file.read_text(encoding='utf-8'))
    if proto.get('attack_preset') not in ('reviewer46','internal46'):
        raise ValueError('Cannot mix attack registries: matched benchmark is not canonical reviewer46')
    observed=readcsv(matched_dir/'matched_psnr_per_attack.csv')
    reference={}
    for row in observed:
        key=(row['host'],row['watermark'],row['method_id'],row['phase'])
        if key in reference:raise RuntimeError(f'Duplicate benchmark case: {key}')
        reference[key]=row
    frozen=proto.get('frozen_steps_by_watermark')
    if not frozen:raise ValueError('Missing frozen steps; do not recalibrate on held-out figures')
    return proto,reference,{'protocol_sha256':sha256(proto_file),
           'per_attack_csv_sha256':sha256(matched_dir/'matched_psnr_per_attack.csv')}


def write_manifest(path,payload):
    path.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matched-dir',type=Path,required=True)
    p.add_argument('--host-dir',type=Path,required=True)
    p.add_argument('--watermark-dir',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=True)
    spec=reviewer_manifest_summary()
    ids=spec['attack_ids']
    assert len(ids)==46 and ids[0]=='clean' and len(set(ids))==46
    attack_fns=dict(reviewer_attacks())
    if tuple(attack_fns)!=tuple(ids):raise RuntimeError('Runtime registry differs from frozen 45-attack order')
    if not set(COMPARISON_ATTACKS)<=set(ids):raise RuntimeError('A figure attack is outside the canonical 45')
    proto,reference,hashes=read_benchmark(a.matched_dir)
    config=JILPQIMConfig.load(a.config)
    cfg_hash=config.sha256()
    # Verify the exact same configuration was used in archived comparisons.
    recorded_hash=proto.get('proposal_configuration_sha256')
    if recorded_hash and cfg_hash!=recorded_hash:
        raise ValueError(f'Configuration SHA mismatch: {cfg_hash} vs archived {recorded_hash}')
    hosts=sorted(a.host_dir.glob('*.bmp'))
    logos=sorted(a.watermark_dir.glob('*.png'))
    baboon=select_named(hosts,'baboon')
    tiffany=select_named(hosts,'tiffany')
    named=[p for p in logos if 'cclogocircle' in ''.join(x for x in p.stem.casefold() if x.isalnum())]
    if len(named)!=1:raise ValueError(f'Expected exactly one CC circle watermark; found {[p.name for p in named]}')
    logo=named[0]
    wm=load_binary_watermark(logo,size=64)
    watermark_name=logo.name
    frozen=proto['frozen_steps_by_watermark'].get(watermark_name,{})
    if not frozen:raise ValueError(f'Missing frozen matched-PSNR configuration for {watermark_name}')
    cache={};observations=[]

    def measured(host_path,method,attack):
        key=(host_path.name,watermark_name,method,attack)
        row=reference.get(key)
        if row is None:raise ValueError(f'No archived matched 45-attack observation for {key}')
        if method not in frozen:raise ValueError(f'No frozen matched-PSNR step for {method}')
        if method not in cache or host_path.name not in cache[method]:
            host=load_rgb(host_path,mode='512')
            step=float(frozen[method]);seed=int(row['evaluation_seed'])
            kw={'method_params':config.to_dict()} if method=='proposal' else {}
            marked,k=embed(method,host,wm,seed=seed,repeat='auto',step=step,**kw)
            cache.setdefault(method,{})[host_path.name]=(host,marked,k,seed,step)
        host,marked,k,seed,step=cache[method][host_path.name]
        if int(row['evaluation_seed'])!=seed:raise ValueError(f'Seed mismatch for {key}')
        received=attack_fns[attack](marked)
        extracted=binmask(extract(received,k))
        nc=float(nc_binary(wm,extracted))
        ber=float(np.mean(np.asarray(wm)!=extracted))
        expected=float(row['nc_bipolar'])
        if not math.isfinite(expected) or abs(nc-expected)>1e-5:
            raise RuntimeError(f'Archived NC mismatch for {key}: replay={nc:.8f}, CSV={expected:.8f}')
        # Same binary/bipolar NC convention as the internal benchmark.
        if abs(nc-(1-2*ber))>1e-6:
            raise RuntimeError(f'NC/BER convention mismatch: {key}: {nc} / {ber}')
        observations.append({'host':host_path.name,'watermark':watermark_name,
            'method':method,'attack':attack,'nc':nc,'ber':ber,
            'matched_step':step,'watermarked_psnr_db':float(psnr(host,marked)),
            'eligible_under_primary_psnr_rule':str(row.get('primary_rank_eligible','')).lower()=='true',
            'source_csv_nc':expected,'evaluation_seed':seed})
        return host,received,extracted,nc,ber

    # Reference 1: five attack rows and 5 columns (original + four actual methods).
    # Do not label images as Guo / Nha / Su: those implementations are NOT in TEMPO.
    fig,axes=plt.subplots(len(COMPARISON_ATTACKS),len(COMPARISON_METHODS)+1,
                          figsize=(8.8,9.2),gridspec_kw={'wspace':0.07,'hspace':0.24})
    for c,title in enumerate(['Original']+[x[1] for x in COMPARISON_METHODS]):
        axes[0,c].set_title(title,fontsize=9,pad=12)
    for ri,attack in enumerate(COMPARISON_ATTACKS):
        ax=axes[ri,0];ax.imshow(wm,cmap='gray',vmin=0,vmax=255,interpolation='nearest');ax.axis('off')
        ax.text(0,-0.13,LABELS.get(attack,attack),transform=ax.transAxes,fontsize=8,va='top')
        for ci,(method,_label) in enumerate(COMPARISON_METHODS,1):
            _,_,rec,nc,_=measured(baboon,method,attack)
            ax=axes[ri,ci];ax.imshow(rec,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
            ax.text(.5,-.045,f'NC={nc:.3f}',transform=ax.transAxes,ha='center',va='top',fontsize=7)
    fig.subplots_adjust(left=.045,right=.995,top=.94,bottom=.04)
    comparison=a.out_dir/'fig_extracted_comparison.pdf'
    fig.savefig(comparison,format='pdf',dpi=250,metadata={'Title':'TEMPO matched-PSNR extracted watermarks; frozen 45-attack registry'})
    plt.close(fig)

    # Reference 2: Tiffany attacked RGB next to the recovered watermark; 3x6 panels
    # per page. Includes clean and ALL 45 attacked conditions (3 pages total).
    tiffany_pdf=a.out_dir/'fig_Tiffani.pdf'
    panels_per_page=18
    with PdfPages(tiffany_pdf,metadata={'Title':'TEMPO Tiffany 45 internal attacks and clean'}) as pdf:
        for first in range(0,len(ids),panels_per_page):
            block=ids[first:first+panels_per_page]
            fig,axes=plt.subplots(6,3,figsize=(11.5,14.1),gridspec_kw={'wspace':.035,'hspace':.37})
            for ax in axes.flat:ax.set_xticks([]);ax.set_yticks([])
            for i,attack in enumerate(block):
                ax=axes.flat[i]
                _,received,rec,nc,ber=measured(tiffany,'proposal',attack)
                left=np.asarray(Image.fromarray(received).resize((240,240),Image.Resampling.LANCZOS))
                right=(np.asarray(Image.fromarray(rec,'L').resize((240,240),Image.Resampling.NEAREST).convert('RGB')))
                ax.imshow(np.concatenate((left,right),axis=1),interpolation='nearest')
                title=LABELS.get(attack,attack.replace('_',' '))
                ax.set_title(f'{title}\nNC={nc:.4f}     BER={ber:.4f}',fontsize=8,pad=4)
                for spine in ax.spines.values():spine.set_visible(True);spine.set_linewidth(.45);spine.set_color('#777777')
            for ax in axes.flat[len(block):]:ax.axis('off')
            fig.subplots_adjust(left=.015,right=.985,top=.965,bottom=.025)
            pdf.savefig(fig,dpi=250)
            plt.close(fig)
    csvpath=a.out_dir/'reference_style_figure_observations.csv'
    with csvpath.open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(observations[0]));writer.writeheader();writer.writerows(observations)
    report={'status':'COMPLETE','figure_role':'result panels only; sample PDFs used for layout reference, not measurements',
      'attack_protocol':spec['protocol_id'],'attack_manifest_sha256':spec['sha256'],
      'internal_attacked_conditions':45,'clean_conditions':1,'external_catalogs_unmodified':True,
      'matched_source':hashes,'proposal_config_sha256':cfg_hash,
      'watermark':watermark_name,'comparison_host':baboon.name,'tiffany_host':tiffany.name,
      'comparison_methods':[m for m,_ in COMPARISON_METHODS],
      'comparison_attack_ids':list(COMPARISON_ATTACKS),
      'tiffany_attack_ids':ids,'observations_csv_sha256':sha256(csvpath),
      'pdfs':{p.name:{'sha256':sha256(p),'bytes':p.stat().st_size} for p in (comparison,tiffany_pdf)},
      'note':'Rows not passing matched-PSNR eligibility are illustrative only; consult observations CSV and primary eligible comparison table.'}
    write_manifest(a.out_dir/'reference_style_figures_manifest.json',report)
    print(json.dumps({'status':'COMPLETE','internal_attacks':45,'pdfs':list(report['pdfs'])},indent=2))

if __name__=='__main__': main()
