#!/usr/bin/env python3
"""Produce selected PDF paper figures directly from completed experiment CSVs.

Does not extrapolate missing experiments, synthesize missing scores, or save all
extracted bitmaps. Each figure cites its numerical CSV in the filename metadata.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

PAPER20=['gaussian_noise','shot_noise','impulse_noise','speckle_noise','defocus_blur',
         'glass_blur','motion_blur','zoom_blur','snow','frost','fog','brightness',
         'contrast','elastic_transform','pixelate','jpeg_compression','scale',
         'rotation','shear','piecewise_affine']


def read(p):
    return list(csv.DictReader(p.open(newline='',encoding='utf8'))) if p.is_file() else []

def save(fig,path):
    fig.tight_layout();fig.savefig(path,format='pdf',bbox_inches='tight')
    plt.close(fig);print('Generated',path.name)

def main(a):
    root=a.results_dir;out=a.out_dir;out.mkdir(parents=True,exist_ok=True)
    cap=read(root/'capacity/payload_capacity_summary.csv')
    if cap:
        fig,ax=plt.subplots(figsize=(6,3.5))
        ax.plot([int(r['payload_bits']) for r in cap],[float(r['mean_attacked_nc']) for r in cap],marker='o')
        ax.set(xlabel='Payload bits',ylabel='Mean attacked bipolar NC',title='Payload capacity / fixed attack protocol')
        ax.grid(alpha=.2);save(fig,out/'capacity_robustness.pdf')
        fig,ax=plt.subplots(figsize=(6,3.5))
        ax.plot([int(r['payload_bits']) for r in cap],[float(r['mean_clean_psnr_db']) for r in cap],marker='o')
        ax.set(xlabel='Payload bits',ylabel='Clean PSNR (dB)',title='Payload / imperceptibility')
        ax.grid(alpha=.2);save(fig,out/'capacity_psnr.pdf')
    sec=read(root/'security/security_cases.csv')
    filtered=[r for r in sec if r['experiment'].startswith('public_carrier_suppression')
              and np.isfinite(float(r['psnr_vs_marked_db']))]
    if filtered:
        fig,ax=plt.subplots(figsize=(6,3.5))
        for name in sorted(set(r['experiment'] for r in filtered)):
            z=[r for r in filtered if r['experiment']==name]
            ax.scatter([float(r['psnr_vs_marked_db']) for r in z],
                       [float(r['ber']) for r in z],label=name.replace('public_carrier_suppression_',''),s=16,alpha=.65)
        ax.set(xlabel='Attacked vs marked PSNR (dB)',ylabel='Extracted payload BER',title='Carrier removal / distortion cost')
        ax.legend(fontsize=7);ax.grid(alpha=.2);save(fig,out/'security_removal_budget.pdf')
    roc=read(root/'detection/heldout_roc.csv')
    if roc:
        fig,ax=plt.subplots(figsize=(4,4))
        ax.plot([float(r['fpr']) for r in roc],[float(r['tpr']) for r in roc])
        ax.plot([0,1],[0,1],ls='--',linewidth=.8)
        ax.set(xlabel='False-positive rate',ylabel='True-positive rate',title='Known-message held-out ROC',xlim=(0,1),ylim=(0,1))
        save(fig,out/'known_message_roc.pdf')
    theoretical=read(root/'theory/joint_carrier_numerical.csv')
    if theoretical:
        fig,ax=plt.subplots(figsize=(4,4))
        x=np.array([float(r['predicted_minimum_energy']) for r in theoretical]);
        y=np.array([float(r['continuous_energy']) for r in theoretical])
        ax.scatter(x,y,s=10,alpha=.5);bound=float(max(max(x),max(y)))
        ax.plot([0,bound],[0,bound],ls='--',linewidth=.8)
        ax.set(xlabel='Analytical minimum energy',ylabel='Observed continuous energy',title='Joint carrier solution')
        save(fig,out/'joint_energy_consistency.pdf')
    mm=read(root/'mmwatermark/official_mmwatermark_raw.csv')
    if mm:
        matrix=np.full((20,5),np.nan)
        for i,attack in enumerate(PAPER20):
            for severity in range(1,6):
                z=[float(r['nc_bipolar']) for r in mm if r.get('canonical_attack_family')==attack
                   and int(r['severity'])==severity]
                if z:matrix[i,severity-1]=float(np.mean(z))
        fig,ax=plt.subplots(figsize=(8,7))
        im=ax.imshow(np.ma.masked_invalid(matrix),aspect='auto',vmin=-1,vmax=1,cmap='coolwarm')
        ax.set_xticks(range(5),labels=[1,2,3,4,5]);ax.set_yticks(range(20),labels=PAPER20,fontsize=7)
        ax.set(xlabel='Official severity',title='Original MMWatermark 20 x 5 / bipolar NC')
        fig.colorbar(im,ax=ax,label='Mean NC');save(fig,out/'mmwatermark_paper100_heatmap.pdf')
        if np.isfinite(matrix).all():
            fig,ax=plt.subplots(figsize=(6,3.5))
            for name,indices in [('Noise',slice(0,4)),('Blur',slice(4,8)),
                                 ('Weather',slice(8,12)),('Digital',slice(12,16)),('Geometric',slice(16,20))]:
                ax.plot(range(1,6),matrix[indices].mean(axis=0),marker='o',label=name)
            ax.set(xlabel='Severity',ylabel='Average bipolar NC',title='MMWatermark paper-100 severity effects')
            ax.legend(fontsize=8);save(fig,out/'mmwatermark_severity.pdf')
    gallery=sorted((root/'gallery').glob('*_extraction_gallery.png')) if (root/'gallery').exists() else []
    if gallery:
        # Save only two selected comparison sheets; raw PNGs are excluded from ZIP.
        ims=[Image.open(p).convert('RGB') for p in gallery[:2]]
        ims[0].save(out/'selected_extraction_examples.pdf',save_all=True,append_images=ims[1:],resolution=120)
        for im in ims:im.close()
    video=read(root/'video_security/temporal_attack_voting.csv')
    if video:
        z=[r for r in video if r['temporal_policy']=='original']
        fig,ax=plt.subplots(figsize=(6,3.5))
        for name in sorted(set(r['attack'] for r in z)):
            grouped={w:[] for w in sorted({int(r['window']) for r in z})}
            for r in z:
                if r['attack']==name: grouped[int(r['window'])].append(float(r['voted_nc']))
            ax.plot(list(grouped),[np.mean(v) if v else np.nan for v in grouped.values()],marker='o',label=name)
        ax.set(xlabel='Voting frames',ylabel='Mean voted NC',title='Real-video temporal recovery')
        ax.legend(fontsize=7);save(fig,out/'temporal_voting_robustness.pdf')
    if not list(out.glob('*.pdf')):print('No completed experiment CSVs/selected galleries yet; no figures generated.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    main(p.parse_args())
