#!/usr/bin/env python3
"""TEMPO: manuscript-quality, vector PDF figures made with Python.

Analytical schematic panels use formulas from the method. Empirical panels are
NEVER fabricated: they require saved observations or perform a documented
embedding/decoding experiment. Missing evidence is reported in a manifest.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image

plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.12})


def export(fig, path):
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path, format='pdf', dpi=200)
    plt.close(fig)
    return {'name':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'size':path.stat().st_size}


def read_csv(path):
    with open(path,newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))


def value(row,*fields):
    for k in fields:
        try:
            if row.get(k,'') != '':
                v=float(row[k]);
                if np.isfinite(v):return v
        except (TypeError,ValueError):pass
    return None


def workflow(out):
    fig, axes=plt.subplots(1,3,figsize=(11.2,4.15),layout='constrained')
    groups=[('a   Embedding',['RGB frame','8 x 8 DCT blocks','Payload / pilot coordinates','Center-QIM targets','Minimum-norm RGB update','Clip / integer correction']),
            ('b   Blind detection',['Received frame','Pilot agreement','Confidence test','Aligned extraction /','bounded geometric recovery','Inverse permutation']),
            ('c   Causal stream',['Timed frame arrivals','Bounded input queue','Embedding / channel','Blind decoding','Frame-wise NC / BER','Causal majority vote'])]
    for ax,(header,steps) in zip(axes,groups):
        ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.text(.06,.97,header,ha='left',va='top',fontweight='bold',fontsize=10)
        top=.84; height=.095; gap=.033
        for j,step in enumerate(steps):
            yy=top-j*(height+gap)
            ax.add_patch(FancyBboxPatch((.08,yy-height),.84,height,boxstyle='round,pad=0.007,rounding_size=.013',
                                      linewidth=.9,edgecolor='#484848',facecolor=('#e9eef2' if j in (2,3) else '#ffffff')))
            ax.text(.5,yy-height/2,step,ha='center',va='center',fontsize=8.5)
            if j<len(steps)-1:ax.add_patch(FancyArrowPatch((.5,yy-height-.003),(.5,yy-height-gap+.008),
                                 arrowstyle='-|>', mutation_scale=10,linewidth=.9,color='#484848'))
    return export(fig,out/'fig01.pdf')


def carrier(out):
    from realtime_watermark.core import dct_basis
    p01=dct_basis(8,0,1);p10=dct_basis(8,1,0)
    arrs=[p01,p10,(p01-p10)/2,(p01+p10)/2]
    titles=['(a) DCT $\\phi_{01}$','(b) DCT $\\phi_{10}$','(c) Payload $a_p$','(d) Pilot $a_s$']
    fig,axes=plt.subplots(2,3,figsize=(10.8,6.35),layout='constrained')
    for ax,x,t in zip(axes.ravel()[:4],arrs,titles):
        vmax=float(np.max(np.abs(x)))
        ax.imshow(x,vmin=-vmax,vmax=vmax,cmap='RdBu_r',interpolation='nearest')
        ax.set_title(t,fontsize=9);ax.set(xticks=[],yticks=[])
    ax=axes[1,1];ax.set_title('(e) Orthogonal carrier coordinates',fontsize=9)
    ax.set(xlim=(-1,1),ylim=(-1,1),xlabel='payload coordinate',ylabel='pilot coordinate')
    ax.axhline(0,c='#888888',lw=.7);ax.axvline(0,c='#888888',lw=.7)
    ax.arrow(-.5,-.4,1,0,head_width=.055,length_includes_head=True,color='#245c75')
    ax.arrow(-.5,-.4,0,.85,head_width=.055,length_includes_head=True,color='#ad6541')
    ax.text(.12,-.53,'payload displacement',fontsize=8)
    ax.text(-.94,.36,'pilot displacement',fontsize=8)
    ax=axes[1,2];ax.set_title('(f) Binary QIM cells and boundaries',fontsize=9)
    for q in range(-3,4):
        ax.axvspan(q-.5,q+.5,facecolor=('#cbdde6' if q%2==0 else '#eedacf'),alpha=.65)
        ax.plot(q,0,'|',c='black',ms=11)
    ax.axvline(1.5,c='#666666',ls=':',lw=1)
    ax.annotate('',xy=(2,0.24),xytext=(1.56,.24),arrowprops={'arrowstyle':'->','lw':1.4})
    ax.text(1.58,.4,'center target',fontsize=8)
    ax.set(xlim=(-3.5,3.5),ylim=(-.4,.65),yticks=[],xlabel='$v/\\Delta$')
    return export(fig,out/'fig02.pdf')


def montage(out,hostdir,wmdir):
    hosts=sorted(x for x in hostdir.glob('*.bmp') if not any(s in x.stem.lower() for s in ('lenna','lena')))
    wms=sorted(wmdir.glob('*.png'))
    if len(hosts)!=12 or len(wms)!=2:raise ValueError(f'Expected 12 hosts and 2 watermarks, found {len(hosts)} and {len(wms)}')
    fig,axes=plt.subplots(4,4,figsize=(11,10.5),layout='constrained')
    panels=[(p,p.stem) for p in hosts]+[(p,p.stem) for p in wms]
    for ax,(file,name) in zip(axes.flat,panels):
        with Image.open(file) as im: ax.imshow(im.convert('RGB'))
        ax.set_title(name,fontsize=8);ax.axis('off')
    for ax in list(axes.flat)[len(panels):]:ax.axis('off')
    return export(fig,out/'host_and_watermark_images.pdf')


def hyperparams(out,root):
    source=root/'joint_sensitivity'/'joint_parameter_coupling.csv'
    if not source.is_file():raise FileNotFoundError(f'Measured joint-sensitivity CSV missing: {source}')
    rows=[r for r in read_csv(source) if r.get('attack')!='clean' and value(r,'nc') is not None]
    if not rows:raise ValueError('No attacked NC data in joint sensitivity')
    fig,axes=plt.subplots(2,2,figsize=(10.8,7.9),layout='constrained')
    counts=sorted({int(float(r['pilot_count'])) for r in rows})
    pilot=64
    if pilot not in counts:raise ValueError('Manuscript figure requires measured 64-pilot cells')
    group=defaultdict(list)
    for r in rows:
        k=(int(float(r['pilot_count'])),float(r['step']),float(r['pilot_step']))
        group[k].append(float(r['nc']))
    xs=sorted({x[1] for x in group if x[0]==pilot});ys=sorted({x[2] for x in group if x[0]==pilot})
    dat=np.array([[np.mean(group.get((pilot,x,y),[np.nan])) for x in xs] for y in ys]);ax=axes[0,0]
    im=ax.imshow(dat,aspect='auto',origin='lower',vmin=0,vmax=1,cmap='viridis')
    ax.set(xticks=range(len(xs)),xticklabels=[f'{x:.2f}' for x in xs],yticks=range(len(ys)),yticklabels=[f'{y:.2f}' for y in ys],xlabel='Payload QIM step',ylabel='Pilot QIM step')
    ax.set_title(f'(a) Joint steps, {pilot} pilot blocks');fig.colorbar(im,ax=ax,label='Attacked NC',shrink=.8)
    for i in range(len(ys)):
        for j in range(len(xs)):
            if np.isfinite(dat[i,j]):ax.text(j,i,f'{dat[i,j]:.3f}',ha='center',va='center',color='white' if dat[i,j]<.65 else 'black',fontsize=8)
    ax=axes[0,1];psnr=defaultdict(list)
    # Quality values occur once per attack and are averaged by operating point.
    for r in rows:
        if value(r,'psnr') is not None:psnr[(float(r['step']),int(float(r['pilot_count'])))].append((float(r['psnr']),float(r['nc'])))
    for n in counts:
        pts=sorted([(np.mean(p,axis=0),k[0]) for k,p in psnr.items() if k[1]==n],key=lambda z:z[1])
        if pts:ax.plot([p[0][0] for p in pts],[p[0][1] for p in pts],marker='o',lw=1.1,label=f'{n} pilots')
    ax.set(xlabel='Embedding PSNR (dB)',ylabel='Mean attacked NC',title='(b) Quality versus robustness');ax.legend(fontsize=7)
    ax=axes[1,0];bycount=defaultdict(list)
    for r in rows:bycount[int(float(r['pilot_count']))].append(float(r['nc']))
    ax.plot(counts,[np.mean(bycount[n]) for n in counts],marker='o',c='#245c75')
    ax.set(xticks=counts,xlabel='Pilot count',ylabel='Mean attacked NC',title='(c) Synchronization density');ax.grid(axis='y',alpha=.2)
    ax=axes[1,1];byattack=defaultdict(lambda:defaultdict(list))
    for r in rows:byattack[r['attack']][int(float(r['pilot_count']))].append(float(r['nc']))
    for name,d in sorted(byattack.items()):ax.plot(sorted(d),[np.mean(d[k]) for k in sorted(d)],marker='o',label=name)
    ax.set(xticks=counts,xlabel='Pilot count',ylabel='NC',title='(d) Interactions with attack type')
    ax.legend(fontsize=7)
    return export(fig,out/'proposal_hyperparameter_multi_panel.pdf')


def baboon(out,hostdir,wmdir,config_path,include_baselines=True):
    from realtime_watermark.core import load_binary_watermark,nc_binary,bit_accuracy,psnr
    from realtime_watermark.attack_manifest import reviewer_attacks
    from realtime_watermark.registry import embed,extract,COMMON_IMAGE_METHODS
    from realtime_watermark.proposals.jilp_qim.config import JILPQIMConfig
    hostfile=next((p for p in hostdir.glob('*.bmp') if 'baboon' in p.stem.lower()),None)
    wmfile=next((p for p in wmdir.glob('*.png') if 'cc.logo.circle' in p.name.lower()),None)
    if not hostfile or not wmfile:raise FileNotFoundError('Baboon or CC-logo image missing')
    host=np.asarray(Image.open(hostfile).convert('RGB'))
    wm=load_binary_watermark(wmfile,size=64)
    cfg=JILPQIMConfig.load(config_path)
    attacks=dict(reviewer_attacks())
    if 'jpeg30' not in attacks:raise ValueError('The canonical reviewer attack preset has no jpeg30')
    marked,key=embed('proposal',host,wm,seed=2026,repeat=1,method_params=cfg.to_dict())
    attacked=attacks['jpeg30'](marked)
    fig,axes=plt.subplots(1,3,figsize=(11,3.75),layout='constrained')
    for ax,img,title in zip(axes,[host,marked,attacked],['(a) Original Baboon',f'(b) Watermarked, {psnr(host,marked):.2f} dB','(c) JPEG quality 30']):
        ax.imshow(img);ax.set_title(title,fontsize=9);ax.axis('off')
    a=export(fig,out/'baboon_attacked_host.pdf')
    methods=(list(COMMON_IMAGE_METHODS) if include_baselines else ['proposal'])
    recovered=[];errors={};names=[]
    for method in methods:
        try:
            if method=='proposal':yk=marked;kk=key
            else:yk,kk=embed(method,host,wm,seed=2026)
            received=attacks['jpeg30'](yk)
            rec=np.asarray(extract(received,kk),dtype=np.uint8)
            score=float(nc_binary(wm,rec))
            recovered.append(rec);names.append(f'{method.replace("dct_", "")}\nNC={score:.3f}')
        except Exception as exc:
            errors[method]=f'{type(exc).__name__}: {str(exc)[:150]}'
    if 'proposal' not in methods or not recovered: raise RuntimeError('Cannot render extraction gallery without real decodes')
    allims=[wm]+recovered;allnames=['Original CC watermark']+names
    fig,axes=plt.subplots(1,len(allims),figsize=(max(7,1.8*len(allims)),2.65),layout='constrained')
    for ax,im,name in zip(np.atleast_1d(axes),allims,allnames):
        ax.imshow(np.asarray(im).reshape(64,64),cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        ax.set_title(name,fontsize=7);ax.axis('off')
    b=export(fig,out/'baboon_extracted_methods.pdf')
    return [a,b],errors


def result_gallery(out,root):
    """Render actual matched-Baboon recovered images, never run the decoder implicitly."""
    folder=root/'result_figure_sources'/'baboon'
    marker=folder/'gallery_manifest.json'
    if not marker.is_file():raise FileNotFoundError('Matched-gallery evidence unavailable: '+str(marker))
    meta=json.loads(marker.read_text())
    if meta.get('status')!='complete':raise ValueError('Matched gallery is incomplete')
    bywm=defaultdict(list)
    for item in meta['panels']:bywm[item['watermark']].append(item)
    produced=[]
    for wm,items in sorted(bywm.items()):
        wm_dir=folder/Path(wm).stem
        original=np.asarray(Image.open(wm_dir/'original.png').convert('L'))
        ordered=sorted(items,key=lambda z:(z['method']!='proposal',z['method']))
        images=[original]+[np.asarray(Image.open(wm_dir/(x['method']+'_decoded.png')).convert('L')) for x in ordered]
        short={'proposal':'Proposed','dct_dm_qim_chen2001':'DM-QIM',
               'dct_stdm_qim_chen2001':'STDM-QIM','dct_iss_malvar2003':'ISS',
               'ca_qim_a2_mao2024':'CA-QIM-A2',
               'haar_hd_svd_qim_adapter2024':'Independent Haar-HD-SVD QIM'}
        labels=['Original']+[f"{short.get(x['method'],x['method'])} ({x['detector_type'].replace('_','-')})\nNC={x['nc']:.3f}, PSNR={x['psnr_db']:.1f} dB" for x in ordered]
        fig,axes=plt.subplots(1,len(images),figsize=(max(9,2.3*len(images)),2.75),layout='constrained')
        for ax,arr,label in zip(np.atleast_1d(axes),images,labels):
            ax.imshow(arr,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            ax.set_title(label,fontsize=7);ax.axis('off')
        name='baboon_extracted_methods.pdf' if wm==next(iter(sorted(bywm))) else 'baboon_extracted_'+Path(wm).stem+'.pdf'
        produced.append(export(fig,out/name))
    # Source watermarked/attacked images from the same run, never standalone re-embedding.
    first=next(iter(sorted(bywm)))
    candidates=sorted(bywm[first],key=lambda z:(z['method']!='proposal',z['method']))
    if not candidates:raise ValueError('No gallery panels')
    src=folder/Path(first).stem
    original=np.asarray(Image.open(folder/'host.png').convert('RGB'))
    item=candidates[0];method=item['method']
    watermarked=np.asarray(Image.open(src/(method+'_marked.png')).convert('RGB'))
    attacked=np.asarray(Image.open(src/(method+'_attacked.png')).convert('RGB'))
    fig,axes=plt.subplots(1,3,figsize=(9.5,3.25),layout='constrained')
    for ax,arr,title in zip(axes,[original,watermarked,attacked],['(a) Host','(b) Watermarked','(c) JPEG quality 30']):
        ax.imshow(arr);ax.set_title(title,fontsize=9);ax.axis('off')
    produced.append(export(fig,out/'baboon_attacked_host.pdf'))
    return produced


def roc(out,root):
    files=[root/'detection'/'heldout_roc.csv',root/'reviewer_experiments'/'roc.csv']
    source=next((p for p in files if p.is_file()),None)
    if source is None:raise FileNotFoundError('No measured ROC CSV')
    rows=read_csv(source)
    x=[];y=[]
    for r in rows:
        f=value(r,'fpr','FPR');t=value(r,'tpr','TPR')
        if f is not None and t is not None:x.append(f);y.append(t)
    if len(x)<2:raise ValueError(f'ROC observations missing from {source}')
    fig,ax=plt.subplots(figsize=(4.8,4.1));ax.plot(x,y,lw=1.6,label='Observed detection')
    ax.plot([0,1],[0,1],ls=':',lw=.8,c='gray',label='Chance')
    ax.set(xlabel='False-positive rate',ylabel='True-positive rate',xlim=(-.01,1),ylim=(-.01,1.01),title='Claim-conditioned watermark detection')
    ax.legend(fontsize=8);return export(fig,out/'reviewer_roc.pdf')


def capacity(out,root):
    file=root/'reviewer_experiments'/'capacity_by_payload_attack.csv'
    if not file.is_file():raise FileNotFoundError('No measured capacity CSV')
    rows=read_csv(file);d=defaultdict(list)
    for r in rows:
        n=value(r,'payload_bits');ms=value(r,'mean_embed_ms');nc=value(r,'mean_nc');
        if n is not None and ms is not None and nc is not None:d[(r['attack'],int(n))].append((ms,nc))
    if not d:raise ValueError('No usable capacity metrics')
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
    for attack in sorted({k[0] for k in d}):
        pairs=sorted([(n,np.mean(d[(attack,n)],axis=0)) for a,n in d if a==attack])
        axes[0].plot([n for n,v in pairs],[v[0] for n,v in pairs],marker='o',label=attack)
        axes[1].plot([n for n,v in pairs],[v[1] for n,v in pairs],marker='o',label=attack)
    axes[0].set(xlabel='Payload (bits)',ylabel='Mean embedding time (ms)',title='(a) Payload cost')
    axes[1].set(xlabel='Payload (bits)',ylabel='Mean recovered NC',title='(b) Capacity sensitivity')
    axes[1].legend(fontsize=7)
    return export(fig,out/'reviewer_capacity_latency.pdf')



def orthogonality(out,root):
    file=root/'reviewer_experiments'/'orthogonality_and_integer.csv'
    geom=root/'reviewer_experiments'/'geometry_and_targeted_attacks.csv'
    if not file.is_file() or not geom.is_file():
        raise FileNotFoundError('Measured integer/geometry carrier files missing')
    integer=read_csv(file); geometric=read_csv(geom)
    leak=[value(r,'pilot_coordinate_difference_final_vs_pilot_stage_mean') for r in integer]
    leak=[x for x in leak if x is not None]
    if not leak:raise ValueError('No integer carrier displacement recorded')
    byattack=defaultdict(list)
    for r in geometric:
        if r.get('status')=='ok' and r.get('attack')!='clean':
            x=value(r,'pilot_blocks_payload_coordinate_rmse')
            y=value(r,'pilot_blocks_pilot_coordinate_rmse')
            if x is not None and y is not None:byattack[r['attack']].append((x,y))
    if not byattack:raise ValueError('No measured geometry coordinate errors')
    top=sorted(byattack,key=lambda z:-np.mean([p[0] for p in byattack[z]]))[:8]
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.1),layout='constrained')
    axes[0].hist(leak,bins=min(12,max(3,len(leak)//2)),edgecolor='white',color='#486c81')
    axes[0].set(xlabel='Absolute pilot displacement after payload step',ylabel='Host / watermark observations',
        title='(a) Integer-domain interaction')
    y=np.arange(len(top));w=.37
    axes[1].barh(y-w/2,[np.mean([v[0] for v in byattack[a]]) for a in top],height=w,label='Payload coordinate')
    axes[1].barh(y+w/2,[np.mean([v[1] for v in byattack[a]]) for a in top],height=w,label='Pilot coordinate')
    axes[1].set(yticks=y,yticklabels=top,xlabel='Coordinate RMSE after geometry',title='(b) Resampling disturbance')
    axes[1].invert_yaxis();axes[1].legend(fontsize=7)
    fig.text(.5,-.025,'Observed coordinate movement, not a causal cross-carrier leakage proof',ha='center',fontsize=8)
    return export(fig,out/'reviewer_orthogonality.pdf')


def stream_deadlines(out,root):
    file=root/'throughput_audit'/'wallclock_throughput_audit.csv'
    if not file.is_file():raise FileNotFoundError('No wall-clock throughput audit CSV')
    rows=read_csv(file)
    selected=[r for r in rows if r.get('resolution')=='1280x720' and r.get('channel') in ('clean','jpeg70')]
    if not selected:raise ValueError('No measured 720p streams')
    names=[f"{r.get('watermark','')} / {r['channel']}" for r in selected]
    fps=[value(r,'effective_output_fps') for r in selected]
    ontime=[value(r,'on_time_completion_ratio') for r in selected]
    if any(v is None for v in fps+ontime):raise ValueError('Stream metric missing')
    fig,axes=plt.subplots(1,2,figsize=(9.3,3.9),layout='constrained')
    x=np.arange(len(selected));axes[0].bar(x,fps,color='#486c81')
    axes[0].axhline(30,color='#a54c40',ls='--',lw=1,label='30 FPS target')
    axes[0].set(ylabel='Effective output FPS',xticks=x,xticklabels=names,title='(a) Measured output rate')
    axes[0].tick_params(axis='x',labelrotation=50);axes[0].legend(fontsize=7)
    axes[1].bar(x,np.asarray(ontime)*100,color='#486c81')
    axes[1].set(ylabel='On-time completion (%)',xticks=x,xticklabels=names,ylim=(0,101),title='(b) Deadline behavior')
    axes[1].tick_params(axis='x',labelrotation=50)
    return export(fig,out/'reviewer_realtime_720p.pdf')

def reference_style_pdf(out,root,filename):
    """Include an already verified experiment PDF, never remake results here."""
    source=root/'result_comparison_figures'/filename
    manifest=source.parent/'reference_style_figures_manifest.json'
    if not manifest.is_file() or not source.is_file():
        raise FileNotFoundError('Verified result figure or provenance missing: '+str(source))
    evidence=json.loads(manifest.read_text(encoding='utf-8'))
    recorded=evidence.get('pdfs',{}).get(filename,{}).get('sha256')
    if evidence.get('status')!='COMPLETE' or not recorded or hashlib.sha256(source.read_bytes()).hexdigest()!=recorded:
        raise RuntimeError('Result PDF SHA256/provenance mismatch: '+filename)
    destination=out/filename
    destination.parent.mkdir(parents=True,exist_ok=True)
    if source.resolve()!=destination.resolve():shutil.copyfile(source,destination)
    return {'name':filename,'sha256':recorded,'size':destination.stat().st_size}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results-root',required=True);ap.add_argument('--host-dir',required=True)
    ap.add_argument('--watermark-dir',required=True);ap.add_argument('--config',required=True)
    ap.add_argument('--out-dir',required=True);ap.add_argument('--include-nonresult-figures',action='store_true')
    args=ap.parse_args()
    root=Path(args.results_root);hostdir=Path(args.host_dir);wmdir=Path(args.watermark_dir)
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    done=[];missing={}
    stages={'fig_extracted_comparison.pdf':lambda:reference_style_pdf(out,root,'fig_extracted_comparison.pdf'),
            'fig_Tiffani.pdf':lambda:reference_style_pdf(out,root,'fig_Tiffani.pdf'),
            'proposal_hyperparameter_multi_panel.pdf':lambda:hyperparams(out,root),
            'baboon_extracted_methods.pdf':lambda:result_gallery(out,root),
            'reviewer_roc.pdf':lambda:roc(out,root),
            'reviewer_capacity_latency.pdf':lambda:capacity(out,root),
            'reviewer_orthogonality.pdf':lambda:orthogonality(out,root),
            'reviewer_realtime_720p.pdf':lambda:stream_deadlines(out,root)}
    if args.include_nonresult_figures:
        stages={'fig01.pdf':lambda:workflow(out),'fig02.pdf':lambda:carrier(out),
                'host_and_watermark_images.pdf':lambda:montage(out,hostdir,wmdir),**stages}
    for name,fn in stages.items():
        try:
            result=fn()
            if isinstance(result,list):
                done.extend(result)
            elif isinstance(result,tuple):
                for obj in result[0]:done.append(obj)
                if result[1]:missing['baseline_gallery_errors']=result[1]
            else:done.append(result)
            print('GENERATED',name)
        except Exception as exc:
            missing[name]=f'{type(exc).__name__}: {exc}'
            print('UNAVAILABLE',name,missing[name],file=sys.stderr)
    manifest={'generator':'Python/matplotlib','format':'true vector PDF',
        'generated':done,'unavailable':missing,'results_root':str(root),
        'config':str(args.config),'no_placeholder_figures':True,
        'notes':['Analysis plots use only executed CSV observations.',
                 'Result comparison/Tiffany PDFs replay archived matched-PSNR observations and verify NC/BER.']}
    (out/'figures_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'generated':len(done),'unavailable':list(missing)},indent=2))
    return 0  # missing figures remain recorded, not a scientific acceptance gate

if __name__=='__main__':raise SystemExit(main())
