"""Reproducible 720p JPEG70 CPU microbenchmark. Not a live-stream FPS claim."""
import argparse,csv,json,os,time
from pathlib import Path
import cv2,numpy as np
from realtime_watermark.core import load_rgb,load_binary_watermark,nc_binary,psnr
from realtime_watermark.registry import embed
from realtime_watermark.proposals.jilp_qim.method import extract_jilp_qim
from realtime_watermark.methods import warmup_numba
from scripts.method_revision_study import params_for,attack_rgb

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repeats',type=int,default=12);ap.add_argument('--output',default='results/method_revision/timing_720.csv'); ap.add_argument('--hosts',nargs='+',default=['lenna.bmp','pepper.bmp']); ap.add_argument('--watermarks',nargs='+',default=['wm.png','Cc.logo.circle.png'])
    args=ap.parse_args();dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1);warmup_numba()
    root=Path(__file__).resolve().parents[1]/'data/user_dataset'
    hosts=args.hosts;wms=args.watermarks
    rows=[]
    for hn in hosts:
        host=cv2.resize(load_rgb(root/'host'/hn,mode='full'),(1280,720),interpolation=cv2.INTER_LINEAR)
        for wn in wms:
            wm=load_binary_watermark(root/'watermark'/wn,size=64)
            # Warm both 720p paths so that one-time JIT compilation does not
            # count toward measured FPS. The measured CPU pipeline remains
            # embed -> JPEG70 encode/decode -> blind extract.
            for warm_variant in ('baseline', 'candidate'):
                warm_image, warm_key=embed('proposal',host,wm,seed=2026,method_params=params_for(warm_variant))
                extract_jilp_qim(attack_rgb(warm_image,'jpeg70'),warm_key,sync=True)
            # Alternating order reduces temperature/drift bias but not OS variance.
            for i in range(args.repeats):
                for variant in (("baseline","candidate") if i%2==0 else ("candidate","baseline")):
                    start=time.perf_counter_ns()
                    marked,key=embed('proposal',host,wm,seed=2026,method_params=params_for(variant))
                    t_embed=time.perf_counter_ns()
                    image=attack_rgb(marked,'jpeg70')
                    t_codec=time.perf_counter_ns()
                    recovered=extract_jilp_qim(image,key,sync=True)
                    end=time.perf_counter_ns()
                    rows.append(dict(host=hn,watermark=wn,iteration=i,variant=variant,
                         embed_ms=(t_embed-start)/1e6,codec_ms=(t_codec-t_embed)/1e6,
                         extract_ms=(end-t_codec)/1e6,total_ms=(end-start)/1e6,
                         nc=nc_binary(wm,recovered),psnr=psnr(host,marked)))
            print('finished',hn,wn,flush=True)
    with dest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for v in ('baseline','candidate'):
        rr=[r for r in rows if r['variant']==v]
        total=np.array([r['total_ms'] for r in rr]); print(v,dict(median_ms=float(np.median(total)),p95_ms=float(np.percentile(total,95)),
                      effective_fps=float(1000/np.mean(total)),nc_mean=float(np.mean([r['nc'] for r in rr])),
                      psnr_mean=float(np.mean([r['psnr'] for r in rr]))))
if __name__=='__main__':main()
