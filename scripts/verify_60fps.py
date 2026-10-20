"""60Hz causal benchmark: records both throughput and hard per-frame deadlines."""
import sys,os,time,io,json,gc,argparse
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/hartley_frequency_preserving48_candidate.json');p.add_argument('--frames',type=int,default=120);p.add_argument('--rate',type=int,default=60);p.add_argument('--watermarks',default='wm.png,Cc.logo.circle.png');p.add_argument('--repeats',type=int,default=2);p.add_argument('--codec',choices=['pillow','cv2'],default='pillow');p.add_argument('--output',default='/mnt/data/work60/new_60fps_results.json');a=p.parse_args()
ROOT=Path(__file__).resolve().parents[1]
from realtime_watermark.stream_pipeline import _synthetic_frames,_run_realtime,summarize_stream_run, StreamPacket
import realtime_watermark.stream_pipeline as stream
from realtime_watermark.core import load_binary_watermark,psnr,attack_jpeg
from realtime_watermark.fair_benchmark import calibrate_step_to_psnr
from realtime_watermark.registry import embed
from realtime_watermark.proposals.jilp_qim.config import JILPQIMConfig
from realtime_watermark.proposals.jilp_qim.method import extract_jilp_qim
import cv2
cv2.setNumThreads(1)
os.environ["TEMPO_STREAM_JPEG_BACKEND"] = "opencv" if a.codec=="cv2" else "pillow"
frames=_synthetic_frames(1280,720,a.frames)
params=JILPQIMConfig.load(ROOT/a.config).to_dict()
rows=[]
for wf in a.watermarks.split(','):
 wm=load_binary_watermark(ROOT/'data/user_dataset/watermark'/wf)
 step, pcal=calibrate_step_to_psnr('proposal',frames[0],wm,target_psnr=45.,seed=2026,iterations=5,method_params=params)
 marked,key=embed('proposal',frames[0],wm,step=step,seed=2026,method_params=params)
 _=extract_jilp_qim(marked,key,sync=True);_=stream.apply_stream_channel(marked,'jpeg70',frame_id=0,seed=2026)
 print('START',wf,'calib',pcal,'step',step,'threads',os.environ.get('JILP_NUM_THREADS'),'codec',a.codec,flush=True)
 for trial in range(a.repeats):
  traces,drops,duration=_run_realtime(frames,method_id='proposal',watermark=wm,channel='jpeg70',target_fps=a.rate,queue_size=2,seed=2026,step=step,method_params=params)
  # independent off-path result diagnostics
  for r in traces:
   if not r.get('queue_dropped'):
    y,_=embed('proposal',frames[r['frame_id']],wm,step=step,seed=2026,method_params=params)
    r['psnr']=psnr(frames[r['frame_id']],y)
  s=summarize_stream_run(traces,method_id='proposal',channel='jpeg70',watermark=wm,target_fps=a.rate,resolution=(1280,720),pace_mode='realtime',queue_size=2,queue_drops=drops,total_duration_s=duration,calibrated_step=step,calibration_psnr=pcal)
  row={'config':a.config,'watermark':wf,'trial':trial,'threads':os.environ.get('JILP_NUM_THREADS'),'codec':a.codec,'duration':duration,**s}
  rows.append(row)
  print('RESULT',wf,'trial',trial,'fps',round(s['effective_output_fps'],3),'processed',s['frames_processed'],'/',a.frames,'ontime',s['frames_completed_on_time'],'/',a.frames,'drops',drops,'p95 lat',round(s['latency_ms_p95'],3),'p95 service',round(s['service_ms_p95'],3),'psnr',round(s['mean_embedding_psnr'],3),'nc',round(s['single_frame_nc_mean'],4),flush=True)
  del traces;gc.collect()
 a_path=Path(a.output);a_path.parent.mkdir(exist_ok=True,parents=True);a_path.write_text(json.dumps(rows,indent=2,allow_nan=True))
