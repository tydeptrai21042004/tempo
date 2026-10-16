#!/usr/bin/env python3
"""Run genuine multi-video, both-watermark video evidence without synthetic fallback."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from realtime_watermark.registry import COMMON_VIDEO_METHODS

ROOT=Path(__file__).resolve().parents[1]


def read_csv(path):
    if not path.is_file():
        raise RuntimeError(f'Missing CSV from required video experiment: {path}')
    with path.open(newline='',encoding='utf8') as f:
        return list(csv.DictReader(f))


def write_csv(path,rows):
    if not rows:
        raise ValueError(f'Refusing to create empty evidence: {path}')
    keys=list(dict.fromkeys(k for r in rows for k in r))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def execute(command):
    print('+',' '.join(str(x) for x in command),flush=True)
    subprocess.run(command,check=True,cwd=ROOT)


def run(corpus_path:Path,watermark_dir:Path,config:Path,video_out:Path,stream_out:Path,
        *,mode:str,max_frames:int,stream_frames:int,codec:str,python_bin:str,
        resolutions:str='512x512,1280x720',target_fps:int=30,
        stream_channels:str='clean,jpeg70,jpeg40',resume:bool=False):
    corpus=json.loads(corpus_path.read_text())
    videos=corpus.get('videos',[])
    if corpus.get('status')!='complete' or not videos:
        raise ValueError('Video corpus has not been successfully verified')
    if any(v.get('status')!='verified' for v in videos):
        raise ValueError('Unverified source in video corpus')
    if mode=='full' and not any(v.get('native_720p') for v in videos):
        raise ValueError('Full 720p video evidence requires genuinely native 720p source')
    watermarks=sorted(watermark_dir.glob('*.png'))
    if len(watermarks)!=2:
        raise ValueError(f'Exactly two watermarks required, found {len(watermarks)}')
    video_out.mkdir(parents=True,exist_ok=True)
    stream_out.mkdir(parents=True,exist_ok=True)
    all_video,all_stream,all_frames,coverage=[],[],[],[]
    for clip in videos:
        prepared=Path(clip['prepared_path'])
        if not prepared.is_file():
            raise FileNotFoundError(prepared)
        clip_out=video_out/clip['id']
        methods='all' if mode=='full' else 'proposal'
        summary_path=clip_out/'all_watermarks_video_summary.csv'
        expected_methods=set(COMMON_VIDEO_METHODS) if mode=='full' else {'proposal'}
        expected_pairs={(watermark.name,method) for watermark in watermarks for method in expected_methods}
        use_cached_video=False
        if resume and summary_path.is_file():
            try:
                cached=read_csv(summary_path)
                protocol=json.loads((clip_out/'all_watermarks_video_protocol.json').read_text())
                from realtime_watermark.proposals.jilp_qim.config import JILPQIMConfig
                protocol_ok=(protocol.get('proposal_configuration_sha256')==JILPQIMConfig.load(config).sha256()
                    and protocol.get('input_video')==str(prepared)
                    and protocol.get('frames')==max_frames
                    and protocol.get('codec')==codec
                    and set(protocol.get('methods',[]))==expected_methods)
                actual_pairs={(x['watermark_name'],x['method_id']) for x in cached}
                use_cached_video=(protocol_ok and len(cached)==len(expected_pairs) and actual_pairs==expected_pairs and all(
                    2<=int(x['frame_count'])<=max_frames and
                    Path(x['per_frame_csv']).is_file() and Path(x['recovered_watermark']).is_file() and
                    len(read_csv(Path(x['per_frame_csv'])))==int(x['frame_count']) for x in cached))
            except (ValueError,KeyError,OSError,RuntimeError):
                use_cached_video=False
        if use_cached_video:
            print(f'RESUME verified video rows: {clip["id"]} ({len(expected_pairs)} pairs)',flush=True)
        else:
            execute([python_bin,str(ROOT/'scripts/run_multiwatermark_video.py'),
                '--watermark-dir',str(watermark_dir),'--proposal-config',str(config),
                '--out-dir',str(clip_out),'--input',str(prepared),
                '--max-frames',str(max_frames),'--warmup-frames',str(min(4,max_frames//4)),
                '--methods',methods,'--codec',codec])
        video_rows=read_csv(clip_out/'all_watermarks_video_summary.csv')
        pairs={(r['watermark_name'],r['method_id']) for r in video_rows}
        if pairs!=expected_pairs or len(video_rows)!=len(expected_pairs):
            raise RuntimeError(f'Incomplete or duplicate watermark-method pairs for video {clip["id"]}: {pairs}')
        for row in video_rows:
            if int(row['frame_count'])<2 or int(row['frame_count'])>max_frames:
                raise RuntimeError(f'Invalid frame count for {clip["id"]}: {row["frame_count"]}')
            row.update({'video_id':clip['id'],'real_video':str(clip['is_real_source_video']),
                        'source_sha256':clip['original_sha256'], 'derived_512':True})
            all_video.append(row)
            frame_path=Path(row['per_frame_csv'])
            per_frames=read_csv(frame_path)
            if len(per_frames)!=int(row['frame_count']):
                raise RuntimeError(f'Missing per-frame video metrics: {frame_path}')
            for r in per_frames:
                r.update({'video_id':clip['id'],'watermark_name':row['watermark_name'],
                    'method_id':row['method_id'],'source_sha256':clip['original_sha256']})
            all_frames.extend(per_frames)
        for watermark in watermarks:
            safe_wm=''.join(c if c.isalnum() else '_' for c in watermark.stem)
            job=stream_out/f'{clip["id"]}_{safe_wm}'
            # Always evaluate native/decode source for streaming. When resized, annotate it.
            src_path=Path(clip['original_path'])
            desired=['512x512']
            if mode=='full' and clip['native_720p'] and '1280x720' in resolutions:
                desired.append('1280x720')
            required_channels={x.strip() for x in stream_channels.split(',')}
            expected_stream_conditions={(res,channel) for res in desired for channel in required_channels}
            stream_summary=job/'ALL_STREAM_realtime_summary.csv'
            trace_path=job/'ALL_STREAM_realtime_trace.csv'
            stream_manifest=job/'ALL_STREAM_run_manifest.json'
            use_cached_stream=False
            if resume and all(p.is_file() for p in (stream_summary,trace_path,stream_manifest)):
                try:
                    cached_rows=read_csv(stream_summary)
                    manifest=json.loads(stream_manifest.read_text())
                    use_cached_stream=(
                        len(cached_rows)==len(expected_stream_conditions) and
                        {(x['resolution'],x['channel']) for x in cached_rows}==expected_stream_conditions and
                        all(x.get('payload_bits')=='4096' and x.get('pace_mode')=='realtime' and
                            x.get('effective_output_fps','')!='' for x in cached_rows) and
                        manifest.get('payload_bits')==4096 and manifest.get('pace_mode')=='realtime' and
                        manifest.get('source_video')==str(src_path) and
                        manifest.get('frames_requested')==stream_frames and
                        set(manifest.get('channels',[]))==required_channels and
                        set(manifest.get('methods',[]))=={'proposal'} and
                        bool(read_csv(trace_path)))
                except (ValueError,KeyError,OSError,RuntimeError):
                    use_cached_stream=False
            if use_cached_stream:
                print(f'RESUME verified stream: {job.name}',flush=True)
            else:
                execute([python_bin,str(ROOT/'scripts/live_stream_benchmark.py'),
                    '--watermark',str(watermark),'--proposal-config',str(config),
                    '--methods','proposal','--workdir',str(job),
                    '--input',str(src_path),'--frames',str(stream_frames),
                    '--wm-size','64','--resolutions',','.join(desired),
                    '--target-fps',str(target_fps),'--channels',stream_channels,
                    '--queue-size','2','--pace-mode','realtime'])
            stream_rows=read_csv(job/'ALL_STREAM_realtime_summary.csv')
            trace_rows=read_csv(job/'ALL_STREAM_realtime_trace.csv')
            required_channels={x.strip() for x in stream_channels.split(',')}
            measured={(r['resolution'],r['channel']) for r in stream_rows}
            for res in desired:
                if not required_channels.issubset({channel for size,channel in measured if size==res}):
                    raise RuntimeError(f'Missing stream conditions for {clip["id"]} / {res}')
            for r in stream_rows:
                if r.get('payload_bits')!='4096':
                    raise RuntimeError(f'Wrong stream payload: {r}')
                r.update({'video_id':clip['id'],'watermark_name':watermark.name,
                          'source_sha256':clip['original_sha256'],
                          'native_input_resolution':f'{clip["original_video"]["width"]}x{clip["original_video"]["height"]}',
                          'resized_for_benchmark':str(r['resolution']!=f'{clip["original_video"]["width"]}x{clip["original_video"]["height"]}'),
                          'real_video':str(clip['is_real_source_video'])})
                all_stream.append(r)
            coverage.append({'video_id':clip['id'],'watermark_name':watermark.name,
                'stream_resolution':','.join(desired),'tested_channels':stream_channels,
                'framewise_method_count':len({r['method_id'] for r in video_rows}),
                'source_is_real_video':clip['is_real_source_video'],
                'native_720p':clip['native_720p'],'stream_trace_rows':len(trace_rows)})
    write_csv(video_out/'all_watermarks_video_summary.csv',all_video)
    write_csv(video_out/'all_clips_per_frame.csv',all_frames)
    write_csv(video_out/'video_corpus_stream_summary.csv',all_stream)
    write_csv(video_out/'video_corpus_coverage.csv',coverage)
    status={'status':'complete','source_count':len(videos),'watermark_count':2,
        'framewise_rows':len(all_frames),'video_summary_rows':len(all_video),
        'stream_summary_rows':len(all_stream),'clip_ids':[c['id'] for c in videos],
        'source_provenance':str(corpus_path.resolve()),'native_720p_sources':[v['id'] for v in videos if v['native_720p']],
        'mode':mode,'synthetic_fallback':False,
        'report_limitations': 'Framewise track uses a shared video codec and common adapter; live stream uses causal wall-clock simulation with source frames loaded before pacing; resampling is explicitly reported. Full real-camera acquisition timing is not measured.'}
    (video_out/'video_corpus_evidence_manifest.json').write_text(json.dumps(status,indent=2),encoding='utf8')
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',required=True)
    p.add_argument('--watermark-dir',required=True)
    p.add_argument('--proposal-config',required=True)
    p.add_argument('--video-out',required=True)
    p.add_argument('--stream-out',required=True)
    p.add_argument('--mode',choices=['quick','full'],default='full')
    p.add_argument('--max-frames',type=int,default=120)
    p.add_argument('--stream-frames',type=int,default=120)
    p.add_argument('--codec',default='mp4v')
    p.add_argument('--python',default=sys.executable)
    p.add_argument('--resolutions',default='512x512,1280x720')
    p.add_argument('--channels',default='clean,jpeg70,jpeg40')
    p.add_argument('--resume',action='store_true',help='Skip individually verified completed video and stream jobs')
    args=p.parse_args()
    result=run(corpus_path=Path(args.corpus),watermark_dir=Path(args.watermark_dir),
        config=Path(args.proposal_config),video_out=Path(args.video_out),
        stream_out=Path(args.stream_out),mode=args.mode,max_frames=args.max_frames,
        stream_frames=args.stream_frames,codec=args.codec,python_bin=args.python,
        resolutions=args.resolutions,stream_channels=args.channels,resume=args.resume)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
