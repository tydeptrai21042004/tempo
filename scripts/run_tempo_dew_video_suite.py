#!/usr/bin/env python3
"""Final paper comparison: TEMPO versus Langelaar--Lagendijk DEW on real video.

Two separately labeled payload tracks: 256-bit matched TEMPO/DEW and
4096-bit TEMPO-native. Never calibrate from evaluation frames; never use
synthetic fallback or claim compressed-bitstream DEW reproduction.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from realtime_watermark.registry import REALTIME_VIDEO_BASELINES, PROPOSAL_METHOD
from realtime_watermark.core import load_binary_watermark, save_json
from realtime_watermark.proposals.jilp_qim.config import JILPQIMConfig
from realtime_watermark.stream_pipeline import load_stream_frames, run_stream_benchmark


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError('No measured rows; refusing to create empty results')
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def main(a: argparse.Namespace) -> None:
    corpus_path = Path(a.corpus)
    corpus = json.loads(corpus_path.read_text(encoding='utf-8'))
    videos = corpus.get('videos', [])
    if corpus.get('status') != 'complete' or not videos:
        raise RuntimeError('Complete, verified real-video corpus required')
    if a.protocol == 'full' and (len(videos) < 4 or not any(v.get('native_720p') for v in videos)):
        raise RuntimeError('Full reviewer video protocol requires at least four clips and one native-HD source')
    if len(REALTIME_VIDEO_BASELINES) != 1 or REALTIME_VIDEO_BASELINES[0] != 'dew_langelaar2001':
        raise RuntimeError('Final video baseline registry must contain DEW only')
    marks = sorted(Path(a.watermark_dir).glob('*.png'))
    if len(marks) != 2:
        raise RuntimeError('Exactly two original binary watermarks are required')
    conf = JILPQIMConfig.load(a.config)
    root = Path(a.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    traces = []
    coverage = []
    channels = tuple(x.strip() for x in a.channels.split(',') if x.strip())
    if not channels:
        raise ValueError('No stream channels requested')
    for clip in videos:
        if clip.get('status') != 'verified':
            raise RuntimeError(f'Unverified video corpus entry: {clip.get("id")}')
        if a.protocol == 'full' and not clip.get('is_real_source_video', False):
            raise RuntimeError(f'Nonpublic fixture not eligible for real-video paper: {clip.get("id")}')
        path = Path(clip['original_path'])
        if not path.is_file() or sha(path) != clip.get('original_sha256'):
            raise RuntimeError(f'Missing or changed source video: {path}')
        resolutions = [(512, 512)]
        if clip.get('native_720p') and not a.no_native_hd:
            resolutions.append((1280, 720))
        for width, height in resolutions:
            frames = load_stream_frames(path, width=width, height=height, max_frames=a.frames)
            if len(frames) < 2:
                raise RuntimeError(f'Not enough decoded frames in {path}')
            for mark in marks:
                for payload_side, methods, track in (
                    (16, ('proposal', 'dew_langelaar2001'), 'matched_payload_256'),
                    (64, ('proposal',), 'native_tempo_4096'),
                ):
                    if a.only_matched and track != 'matched_payload_256':
                        continue
                    watermark = load_binary_watermark(mark, size=payload_side)
                    out = root / 'runs' / str(clip['id']) / mark.stem / f'{width}x{height}' / track
                    # Frozen means NO call to calibrate_step_to_psnr on test video.
                    # DEW zeroes the selected DCT coefficients (not the margin adapter).
                    result = run_stream_benchmark(
                        frames=frames, watermark=watermark, out_dir=out,
                        methods=methods, channels=channels,
                        target_fps=float(a.fps), queue_size=int(a.queue_size),
                        seed=int(a.seed), pace_mode='realtime',
                        proposal_params=conf.to_dict(), calibration_mode='frozen',
                        paper_dew=True,
                    )
                    expected = {(m, c) for m in methods for c in channels}
                    observed = {(r['method_id'], r['channel']) for r in result['summary']}
                    if expected != observed or len(result['summary']) != len(expected):
                        raise RuntimeError(f'Incomplete video method/channel coverage: {clip["id"]} {track}')
                    for row in result['summary']:
                        rows.append({**row, 'video_id':clip['id'], 'watermark_name':mark.name,
                            'source_sha256':sha(path), 'watermark_source_sha256':sha(mark),
                            'protocol_track':track, 'baseline_implementation':
                            'decoded_frame_paper_zeroing_DEW' if row['method_id']=='dew_langelaar2001' else 'TEMPO_frozen',
                            'source_real_verified':bool(clip.get('is_real_source_video'))})
                    for trace in result['trace']:
                        traces.append({**trace, 'video_id':clip['id'], 'watermark_name':mark.name,
                            'protocol_track':track})
                    coverage.append({'video_id':clip['id'], 'watermark':mark.name,
                        'resolution':f'{width}x{height}', 'track':track,
                        'methods':','.join(methods),'channels':','.join(channels),
                        'frames_decoded':len(frames), 'summary_rows':len(result['summary'])})
    if not any(r['method_id']=='dew_langelaar2001' for r in rows):
        raise RuntimeError('Missing measured DEW baseline')
    write_csv(root/'video_comparison_summary.csv', rows)
    write_csv(root/'video_comparison_trace.csv', traces)
    write_csv(root/'video_comparison_coverage.csv', coverage)
    save_json(root/'video_comparison_manifest.json', {
        'status':'complete', 'track':'real_video_realtime', 'baseline_ids':list(REALTIME_VIDEO_BASELINES),
        'matched_payload_bits':256, 'tempo_native_payload_bits':4096,
        'baseline_fidelity':'DEW DCT energy differential paper-zeroing on decoded RGB frames; NOT compressed bitstream DEW',
        'calibration_mode':'frozen', 'test_video_calibration':False, 'pace_mode':'realtime',
        'detector_classes':{'proposal':'blind','dew_langelaar2001':'blind'},
        'note':'No matched PSNR claim: paper-style DEW has its own embedding distortion.',
        'channels':channels,'target_fps':a.fps,'frames_requested':a.frames,
        'corpus_sha256':sha(corpus_path), 'config_sha256':conf.sha256(),
        'source_count':len(videos), 'watermark_count':len(marks),
        'summary_rows':len(rows),'trace_rows':len(traces),
        'synthetic_fallback':False, 'neural_baselines':False,
    })
    print('Final TEMPO-DEW real-video evidence:',root, 'summary rows',len(rows), flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--watermark-dir',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/manuscript_frozen.json')
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--frames',type=int,default=300)
    p.add_argument('--fps',type=float,default=30)
    p.add_argument('--queue-size',type=int,default=2)
    p.add_argument('--channels',default='clean,jpeg90,jpeg70,jpeg40')
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--protocol',choices=('full','development_check'),default='full')
    p.add_argument('--no-native-hd',action='store_true')
    p.add_argument('--only-matched',action='store_true')
    main(p.parse_args())
