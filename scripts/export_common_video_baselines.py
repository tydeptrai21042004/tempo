#!/usr/bin/env python3
"""Verify and export existing *same-source* video comparisons, without rerunning methods.

The export preserves blind/semi-blind/non-blind decoder assumptions and the
measured (unmatched) embedding PSNRs. It is derived evidence, NOT a new trial.
External Kodak experiments are unrelated and remain TEMPO-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

# Required classes from the actual native disclosures; never default to blind.
from realtime_watermark.paper_native import NATIVE_DISCLOSURES
from realtime_watermark.registry import COMMON_VIDEO_METHODS


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f'Original measured CSV unavailable: {path}')
    with path.open(newline='', encoding='utf-8') as stream:
        result = list(csv.DictReader(stream))
    if not result:
        raise ValueError(f'Empty measured CSV: {path}')
    return result


def write_csv(path: Path, entries: list[dict]) -> None:
    if not entries:
        raise ValueError(f'Cannot export empty evidence: {path}')
    keys = list(dict.fromkeys(key for item in entries for key in item))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(entries)


def _verified_per_frame_file(video_dir: Path, item: dict) -> Path:
    """Use a relative measured artifact, not a stale absolute Kaggle path."""
    original = Path(item['per_frame_csv'])
    clip = item['video_id']
    path_parts = original.parts
    if 'video' not in path_parts:
        raise ValueError(f'Not an archived video per-frame path: {original}')
    # Prefix is intentionally ignored because Kaggle run IDs change.
    index = len(path_parts) - 1 - path_parts[::-1].index('video')
    tail = path_parts[index + 1:]
    if len(tail) < 4 or tail[0] != clip or tail[-1] != 'per_frame.csv' or tail[-2] != item['method_id']:
        raise ValueError(f'Per-frame path does not match measured method/video: {original}')
    if any(part in {'..', '.', ''} for part in tail):
        raise ValueError('Unsafe per-frame path')
    return video_dir.joinpath(*tail)


def detector_class(method_id: str) -> str:
    if method_id == 'proposal':
        return 'blind'
    disclosure = NATIVE_DISCLOSURES.get(method_id)
    if disclosure is None:
        raise ValueError(f'No verified detector-class disclosure for {method_id}')
    return disclosure.detector_type


def validate(video_dir: Path, corpus_path: Path, *, full: bool = True) -> dict:
    video_dir = Path(video_dir)
    corpus_path = Path(corpus_path)
    corpus = json.loads(corpus_path.read_text(encoding='utf-8'))
    if corpus.get('status') != 'complete' or not corpus.get('videos'):
        raise ValueError('Only an already-verified, real video corpus can be exported')
    clips = {c['id']: c for c in corpus['videos']}
    if len(clips) != len(corpus['videos']) or any(c.get('status') != 'verified' or not c.get('is_real_source_video') for c in clips.values()):
        raise ValueError('Unverified or duplicate real-video source in corpus')
    if full and (len(clips) < 3 or not any(c.get('native_720p') for c in clips.values())):
        raise ValueError('Full protocol requires at least 3 videos and a native 720p source')

    measured = read_csv(video_dir / 'all_watermarks_video_summary.csv')
    original_frames = read_csv(video_dir / 'all_clips_per_frame.csv')
    methods = set(COMMON_VIDEO_METHODS) if full else {r['method_id'] for r in measured}
    if not methods or not methods.issubset(set(COMMON_VIDEO_METHODS)):
        raise ValueError('The common-source comparison contains an unapproved method')
    expected_classes = {m: detector_class(m) for m in methods}
    watermark_names = {r['watermark_name'] for r in measured}
    if full and len(watermark_names) != 2:
        raise ValueError('Full comparison requires exactly two watermarks')
    if not watermark_names:
        raise ValueError('No watermark observed')
    expected = {(v, w, m) for v in clips for w in watermark_names for m in methods}
    observed = [(r['video_id'], r['watermark_name'], r['method_id']) for r in measured]
    if len(set(observed)) != len(observed) or set(observed) != expected:
        raise ValueError('Missing, duplicate, or unexpected video/watermark/method observations')
    count_frames = Counter((r['video_id'], r['watermark_name'], r['method_id']) for r in original_frames)
    group_frames = defaultdict(set)
    group_psnr = defaultdict(list)
    clip_checks = []
    per_frame_sha = {}
    protocol_sha = {}
    for clip_id, source in sorted(clips.items()):
        protocol_file = video_dir / clip_id / 'all_watermarks_video_protocol.json'
        if not protocol_file.is_file():
            raise FileNotFoundError(f'Original codec protocol missing: {protocol_file}')
        proto = json.loads(protocol_file.read_text(encoding='utf-8'))
        if (proto.get('input_sha256') != source.get('prepared_sha256')
                or proto.get('forced_psnr_calibration') is not False
                or proto.get('common_ecc') is not False
                or set(proto.get('methods', [])) != methods
                or not proto.get('codec')
                or proto.get('watermark_sha256', {}).keys() != watermark_names):
            raise ValueError(f'Protocol mismatch for video {clip_id}')
        if not all(len(v) == 64 for v in proto['watermark_sha256'].values()):
            raise ValueError(f'Watermark SHA256 values absent for {clip_id}')
        clip_rows = [r for r in measured if r['video_id'] == clip_id]
        protocol_sha[clip_id] = sha256(protocol_file)
        for r in clip_rows:
            key = (clip_id, r['watermark_name'], r['method_id'])
            method = r['method_id']
            if (r['detector_type'] != expected_classes[method]
                    or r['source_sha256'] != source['original_sha256']
                    or r.get('derived_512') != 'True'
                    or r.get('real_video') != 'True'
                    or r.get('forced_psnr_calibration') != 'False'
                    or r.get('video_protocol') != 'common_framewise_adapter_native_default_strength'):
                raise ValueError(f'Incorrect detector, source, or protocol assumptions: {key}')
            if r.get('configuration_sha256', '') == '':
                raise ValueError(f'Missing configuration SHA for {key}')
            if r.get('watermark_path') and Path(r['watermark_path']).name != r['watermark_name']:
                raise ValueError(f'Watermark file mismatch in row: {key}')
            if not (math.isfinite(float(r['precodec_mean_psnr'])) and float(r['precodec_mean_psnr']) > 0
                    and math.isfinite(float(r['nc'])) and math.isfinite(float(r['ber']))):
                raise ValueError(f'Non-finite measured result: {key}')
            n = int(r['frame_count'])
            if n < 2 or n > int(proto['frames']) or count_frames[key] != n:
                raise ValueError(f'Frame count differs from archived per-frame CSV: {key}')
            frame_file = _verified_per_frame_file(video_dir, r)
            file_rows = read_csv(frame_file)
            if len(file_rows) != n:
                raise ValueError(f'Per-case frame count mismatch: {frame_file}')
            per_frame_sha['/'.join(frame_file.relative_to(video_dir).parts)] = sha256(frame_file)
            group_frames[(clip_id, r['watermark_name'])].add(n)
            group_psnr[(clip_id, r['watermark_name'])].append(float(r['precodec_mean_psnr']))
        clip_checks.append({'video_id':clip_id,'source_sha256':source['original_sha256'],
            'prepared_sha256':source['prepared_sha256'], 'codec':proto['codec'],
            'requested_frames':proto['frames'], 'methods':sorted(methods),
            'watermark_sha256':proto['watermark_sha256'], 'protocol_sha256':protocol_sha[clip_id]})
    if any(len(vals) != 1 for vals in group_frames.values()):
        raise ValueError('Compared methods used different decoded frame counts')
    if set(count_frames) != expected:
        raise ValueError('Aggregated per-frame CSV has extra or missing experiment keys')
    # All methods evaluated on the same codec settings; PSNRs remain native.
    if len({c['codec'] for c in clip_checks}) != 1:
        raise ValueError('Different codec settings in purported common-codec comparison')
    watermark_hash_by_name = defaultdict(set)
    for check in clip_checks:
        for w, digest in check['watermark_sha256'].items():
            watermark_hash_by_name[w].add(digest)
    if any(len(x) != 1 for x in watermark_hash_by_name.values()):
        raise ValueError('Watermark payload bytes differ between clips')
    return {'rows':measured, 'per_frame_rows':original_frames,
        'methods':sorted(methods), 'detector_classes':expected_classes,
        'watermarks':sorted(watermark_names),'clips':clip_checks,
        'per_frame_sha256':per_frame_sha, 'video_summary_sha256':sha256(video_dir/'all_watermarks_video_summary.csv'),
        'aggregate_frames_sha256':sha256(video_dir/'all_clips_per_frame.csv'),
        'source_corpus_sha256':sha256(corpus_path), 'psnr_range':{
            '/'.join(k): [round(min(v),4),round(max(v),4)] for k,v in group_psnr.items()},
        'distinct_psnr_values':len({round(float(r['precodec_mean_psnr']), 4) for r in measured}),
    }


def export(video_dir: Path, corpus_path: Path, out_dir: Path, *, full: bool = True) -> dict:
    info = validate(video_dir,corpus_path,full=full)  # no outputs until all measurements verify
    out_dir=Path(out_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    main=out_dir/'all_watermarks_video_summary.csv'
    frames=out_dir/'all_clips_per_frame.csv'
    classes=out_dir/'detector_classification.csv'
    write_csv(main, info['rows'])
    write_csv(frames, info['per_frame_rows'])
    write_csv(classes,[{'method_id':m,'detector_type':info['detector_classes'][m],
                        'detector_assumptions_preserved':True,
                        'forced_psnr_calibration':False} for m in info['methods']])
    manifest={'status':'complete','kind':'verified_export_of_prior_measurements_not_new_experiment',
        'comparison':'same_original_video_source_and_codec_unmatched_psnr',
        'full_protocol':full,'video_count':len(info['clips']),
        'watermark_count':len(info['watermarks']),'method_count':len(info['methods']),
        'row_count':len(info['rows']),'frame_row_count':len(info['per_frame_rows']),
        'method_detector_classes':info['detector_classes'],
        'native_method_strengths_preserved':True,'psnr_matched':False,
        'all_methods_blind':all(x=='blind' for x in info['detector_classes'].values()),
        'input_hashes':{'video_summary':info['video_summary_sha256'],
            'aggregate_per_frame':info['aggregate_frames_sha256'],
            'video_corpus':info['source_corpus_sha256']},
        'export_sha256':{'video_summary':sha256(main),'aggregate_per_frame':sha256(frames),
                          'detector_classification':sha256(classes)},
        'protocol_hashes':{x['video_id']:x['protocol_sha256'] for x in info['clips']},
        'per_frame_sha256':info['per_frame_sha256'],
        'warnings':['Results were recorded at method-specific embedding PSNR; not a distortion-matched video ranking.',
            'Semi-blind/non-blind methods retain their original detector assumptions.',
            'Framewise videos were normalized to 512x512; the separate native-720p causal stream test is TEMPO-only.']}
    (out_dir/'source_video_verification.json').write_text(json.dumps({
        'clips':info['clips'],'watermarks':info['watermarks'],
        'psnr_ranges_db':info['psnr_range']},indent=2)+'\n',encoding='utf8')
    (out_dir/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    return manifest


def verify_export(video_dir: Path, corpus_path: Path, out_dir: Path, *, full: bool = True) -> bool:
    """Reviewer check: verify rows/classes AND hashes against original measured data."""
    out_dir=Path(out_dir)
    marker=out_dir/'experiment_manifest.json'
    if not marker.is_file():
        return False
    try:
        m=json.loads(marker.read_text(encoding='utf8'))
        if m.get('status')!='complete' or m.get('psnr_matched') is not False:
            return False
        info=validate(video_dir,corpus_path,full=full)
        actual=read_csv(out_dir/'all_watermarks_video_summary.csv')
        classes=read_csv(out_dir/'detector_classification.csv')
        if actual != info['rows'] or len(read_csv(out_dir/'all_clips_per_frame.csv'))!=len(info['per_frame_rows']):
            return False
        if {r['method_id']:r['detector_type'] for r in classes}!=info['detector_classes']:
            return False
        if m.get('method_detector_classes') != info['detector_classes']:
            return False
        hashes=m['export_sha256']
        for key,filename in [('video_summary','all_watermarks_video_summary.csv'),
            ('aggregate_per_frame','all_clips_per_frame.csv'),
            ('detector_classification','detector_classification.csv')]:
            if sha256(out_dir/filename)!=hashes[key]:
                return False
        return (m.get('input_hashes',{}).get('video_summary')==info['video_summary_sha256']
                and m.get('input_hashes',{}).get('aggregate_per_frame')==info['aggregate_frames_sha256']
                and m.get('row_count')==len(info['rows'])
                and m.get('frame_row_count')==len(info['per_frame_rows']))
    except (ValueError,KeyError,FileNotFoundError,OSError,TypeError,json.JSONDecodeError):
        return False


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video-dir',type=Path,required=True)
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--mode',choices=['full','quick'],default='full')
    a=p.parse_args()
    result=export(a.video_dir,a.corpus,a.out_dir,full=a.mode=='full')
    print(json.dumps({'status':result['status'],'rows':result['row_count'],
        'methods':result['method_detector_classes'],'psnr_matched':result['psnr_matched']},indent=2))

if __name__=='__main__':
    main()
