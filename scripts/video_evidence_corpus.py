#!/usr/bin/env python3
"""Download, verify and normalize genuine source videos for reproducible TEMPO tests.

No synthetic or hidden demo fallback. Offline tests can specify local_path inside the
manifest, but those records are explicitly tagged source_kind=local_test_fixture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda: f.read(1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()


def inspect_video(path: Path) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'Not a readable video file: {path}')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    claimed_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded = 0
    for _ in range(4):
        ok, frame = cap.read()
        if not ok:
            break
        if frame is None or frame.size == 0:
            break
        decoded += 1
    cap.release()
    if width < 64 or height < 64 or decoded == 0 or fps <= 0:
        raise ValueError(f'Invalid video metadata/decode ({width}x{height}, {fps} fps, {decoded} readable): {path}')
    return {'width': width, 'height': height, 'fps': fps, 'reported_frames': claimed_frames,
            'decoded_validation_frames': decoded, 'bytes': path.stat().st_size}


def _download(url: str, destination: Path, limit: int, timeout: int) -> None:
    parsed = urlparse(url)
    if parsed.scheme != 'https' or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError('Only public HTTPS video URLs without credentials are accepted')
    tmp = destination.with_suffix('.download')
    tmp.unlink(missing_ok=True)
    try:
        req = Request(url, headers={'User-Agent': 'TEMPO-Research-Video-Evidence/1.0'})
        with urlopen(req, timeout=timeout) as response, tmp.open('wb') as out:
            final = urlparse(response.geturl())
            if final.scheme != 'https':
                raise ValueError('Video URL redirected away from HTTPS')
            length = int(response.headers.get('Content-Length', 0) or 0)
            if length > limit:
                raise ValueError(f'Video download exceeds {limit} byte limit')
            count = 0
            while True:
                buf = response.read(1024 * 1024)
                if not buf:
                    break
                count += len(buf)
                if count > limit:
                    raise ValueError(f'Video download exceeded {limit} byte limit')
                out.write(buf)
        if tmp.stat().st_size < 1024:
            raise ValueError('Video download is too small to be credible')
        os.replace(tmp, destination)
    finally:
        tmp.unlink(missing_ok=True)


def _normalize(input_path: Path, output: Path, *, frames: int, size: int = 512) -> dict:
    """Spatially letterbox, don't repeat the clip; record resampling explicitly."""
    import cv2
    cap = cv2.VideoCapture(str(input_path))
    native_fps = float(cap.get(cv2.CAP_PROP_FPS))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), native_fps, (size,size))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError('Could not open prepared-video writer')
    n = 0
    try:
        while n < frames:
            ok, frame = cap.read()
            if not ok:
                break
            h,w = frame.shape[:2]
            scale = min(size/w, size/h)
            nw,nh = max(1,round(w*scale)),max(1,round(h*scale))
            resized = cv2.resize(frame,(nw,nh),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
            # Neutral padding, not replicated moving content. Provenance flags synthetic padding.
            target = __import__('numpy').full((size,size,3),128,dtype='uint8')
            x,y = (size-nw)//2,(size-nh)//2
            target[y:y+nh,x:x+nw] = resized
            writer.write(target)
            n += 1
    finally:
        writer.release()
        cap.release()
    if n == 0:
        output.unlink(missing_ok=True)
        raise ValueError(f'No decodable frames: {input_path}')
    return {'frames_written':n, 'normalization':'aspect_preserving_letterbox',
            'prepared_resolution':f'{size}x{size}', 'padded_background_rgb': [128,128,128],
            'synthetic_image_generation':False}


def prepare(*, manifest:Path, cache:Path, prepared:Path, output:Path,
            mode:str, frames:int, timeout:int=30, allow_local:bool=False) -> dict:
    document=json.loads(manifest.read_text(encoding='utf8'))
    sources=document['sources']
    selected = [x for x in sources if mode=='full' or x.get('split')=='quick']
    if not selected:
        raise ValueError('No video sources selected')
    cache.mkdir(parents=True,exist_ok=True)
    prepared.mkdir(parents=True,exist_ok=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    records=[]
    for item in selected:
        name=item['id']
        if not name or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for ch in name):
            raise ValueError(f'Unsafe video identifier: {name}')
        kind='remote_download'
        raw=cache/f'{name}.mp4'
        local=item.get('local_path')
        if local is not None:
            if not allow_local:
                raise ValueError('Local video fixtures require --allow-local-fixtures')
            kind='local_test_fixture'
            local_path=Path(local)
            if not local_path.is_absolute():
                local_path=(manifest.parent/local_path).resolve()
            if not local_path.is_file():
                raise FileNotFoundError(local_path)
            if raw.resolve() != local_path.resolve():
                shutil.copyfile(local_path,raw)
        else:
            # A filename alone cannot establish that a cached file is a real
            # download. Require the URL/checksum sidecar from a successful transfer.
            sidecar=cache/f'{name}.download_provenance.json'
            trusted=False
            if raw.is_file() and sidecar.is_file():
                try:
                    prior=json.loads(sidecar.read_text())
                    trusted=(prior.get('source_url')==item['url'] and
                             prior.get('sha256')==sha256(raw))
                except (ValueError,OSError):
                    trusted=False
            if not trusted:
                raw.unlink(missing_ok=True)
                sidecar.unlink(missing_ok=True)
                _download(item['url'], raw, int(item.get('max_download_mb',100))*1024*1024, timeout)
                sidecar.write_text(json.dumps({'source_url':item['url'],
                    'sha256':sha256(raw),'download_verified_by':'HTTPS transfer',
                    'bytes':raw.stat().st_size},indent=2))
        try:
            metadata=inspect_video(raw)
            if metadata['bytes']>int(item.get('max_download_mb',100))*1024*1024:
                raise ValueError('Cached asset larger than declared maximum')
            if metadata['height']<int(item.get('require_min_height',0)):
                raise ValueError(f'{name} requires native height {item["require_min_height"]}, got {metadata["height"]}')
            checksum=sha256(raw)
            expected=item.get('sha256')
            if expected and checksum.lower()!=str(expected).lower():
                raise ValueError(f'SHA256 mismatch for {name}')
            derived=prepared/f'{name}_512.mp4'
            norm=_normalize(raw,derived,frames=frames)
            checked=inspect_video(derived)
            if checked['width']!=512 or checked['height']!=512:
                raise ValueError('Prepared video does not have required dimensions')
            records.append({'id':name,'status':'verified','source_kind':kind,'source_url':item.get('url'),
                'attribution':item.get('attribution'), 'license_note':item.get('license_note'),
                'original_path':str(raw.resolve()),'original_sha256':checksum,
                'original_video':metadata,'prepared_path':str(derived.resolve()),
                'prepared_sha256':sha256(derived), 'preparation':norm,
                'native_720p':bool(metadata['width']>=1280 and metadata['height']>=720),
                'is_real_source_video':kind=='remote_download'})
        except Exception:
            # Quarantine corrupt and tampered source; never treat as synthetic evidence.
            if kind=='remote_download':
                raw.unlink(missing_ok=True)
                (cache/f'{name}.download_provenance.json').unlink(missing_ok=True)
            raise
    result={'schema':'tempo_video_corpus_v1','status':'complete','mode':mode,
        'required_count':len(selected),'verified_count':len(records),
        'all_real_videos':all(r['is_real_source_video'] for r in records),
        'has_native_720p':any(r['native_720p'] for r in records),
        'manifest_source':str(manifest.resolve()),'videos':records}
    output.write_text(json.dumps(result,indent=2),encoding='utf8')
    if mode=='full' and not result['has_native_720p']:
        raise RuntimeError('Full corpus requires at least one genuinely native 720p clip')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default='configs/video_evidence_sources.json')
    p.add_argument('--cache-dir',default='data/downloaded_videos')
    p.add_argument('--prepared-dir',default='data/prepared_videos')
    p.add_argument('--out',default='reports/video_evidence/corpus.json')
    p.add_argument('--mode',choices=['quick','full'],default='full')
    p.add_argument('--frames',type=int,default=180)
    p.add_argument('--timeout',type=int,default=30)
    p.add_argument('--allow-local-fixtures',action='store_true')
    args=p.parse_args()
    if args.frames<1:
        p.error('frames must be positive')
    result=prepare(manifest=Path(args.manifest),cache=Path(args.cache_dir),
        prepared=Path(args.prepared_dir),output=Path(args.out),
        mode=args.mode,frames=args.frames,timeout=args.timeout,
        allow_local=args.allow_local_fixtures)
    print('Verified video sources:',result['verified_count'],'native 720p:',result['has_native_720p'])

if __name__=='__main__':main()
