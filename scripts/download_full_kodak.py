#!/usr/bin/env python3
"""Fetch and verify ALL 24 original Kodak PCD0992 lossless RGB photographs.

There is deliberately no sample limit, resizing, crop, substitution, or fixture
fallback. A complete 24-image manifest is mandatory before downstream science.
Downloaded SHA256 digests provide run provenance (not a publisher-signed digest).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
import urllib.request
from pathlib import Path
from PIL import Image

NAMES = tuple(f"kodim{i:02d}.png" for i in range(1, 25))
ORIGINS = (
    "https://www.r0k.us/graphics/kodak/kodak/{name}",
    "https://raw.githubusercontent.com/MohamedBakrAli/Kodak-Lossless-True-Color-Image-Suite/master/PhotoCD_PCD0992/{number}.png",
)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def inspect_png(payload: bytes, name: str) -> dict:
    # Pillow verifies full lossless PNG stream and dimensions independently.
    with Image.open(io.BytesIO(payload)) as im:
        if im.format != 'PNG':
            raise ValueError(f'{name}: source is not PNG')
        im.verify()
    with Image.open(io.BytesIO(payload)) as im:
        w, h = im.size
        if (w, h) not in ((768, 512), (512, 768)):
            raise ValueError(f'{name}: unexpected native dimensions {w}x{h}')
        if im.mode not in ('RGB', 'RGBA'):
            raise ValueError(f'{name}: expected true-color PNG, got mode {im.mode}')
        im.load()
    return {'width': w, 'height': h, 'mode': 'RGB/RGBA true color'}


def _get(url: str, timeout: float) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(
        url, headers={'User-Agent': 'TEMPO-Kodak-full-suite/1.0'}), timeout=timeout) as response:
        data = response.read(5_000_001)
    if not 10_000 < len(data) <= 5_000_000:
        raise ValueError('Image payload outside expected size range')
    return data


def download(out_dir: Path, *, timeout: float = 30.0, retries: int = 2, offline: bool = False) -> dict:
    out_dir = Path(out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / 'kodak_full_manifest.json'
    previous = {}
    if manifest_path.is_file():
        try:
            previous = {r['name']: r for r in json.loads(manifest_path.read_text()).get('files', [])}
        except Exception:
            previous = {}
    files = []
    errors = []
    for index, name in enumerate(NAMES, start=1):
        path = out_dir / name
        origin = previous.get(name, {}).get('source_url', '')
        if path.is_file():
            payload = path.read_bytes()
            try:
                metadata = inspect_png(payload, name)
                # If a manifest exists, never accept a file changed in-place.
                prior_sha = previous.get(name, {}).get('sha256')
                if prior_sha and sha256(payload) != prior_sha:
                    raise RuntimeError('Local file SHA256 differs from retained provenance')
            except Exception as exc:
                raise RuntimeError(f'Cached Kodak file invalid: {name}: {exc}') from exc
        else:
            if offline:
                raise FileNotFoundError(f'Offline strict mode: missing {path}')
            problems = []
            for attempt in range(retries + 1):
                for template in ORIGINS:
                    url = template.format(name=name, number=index)
                    try:
                        candidate = _get(url, timeout)
                        metadata = inspect_png(candidate, name)
                        payload, origin = candidate, url
                        temp = path.with_suffix('.download')
                        temp.write_bytes(payload)
                        os.replace(temp, path)
                        break
                    except Exception as exc:
                        problems.append(f'{url}: {type(exc).__name__}: {exc}')
                if path.is_file():
                    break
                if attempt < retries:
                    time.sleep(min(2 ** attempt, 5))
            if not path.is_file():
                errors.append({'file': name, 'attempts': problems})
                continue
        files.append({'name': name, 'path': str(path), 'sha256': sha256(payload),
                      'bytes': len(payload), 'source_url': origin, **metadata})
        print(f'[Kodak] {len(files)}/24 {name} {metadata["width"]}x{metadata["height"]} SHA256={files[-1]["sha256"][:16]}', flush=True)
    if len(files) == 24 and len({r['sha256'] for r in files}) != 24:
        errors.append({'error': 'Duplicate image file hashes found'})
    complete = len(files) == 24 and not errors
    manifest = {'dataset': 'Kodak Lossless True Color Image Suite PCD0992',
                'reference': 'https://www.r0k.us/graphics/kodak/',
                'status': 'COMPLETE_24_OF_24' if complete else 'INCOMPLETE',
                'expected_images': 24, 'verified_images': len(files),
                'crop_policy': 'NONE; original native 768x512 or 512x768 pixels',
                'resize_policy': 'NONE', 'files': files, 'errors': errors,
                'note': 'SHA256 values record downloaded bytes; no publisher-authoritative per-file hashes were available.'}
    temp_manifest = manifest_path.with_suffix('.json.tmp')
    temp_manifest.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf8')
    os.replace(temp_manifest, manifest_path)
    if not complete:
        raise RuntimeError(f'Full Kodak dataset incomplete: {len(files)}/24. See {manifest_path}')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--offline', action='store_true', help='Use verified cached 24 PNGs, do not access the network')
    p.add_argument('--timeout', type=float, default=30.)
    p.add_argument('--retries', type=int, default=2)
    a = p.parse_args()
    download(a.out_dir, timeout=a.timeout, retries=a.retries, offline=a.offline)


if __name__ == '__main__':
    main()
