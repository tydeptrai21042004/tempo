#!/usr/bin/env python3
"""Safely update only hashes for this patch's new or changed repo files."""
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parent
TARGETS=[
    'KAGGLE_TEMPO_STRICT_FULL_ONE_CELL.py',
    'APPLY_TEMPO_SCIENTIFIC_PATCH.py',
    'UPDATE_SOURCE_MANIFEST_FOR_STRICT_PATCH.py',
    'scripts/strict_scientific_gate.py',
    'scripts/benchmark_frozen_720p.py',
    'src/realtime_watermark/stream_pipeline.py',
    'tests/test_strict_scientific_gate.py',
    'tests/test_stream_memory_720p.py',
]
manifest=ROOT/'SOURCE_MANIFEST.sha256'
if not manifest.is_file():raise FileNotFoundError('Run in cloned TEMPO repository root: SOURCE_MANIFEST.sha256 missing')
records={}
for line in manifest.read_text().splitlines():
    if not line.strip():continue
    digest,path=line.split(maxsplit=1)
    records[path.removeprefix('*')]=digest
for path in TARGETS:
    f=ROOT/path
    if not f.is_file():raise FileNotFoundError(f)
    records[path]=hashlib.sha256(f.read_bytes()).hexdigest()
manifest.write_text(''.join(f'{digest}  {path}\n' for path,digest in sorted(records.items())),encoding='utf-8')
print(f'Updated {len(TARGETS)} patch entries while preserving all other recorded hashes.')
