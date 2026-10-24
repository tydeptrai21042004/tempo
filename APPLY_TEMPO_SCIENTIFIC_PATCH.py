#!/usr/bin/env python3
"""Install only the low-memory stream trace fix; keep repository code intact.
Run from the TEMPO repository root after copying the ZIP contents on top.
"""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=ROOT/'src/realtime_watermark/stream_pipeline.py'
if not p.is_file():raise FileNotFoundError(p)
s=p.read_text(encoding='utf8')
marker='# TEMPO_SCIENTIFIC_TRACE_COMPACTION_V1'
if marker not in s:
    rt=s.index('def _run_realtime(')
    anchor='            end = time.perf_counter()\n            with lock:'
    if s[rt:].count(anchor)!=1:
        raise RuntimeError('Upstream stream pipeline has changed: refusing an unsafe automatic edit. Reconcile manually.')
    replacement='''            end = time.perf_counter()
            # TEMPO_SCIENTIFIC_TRACE_COMPACTION_V1
            # Only small watermark arrays and numeric timing are required in
            # traces. Keeping 2 RGB 720p frames for 300 packets uses >1.6 GB.
            # Drop those after service time and PSNR have been computed.
            result.pop("marked", None)
            result.pop("attacked", None)
            with lock:'''
    s=s[:rt]+s[rt:].replace(anchor,replacement,1)
    p.write_text(s,encoding='utf8')
    print('Applied exact-scope stream-memory correction')
else:
    print('Stream-memory correction already present')
# This updates only the new/changed files while keeping all other manifest
# entries and digests fixed. It never signs unverified unrelated source changes.
import runpy
runpy.run_path(str(ROOT/'UPDATE_SOURCE_MANIFEST_FOR_STRICT_PATCH.py'),run_name='__main__')
