# TEMPO recovery for Kaggle SESSION THAT RAN THE FAILED JOB.
# No downloaded external CSV; uses the REAL 5328 already-measured rows.
# Completes the correctly invoked video benchmark and accurately summarizes
# the supported upstream Kodak benchmark (does NOT invent five unsupported cases).
import os, sys, subprocess, json, hashlib, traceback, zipfile, time, base64
from pathlib import Path
from datetime import datetime, timezone
WORK=Path('/kaggle/working')
assert WORK.is_dir(), 'Run this in the Kaggle session containing the previous TEMPO results.'
ROOTS=sorted((p for p in WORK.glob('TEMPO_COMPLETE_FRESH_*') if p.is_dir()), key=lambda p:p.stat().st_mtime,reverse=True)
if not ROOTS: raise RuntimeError('Original /kaggle/working results are missing. Restore the original Kaggle session; do not rerun 5328 trials.')
ROOT=next((p for p in ROOTS if (p/'extra/official_mmwatermark_kodak_24/official_mmwatermark_manifest.json').exists()),None)
if ROOT is None:raise RuntimeError('Official Kodak result manifest not found in any existing TEMPO run directory.')
repo=ROOT/'tempo';report=ROOT/'reports';extra=ROOT/'extra';meta=ROOT/'run_metadata'
result=extra/'official_mmwatermark_kodak_24';original=meta/'status.json'
manifest=result/'official_mmwatermark_manifest.json';raw=result/'official_mmwatermark_raw.csv';preflight=result/'official_mmwatermark_preflight.json'
if not all(p.is_file() for p in (manifest,raw,preflight)):
    raise RuntimeError('Missing recorded official manifest, raw CSV, or unsupported-preflight record; do not guess.')
doc=json.loads(manifest.read_text());pre=json.loads(preflight.read_text())
print('Existing run:',ROOT)
print('Existing measured compatible trials:',doc.get('completed_rows'),'/',doc.get('expected_compatible_rows'))
print('Entire original catalog:',doc.get('completed_rows'),'/',doc.get('expected_full_requested_rows'))
print('Unsupported attack/severity combinations:')
for v in pre.get('unsupported_conditions',[]):print('  -',v.get('attack'),v.get('severity'),str(v.get('reason'))[:400])
source=base64.b64decode('IyEvdXNyL2Jpbi9lbnYgcHl0aG9uMwoiIiJTdW1tYXJpemUgcmVhbCBvZmZpY2lhbCBLb2RhayBNTVdhdGVybWFyayByb3dzIHdpdGhvdXQgaW52ZW50aW5nIHVuYXZhaWxhYmxlIGF0dGFja3MuCgpjb21wbGV0ZV9zdXBwb3J0ZWRfY2F0YWxvZz1UcnVlIG1lYW5zIGFsbCBwcmVmbGlnaHQtc3VwcG9ydGVkIHRyaWFscyBleGlzdC4KZnVsbF9vcmlnaW5hbF9jYXRhbG9nPVRydWUgbWVhbnMgKm5vKiB1cHN0cmVhbSBjb25kaXRpb25zIHdlcmUgZXhjbHVkZWQuClRoZXNlIGFyZSBkaWZmZXJlbnQgY2xhaW1zIGFuZCBNVVNUIE5PVCBiZSBjb25mbGF0ZWQuCiIiIgpmcm9tIF9fZnV0dXJlX18gaW1wb3J0IGFubm90YXRpb25zCmltcG9ydCBhcmdwYXJzZSxjc3YsanNvbixtYXRoLHN0YXRpc3RpY3MKZnJvbSBwYXRobGliIGltcG9ydCBQYXRoCmZyb20gY29sbGVjdGlvbnMgaW1wb3J0IGRlZmF1bHRkaWN0CgoKZGVmIHN1bW1hcml6ZShtYW5pZmVzdF9wYXRoOlBhdGgscmF3X3BhdGg6UGF0aCxvdXQ6UGF0aCk6CiAgICBtPWpzb24ubG9hZHMoUGF0aChtYW5pZmVzdF9wYXRoKS5yZWFkX3RleHQoZW5jb2Rpbmc9J3V0ZjgnKSkKICAgIGlmIG5vdCAobS5nZXQoJ2NvbXBsZXRlJykgYW5kIGxlbihtLmdldCgnaG9zdHMnLFtdKSk9PTI0IGFuZAogICAgICAgICAgICBsZW4oc2V0KG1bJ2hvc3RzJ10pKT09MjQgYW5kIGxlbihtLmdldCgnd2F0ZXJtYXJrcycsW10pKT09MiBhbmQKICAgICAgICAgICAgbGVuKHNldChtWyd3YXRlcm1hcmtzJ10pKT09MiBhbmQgbS5nZXQoJ21ldGhvZHMnKT09Wydwcm9wb3NhbCddKToKICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoJ0FsbCAyNCBLb2RhayBob3N0cywgYm90aCB3YXRlcm1hcmtzIGFuZCBjb21wbGV0ZWQgcHJvcG9zYWwgdHJpYWxzIHJlcXVpcmVkJykKICAgIHVuc3VwcG9ydGVkPW0uZ2V0KCdwcmVmbGlnaHRfdW5zdXBwb3J0ZWRfY29uZGl0aW9ucycsW10pCiAgICBpZiBub3QgaXNpbnN0YW5jZSh1bnN1cHBvcnRlZCxsaXN0KTpyYWlzZSBSdW50aW1lRXJyb3IoJ01hbGZvcm1lZCBwcmVmbGlnaHQgZXhjbHVzaW9ucycpCiAgICByZXF1ZXN0ZWQ9eyhhLGludChzZXYpKSBmb3IgYSBpbiBtWydzZWxlY3RlZF9hdHRhY2tfZnVuY3Rpb25zJ10gZm9yIHNldiBpbiBtWydzZXZlcml0aWVzJ119CiAgICBpZiBsZW4ocmVxdWVzdGVkKSE9bGVuKG1bJ3NlbGVjdGVkX2F0dGFja19mdW5jdGlvbnMnXSkqbGVuKG1bJ3NldmVyaXRpZXMnXSk6CiAgICAgICAgcmFpc2UgUnVudGltZUVycm9yKCdEdXBsaWNhdGUgcmVxdWVzdGVkIGF0dGFjay9zZXZlcml0eScpCiAgICBvbWl0dGVkPXsoeFsnYXR0YWNrJ10saW50KHhbJ3NldmVyaXR5J10pKSBmb3IgeCBpbiB1bnN1cHBvcnRlZH0KICAgIGlmIGxlbihvbWl0dGVkKSE9bGVuKHVuc3VwcG9ydGVkKSBvciBub3Qgb21pdHRlZDw9cmVxdWVzdGVkOgogICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcignRHVwbGljYXRlZCBvciB1bmV4cGVjdGVkIHVuc3VwcG9ydGVkIGNvbmRpdGlvbnMnKQogICAgc3VwcG9ydGVkPXJlcXVlc3RlZC1vbWl0dGVkCiAgICBsYWJlbHM9eydjbGVhbid9fHtmJ21tdzo6e2F9OjpzZXZlcml0eXtzZXZ9JyBmb3IgYSxzZXYgaW4gc3VwcG9ydGVkfQogICAgZXhwZWN0ZWRfcGVyX3BhaXI9bGVuKGxhYmVscykKICAgIGV4cGVjdGVkPTI0KjIqZXhwZWN0ZWRfcGVyX3BhaXIKICAgIGV4cGVjdGVkX2Z1bGw9MjQqMiooMStsZW4ocmVxdWVzdGVkKSkKICAgIGlmIChtLmdldCgnZXhwZWN0ZWRfY29tcGF0aWJsZV9yb3dzJykhPWV4cGVjdGVkIG9yCiAgICAgICAgbS5nZXQoJ2NvbXBsZXRlZF9yb3dzJykhPWV4cGVjdGVkIG9yCiAgICAgICAgbS5nZXQoJ2V4cGVjdGVkX2Z1bGxfcmVxdWVzdGVkX3Jvd3MnKSE9ZXhwZWN0ZWRfZnVsbCBvcgogICAgICAgIGJvb2wobS5nZXQoJ2Z1bGxfY292ZXJhZ2UnKSkgIT0gKG5vdCBvbWl0dGVkIGFuZCBtLmdldCgnY29tcGxldGUnKSBhbmQgZXhwZWN0ZWQ9PWV4cGVjdGVkX2Z1bGwpIG9yCiAgICAgICAgbS5nZXQoJ2ZhaWx1cmVzJykpOgogICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcignTWFuaWZlc3QgY291bnRzL2NvdmVyYWdlIGNsYWltcyBjb250cmFkaWN0IGdlbnVpbmUgY29uZGl0aW9uIGludmVudG9yeScpCiAgICB3aXRoIFBhdGgocmF3X3BhdGgpLm9wZW4obmV3bGluZT0nJyxlbmNvZGluZz0ndXRmOCcpIGFzIGY6CiAgICAgICAgcm93cz1saXN0KGNzdi5EaWN0UmVhZGVyKGYpKQogICAgaWYgbGVuKHJvd3MpIT1leHBlY3RlZDoKICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoZidFeHBlY3RlZCB7ZXhwZWN0ZWR9IGNvbXBsZXRlZCBzdXBwb3J0ZWQgdHJpYWxzLCBnb3Qge2xlbihyb3dzKX0nKQogICAgc2Vlbj1zZXQoKTsgcGVyX2F0dD1kZWZhdWx0ZGljdChsaXN0KTsgcGVyX2hvc3Q9ZGVmYXVsdGRpY3QobGlzdCk7IGJ5X21hcms9ZGVmYXVsdGRpY3QobGlzdCkKICAgIGZvciByb3cgaW4gcm93czoKICAgICAgICBpZiByb3dbJ21ldGhvZF9pZCddIT0ncHJvcG9zYWwnIG9yIHJvd1snaG9zdCddIG5vdCBpbiBtWydob3N0cyddIG9yIHJvd1snd2F0ZXJtYXJrJ10gbm90IGluIG1bJ3dhdGVybWFya3MnXSBvciByb3dbJ2F0dGFjayddIG5vdCBpbiBsYWJlbHM6CiAgICAgICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcignVW5leHBlY3RlZCBtZXRob2QsIEtvZGFrIGhvc3QsIHdhdGVybWFyayBvciBhdHRhY2sgbGFiZWwnKQogICAgICAgIGlmIGludChyb3dbJ3BheWxvYWRfYml0cyddKSE9NDA5NjoKICAgICAgICAgICAgcmFpc2UgUnVudGltZUVycm9yKCdFeHRlcm5hbCBjb21wYXJpc29uIHJlcXVpcmVzIGdlbnVpbmUgNDA5Ni1iaXQgcGF5bG9hZCcpCiAgICAgICAga2V5PShyb3dbJ2hvc3QnXSxyb3dbJ3dhdGVybWFyayddLHJvd1snbWV0aG9kX2lkJ10scm93WydhdHRhY2snXSkKICAgICAgICBpZiBrZXkgaW4gc2VlbjpyYWlzZSBSdW50aW1lRXJyb3IoJ0R1cGxpY2F0ZSBleHRlcm5hbCBiZW5jaG1hcmsgcm93ICcrc3RyKGtleSkpCiAgICAgICAgc2Vlbi5hZGQoa2V5KQogICAgICAgIHBlcl9hdHRbKHJvd1snd2F0ZXJtYXJrJ10scm93WydhdHRhY2snXSldLmFwcGVuZChyb3cpCiAgICAgICAgcGVyX2hvc3RbKHJvd1snd2F0ZXJtYXJrJ10scm93Wydob3N0J10pXS5hcHBlbmQocm93KQogICAgICAgIGJ5X21hcmtbcm93Wyd3YXRlcm1hcmsnXV0uYXBwZW5kKHJvdykKICAgIGZvciBtYXJrIGluIG1bJ3dhdGVybWFya3MnXToKICAgICAgICBmb3IgaG9zdCBpbiBtWydob3N0cyddOgogICAgICAgICAgICBhY3R1YWw9e3JbJ2F0dGFjayddIGZvciByIGluIHBlcl9ob3N0WyhtYXJrLGhvc3QpXX0KICAgICAgICAgICAgaWYgYWN0dWFsIT1sYWJlbHM6CiAgICAgICAgICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoZidNaXNzaW5nIHN1cHBvcnRlZCBhdHRhY2tzIGZvciB7bWFya30ve2hvc3R9OiB7c29ydGVkKGxhYmVscy1hY3R1YWwpfScpCiAgICBvdXQ9UGF0aChvdXQpO291dC5ta2RpcihwYXJlbnRzPVRydWUsZXhpc3Rfb2s9VHJ1ZSkKICAgIGRlZiBjYWxjKHJzKToKICAgICAgICBzdGF0cz17J3RyaWFscyc6bGVuKHJzKX0KICAgICAgICBmb3IgbWV0cmljIGluICgnY2xlYW5fZW1iZWRkaW5nX3BzbnInLCduY19iaXBvbGFyJywnYmVyJywnZW1iZWRfbXMnLCdleHRyYWN0X21zJyk6CiAgICAgICAgICAgIHZhbHVlcz1bZmxvYXQocm93W21ldHJpY10pIGZvciByb3cgaW4gcnNdCiAgICAgICAgICAgIGlmIG5vdCBhbGwobWF0aC5pc2Zpbml0ZSh2KSBmb3IgdiBpbiB2YWx1ZXMpOnJhaXNlIFJ1bnRpbWVFcnJvcignTm9uZmluaXRlICcrbWV0cmljKQogICAgICAgICAgICBzdGF0c1snbWVhbl8nK21ldHJpY109c3RhdGlzdGljcy5mbWVhbih2YWx1ZXMpCiAgICAgICAgICAgIHN0YXRzWydzdGRfJyttZXRyaWNdPXN0YXRpc3RpY3Muc3RkZXYodmFsdWVzKSBpZiBsZW4odmFsdWVzKT4xIGVsc2UgMC4wCiAgICAgICAgICAgIHN0YXRzWydtaW5fJyttZXRyaWNdPW1pbih2YWx1ZXMpCiAgICAgICAgICAgIHN0YXRzWydtYXhfJyttZXRyaWNdPW1heCh2YWx1ZXMpCiAgICAgICAgcmV0dXJuIHN0YXRzCiAgICBvdXRwdXRzPXsKICAgICAgICAnYnlfd2F0ZXJtYXJrX2F0dGFjay5jc3YnOlt7J3dhdGVybWFyayc6a1swXSwnYXR0YWNrJzprWzFdLCoqY2FsYyhycyl9IGZvciBrLHJzIGluIHNvcnRlZChwZXJfYXR0Lml0ZW1zKCkpXSwKICAgICAgICAnYnlfd2F0ZXJtYXJrX2hvc3QuY3N2Jzpbeyd3YXRlcm1hcmsnOmtbMF0sJ2hvc3QnOmtbMV0sKipjYWxjKHJzKX0gZm9yIGsscnMgaW4gc29ydGVkKHBlcl9ob3N0Lml0ZW1zKCkpXSwKICAgICAgICAnYnlfd2F0ZXJtYXJrX292ZXJhbGwuY3N2Jzpbeyd3YXRlcm1hcmsnOmssKipjYWxjKHJzKX0gZm9yIGsscnMgaW4gc29ydGVkKGJ5X21hcmsuaXRlbXMoKSldLAogICAgfQogICAgZm9yIGZpbGVuYW1lLGRhdGEgaW4gb3V0cHV0cy5pdGVtcygpOgogICAgICAgIHdpdGggKG91dC9maWxlbmFtZSkub3BlbigndycsbmV3bGluZT0nJyxlbmNvZGluZz0ndXRmOCcpIGFzIGY6CiAgICAgICAgICAgIHdyaXRlcj1jc3YuRGljdFdyaXRlcihmLGZpZWxkbmFtZXM9bGlzdChkYXRhWzBdKSk7d3JpdGVyLndyaXRlaGVhZGVyKCk7d3JpdGVyLndyaXRlcm93cyhkYXRhKQogICAgd2l0aCAob3V0Lyd1cHN0cmVhbV91bnN1cHBvcnRlZF9jb25kaXRpb25zLmNzdicpLm9wZW4oJ3cnLG5ld2xpbmU9JycsZW5jb2Rpbmc9J3V0ZjgnKSBhcyBmOgogICAgICAgIHdyaXRlcj1jc3YuRGljdFdyaXRlcihmLGZpZWxkbmFtZXM9WydhdHRhY2snLCdzZXZlcml0eScsJ3JlYXNvbiddKTt3cml0ZXIud3JpdGVoZWFkZXIoKQogICAgICAgIGZvciB4IGluIHVuc3VwcG9ydGVkOndyaXRlci53cml0ZXJvdyh7azp4LmdldChrLCcnKSBmb3IgayBpbiAoJ2F0dGFjaycsJ3NldmVyaXR5JywncmVhc29uJyl9KQogICAgcmVzdWx0PXsKICAgICAgICAnc3RhdHVzJzonQ09NUExFVEUnIGlmIG5vdCBvbWl0dGVkIGVsc2UgJ0NPTVBMRVRFX1NVUFBPUlRFRF9DQVRBTE9HX09OTFknLAogICAgICAgICdwcm92aWRlcic6J29mZmljaWFsX01NV2F0ZXJtYXJrX1JvYnVzdG5lc3MnLAogICAgICAgICdkYXRhc2V0JzonS29kYWsgMjQgbG9zc2xlc3MgNTEyeDUxMiBjZW50ZXItY3JvcHBlZCBSR0IgQk1QcycsCiAgICAgICAgJ21ldGhvZHMnOlsncHJvcG9zYWwnXSwncGF5bG9hZF9iaXRzJzo0MDk2LCdob3N0cyc6MjQsJ3dhdGVybWFya3MnOjIsCiAgICAgICAgJ3N1cHBvcnRlZF9jb25kaXRpb25zX3Blcl9wYWlyJzpleHBlY3RlZF9wZXJfcGFpciwKICAgICAgICAnb3JpZ2luYWxfcmVxdWVzdGVkX2NvbmRpdGlvbnNfcGVyX3BhaXInOjErbGVuKHJlcXVlc3RlZCksCiAgICAgICAgJ3RyaWFscyc6bGVuKHJvd3MpLCdjb21wbGV0ZWRfc3VwcG9ydGVkX3Jvd3MnOmxlbihyb3dzKSwnZXhwZWN0ZWRfc3VwcG9ydGVkX3Jvd3MnOmV4cGVjdGVkLAogICAgICAgICd1bmF2YWlsYWJsZV91cHN0cmVhbV9jb25kaXRpb25zJzpsZW4odW5zdXBwb3J0ZWQpLAogICAgICAgICdtaXNzaW5nX3N1cHBvcnRlZF90cmlhbHMnOjAsJ2Z1bGxfb3JpZ2luYWxfY2F0YWxvZyc6bm90IGJvb2wodW5zdXBwb3J0ZWQpLAogICAgICAgICdhbGxfc3VwcG9ydGVkX3RyaWFsc19jb21wbGV0ZSc6VHJ1ZSwKICAgICAgICAnd2FybmluZyc6KCdVbnN1cHBvcnRlZCB1cHN0cmVhbSBhdHRhY2svc2V2ZXJpdHkgY29uZGl0aW9ucyBoYXZlIG5vIG1ldHJpY3M7IHNlZSAnCiAgICAgICAgICAgICAgICAgICAndXBzdHJlYW1fdW5zdXBwb3J0ZWRfY29uZGl0aW9ucy5jc3YuIERvIG5vdCBjbGFpbSBmdWxsIHJlcXVlc3RlZCBjb3ZlcmFnZS4nIGlmIHVuc3VwcG9ydGVkIGVsc2UgJycpLAogICAgICAgICdvdXRwdXRzJzpsaXN0KG91dHB1dHMpK1sndXBzdHJlYW1fdW5zdXBwb3J0ZWRfY29uZGl0aW9ucy5jc3YnXSwKICAgIH0KICAgIChvdXQvJ3N1bW1hcnlfbWFuaWZlc3QuanNvbicpLndyaXRlX3RleHQoanNvbi5kdW1wcyhyZXN1bHQsaW5kZW50PTIpKydcbicsZW5jb2Rpbmc9J3V0ZjgnKQogICAgcmV0dXJuIHJlc3VsdAoKCmRlZiBtYWluKCk6CiAgICBhcD1hcmdwYXJzZS5Bcmd1bWVudFBhcnNlcihkZXNjcmlwdGlvbj1fX2RvY19fKQogICAgYXAuYWRkX2FyZ3VtZW50KCctLW1hbmlmZXN0Jyx0eXBlPVBhdGgscmVxdWlyZWQ9VHJ1ZSkKICAgIGFwLmFkZF9hcmd1bWVudCgnLS1yYXcnLHR5cGU9UGF0aCxyZXF1aXJlZD1UcnVlKQogICAgYXAuYWRkX2FyZ3VtZW50KCctLW91dC1kaXInLHR5cGU9UGF0aCxyZXF1aXJlZD1UcnVlKQogICAgYT1hcC5wYXJzZV9hcmdzKCkKICAgIHByaW50KGpzb24uZHVtcHMoc3VtbWFyaXplKGEubWFuaWZlc3QsYS5yYXcsYS5vdXRfZGlyKSxpbmRlbnQ9MikpCgppZiBfX25hbWVfXz09J19fbWFpbl9fJzptYWluKCkK')
recovery_code=meta/'summarize_supported_kodak_recovery.py';recovery_code.write_bytes(source)
python_path=repo/'.kaggle_python'
py=python_path.read_text().strip() if python_path.is_file() else sys.executable
if not Path(py).exists():py=sys.executable
entries={};logs={};started=datetime.now(timezone.utc).isoformat()

def run(name,command):
    print('\n===',name,'===\n',' '.join(map(str,command)),flush=True)
    logfile=meta/(name+'.log')
    with logfile.open('w',encoding='utf8') as out:
        p=subprocess.run(list(map(str,command)),cwd=repo,stdout=out,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONUNBUFFERED':'1'})
    loglines=logfile.read_text(errors='replace').splitlines()
    print('\n'.join(loglines[-25:]),flush=True)
    entries[name]={'returncode':p.returncode,'status':'COMPLETED' if p.returncode==0 else 'FAILED','log':str(logfile)}
    if p.returncode!=0:raise RuntimeError(name+' failed, see '+str(logfile))
    return True

try:
    if (doc.get('completed_rows')!=doc.get('expected_compatible_rows') or not doc.get('complete')
        or len(doc.get('hosts',[]))!=24 or len(doc.get('watermarks',[]))!=2 or doc.get('methods')!=['proposal']):
        raise RuntimeError('Existing official results are not complete for all supported conditions.')
    run('RECOVER_OFFICIAL_24_KODAK_SUMMARIES',[py,recovery_code,
      '--manifest',manifest,'--raw',raw,'--out-dir',result/'summaries'])
    # Rerun ONLY broken original video benchmark. 4096-bit algorithm-only stage
    # already succeeded in the original run and is preserved unchanged.
    video_manifest=report/'video_sources/corpus.json'
    if not video_manifest.is_file():raise RuntimeError('No original real-video corpus manifest')
    video=None
    for item in json.loads(video_manifest.read_text()).get('videos',[]):
        p=Path(item.get('original_path') or '')
        if item.get('native_720p') is True and item.get('is_real_source_video') is True and p.is_file():
            with p.open('rb') as handle:
                digest=hashlib.file_digest(handle,'sha256').hexdigest()
            if digest==item.get('original_sha256'):video=p;break
    if video is None:raise RuntimeError('Verified original 720p video not present; requires the same Kaggle session.')
    config=report/'configs/proposal_selected_all_watermarks.json';marks=ROOT/'selected_data/watermark'
    if not config.is_file() or len(list(marks.glob('*.png')))!=2:raise RuntimeError('Missing frozen config or both 4096-bit watermark PNGs')
    run('RECOVER_FROZEN_720P',[py,repo/'scripts/benchmark_frozen_720p.py',
      '--video',video,'--watermark-dir',marks,'--config',config,
      '--out-dir',extra/'frozen_720p_30fps','--frames',300,'--fps',32])
    summary=json.loads((result/'summaries/summary_manifest.json').read_text())
    records={'status':'RECOVERED_SUPPORTED_BENCHMARK_AND_VIDEO','started':started,
      'previous_run_dir':str(ROOT),'original_catalog_full_coverage':summary['full_original_catalog'],
      'supported_trials_completed':summary['completed_supported_rows'],
      'unsupported_conditions':summary['unavailable_upstream_conditions'],'stages':entries,
      'disclosure':'Five unsupported original MMWatermark conditions have no numerical results; preserved as excluded.'}
except BaseException as e:
    records={'status':'RECOVERY_FAILED','started':started,'error':repr(e),'stages':entries}
    (meta/'recovery_exception.txt').write_text(traceback.format_exc())
finally:
    (meta/'recovery_status.json').write_text(json.dumps(records,indent=2))
    outfile=WORK/('TEMPO_RECOVERED_EVIDENCE_'+ROOT.name.rsplit('_',1)[-1]+'.zip')
    part=Path(str(outfile)+'.part')
    with zipfile.ZipFile(part,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for folder in ('reports','extra','run_metadata','selected_data'):
            base=ROOT/folder
            if not base.exists():continue
            for f in base.rglob('*'):
                if f.is_file() and not f.is_symlink() and f.suffix.lower() not in ('.zip','.mp4','.avi','.mov','.mkv','.part','.tmp'):
                    z.write(f,str(Path(folder)/f.relative_to(base)))
    part.replace(outfile)
    print('\nEvidence ZIP:',outfile,'size MB',round(outfile.stat().st_size/1e6,2))
    print('Recovery status:',records['status'])
if records['status']!='RECOVERED_SUPPORTED_BENCHMARK_AND_VIDEO':
    raise RuntimeError('Recovery failed: '+str(meta/'recovery_status.json'))
