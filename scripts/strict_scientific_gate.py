#!/usr/bin/env python3
"""Fail-closed scientific audit for TEMPO original-MMW and measured 720p FPS.

Reads real evidence; never changes scores. It intentionally distinguishes runtime
completion from experimental coverage or a claim that a target was met.
"""
from __future__ import annotations
import argparse, ast, csv, json, math, sys
from pathlib import Path

# 23 genuine (image, severity) entry points in the original image_perturb.py.
# Legacy *_224 helpers and motion_blur_ori are not 512x512 benchmark entry
# points; they are accounted for in catalog but cannot silently replace these.
OFFICIAL_IMAGE_ATTACKS = frozenset(('gaussian_noise','shot_noise','impulse_noise',
    'speckle_noise','gaussian_blur','glass_blur','defocus_blur','motion_blur',
    'zoom_blur','fog','frost','snow','spatter','brightness','contrast','saturate',
    'pixelate','jpeg_compression','elastic_transform','scale','rotate_img',
    'shear','piecewise_affine'))
LEGACY_HELPERS = frozenset(('frost_224', 'snow_224', 'motion_blur_ori'))


def read_json(p):
    return json.loads(p.read_text(encoding='utf-8')) if p.is_file() else None


def read_csv(p):
    if not p.is_file():
        return None
    with p.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def number(value):
    try: n = float(value)
    except (TypeError, ValueError): return None
    return n if math.isfinite(n) else None


def is_true(v):
    return v is True or str(v).strip().lower() in ('true', '1', 'yes')


def audit(root: Path, *, min_fps=30., min_psnr=50., min_frames=300.,
          required_watermarks=2, require_reviewer=True):
    root = Path(root)
    reports = root/'reports'
    extra = root/'extra'
    failures=[]; notes=[]
    def check(condition, message):
        if not condition: failures.append(message)
        return bool(condition)
    out = {'requirements': {'fps_min': min_fps, 'psnr_min_db': min_psnr,
           'minimum_stream_frames': min_frames, 'watermarks': required_watermarks},
           'checks': {}, 'failures': failures, 'notes': notes}

    # Original upstream code is authoritative: don't count a reduced or renamed list.
    original = root/'MMWatermark-Robustness'/'perturbations'/'image_perturb.py'
    names=set();severity_api=set()
    if original.is_file():
        tree=ast.parse(original.read_text(encoding='utf-8'))
        functions=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
        names={n.name for n in functions}
        severity_api={n.name for n in functions
                      if len(n.args.args)>=2
                      and n.args.args[1].arg in ('severity','level')}
    check(bool(names), 'Original MMWatermark image_perturb.py missing from cloned upstream checkout')
    missing_source=sorted(OFFICIAL_IMAGE_ATTACKS-names)
    check(not missing_source, f'Original upstream changed or missing required attack functions: {missing_source}')
    unexpected_severity_api=sorted(severity_api - OFFICIAL_IMAGE_ATTACKS - LEGACY_HELPERS)
    check(not unexpected_severity_api,
          f'New upstream severity attacks must be classified and covered: {unexpected_severity_api}')
    notes.append('Legacy upstream frost_224, snow_224 and motion_blur_ori are not interchangeable 512x512 attack conditions.')
    out['checks']['upstream_catalog']={'expected_512x512_image_attacks':len(OFFICIAL_IMAGE_ATTACKS),
       'source_functions_found': len(names), 'missing_from_source':missing_source,
       'legacy_helpers_in_source':sorted(LEGACY_HELPERS & names),
       'new_unclassified_severity_functions':unexpected_severity_api}

    manifest=read_json(extra/'external_official_mmwatermark'/'official_mmwatermark_manifest.json')
    rows=read_csv(extra/'external_official_mmwatermark'/'official_mmwatermark_raw.csv')
    mm={'manifest_present':manifest is not None,'raw_csv_present':rows is not None}
    if not check(manifest is not None and rows is not None, 'Original MMWatermark manifest or raw CSV missing'):
        out['checks']['mmwatermark']=mm
    else:
        actual_attacks=set(manifest.get('selected_attack_functions',[]))
        sev=set(manifest.get('severities',[]))
        hosts=set(manifest.get('hosts',[]))
        watermarks=set(manifest.get('watermarks',[]))
        methods=set(manifest.get('methods',[]))
        expected=(len(hosts)*len(watermarks)*len(methods)*(1+len(OFFICIAL_IMAGE_ATTACKS)*5))
        sample_keys=set()
        duplicates=[];invalid=[]
        for row in rows:
            key=(row.get('host'),row.get('watermark'),row.get('method_id'),row.get('attack'))
            if key in sample_keys: duplicates.append(key)
            sample_keys.add(key)
            for field in ('clean_embedding_psnr','nc_bipolar','ber','embed_ms','extract_ms'):
                if number(row.get(field)) is None: invalid.append((key,field))
        expected_keys={(h,w,m,'clean') for h in hosts for w in watermarks for m in methods}
        expected_keys.update((h,w,m,f'mmw::{a}::severity{s}') for h in hosts
                             for w in watermarks for m in methods
                             for a in OFFICIAL_IMAGE_ATTACKS for s in range(1,6))
        missing_keys=expected_keys-sample_keys
        unexpected_keys=sample_keys-expected_keys
        mm.update({'listed_attacks':sorted(actual_attacks),'hosts':len(hosts),
          'watermarks':len(watermarks),'rows':len(rows),'expected_rows':expected,
          'reported_full_coverage':manifest.get('full_coverage'),
          'missing_rows':len(missing_keys),'unexpected_rows':len(unexpected_keys),
          'duplicates':len(duplicates),'invalid_numeric_values':len(invalid),
          'unsupported_conditions':len(manifest.get('preflight_unsupported_conditions',[])),
          'failed_trials':len(manifest.get('failures',[])),
          'upstream_sha':manifest.get('official_commit')})
        check(actual_attacks==OFFICIAL_IMAGE_ATTACKS, f'Original attack list incomplete: missing {sorted(OFFICIAL_IMAGE_ATTACKS-actual_attacks)}, extra {sorted(actual_attacks-OFFICIAL_IMAGE_ATTACKS)}')
        check(sev==set(range(1,6)), 'Not all five original severity levels were requested')
        check(len(hosts)==9 and len(watermarks)==required_watermarks and methods=={'proposal'},
              'MMW must use nine held-out TEMPO hosts, both watermarks, proposal only')
        check(expected==len(rows)==manifest.get('expected_full_requested_rows') and
              len(missing_keys)==0 and not unexpected_keys and not duplicates and not invalid,
              f'Original MMW row coverage invalid: {len(rows)}/{expected}, missing={len(missing_keys)}, extra={len(unexpected_keys)}, duplicates={len(duplicates)}, invalid={len(invalid)}')
        check(is_true(manifest.get('full_coverage')) and not manifest.get('failures')
              and not manifest.get('preflight_unsupported_conditions'),
              'Original MMW reported incomplete upstream attack coverage')
        out['checks']['mmwatermark']=mm

    # Frozen config integrity for new 720p experiment.
    fps_doc=read_json(extra/'frozen_720p_30fps'/'frozen_fps_manifest.json')
    fps={'manifest_present':fps_doc is not None}
    if not check(fps_doc is not None,'Measured frozen-config 720p JPEG70 benchmark missing'):
        out['checks']['frozen_720p_jpeg70']=fps
    else:
        dims=fps_doc.get('source_native_resolution') or []
        check(len(dims)==2 and dims[0]>=1280 and dims[1]>=720,
              'FPS input is not verified native-resolution 720p')
        # Evidence must trace to the original verified native-resolution
        # download, not a synthetic fixture or 512x512 prepared video.
        corpus=read_json(reports/'video_sources'/'corpus.json') or {}
        trusted=[v for v in corpus.get('videos', [])
                 if v.get('is_real_source_video') is True
                 and v.get('native_720p') is True
                 and v.get('original_sha256') == fps_doc.get('video_sha256')
                 and v.get('original_path') == fps_doc.get('video')]
        check(bool(trusted), 'Frozen 720p FPS video not provenance-matched to a verified original real native 720p download')
        cases=fps_doc.get('cases',[])
        fps['cases']=cases
        expected_wms=set((manifest or {}).get('watermarks', []))
        observed_wms={c.get('watermark') for c in cases}
        check(len(cases)==required_watermarks and observed_wms==expected_wms,
              f'FPS watermarks {sorted(str(x) for x in observed_wms)} do not match all original benchmark watermarks')
        for c in cases:
            tag=c.get('watermark','unknown')
            frames=number(c.get('frames_generated'))
            processed=number(c.get('frames_processed'))
            drops=number(c.get('frames_dropped_queue'))
            channel_drops=number(c.get('frames_dropped_channel'))
            fps_value=number(c.get('effective_output_fps'))
            p95=number(c.get('service_ms_p95'))
            latency95=number(c.get('latency_ms_p95'))
            ontime=number(c.get('on_time_completion_ratio'))
            check(c.get('resolution')=='1280x720' and c.get('channel')=='jpeg70' and c.get('pace_mode')=='realtime'
                  and c.get('method_id')=='proposal' and int(c.get('payload_bits') or 0)==4096,
                  f'{tag}: invalid real-video FPS protocol')
            check(frames is not None and frames>=min_frames and processed==frames and
                  drops==0 and channel_drops==0,
                  f'{tag}: frame loss or inadequate frame sample')
            # Producer may target 32 FPS; enforce the requested 30-FPS
            # end-to-end p95 deadline directly, not the stricter 32-FPS
            # per-frame deadline counter. Do not hide dropped frames.
            check(fps_value is not None and fps_value>=min_fps and p95 is not None and p95<=1000/min_fps and latency95 is not None and latency95<=1000/min_fps,
                  f'{tag}: measured FPS={fps_value}, service p95={p95} ms, e2e latency p95={latency95} ms; needs >= {min_fps} FPS and <= {1000/min_fps:.3f} ms')
            check(is_true(c.get('real_video')) and is_true(c.get('frozen_proposal')),
                  f'{tag}: video not real or step not frozen to selected image configuration')
        out['checks']['frozen_720p_jpeg70']=fps

    image_rows=read_csv(reports/'all_methods_all_metrics'/'summary_by_method_watermark.csv')
    proposal=[r for r in (image_rows or []) if r.get('method_id')=='proposal']
    quality={'rows':len(proposal),'values':[]}
    check(len(proposal)>=required_watermarks,'Proposal final held-out image summary absent or missing watermarks')
    for r in proposal:
        mean_psnr=number(r.get('mean_embedding_psnr'))
        clean_acc=number(r.get('mean_clean_bit_accuracy'))
        quality['values'].append({'watermark':r.get('watermark_name'),'mean_psnr':mean_psnr,'clean_accuracy':clean_acc})
        check(mean_psnr is not None and mean_psnr>=min_psnr,
              f"{r.get('watermark_name')}: final clean mean PSNR {mean_psnr} below {min_psnr}")
        check(clean_acc is not None and clean_acc>=1.0-1e-9,
              f"{r.get('watermark_name')}: clean bit accuracy {clean_acc}, expected 1")
    out['checks']['quality']=quality

    if require_reviewer:
        reviewer=read_json(reports/'reviewer_completeness.json')
        ledger=read_json(extra/'reviewer_expansion'/'expansion_ledger.json')
        out['checks']['reviewer']={'completeness':reviewer,'expansion_status':(ledger or {}).get('status')}
        check(reviewer is not None,'Reviewer completeness report missing')
        if reviewer is not None:
            check(not reviewer.get('failed_checks')
                  and is_true(reviewer.get('execution_evidence_complete'))
                  and is_true(reviewer.get('scientifically_complete_for_resubmission')),
                  f"Reviewer completeness checks failed: {reviewer.get('failed_checks')}")
            check(not reviewer.get('requires_external_or_manuscript_action'),
                  'Reviewer manifest still lists external/manuscript actions')
        check(ledger is not None and ledger.get('status')=='complete' and not ledger.get('incomplete_stages'),
              'Additional reviewer expansion not fully complete')
    out['status']='PASS' if not failures else 'FAIL'
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--out',type=Path,default=None)
    p.add_argument('--target-fps',type=float,default=30.)
    p.add_argument('--target-psnr',type=float,default=50.)
    p.add_argument('--allow-incomplete-reviewer',action='store_true')
    a=p.parse_args()
    doc=audit(a.run_root,min_fps=a.target_fps,min_psnr=a.target_psnr,
              require_reviewer=not a.allow_incomplete_reviewer)
    dest=a.out or (a.run_root/'run_metadata'/'scientific_gate.json')
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(doc,indent=2,default=str)+'\n',encoding='utf8')
    print(json.dumps({'status':doc['status'],'failures':doc['failures'], 'report':str(dest)},indent=2))
    if doc['status']!='PASS':sys.exit(1)

if __name__=='__main__':main()
