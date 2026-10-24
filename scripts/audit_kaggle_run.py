#!/usr/bin/env python3
"""Inspect existing TEMPO Kaggle evidence. Does NOT rerun, tune or invent data.

Example:
 python scripts/audit_kaggle_run.py --run-root /kaggle/working/TEMPO_FRESH_...
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path


def load_json(p: Path):
    return json.loads(p.read_text(encoding='utf8')) if p.is_file() else None


def load_csv(p: Path):
    with p.open(encoding='utf8', newline='') as f:
        return list(csv.DictReader(f))


def fnum(value):
    try:
        result=float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def audit(run_root: Path, psnr_target: float=50.0, fps_target: float=30.0):
    root=Path(run_root).resolve()
    reports=root/'reports' if (root/'reports').is_dir() else root
    evidence_dir=root if (root/'reports').is_dir() else root.parent
    info={'run_root':str(root),'thresholds':{'clean_mean_psnr_db':psnr_target,
         'measured_720p_jpeg70_effective_fps':fps_target},'findings':[],
         'stages':{},'reviewer_completeness':{},'official_mmwatermark':{},
         'proposal_image_metrics':[],'fps_variants':{},'real_video_720p_jpeg70':{}}
    def flag(level,message):
        info['findings'].append({'severity':level,'message':message})

    meta=load_json(evidence_dir/'run_metadata/status.json')
    if meta is None:
        flag('unknown','run_metadata/status.json unavailable; stage exit codes not independently verified')
    else:
        info['source_git_commit']=meta.get('git_commit')
        for key,record in meta.get('status',{}).items():
            state=record.get('status') if isinstance(record,dict) else str(record)
            info['stages'][key]=state
            if str(state).startswith(('FAILED','STOPPED')):
                flag('failure',f'{key}: {state}; inspect run_metadata/{key}.log')
            elif state=='NOT_RUN':
                reason=record.get('reason','') if isinstance(record,dict) else ''
                flag('not_run',f'{key}: {reason}')
            elif state in ('PARTIAL_CHECK_EXPANSION_LEDGER','MISSING'):
                flag('incomplete',f'{key}: {state}')

    completeness=load_json(reports/'reviewer_completeness.json')
    if completeness is None:
        flag('unknown','reviewer_completeness.json unavailable; no scientific-coverage claim possible')
    else:
        failures=completeness.get('failed_checks',[])
        info['reviewer_completeness']={
            'execution_evidence_complete':completeness.get('execution_evidence_complete'),
            'scientifically_complete_for_resubmission':completeness.get('scientifically_complete_for_resubmission'),
            'failed_checks':failures,
            'failure_reasons':{x:completeness.get('failure_reasons',{}).get(x) for x in failures},
            'requires_external_or_manuscript_action':completeness.get('requires_external_or_manuscript_action',{})}
        for item in failures:
            flag('incomplete',f'executable evidence missing: {item}: '
                 f"{completeness.get('failure_reasons',{}).get(item,'')}")
        if completeness.get('requires_external_or_manuscript_action'):
            flag('action_needed','Some reviewer requests require genuine external model results or manuscript work')

    expansion=load_json(evidence_dir/'extra/reviewer_expansion/expansion_ledger.json')
    if expansion:
        info['reviewer_expansion']={'status':expansion.get('status'),
                                   'incomplete_stages':expansion.get('incomplete_stages',[]),
                                   'stage_statuses':{k:v.get('status') for k,v in expansion.get('stages',{}).items()}}
        for item in expansion.get('incomplete_stages',[]):
            flag('incomplete',f'extra reviewer expansion stage not complete: {item}')

    official=load_json(evidence_dir/'extra/external_official_mmwatermark/official_mmwatermark_manifest.json')
    if official is None:
        flag('unknown','Official MMWatermark coverage not verifiable from stage exit status alone')
    else:
        unknown=official.get('preflight_unsupported_conditions',[])
        trial_errors=official.get('failures',[])
        listed=official.get('selected_attack_functions',[])
        # The legacy adapter silently omitted rotate_img, so a legacy manifest
        # can truthfully be full for its *reduced* list but not the corrected
        # original upstream benchmark. Do not mistake the two scopes.
        missing_rotation='rotate_img' not in listed
        if missing_rotation:
            flag('incomplete','Original upstream rotate_img (5 severity levels) was not included in this MMWatermark run; rerun with the corrected adapter')
        info['official_mmwatermark']={
            'full_coverage':bool(official.get('full_coverage')) and not missing_rotation,
            'reported_full_coverage':bool(official.get('full_coverage')),
            'rotate_img_included':not missing_rotation,
            'compatible_complete':bool(official.get('complete')),
            'completed_rows':official.get('completed_rows'),
            'expected_full_requested_rows':official.get('expected_full_requested_rows'),
            'expected_compatible_rows':official.get('expected_compatible_rows'),
            'preflight_unsupported_count':len(unknown),'failed_trials':len(trial_errors),
            'source_commit':official.get('official_commit')}
        if unknown: flag('incomplete',f'{len(unknown)} upstream attack/severity pairs unsupported; check official_mmwatermark_preflight.json')
        if trial_errors: flag('failure',f'{len(trial_errors)} official MMWatermark trials failed')
        if not official.get('full_coverage'):
            flag('incomplete','MMWatermark stage exited but did not cover all requested upstream conditions')

    image_path=reports/'all_methods_all_metrics/summary_by_method_watermark.csv'
    if not image_path.is_file():
        flag('unknown','Final proposal image-quality CSV absent; search-stage PSNR is not final held-out PSNR')
    else:
        proposal=[r for r in load_csv(image_path) if r.get('method_id')=='proposal']
        if not proposal: flag('unknown','No proposal method rows in final image-quality summary')
        for row in proposal:
            summary={key:row.get(key) for key in ('watermark_name','host_count','mean_embedding_psnr',
                'mean_embedding_ssim','mean_clean_bit_accuracy','mean_attack_nc','global_worst_nc')}
            info['proposal_image_metrics'].append(summary)
            value=fnum(row.get('mean_embedding_psnr'))
            if value is None:
                flag('unknown',f"Missing mean PSNR for {row.get('watermark_name','watermark')}")
            elif value < psnr_target:
                flag('performance',f"Final {row.get('watermark_name')} mean PSNR {value:.3f} dB < {psnr_target:g} dB goal; do not substitute hyperparameter-search results")
            nc=fnum(row.get('mean_clean_bit_accuracy'))
            if nc is not None and nc < 1.0-1e-9:
                flag('performance',f"Clean bit accuracy {nc:.6f} < 1 for {row.get('watermark_name')}")

    fps_path=evidence_dir/'extra/fps_variants/fps_ablation_summary.csv'
    if fps_path.is_file():
        variants=load_csv(fps_path)
        info['fps_variants']={'rows':len(variants),
                              'failed_observations':sum(int(float(r.get('failures') or 0)) for r in variants),
                              'note':'Operation-level FPS estimates only; not 720p wall-clock effective FPS'}
        if info['fps_variants']['failed_observations']:
            flag('incomplete',f"{info['fps_variants']['failed_observations']} FPS variant observations failed")
    else:
        flag('unknown','fps_ablation_summary.csv unavailable')

    stream_path=reports/'throughput_audit/wallclock_throughput_audit.csv'
    if stream_path.is_file():
        records=[r for r in load_csv(stream_path) if r.get('resolution')=='1280x720'
                 and r.get('channel','').lower()=='jpeg70' and r.get('source_kind')=='real_video']
        good=[];bad=[]
        for r in records:
            fps=fnum(r.get('effective_output_fps'))
            p95=fnum(r.get('service_ms_p95'))
            if fps is not None and p95 is not None:
                record={'video':r.get('source_video'),'watermark':r.get('watermark'),
                        'fps':fps,'service_ms_p95':p95,'meets_effective_fps_target':fps>=fps_target,
                        'meets_p95_deadline':p95 <= 1000/fps_target}
                good.append(record)
                if fps < fps_target or p95>1000/fps_target:bad.append(record)
        info['real_video_720p_jpeg70']={'observations':good,'qualifying_cases':len(good)-len(bad),
                                        'total_cases':len(good),'passes_all_cases':bool(good) and not bad}
        if not good:flag('incomplete','No actual real-video 720p JPEG70 timing records found')
        if bad:flag('performance',f'{len(bad)}/{len(good)} real-video 720p JPEG70 cases missed effective FPS or p95 deadline')
    else:
        flag('unknown','wallclock_throughput_audit.csv unavailable; no 720p deadline confirmation')

    return info


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--out',type=Path,default=None)
    p.add_argument('--target-psnr',type=float,default=50.0)
    p.add_argument('--target-fps',type=float,default=30.0)
    p.add_argument('--fail-on-critical',action='store_true',help='Exit 1 only for execution failures or missing executable evidence')
    args=p.parse_args()
    doc=audit(args.run_root,args.target_psnr,args.target_fps)
    print(json.dumps(doc,indent=2,ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(doc,indent=2,ensure_ascii=False),encoding='utf8')
    if args.fail_on_critical and any(x['severity'] in ('failure','incomplete') for x in doc['findings']):
        raise SystemExit(1)

if __name__=='__main__':main()
