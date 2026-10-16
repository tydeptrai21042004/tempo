#!/usr/bin/env python3
"""Reviewer evidence ledger. Never equate script success with publication completeness."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path


def rows(path):
    if not path.is_file():return []
    with path.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))


def check(root:Path,mode='full'):
    full=mode=='full';checks={};reasons={}
    def record(name,ok,explanation=''):
        checks[name]=bool(ok)
        if not ok:reasons[name]=explanation or 'Missing or incomplete executable evidence'
    split=root/'holdout'/'split_manifest.json'
    sp=json.loads(split.read_text()) if split.exists() else {}
    dev={x['sha256'] for x in sp.get('development',[])}
    test={x['sha256'] for x in sp.get('test',[])}
    record('data_disjoint',bool(dev and test and not(dev&test)),'Development/test split absent or overlapping')
    image=rows(root/'all_methods_all_metrics'/'all_watermarks_all_methods_all_metrics.csv')
    expected_hosts={x['name'] for x in sp.get('test',[])}
    image_hosts={r.get('host') for r in image}
    image_wms={r.get('watermark_name') for r in image}
    record('main_image_csv',bool(image),'Main image benchmark missing')
    record('main_payload_4096',bool(image) and all(int(float(r.get('payload_bits') or 0))==4096 for r in image),
           'Not all primary observations use 4096 bits')
    record('image_watermark_count',len(image_wms)==2,'Both watermarks must appear in main image results')
    if full:record('all_heldout_hosts',expected_hosts.issubset(image_hosts),'Not all held-out hosts evaluated')
    matched=root/'matched_psnr'/'matched_psnr_primary_eligible_pairs.csv'
    pairs=rows(matched)
    record('matched_psnr_eligible',bool(pairs) and all(str(r.get('primary_rank_eligible')).lower()=='true' for r in pairs),
           'No eligible matched-PSNR rows, or ineligible rows leaked to primary table')
    attacker=root/'reviewer_experiments'/'reviewer_manifest.json'
    if attacker.is_file():
        evidence=json.loads(attacker.read_text())
        ok=evidence.get('expected_attack_rows')==evidence.get('actual_attack_rows') and evidence.get('failed_attack_rows')==0
    else:ok=False
    record('offgrid_and_targeted_attacks',ok,'Supplementary geometric and targeted attack matrix incomplete')
    record('integer_orthogonality_diagnostics',bool(rows(root/'reviewer_experiments'/'orthogonality_and_integer.csv')),
           'Pilot leakage / integer diagnostics not recorded')
    record('capacity_sensitivity',bool(rows(root/'reviewer_experiments'/'capacity_sensitivity.csv')),
           'Payload capacity experiment absent')
    detection=root/'detection'/'detection_manifest.json'
    dm=json.loads(detection.read_text()) if detection.exists() else {}
    record('threshold_calibrated_detection',bool(dm.get('status')=='complete' and dm.get('positive_samples',0)>0 and dm.get('negative_samples',0)>0),
           'Independent-development threshold plus held-out ROC missing')
    record('temporal_voting',bool(rows(root/'temporal_limits'/'temporal_vote_results.csv')),
           'Temporal voting under correlated/uncorrelated error settings absent')
    record('joint_parameter_coupling',bool(rows(root/'joint_sensitivity'/'joint_parameter_coupling.csv')),
           'Joint parameter sensitivity absent')
    record('visual_extraction_gallery',bool(rows(root/'galleries'/'gallery_data.csv')),
           'Actual extracted watermark figures absent')
    record('host_cluster_ci',bool(rows(root/'statistical_evidence'/'method_host_cluster_ci.csv')),
           'Paired/host-cluster uncertainty summary missing')
    record('paired_effects',bool(rows(root/'statistical_evidence'/'paired_proposal_effects.csv')),
           'Paired host-level method comparison missing')
    streams=list((root/'live_stream').glob('*/ALL_STREAM_run_manifest.json'))
    expected_n=2 if full else 1
    stream_manifest=[json.loads(p.read_text()) for p in streams]
    record('wall_clock_4096',len(stream_manifest)>=expected_n and all(
      m.get('pace_mode')=='realtime' and m.get('payload_bits')==4096 for m in stream_manifest),
      'Need actual 4096-bit wall-clock timing for all two watermarks in full mode')
    record('stream_traces',len(streams)>=expected_n and all(
      (p.parent/'ALL_STREAM_realtime_trace.csv').is_file() and
      (p.parent/'ALL_STREAM_realtime_trace.csv').stat().st_size>100 for p in streams),
      'Per-frame trace missing')
    record('video_main',bool(rows(root/'video'/'all_watermarks_video_summary.csv')),
           'Framewise video benchmark absent')
    record('structural_ablation',any((root/'ablation').rglob('*.csv')),
           'No structural ablation CSVs')
    # These items are intentionally NOT checked off without external evidence.
    remaining={
      'recent_independent_blind_video_baselines':'Requires faithful recent-method implementations, provenance and matched protocol; no implementation invented',
      'modern_deep_learning_watermark_baselines':'Example-review lesson only; 4096-bit adaptations must be clearly documented and validated',
      'recent_baseline_visual_gallery':'Not claimable until real baseline reconstructions are imported',
      'trained_adversarial_generators':'Intentional DCT zeroing included; no unsupported claim that diffusion/generative attacks were run',
      'final_manuscript_abstract_and_table_reconciliation':'Requires manuscript update using final measured CSVs, not code alone',
      'crypto_security_guarantee':'Arnold scrambling is not encryption; ROC and wrong-key tests do not imply cryptographic authenticity',
      'universal_affine_geometric_invariance':'Unsupported; algorithm has a finite geometric candidate family',
      'global_integer_optimality':'Unsupported; integer-domain projection is bounded heuristic',
    }
    reviewer_map={
      'Original Reviewer 1':{'throughput consistency':'final_manuscript_abstract_and_table_reconciliation',
        'matched distortion':'matched_psnr_eligible','holdout and reproducibility':'data_disjoint',
        'uncertainty':'host_cluster_ci','paired benchmark':'paired_effects'},
      'Original Reviewer 2':{'integer decoupling':'integer_orthogonality_diagnostics',
        'temporal majority limits':'temporal_voting','joint parameter coupling':'joint_parameter_coupling',
        'strong geometry':'offgrid_and_targeted_attacks','capacity':'capacity_sensitivity',
        'adversarial pilot interference':'offgrid_and_targeted_attacks'},
      'Original Reviewer 5':{'off-grid rotation':'offgrid_and_targeted_attacks',
        'targeted DCT suppression':'offgrid_and_targeted_attacks',
        'ROC/FPR':'threshold_calibrated_detection','recent video baselines':'recent_independent_blind_video_baselines',
        'live 4096-bit FPS':'wall_clock_4096','matched PSNR':'matched_psnr_eligible'},
      'Example DWT-Hessenberg review':{'independent tuning':'data_disjoint',
        'wrong-key security':'threshold_calibrated_detection','noise severity':'offgrid_and_targeted_attacks',
        'host-level statistics':'host_cluster_ci','extracted image gallery':'visual_extraction_gallery',
        'modern deep models':'modern_deep_learning_watermark_baselines'},
    }
    status={section:{topic:('executed' if checks.get(key,False) else 'requires_external_or_more_evidence' if key in remaining else 'not_verified')
                     for topic,key in topics.items()} for section,topics in reviewer_map.items()}
    failed=[k for k,v in checks.items() if not v]
    return {'run_mode':mode,'execution_evidence_complete':not failed,'scientifically_complete_for_resubmission':not failed and not remaining,
            'checks':checks,'failed_checks':failed,'failure_reasons':reasons,
            'reviewer_item_coverage':status,'requires_external_or_manuscript_action':remaining,
            'important':'quick mode is a smoke protocol only. A complete run is not a claim all reviewers have been satisfied.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results-dir',required=True)
    p.add_argument('--mode',choices=['quick','full'],default='full');p.add_argument('--strict',action='store_true',help='nonzero exit if an executable evidence check is missing');a=p.parse_args()
    root=Path(a.results_dir);root.mkdir(parents=True,exist_ok=True)
    report=check(root,a.mode)
    (root/'reviewer_completeness.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({'execution_evidence_complete':report['execution_evidence_complete'],
                       'missing':report['failed_checks'],'publication_evidence_complete':False},indent=2))
    if a.strict and report['failed_checks']:raise SystemExit(1)
