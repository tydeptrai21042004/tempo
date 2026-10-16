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
    pair_manifest_path=root/'matched_psnr'/'matched_psnr_strict_paired_manifest.json'
    pair_manifest=json.loads(pair_manifest_path.read_text()) if pair_manifest_path.is_file() else {}
    pair_rows=rows(root/'matched_psnr'/'matched_psnr_strict_paired_attacks.csv')
    record('matched_psnr_same_case_paired',pair_manifest.get('status')=='complete' and bool(pair_rows)
        and pair_manifest.get('total_paired_attack_rows')==len(pair_rows)
        and all(r.get('proposal_psnr_eligible')=='True' and r.get('baseline_psnr_eligible')=='True' for r in pair_rows),
        'Matched PSNR conclusions must use the intersection of eligible host/watermark/attack conditions')
    attacker=root/'reviewer_experiments'/'reviewer_manifest.json'
    if attacker.is_file():
        evidence=json.loads(attacker.read_text())
        ok=evidence.get('expected_attack_rows')==evidence.get('actual_attack_rows') and evidence.get('failed_attack_rows')==0
    else:ok=False
    record('offgrid_and_targeted_attacks',ok,'Supplementary geometric and targeted attack matrix incomplete')
    orth=rows(root/'reviewer_experiments'/'orthogonality_and_integer.csv')
    record('integer_orthogonality_diagnostics',bool(orth) and
        (not full or expected_hosts.issubset({r.get('host') for r in orth})) and
        (not full or {r.get('watermark') for r in orth}==image_wms) and all(
        all(field in r and r[field]!='' for field in (
            'pilot_lattice_unit_updates','pilot_lattice_corrected_blocks',
            'pilot_lattice_unresolved','pilot_coordinate_difference_final_vs_pilot_stage_mean')) for r in orth),
           'Integer pilot counters must come from actual integer_pilot_* params, not missing keys defaulted to zero')
    geo=rows(root/'reviewer_experiments'/'geometry_and_targeted_attacks.csv')
    record('sync_harm_and_raw_vs_aligned',bool(geo) and all(
        'sync_harmed_nc' in r and r.get('pilot_score_raw','')!='' and
        r.get('raw_bit_accuracy','')!='' and r.get('sync_gain_nc','')!=''
        for r in geo if r.get('status')=='ok'),
        'Raw-versus-registered interference and failure diagnostics absent')
    boundary_file=root/'integer_boundaries'/'boundary_manifest.json'
    boundary=json.loads(boundary_file.read_text()) if boundary_file.is_file() else {}
    attempts=rows(root/'integer_boundaries'/'embedding_attempts.csv')
    summary=rows(root/'integer_boundaries'/'rejection_summary.csv')
    required_intensity={'black_0','white_255'} if not full else {
        'black_0','white_255','shadow_0to3','highlight_252to255','gray_128','rgb_channel_saturation'}
    record('integer_rejection_denominators',boundary.get('status')=='complete' and bool(attempts) and
        len(attempts)==boundary.get('attempts') and bool(summary) and
        required_intensity.issubset({r.get('scenario') for r in attempts}) and
        all(r.get('attempted')=='1' and r.get('status') in {'accepted','rejected_by_integer_constraints'}
            for r in attempts) and (not full or len(boundary.get('watermarks',[]))==2),
        'Extreme intensity trials, accurate rejection denominators, or both watermarks missing')
    capacity=rows(root/'reviewer_experiments'/'capacity_sensitivity.csv')
    cap_groups={(r.get('watermark'),r.get('payload_bits'),r.get('attack')) for r in capacity}
    required_sizes={'256','1024','2304','4096'}
    record('capacity_attack_vs_payload',bool(capacity) and required_sizes.issubset({r[1] for r in cap_groups})
        and (not full or expected_hosts.issubset({r.get('host') for r in capacity}))
        and (not full or {r.get('watermark') for r in capacity}==image_wms)
        and {'clean','jpeg40','rotate_2p5'}.issubset({r[2] for r in cap_groups}),
        'Capacity study must vary payload under JPEG and geometric attacks (not only clean)')
    record('capacity_sensitivity',bool(capacity),'Payload capacity experiment absent')
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
    stream_summaries=[rows(p.parent/'ALL_STREAM_realtime_summary.csv') for p in streams]
    required_stream_conditions={('1280x720','clean'),('1280x720','jpeg70')} if full else {('512x512','clean')}
    record('720p_deadline_stress',len(stream_summaries)>=expected_n and all(
       required_stream_conditions.issubset({(r.get('resolution'),r.get('channel')) for r in ss})
       and all(r.get('pace_mode')=='realtime' and r.get('payload_bits')=='4096' and
               r.get('service_ms_p95','')!='' and r.get('effective_output_fps','')!=''
               for r in ss) for ss in stream_summaries),
       'Required 720p JPEG70/clean wall-clock p95/FPS observations missing')
    ta_path=root/'throughput_audit'/'wallclock_throughput_manifest.json'
    ta=json.loads(ta_path.read_text()) if ta_path.is_file() else {}
    record('throughput_formula_reconciled',ta.get('status')=='complete' and ta.get('observations',0)>0 and
           ta.get('throughput_formula','').startswith('4096 * actual effective output FPS'),
           'Failed to recompute payload bps from payload bits times observed effective output FPS')
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
      'independent_real_video_validation':'Real video is optional. Synthetic moving frames do NOT constitute real-video evidence; set VIDEO_INPUT to a verified video',
      'trained_adversarial_generators':'Intentional DCT zeroing included; no unsupported claim that diffusion/generative attacks were run',
      'final_manuscript_abstract_and_table_reconciliation':'Requires manuscript update using final measured CSVs, not code alone',
      'crypto_security_guarantee':'Arnold scrambling is not encryption; ROC and wrong-key tests do not imply cryptographic authenticity',
      'universal_affine_geometric_invariance':'Unsupported; algorithm has a finite geometric candidate family',
      'global_integer_optimality':'Unsupported; integer-domain projection is bounded heuristic',
    }
    reviewer_map={
      'Original Reviewer 1':{'throughput consistency':'final_manuscript_abstract_and_table_reconciliation',
        'matched distortion':'matched_psnr_eligible','holdout and reproducibility':'data_disjoint',
        'uncertainty':'host_cluster_ci','paired benchmark':'matched_psnr_same_case_paired'},
      'Original Reviewer 2':{'integer decoupling':'integer_orthogonality_diagnostics',
        'integer rejection extremes':'integer_rejection_denominators',
        'temporal majority limits':'temporal_voting','joint parameter coupling':'joint_parameter_coupling',
        'strong geometry':'offgrid_and_targeted_attacks','capacity':'capacity_attack_vs_payload',
        'adversarial pilot interference':'offgrid_and_targeted_attacks'},
      'Original Reviewer 5':{'off-grid rotation':'offgrid_and_targeted_attacks',
        'targeted DCT suppression':'offgrid_and_targeted_attacks',
        'ROC/FPR':'threshold_calibrated_detection','recent video baselines':'recent_independent_blind_video_baselines',
        'live 4096-bit FPS':'720p_deadline_stress','matched PSNR':'matched_psnr_same_case_paired'},
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
