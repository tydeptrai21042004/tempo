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
    carrier_rows=rows(root/'reviewer_experiments'/'targeted_carrier_interference.csv')
    targets={(r.get('target'),r.get('fraction')) for r in carrier_rows}
    record('targeted_carrier_attenuation',bool(carrier_rows) and
        {('payload','1.0'),('pilot','1.0')}.issubset(targets) and
        all(r.get('scope')=='targeted_public_carrier_probe_not_generic_ranking'
            and r.get('payload_bits')=='4096' for r in carrier_rows),
        'Public payload/pilot carrier attenuation experiment missing; must stay outside shared 46-condition ranking')
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
    # Non-HD sources are intentionally evaluated at 512x512 only. Do not
    # incorrectly demand native 720p observations for every source.
    all_streams_valid=all(
        ('512x512','clean') in {(r.get('resolution'),r.get('channel')) for r in ss}
        and all(r.get('pace_mode')=='realtime' and r.get('payload_bits')=='4096'
                and r.get('service_ms_p95','')!='' and r.get('effective_output_fps','')!=''
                for r in ss) for ss in stream_summaries)
    hd_streams=sum({('1280x720','clean'),('1280x720','jpeg70')}.issubset(
        {(r.get('resolution'),r.get('channel')) for r in ss}) for ss in stream_summaries)
    record('720p_deadline_stress',len(stream_summaries)>=expected_n and
        all_streams_valid and (not full or hd_streams>=2),
        'Required 512x512 on each stream plus 720p clean/JPEG70 on native HD streams missing')
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
    # Genuine downloaded-video requirement: no synthetic replacement counts as evidence.
    cp=root/'video_sources'/'corpus.json'
    corpus=json.loads(cp.read_text()) if cp.is_file() else {}
    clips=corpus.get('videos',[])
    real_ok=(corpus.get('status')=='complete' and bool(clips)
             and all(c.get('status')=='verified' and c.get('is_real_source_video')
                     and len(c.get('original_sha256',''))==64 and c.get('preparation',{}).get('frames_written',0)>0
                     for c in clips))
    if full:real_ok=real_ok and len(clips)>=3 and any(c.get('native_720p') for c in clips)
    record('downloaded_real_video_corpus',real_ok,
           'Real downloaded clips with SHA256/valid decoding are required; full mode needs native 720p')
    evidence=root/'video'/'video_corpus_evidence_manifest.json'
    vm=json.loads(evidence.read_text()) if evidence.exists() else {}
    vrows=rows(root/'video'/'all_watermarks_video_summary.csv')
    frows=rows(root/'video'/'all_clips_per_frame.csv')
    sr=rows(root/'video'/'video_corpus_stream_summary.csv')
    allids={c['id'] for c in clips}
    allpairs={(r.get('video_id'),r.get('watermark_name')) for r in vrows if r.get('method_id')=='proposal'}
    expected_pairs=allids and {(id_,wm) for id_ in allids for wm in image_wms}
    record('video_both_watermarks_all_clips',bool(real_ok and vm.get('status')=='complete' and
           expected_pairs and expected_pairs.issubset(allpairs) and len(frows)>=len(vrows)*2),
           'Each downloaded video must produce per-frame evidence for both watermark files')
    stream_pairs={(r.get('video_id'),r.get('watermark_name')) for r in sr}
    native_ids={c['id'] for c in clips if c.get('native_720p')}
    have_hd={(r.get('video_id'),r.get('watermark_name')) for r in sr
             if r.get('resolution')=='1280x720' and r.get('channel')=='jpeg70'}
    record('real_video_stream_deadlines',bool(sr and expected_pairs and expected_pairs.issubset(stream_pairs)
           and (not full or {(id_,w) for id_ in native_ids for w in image_wms}.issubset(have_hd))
           and all(r.get('service_ms_p95','')!='' and r.get('effective_output_fps','')!='' and
                   r.get('real_video')=='True' for r in sr)),
           'Real-video 4096-bit wall-clock FPS/p95 missing; must include both watermarks on native HD clip')
    record('constant_recovery_accounted',bool(vrows) and all(
        r.get('recovered_is_constant') in {'True','False'} and
        r.get('ber','')!='' and r.get('nc','')!='' for r in vrows),
        'Video benchmark must score constant extracted watermarks, never abort or omit them')
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
        'uncertainty':'host_cluster_ci','paired benchmark':'matched_psnr_same_case_paired'},
      'Original Reviewer 2':{'integer decoupling':'integer_orthogonality_diagnostics',
        'integer rejection extremes':'integer_rejection_denominators',
        'real-video temporal evidence':'video_both_watermarks_all_clips','temporal majority limits':'temporal_voting','joint parameter coupling':'joint_parameter_coupling',
        'strong geometry':'offgrid_and_targeted_attacks','capacity':'capacity_attack_vs_payload',
        'adversarial pilot interference':'targeted_carrier_attenuation'},
      'Original Reviewer 5':{'off-grid rotation':'offgrid_and_targeted_attacks',
        'targeted DCT suppression':'targeted_carrier_attenuation',
        'ROC/FPR':'threshold_calibrated_detection','recent video baselines':'recent_independent_blind_video_baselines',
        'live 4096-bit FPS':'real_video_stream_deadlines','matched PSNR':'matched_psnr_same_case_paired'},
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
