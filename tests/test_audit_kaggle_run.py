"""Regression tests for the diagnostic audit; no model training or image recomputation."""
import importlib.util
import json
from pathlib import Path

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/audit_kaggle_run.py'
spec=importlib.util.spec_from_file_location('audit_kaggle_run',SCRIPT)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_missing_files_not_reported_as_success(tmp_path):
    run=tmp_path/'TEMPO_FRESH_test'
    (run/'reports').mkdir(parents=True)
    result=mod.audit(run)
    assert result['proposal_image_metrics']==[]
    assert result['official_mmwatermark']=={}
    assert any(x['severity']=='unknown' for x in result['findings'])


def test_incomplete_upstream_not_hidden_by_successful_exit(tmp_path):
    run=tmp_path/'TEMPO_FRESH_test'
    (run/'reports').mkdir(parents=True)
    (run/'extra/external_official_mmwatermark').mkdir(parents=True)
    m={'full_coverage':False,'complete':True,'completed_rows':500,'expected_compatible_rows':500,
       'expected_full_requested_rows':550,
       'preflight_unsupported_conditions':[{'attack':'foo','severity':1}], 'failures':[]}
    (run/'extra/external_official_mmwatermark/official_mmwatermark_manifest.json').write_text(json.dumps(m))
    result=mod.audit(run)
    assert result['official_mmwatermark']['full_coverage'] is False
    assert any(x['severity']=='incomplete' and 'upstream' in x['message'] for x in result['findings'])


def test_fps_and_psnr_are_independent_targets(tmp_path):
    run=tmp_path/'TEMPO_FRESH_test'
    (run/'reports/all_methods_all_metrics').mkdir(parents=True)
    (run/'reports/throughput_audit').mkdir(parents=True)
    (run/'reports/all_methods_all_metrics/summary_by_method_watermark.csv').write_text(
      'method_id,watermark_name,mean_embedding_psnr,mean_clean_bit_accuracy\nproposal,wm.png,45.3,1\n')
    (run/'reports/throughput_audit/wallclock_throughput_audit.csv').write_text(
      'source_kind,resolution,channel,source_video,watermark,effective_output_fps,service_ms_p95\n'
      'real_video,1280x720,jpeg70,bunny.mp4,wm.png,19.5,52\n')
    result=mod.audit(run)
    assert len([x for x in result['findings'] if x['severity']=='performance'])==2
    assert result['real_video_720p_jpeg70']['passes_all_cases'] is False


def test_legacy_manifest_full_status_does_not_certify_missing_rotation(tmp_path):
    from pathlib import Path
    import importlib.util
    spec=importlib.util.spec_from_file_location('audit_again',Path(__file__).resolve().parents[1]/'scripts/audit_kaggle_run.py')
    audit=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    import json
    (tmp_path/'reports').mkdir()
    folder=tmp_path/'extra'/'external_official_mmwatermark'
    folder.mkdir(parents=True)
    (folder/'official_mmwatermark_manifest.json').write_text(json.dumps({
        'full_coverage':True,'complete':True,'selected_attack_functions':['gaussian_noise'],
        'preflight_unsupported_conditions':[],'failures':[], 'completed_rows':1,
        'expected_full_requested_rows':1,
    }))
    info=audit.audit(tmp_path)
    assert info['official_mmwatermark']['reported_full_coverage'] is True
    assert info['official_mmwatermark']['full_coverage'] is False
    assert any('rotate_img' in x['message'] for x in info['findings'])
