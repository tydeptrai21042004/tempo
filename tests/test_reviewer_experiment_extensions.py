"""Unit tests for evaluation correctness, not just successful script execution."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import pytest


ROOT=Path(__file__).resolve().parents[1]
def script(name):
    p=ROOT/'scripts'/name
    s=importlib.util.spec_from_file_location('exp_'+name.replace('.','_'),p)
    m=importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_temporal_iid_reference_at_independent_half_error():
    from math import comb
    import sys
    sys.path.insert(0,str(ROOT/'scripts'))
    temporal=script('reviewer_temporal_limits.py')
    assert temporal.iid_majority_ber(.5,5)==pytest.approx(.5)
    assert temporal.iid_majority_ber(.1,3)==pytest.approx(.028)


def test_detection_calibration_does_not_see_test_results():
    detector=script('reviewer_detection_protocol.py')
    threshold=detector.calibrate([.50,.55,.63,.64],.05)
    assert threshold>.64
    roc,auc=detector.roc_from_test([
        {'is_positive':True,'score_bit_accuracy':.80},
        {'is_positive':False,'score_bit_accuracy':.55},
    ])
    assert auc==pytest.approx(1.)
    assert roc[0]['fpr']==0.


def test_external_baseline_missing_provenance_rejected():
    importer=script('import_external_baseline_results.py')
    with pytest.raises(ValueError,match='missing columns'):
        importer.validate([{'method_id':'recent'}],{'Girl.bmp'},{'wm.png'})


def test_external_baseline_capacity_and_duplicate_rejected():
    imp=script('import_external_baseline_results.py')
    row={k:'x' for k in imp.REQUIRED}
    row.update(method_id='recent',reference_doi='10.1234/example',
        implementation_url='https://example.org',implementation_commit='commit123',
        detector_class='blind',host='Girl.bmp',watermark='wm.png',attack='jpeg40',
        payload_bits='4096',psnr='45',nc='.8',ber='.1',embed_ms='1',extract_ms='2',
        source_result_sha256='a'*64)
    assert len(imp.validate([row],{'Girl.bmp'},{'wm.png'}))==1
    with pytest.raises(ValueError,match='Duplicate'):
        imp.validate([row,row],{'Girl.bmp'},{'wm.png'})
    bad=dict(row,payload_bits='256')
    with pytest.raises(ValueError,match='4096'):
        imp.validate([bad],{'Girl.bmp'},{'wm.png'})


def test_completion_ledger_never_claims_full_when_empty(tmp_path):
    m=script('reviewer_completeness.py')
    out=m.check(tmp_path,mode='full')
    assert out['execution_evidence_complete'] is False
    assert out['scientifically_complete_for_resubmission'] is False
    assert 'recent_independent_blind_video_baselines' in out['requires_external_or_manuscript_action']
    assert 'main_payload_4096' in out['failed_checks']
