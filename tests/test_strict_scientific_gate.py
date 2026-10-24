from __future__ import annotations
import csv, json, sys
from pathlib import Path
import importlib.util
SPEC=importlib.util.spec_from_file_location('strict_scientific_gate',Path(__file__).resolve().parents[1]/'scripts/strict_scientific_gate.py')
MOD=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MOD)


def save_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document),encoding='utf8')


def save_rows(path,rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=rows[0]);wr.writeheader();wr.writerows(rows)


def fake_run(root, *,fps=31.0,p95=20.0,full=True):
    source=root/'MMWatermark-Robustness/perturbations/image_perturb.py'
    source.parent.mkdir(parents=True,exist_ok=True)
    source.write_text('\n'.join(f'def {name}(x,severity):return x' for name in MOD.OFFICIAL_IMAGE_ATTACKS))
    watermarks=['watermark_001_Cc.logo.circle.png','watermark_002_wm.png']
    hosts=[f'host{i}.bmp' for i in range(9)]
    rows=[]
    for h in hosts:
        for w in watermarks:
            for a in ['clean']+[f'mmw::{name}::severity{i}' for name in MOD.OFFICIAL_IMAGE_ATTACKS for i in range(1,6)]:
                rows.append({'host':h,'watermark':w,'method_id':'proposal','attack':a,
                             'clean_embedding_psnr':51,'nc_bipolar':0.7,'ber':0.1,'embed_ms':1,'extract_ms':1})
    if not full:rows.pop()
    manifest={'full_coverage':full,'selected_attack_functions':sorted(MOD.OFFICIAL_IMAGE_ATTACKS),
              'severities':list(range(1,6)),'hosts':hosts,'watermarks':watermarks,
              'methods':['proposal'],'expected_full_requested_rows':2088,
              'preflight_unsupported_conditions':[],'failures':[]}
    base=root/'extra/external_official_mmwatermark'
    save_json(base/'official_mmwatermark_manifest.json',manifest)
    save_rows(base/'official_mmwatermark_raw.csv',rows)
    save_json(root/'extra/frozen_720p_30fps/frozen_fps_manifest.json',{'source_native_resolution':[1280,720],'video':'/tmp/native.mp4','video_sha256':'abc123','cases':[
        {'watermark':w,'frames_generated':300,'frames_processed':300,
         'frames_dropped_queue':0,'frames_dropped_channel':0,'on_time_completion_ratio':1.0,
         'effective_output_fps':fps,'service_ms_p95':p95,'latency_ms_p95':p95,'resolution':'1280x720',
         'channel':'jpeg70','pace_mode':'realtime','method_id':'proposal',
         'payload_bits':4096,'real_video':True,'frozen_proposal':True} for w in watermarks]})
    save_rows(root/'reports/all_methods_all_metrics/summary_by_method_watermark.csv',[
        {'method_id':'proposal','watermark_name':w,'mean_embedding_psnr':51,
         'mean_clean_bit_accuracy':1.0} for w in watermarks])
    save_json(root/'reports/video_sources/corpus.json', {'videos':[
        {'is_real_source_video':True,'native_720p':True,
         'original_sha256':'abc123','original_path':'/tmp/native.mp4'}]})
    save_json(root/'reports/reviewer_completeness.json',{
        'failed_checks':[],'execution_evidence_complete':True,
        'scientifically_complete_for_resubmission':True,
        'requires_external_or_manuscript_action':{}})
    save_json(root/'extra/reviewer_expansion/expansion_ledger.json',{'status':'complete','incomplete_stages':[]})


def test_passes_when_all_true(tmp_path):
    fake_run(tmp_path)
    assert MOD.audit(tmp_path)['status']=='PASS'


def test_missing_attack_trial_is_not_full(tmp_path):
    fake_run(tmp_path,full=False)
    doc=MOD.audit(tmp_path)
    assert doc['status']=='FAIL'
    assert doc['checks']['mmwatermark']['missing_rows']==1


def test_sub_30_fps_fails(tmp_path):
    fake_run(tmp_path,fps=29.9)
    assert MOD.audit(tmp_path)['status']=='FAIL'


def test_slow_p95_fails(tmp_path):
    fake_run(tmp_path,p95=36.0)
    assert MOD.audit(tmp_path)['status']=='FAIL'


def test_unsupported_severity_fails(tmp_path):
    fake_run(tmp_path)
    p=tmp_path/'extra/external_official_mmwatermark/official_mmwatermark_manifest.json'
    x=json.loads(p.read_text());x['preflight_unsupported_conditions']=[{'attack':'frost','severity':3}]
    save_json(p,x)
    assert MOD.audit(tmp_path)['status']=='FAIL'


def test_false_clean_success_fails(tmp_path):
    fake_run(tmp_path)
    p=tmp_path/'reports/all_methods_all_metrics/summary_by_method_watermark.csv'
    rows=MOD.read_csv(p);rows[0]['mean_clean_bit_accuracy']='0.99';save_rows(p,rows)
    assert MOD.audit(tmp_path)['status']=='FAIL'


def test_psnr_below_50_fails(tmp_path):
    fake_run(tmp_path)
    p=tmp_path/'reports/all_methods_all_metrics/summary_by_method_watermark.csv'
    rows=MOD.read_csv(p);rows[0]['mean_embedding_psnr']='49.99';save_rows(p,rows)
    assert MOD.audit(tmp_path)['status']=='FAIL'


def test_missing_reviewer_evidence_fails(tmp_path):
    fake_run(tmp_path)
    (tmp_path/'reports/reviewer_completeness.json').unlink()
    assert MOD.audit(tmp_path)['status']=='FAIL'


def test_new_unclassified_upstream_attack_fails(tmp_path):
    fake_run(tmp_path)
    source=tmp_path/'MMWatermark-Robustness/perturbations/image_perturb.py'
    source.write_text(source.read_text()+'\ndef brand_new_attack(image, severity): return image\n')
    result=MOD.audit(tmp_path)
    assert result['status']=='FAIL'
    assert 'brand_new_attack' in result['checks']['upstream_catalog']['new_unclassified_severity_functions']


def test_non_native_unverified_fps_video_fails(tmp_path):
    fake_run(tmp_path)
    p=tmp_path/'reports/video_sources/corpus.json'
    record=json.loads(p.read_text())
    record['videos'][0]['native_720p']=False
    save_json(p,record)
    assert MOD.audit(tmp_path)['status']=='FAIL'
