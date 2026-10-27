"""Regression checks for TEMPO-only Kodak and detector-honest video evidence."""
from __future__ import annotations
import csv
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]


def script(name):
    spec=importlib.util.spec_from_file_location('tempotest_'+name.replace('.','_'),ROOT/'scripts'/name)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def csvwrite(path,entries):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf8') as out:
        writer=csv.DictWriter(out,fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)


def fixture_video(tmp_path):
    base=tmp_path/'reports'
    video=base/'video';video.mkdir(parents=True)
    source={'id':'clip','status':'verified','is_real_source_video':True,
            'original_sha256':'a'*64,'prepared_sha256':'b'*64,'native_720p':False}
    corpus=base/'video_sources'/'corpus.json';corpus.parent.mkdir()
    corpus.write_text(json.dumps({'status':'complete','videos':[source]}))
    wm='test_logo.png'
    methods=['proposal','ca_qim_a2_mao2024']
    proto={'input_sha256':'b'*64,'forced_psnr_calibration':False,'common_ecc':False,
           'methods':methods,'codec':'mp4v','frames':2,
           'watermark_sha256':{wm:'c'*64}}
    folder=video/'clip';folder.mkdir()
    (folder/'all_watermarks_video_protocol.json').write_text(json.dumps(proto))
    rows=[];frames=[]
    for method,psnr,cls in [('proposal',44.1,'blind'),('ca_qim_a2_mao2024',38.4,'semi_blind')]:
        file=folder/'wm'/method/'per_frame.csv'
        csvwrite(file,[{'frame':i,'nc':0.5} for i in range(2)])
        rows.append({'video_id':'clip','watermark_name':wm,'method_id':method,
            'detector_type':cls,'source_sha256':'a'*64,
            'derived_512':'True','real_video':'True','forced_psnr_calibration':'False',
            'video_protocol':'common_framewise_adapter_native_default_strength',
            'configuration_sha256':'d'*64,'watermark_path':'/kaggle/input/'+wm,
            'precodec_mean_psnr':psnr,'nc':0.5,'ber':0.25,'frame_count':2,
            'per_frame_csv':f'/kaggle/working/run/reports/video/clip/wm/{method}/per_frame.csv'})
        frames.extend({'video_id':'clip','watermark_name':wm,'method_id':method,'nc':0.5} for _ in range(2))
    csvwrite(video/'all_watermarks_video_summary.csv',rows)
    csvwrite(video/'all_clips_per_frame.csv',frames)
    return video,corpus,base/'video_common_baselines'


def test_preserves_semi_blind_and_unmatched_psnr(tmp_path):
    s=script('export_common_video_baselines.py')
    video,corpus,out=fixture_video(tmp_path)
    manifest=s.export(video,corpus,out,full=False)
    assert manifest['method_detector_classes']['ca_qim_a2_mao2024']=='semi_blind'
    assert manifest['all_methods_blind'] is False
    assert manifest['psnr_matched'] is False
    assert manifest['row_count']==2
    assert s.verify_export(video,corpus,out,full=False)
    exported=s.read_csv(out/'all_watermarks_video_summary.csv')
    assert [float(x['precodec_mean_psnr']) for x in exported]==[44.1,38.4]


def test_reject_corrupt_detector_class_and_export_hash(tmp_path):
    s=script('export_common_video_baselines.py')
    video,corpus,out=fixture_video(tmp_path)
    entries=s.read_csv(video/'all_watermarks_video_summary.csv')
    entries[-1]['detector_type']='blind'
    csvwrite(video/'all_watermarks_video_summary.csv',entries)
    with pytest.raises(ValueError,match='Incorrect detector'):
        s.export(video,corpus,out,full=False)
    entries[-1]['detector_type']='semi_blind'
    csvwrite(video/'all_watermarks_video_summary.csv',entries)
    s.export(video,corpus,out,full=False)
    assert s.verify_export(video,corpus,out,full=False)
    with (out/'all_watermarks_video_summary.csv').open('a') as file:
        file.write('\n')
    assert not s.verify_export(video,corpus,out,full=False)


def test_full_comparison_does_not_reject_semi_blind_methods():
    from realtime_watermark.registry import COMMON_VIDEO_METHODS
    s=script('export_common_video_baselines.py')
    assert len(COMMON_VIDEO_METHODS)==5
    assert s.detector_class('ca_qim_a2_mao2024')=='semi_blind'
    assert s.detector_class('proposal')=='blind'
    assert s.detector_class('dct_dm_qim_chen2001')=='blind'
    assert s.detector_class('dct_spread_spectrum_cox1997')=='non_blind'
    with pytest.raises(ValueError,match='disclosure'):
        s.detector_class('unknown_unverified_method')


def test_gallery_unconditionally_uses_primary_matched_results():
    pipeline=(ROOT/'scripts/kaggle_pipeline.sh').read_text()
    # Only the independent adapter suite remains behind its optional flag.
    assert 'stage "9C2G. VERIFIED BABOON GALLERY FROM PRIMARY MATCHED-PSNR ROWS"' in pipeline
    assert '--matched-dir "$REPORT_ROOT/matched_psnr"' in pipeline
    assert '--matched-dir "$REPORT_ROOT/modern_baselines_matched_psnr"' in pipeline
    assert 'TEMPO_RUN_INDEPENDENT_HAAR_QIM_CONTROL:-0' in pipeline
    assert 'scripts/export_common_video_baselines.py' in pipeline
    assert 'video_common_baselines' in pipeline
    assert 'video_common_blind_baselines' not in (ROOT/'scripts/reviewer_completeness.py').read_text()


def test_external_uses_only_tempo_and_adapter_is_optional():
    bridge=(ROOT/'scripts/run_mmwatermark_in_official_repo.py').read_text()
    official=(ROOT/'scripts/run_official_mmwatermark_tempo.py').read_text()
    pipeline=(ROOT/'scripts/kaggle_pipeline.sh').read_text()
    assert "'--methods','proposal'" in bridge
    assert "selected_methods != ['proposal']" in official
    assert 'diagnose_haar_adapter_video.py' in pipeline
    assert 'adapter_recovery_diagnostic.json' in pipeline
    assert 'RUN_INTERNAL_KODAK_109' not in pipeline
