"""Fail-closed audit of the upstream-supported catalog, without inventing missing rows."""
from pathlib import Path
import csv, json, sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from summarize_official_kodak import summarize


def make(tmp_path, missing=None, corrupt=None):
    hosts=[f'kodim{i:02d}.bmp' for i in range(1,25)]
    marks=['watermark_001_Cc.logo.circle.png','watermark_002_wm.png']
    requested=['jpeg_compression','frost']
    omits=[{'attack':'frost','severity':s,'reason':'simulated upstream incompatibility'} for s in range(1,6)]
    m={'complete':True,'full_coverage':False,'hosts':hosts,'watermarks':marks,
       'methods':['proposal'],'selected_attack_functions':requested,'severities':[1,2,3,4,5],
       'expected_full_requested_rows':24*2*11,
       'expected_compatible_rows':24*2*6,
       'completed_rows':24*2*6,
       'preflight_unsupported_conditions':omits,'failures':[]}
    mf=tmp_path/'manifest.json';mf.write_text(json.dumps(m))
    csvpath=tmp_path/'raw.csv'
    labels=['clean']+[f'mmw::jpeg_compression::severity{s}' for s in range(1,6)]
    cols=['host','watermark','method_id','attack','payload_bits',
          'clean_embedding_psnr','nc_bipolar','ber','embed_ms','extract_ms']
    with csvpath.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for wm in marks:
            for host in hosts:
                for a in labels:
                    if missing==(wm,host,a):continue
                    w.writerow({'host':host,'watermark':wm,'method_id':'proposal','attack':a,
                      'payload_bits':4096 if corrupt!=(wm,host,a) else 256,
                      'clean_embedding_psnr':44.16,'nc_bipolar':.83,'ber':.1,
                      'embed_ms':5.0,'extract_ms':2.0})
    return mf,csvpath


def test_all_24_hosts_both_watermarks_all_supported_and_disclosed(tmp_path):
    m,c=make(tmp_path)
    d=summarize(m,c,tmp_path/'summary')
    assert d['status']=='COMPLETE_SUPPORTED_CATALOG_ONLY'
    assert d['completed_supported_rows']==288
    assert d['full_original_catalog'] is False
    assert d['unavailable_upstream_conditions']==5
    assert len(list(csv.DictReader((tmp_path/'summary/upstream_unsupported_conditions.csv').open())))==5


def test_missing_supported_row_rejected(tmp_path):
    mark='watermark_001_Cc.logo.circle.png';host='kodim01.bmp'
    m,c=make(tmp_path,missing=(mark,host,'clean'))
    with pytest.raises(RuntimeError,match='Expected 288'):
        summarize(m,c,tmp_path/'summary')


def test_fake_payload_rejected(tmp_path):
    mark='watermark_001_Cc.logo.circle.png';host='kodim01.bmp'
    m,c=make(tmp_path,corrupt=(mark,host,'clean'))
    with pytest.raises(RuntimeError,match='4096-bit'):
        summarize(m,c,tmp_path/'summary')


def test_runner_has_separate_video_commands_and_disclosed_unsupported():
    s=(ROOT/'KAGGLE_TEMPO_STRICT_FULL_ONE_CELL.py').read_text()
    assert "REPO/'scripts/benchmark_frozen_720p.py',\n" in s
    assert "REPO/'scripts/benchmark_frozen_720p.py','scripts/benchmark_algorithm_only_720p.py'" not in s
    assert "REPO/'scripts/benchmark_algorithm_only_720p.py'" in s
    assert "'--allow-partial'" in s
    assert "official.get('complete')" in s
    assert "official.get('full_coverage',False)" in s
    assert 'strict_scientific_gate.py' not in s
