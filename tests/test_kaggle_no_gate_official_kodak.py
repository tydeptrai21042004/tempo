"""End-to-end protocol contracts: separate datasets, only official Kodak, no gate."""
from pathlib import Path
import csv, json, sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from summarize_official_kodak import summarize


def test_full_runner_uses_repository_internal_and_only_kodak_official():
    source=(ROOT/'KAGGLE_TEMPO_STRICT_FULL_ONE_CELL.py').read_text()
    assert "'IMAGE_ATTACK_PRESET':'reviewer46'" in source
    assert "'HOST_LIMIT':'0'" in source
    assert "'HP_HOSTS':'3'" in source
    assert "'WM_SIZE':'64'" in source
    assert "'scripts/kaggle_pipeline.sh','full'" in source
    assert "'12_official_mmwatermark_kodak_24'" in source
    assert "'scripts/prepare_kodak_official_512.py'" in source
    assert "'scripts/download_full_kodak.py'" in source
    assert 'run_full_kodak_external.py' not in source.replace('Never run run_full_kodak_external.py','')
    assert "'scripts/strict_scientific_gate.py'" not in source
    assert "launch('10_strict_scientific_gate'" not in source
    assert 'scientific_gate.json' not in source
    assert "'13_summarize_official_kodak'" in source
    assert "'09b_algorithm_only_4096_720p_jpeg70'" in source
    assert 'EXTERNAL_BASELINES_CSV' not in source


def test_official_summary_fails_on_missing_row_and_reports_all(tmp_path):
    host_names=[f'kodim{i:02d}.bmp' for i in range(1,25)]
    wm_names=['watermark_001_Cc.logo.circle.png','watermark_002_wm.png']
    attack_names=['clean','mmw::jpeg_compression::severity1']
    manifest={'full_coverage':True,'hosts':host_names,'watermarks':wm_names,
              'methods':['proposal'],'complete':True,'expected_compatible_rows':96,'failures':[],
              'preflight_unsupported_conditions':[], 'selected_attack_functions':['jpeg_compression'],
              'severities':[1],'completed_rows':96,'expected_full_requested_rows':96}
    manifest_file=tmp_path/'manifest.json';manifest_file.write_text(json.dumps(manifest))
    raw=tmp_path/'official.csv'
    with raw.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['host','watermark','method_id','attack','payload_bits',
            'clean_embedding_psnr','nc_bipolar','ber','embed_ms','extract_ms'])
        w.writeheader()
        for wm in wm_names:
            for host in host_names:
                for a in attack_names:
                    w.writerow({'host':host,'watermark':wm,'method_id':'proposal','attack':a,
                        'payload_bits':4096,'clean_embedding_psnr':44,'nc_bipolar':1,'ber':0,
                        'embed_ms':4.5,'extract_ms':2.0})
    result=summarize(manifest_file,raw,tmp_path/'out')
    assert result['status']=='COMPLETE' and result['trials']==96
    assert len(list(csv.DictReader((tmp_path/'out/by_watermark_attack.csv').open())))==4
    raw.write_text(raw.read_text().rsplit('\n',2)[0]+'\n')
    with pytest.raises(RuntimeError,match='Expected 96'):
        summarize(manifest_file,raw,tmp_path/'out')
