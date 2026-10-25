import json
from pathlib import Path
import numpy as np
from realtime_watermark.attack_manifest import (reviewer_attacks,reviewer_manifest_summary,
    unified_attacks)
from realtime_watermark.core import default_attacks


def test_unified_46_is_under_50_and_subset_of_full():
    sub=reviewer_attacks(); main=unified_attacks()
    ids=[name for name,_ in sub]
    assert len(ids)==46 and len(set(ids))==46 and ids[0]=='clean'
    assert set(ids).issubset(set(name for name,_ in main))
    assert [x[0] for x in default_attacks('reviewer46')]==ids
    assert reviewer_manifest_summary()['attacked_conditions']==45


def test_canonical_operators_match_original_exactly():
    root=Path(__file__).resolve().parents[1]
    a=json.loads((root/'configs/attacks/unified_109.json').read_text())
    b=json.loads((root/'configs/attacks/reviewer_unified_46.json').read_text())
    ref={x['id']:x for x in a['attacks']}
    assert all(ref[e['id']]==e for e in b['attacks'])


def test_rgb_smoke_and_determinism():
    from realtime_watermark.attack_manifest import reviewer_attacks
    rgb=np.arange(64*64*3,dtype=np.uint8).reshape((64,64,3))
    funcs=dict(reviewer_attacks())
    for name in ('clean','jpeg70','rotate_2p5','affine_shear005','gaussian_variance_0p003'):
        out=funcs[name](rgb)
        assert out.dtype==np.uint8 and out.shape==rgb.shape
        assert np.array_equal(out,funcs[name](rgb))


def test_pipeline_uses_single_internal_attack_protocol():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    script=(root/'scripts/kaggle_pipeline.sh').read_text()
    assert 'HP_ATTACK_PRESET="reviewer46"' in script
    assert 'IMAGE_ATTACK_PRESET="reviewer46"' in script
    assert '--attack-preset "$IMAGE_ATTACK_PRESET"' in script
    assert 'scripts/audit_internal_unified_coverage.py' in script
    assert 'TEMPO_RUN_PAPER_NATIVE_SEPARATELY' in script
    assert 'configs/attacks/reviewer_unified_46.json' in script
    # The Kodak suite is separate and preserves its full 109-attack option.
    kodak=(root/'scripts/run_full_kodak_external.py').read_text()
    assert "if profile == 'unified109':" in kodak
    assert 'return unified_attacks()' in kodak


def test_coverage_audit_detects_missing_rows(tmp_path):
    import csv, sys
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'scripts'))
    from audit_internal_unified_coverage import audit
    from realtime_watermark.attack_manifest import reviewer_manifest_summary
    names=reviewer_manifest_summary()['attack_ids']
    a=tmp_path/'all_methods_all_metrics';a.mkdir()
    with (a/'all_watermarks_all_methods_all_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['host','watermark_name','method_id','payload_bits','attack'])
        w.writeheader()
        for name in names:
            w.writerow({'host':'h','watermark_name':'w','method_id':'proposal','payload_bits':4096,'attack':name})
    m=tmp_path/'matched_psnr';m.mkdir()
    with (m/'matched_psnr_per_attack.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['host','watermark','method_id','phase'])
        w.writeheader()
        for name in names:
            w.writerow({'host':'h','watermark':'w','method_id':'proposal','phase':name})
    assert audit(tmp_path)['status']=='COMPLETE'
    with (m/'matched_psnr_per_attack.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['host','watermark','method_id','phase'])
        w.writeheader()
        for name in names[:-1]:
            w.writerow({'host':'h','watermark':'w','method_id':'proposal','phase':name})
    assert audit(tmp_path)['status']=='INCOMPLETE'
