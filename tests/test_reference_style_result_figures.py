"""No extra attack set, numerical invention, or misleading paper labels."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts'/'export_reference_style_result_figures.py'
spec=importlib.util.spec_from_file_location('reference_result_figures',SCRIPT)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_exact_canonical_45_nonclean_conditions():
    from realtime_watermark.attack_manifest import reviewer_attacks,reviewer_manifest_summary
    ids=[name for name,_ in reviewer_attacks()]
    assert len(ids)==46 and ids[0]=='clean'
    assert len(set(ids))==46
    assert reviewer_manifest_summary()['attacked_conditions']==45
    assert set(module.COMPARISON_ATTACKS).issubset(ids)


def test_binary_decodes_do_not_break_bipolar_nc():
    from realtime_watermark.core import nc_binary
    orig=np.tile(np.array([[0,255],[255,0]],dtype=np.uint8),(32,32))
    recovered=module.binmask(orig)
    assert set(np.unique(recovered))=={0,255}
    assert nc_binary(orig,recovered)==pytest.approx(1.0)
    recovered=module.binmask((orig>127).astype(np.uint8))
    assert nc_binary(orig,recovered)==pytest.approx(1.0)


def test_reject_noncanonical_matched_benchmark(tmp_path):
    import json,csv
    root=tmp_path/'matched';root.mkdir()
    (root/'matched_psnr_protocol.json').write_text(json.dumps({'attack_preset':'unified',
        'frozen_steps_by_watermark':{'logo.png':{'proposal':17}}}))
    with (root/'matched_psnr_per_attack.csv').open('w',newline='') as f:
        csv.DictWriter(f,fieldnames=['host','watermark','method_id','phase']).writeheader()
    with pytest.raises(ValueError,match='Cannot mix attack registries'):
        module.read_benchmark(root)
