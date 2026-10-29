from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from realtime_watermark.registry import ROBUSTNESS_BASELINES, REALTIME_VIDEO_BASELINES
from realtime_watermark.stream_pipeline import run_stream_benchmark, _method_params
from realtime_watermark.core import make_demo_host, make_demo_watermark


def test_disjoint_paper_baseline_sets():
    assert len(ROBUSTNESS_BASELINES)==5
    assert REALTIME_VIDEO_BASELINES == ('dew_langelaar2001',)
    assert not set(ROBUSTNESS_BASELINES)&set(REALTIME_VIDEO_BASELINES)


def test_dew_paper_zeroing_fidelity_mode():
    assert _method_params('dew_langelaar2001',None,paper_dew=True)['embedding_mode']=='paper_zeroing'
    assert _method_params('dew_langelaar2001',None,paper_dew=False)['embedding_mode']=='minimum_energy_removal'


def test_frozen_stream_never_calibrates_on_test_frame(tmp_path:Path):
    frames=[make_demo_host(128,128) for _ in range(2)]
    wm=make_demo_watermark(8)
    with patch('realtime_watermark.stream_pipeline.calibrate_step_to_psnr',
               side_effect=AssertionError('test leakage')):
        result=run_stream_benchmark(frames=frames,watermark=wm,out_dir=tmp_path,
            methods=['dew_langelaar2001'],channels=['clean'],target_fps=1000,
            seed=12,queue_size=2,pace_mode='virtual',
            calibration_mode='frozen',paper_dew=True)
    assert result['protocol']['calibration_mode']=='frozen'
    assert result['calibration'][0]['calibrated_step'] is None
    assert 'paper_zeroing' in result['calibration'][0]['method_params_json']


def test_development_requires_separate_frames(tmp_path):
    with pytest.raises(ValueError,match='development'):
        run_stream_benchmark(frames=[make_demo_host(128,128)],watermark=make_demo_watermark(8),
            out_dir=tmp_path,calibration_mode='development')


def test_reviewer_runner_refuses_learned_native(tmp_path:Path):
    import ast
    src=(Path(__file__).resolve().parents[1] / 'scripts/run_tempo_evidence.py').read_text()
    module=ast.parse(src)
    dictionary=next(x.value for x in module.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SCRIPTS' for t in x.targets))
    assert 'learned_native' not in ast.literal_eval(dictionary)
