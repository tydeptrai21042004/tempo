from dataclasses import replace
import numpy as np
import pytest
from realtime_watermark.core import bit_accuracy
from realtime_watermark.registry import embed, extract
from realtime_watermark.proposals.jilp_qim.config import DEFAULT_CONFIG
from realtime_watermark.proposals.jilp_qim.hartley_carrier import payload_pair,weighted_hartley_cosine_conv1d
from realtime_watermark.proposals.jilp_qim import engine


def test_hartley_convolution_exact_stencil():
    impulse = np.zeros((8,8)); impulse[3,4] = 1
    a = weighted_hartley_cosine_conv1d(impulse,0)
    assert np.allclose(a[1:6,4],[0.125,0.25,0.25,0.25,0.125])
    assert np.isclose(np.sum(a),1.0)


@pytest.mark.parametrize('alpha',[0.,0.3,0.6,1.0])
def test_payload_carrier_orthogonality(alpha):
    ba,bb=payload_pair(alpha)
    payload=(ba-bb)/np.sqrt(2)
    pilot=(engine.PILOT_BA-engine.PILOT_BB)/np.sqrt(2)
    assert abs(np.sum(payload*pilot)) < 1e-5
    assert abs(np.sum(ba*bb)) < 1e-5
    assert abs(np.linalg.norm(payload)-1) < 1e-5


def test_hartley_config_backward_compatible():
    assert 'hartley_mix' not in DEFAULT_CONFIG.to_dict()
    cfg=replace(DEFAULT_CONFIG,hartley_mix=0.6)
    assert cfg.to_dict()['hartley_mix']==0.6
    with pytest.raises(ValueError): replace(DEFAULT_CONFIG,hartley_mix=1.2).validated()


def test_hartley_blind_clean_roundtrip(host128,wm16):
    cfg=replace(DEFAULT_CONFIG,hartley_mix=0.6)
    marked,key=embed('proposal',host128,wm16,seed=321,method_params=cfg.to_dict())
    assert key.params['hartley_mix']==0.6
    assert bit_accuracy(wm16,extract(marked,key))==1.0
