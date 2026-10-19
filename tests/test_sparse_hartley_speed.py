"""Correctness invariants for the opt-in, compute-limited Hartley convolution carrier."""
from dataclasses import replace
import numpy as np
import pytest
from realtime_watermark.core import bit_accuracy, make_demo_host, make_demo_watermark
from realtime_watermark.registry import embed, extract
from realtime_watermark.proposals.jilp_qim.config import DEFAULT_CONFIG
from realtime_watermark.proposals.jilp_qim.sparse_hartley_engine import sparse_carrier
from realtime_watermark.proposals.jilp_qim import engine, method

@pytest.mark.parametrize('support',[16,32,40,48,56])
def test_support_and_pilot_orthogonality(support):
    ci,cj,cc=sparse_carrier(0.6,support)
    assert len(ci)==support and len(cj)==support
    q=np.zeros((8,8),dtype=np.float64)
    q[ci,cj]=cc*np.sqrt(2)
    pilot=(engine.PILOT_BA-engine.PILOT_BB)/np.sqrt(2)
    assert np.max(np.abs(q+q.T))<1e-9
    assert abs(float((q*pilot).sum()))<1e-7
    assert abs(np.linalg.norm(q)-1)<1e-7

@pytest.mark.parametrize('support',[16,32,40,48])
def test_sparse_registry_roundtrip_and_stream_confidence(support):
    host=make_demo_host(256,256)
    wm=make_demo_watermark(16)
    cfg=replace(DEFAULT_CONFIG,hartley_mix=0.6,hartley_support=support)
    marked,key=embed('proposal',host,wm,seed=41,method_params=cfg.to_dict())
    assert key.params['hartley_support']==support
    assert key.params['integer_lattice_unresolved']==0
    assert key.params['integer_lattice_final_guard_failures']==0
    assert np.array_equal(extract(marked,key),wm)
    params=method._normalize_jilp_key(key)
    bits,pos,conf=method._extract_payload_bits_confidence(marked,params)
    assert len(bits)==wm.size and len(conf)==wm.size
    assert np.all(np.isfinite(conf))
    assert np.all((conf>=0)&(conf<=1+1e-6))

def test_default_config_backward_compatibility():
    assert 'hartley_support' not in DEFAULT_CONFIG.to_dict()
    with pytest.raises(ValueError):
        replace(DEFAULT_CONFIG,hartley_support=41).validated()
