"""Opt-in frequency-preserving sparse Hartley proposal regression tests."""
from dataclasses import replace
import numpy as np
import pytest
from realtime_watermark.core import dct_basis, make_demo_host, make_demo_watermark
from realtime_watermark.registry import embed, extract
from realtime_watermark.proposals.jilp_qim.config import DEFAULT_CONFIG
from realtime_watermark.proposals.jilp_qim.sparse_hartley_engine import sparse_carrier
from realtime_watermark.proposals.jilp_qim import method

@pytest.mark.parametrize('mode',['sparse','frequency_preserving'])
def test_fixed_48_basis(mode):
    ci,cj,coeff=sparse_carrier(.6,48,mode)
    q=np.zeros((8,8));q[ci,cj]=coeff*np.sqrt(2)
    p=(dct_basis(8,0,1)+dct_basis(8,1,0))/np.sqrt(2)
    assert len(coeff)==48
    assert np.max(np.abs(q+q.T))<1e-11
    assert np.linalg.norm(q)==pytest.approx(1.0,abs=1e-10)
    assert float(np.sum(p*q))==pytest.approx(0,abs=1e-10)
    assert np.array_equal(sparse_carrier(.6,48,mode)[2],coeff)

def test_roundtrip_and_serialized_key():
    host=make_demo_host(256,256)
    wm=make_demo_watermark(16)
    cfg=replace(DEFAULT_CONFIG,hartley_mix=.6,hartley_support=48,
                hartley_carrier_mode='frequency_preserving').validated()
    marked,key=embed('proposal',host,wm,seed=720,method_params=cfg.to_dict())
    assert key.params['hartley_carrier_mode']=='frequency_preserving'
    assert key.params['integer_lattice_unresolved']==0
    assert np.array_equal(extract(marked,key),wm)
    # The key is also accepted after JSON roundtrip, without original host.
    import json
    serial=json.loads(json.dumps(key.to_dict())) if hasattr(key,'to_dict') else None
    if serial is not None:
        assert np.array_equal(extract(marked,serial),wm)
    raw=method._normalize_jilp_key(key)
    bits,pos,conf=method._extract_payload_bits_confidence(marked,raw)
    assert len(bits)==wm.size==len(conf)
    assert np.all(np.isfinite(conf))

def test_original_frozen_config_is_unchanged():
    assert DEFAULT_CONFIG.sha256()=='8f54c9e28722d9698396c0390a654aabdbea0ac8da74b8f81adeb7a51e611d92'
    assert 'hartley_carrier_mode' not in DEFAULT_CONFIG.to_dict()
    assert 'hartley_support' not in DEFAULT_CONFIG.to_dict()
    assert 'hartley_mix' not in DEFAULT_CONFIG.to_dict()
    with pytest.raises(ValueError):
        replace(DEFAULT_CONFIG,hartley_carrier_mode='frequency_preserving').validated()
    with pytest.raises(ValueError):
        replace(DEFAULT_CONFIG,hartley_mix=.6,hartley_support=48,hartley_carrier_mode='invalid').validated()
