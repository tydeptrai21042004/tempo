"""Regression checks for the new sparse convolutional joint center-QIM method."""
import numpy as np
import pytest
from realtime_watermark.core import load_binary_watermark,load_rgb,nc_binary
from realtime_watermark.proposals.jilp_qim.config import JILPQIMConfig
from realtime_watermark.proposals.jilp_qim.method import embed_proposal,extract_jilp_qim
from realtime_watermark.proposals.jilp_qim.sparse_hartley_engine import sparse_carrier
from realtime_watermark.proposals.jilp_qim.engine import PILOT_BA,PILOT_BB
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('support',[24,32,48,56])
def test_sparse_carrier_orthogonal_and_normalized(support):
    ci,cj,coeff=sparse_carrier(.25,support,'frequency_preserving')
    p=.5*(PILOT_BA-PILOT_BB)
    q=np.zeros((8,8),dtype=np.float64);q[ci,cj]=coeff
    assert len(ci)==support
    assert abs(np.sum(q*p))<1e-6
    assert abs(np.sum(q*q)-.5)<1e-6
    assert np.allclose(q,-q.T,atol=1e-7)

@pytest.mark.parametrize('support',[32,48])
def test_sparse_joint_center_clean_recovery(support):
    image=load_rgb(ROOT/'data/user_dataset/host'/'Girl.bmp')
    wm=load_binary_watermark(ROOT/'data/user_dataset/watermark'/'wm.png',size=64)
    cfg=JILPQIMConfig.load(ROOT/'configs/frozen_selected_20261021.json')
    args={**cfg.to_dict(), 'joint_embedding':'orthogonal_joint_sparse_center','hartley_support':support,
          'hartley_mix':.25,'hartley_carrier_mode':'frequency_preserving','phase_fast_threshold':.65}
    marked,key=embed_proposal(image,wm,method_params=args)
    extracted=extract_jilp_qim(marked,key,sync=True)
    assert marked.shape==image.shape
    assert nc_binary(wm,extracted)==1.0
    assert key.params['joint_embedding']=='orthogonal_joint_sparse_center'

@pytest.mark.parametrize('mode',["center_qim","minimum_distortion"])
def test_old_dense_method_configuration_still_supported(mode):
    cfg=JILPQIMConfig(projection_mode=mode,hartley_support=0,joint_embedding='sequential')
    cfg.validated()
