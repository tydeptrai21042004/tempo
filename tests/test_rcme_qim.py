"""RCME candidate regression tests; does not claim attack robustness."""
from __future__ import annotations
from dataclasses import replace
import numpy as np
import pytest
from realtime_watermark.core import bit_accuracy
from realtime_watermark.registry import embed, extract
from realtime_watermark.proposals.rcme_qim import RCMEConfig
from realtime_watermark.proposals.rcme_kernel import _allocate_margins
from realtime_watermark.proposals.jilp_qim.config import DEFAULT_CONFIG


def test_global_budget_is_feasible_and_minimizes_continuous_energy():
    step=17.78404328715303
    epsilon=2.28
    reference_fraction=.45
    s=np.array([-8.,-4.,-1.,0.,2.,5.,7.,9.],dtype=np.float64)
    m=_allocate_margins(s,step,reference_fraction,epsilon)
    reference=reference_fraction*step
    assert np.all(m > epsilon)
    assert np.all(m < .49*step+1e-9)
    assert np.sum(1.0/(m-epsilon)**2) <= len(s)/(reference-epsilon)**2 * (1+1e-5)
    energy=lambda rho: np.sum(np.maximum(0,rho-s)**2)
    assert energy(m) <= energy(np.full_like(s,reference)) + 1e-5
    assert np.std(m)>.005  # actual allocation, not a renamed uniform encoder


def test_rcme_roundtrip_and_original_unchanged(host128,wm16):
    watermarked,key=embed('rcme_qim',host128,wm16,seed=10)
    assert bit_accuracy(wm16,extract(watermarked,key))==1.0
    assert key.params['pilot_count']==0
    assert key.params['sync_search_enabled'] is False
    assert key.params['achieved_budget_mean'] <= key.params['reliability_budget_mean']*(1+1e-5)
    assert 'allocated_margins' not in key.params
    assert key.method_id=='rcme_qim'
    original_cfg=replace(DEFAULT_CONFIG,enable_pilots=False,enable_sync_search=False)
    orig,origkey=embed('proposal',host128,wm16,seed=10,method_params=original_cfg.to_dict())
    assert bit_accuracy(wm16,extract(orig,origkey))==1.0
    assert origkey.method_id=='proposal'


def test_rcme_spare_capacity_roundtrip(host192,wm16):
    marked,key=embed('rcme_qim',host192,wm16,seed=90)
    assert np.array_equal(extract(marked,key),wm16)
    assert key.params['schedule_storage']=='regenerate_from_seed'
    assert key.params['capacity']>wm16.size


def test_rcme_rejects_invalid_config(host128,wm16):
    with pytest.raises(ValueError):
        embed('rcme_qim',host128,wm16,method_params={'reference_margin_fraction': .49})
    with pytest.raises(ValueError):
        embed('rcme_qim',host128,wm16,method_params={'enable_pilots':True})
    assert RCMEConfig().step>0
