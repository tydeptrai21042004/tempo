"""The joint Hartley/DCT candidate is opt-in and leaves the frozen method intact."""
from dataclasses import replace
import numpy as np
import pytest
from realtime_watermark.core import dct_basis, make_demo_host, make_demo_watermark
from realtime_watermark.proposals.jilp_qim.config import DEFAULT_CONFIG, JILPQIMConfig
from realtime_watermark.proposals.jilp_qim.sparse_hartley_engine import sparse_carrier
from realtime_watermark.registry import embed, extract


def _joint():
    return replace(DEFAULT_CONFIG, hartley_mix=.6, hartley_support=48,
                   hartley_carrier_mode='frequency_preserving',
                   joint_embedding='orthogonal_joint').validated()


def test_joint_carriers_are_orthogonal_and_normalized():
    ci, cj, coeff=sparse_carrier(.6,48,'frequency_preserving')
    q=np.zeros((8,8)); q[ci,cj]=coeff
    p=(dct_basis(8,0,1)+dct_basis(8,1,0))/2
    assert np.sum(p*q)==pytest.approx(0,abs=1e-12)
    assert np.sum(p*p)==pytest.approx(.5,abs=1e-10)
    assert np.sum(q*q)==pytest.approx(.5,abs=1e-10)
    dp,ds=.42,-.27
    delta=2*dp*q + 2*ds*p
    assert float(np.sum(delta*q))==pytest.approx(dp,abs=1e-10)
    assert float(np.sum(delta*p))==pytest.approx(ds,abs=3e-8)
    # Minimum squared norm among feasible perturbations (random nullspace).
    rng=np.random.default_rng(2026)
    extra=rng.standard_normal((8,8))
    extra-=2*np.sum(extra*q)*q+2*np.sum(extra*p)*p
    assert np.sum((delta+extra)**2) >= np.sum(delta**2)-1e-10


def test_joint_is_opt_in_and_invalid_combinations_rejected():
    assert _joint().sha256()!=DEFAULT_CONFIG.sha256()
    assert 'joint_embedding' not in DEFAULT_CONFIG.to_dict()
    assert JILPQIMConfig.from_mapping(_joint().to_dict()).joint_embedding=='orthogonal_joint'
    assert DEFAULT_CONFIG.sha256()=='8f54c9e28722d9698396c0390a654aabdbea0ac8da74b8f81adeb7a51e611d92'
    with pytest.raises(ValueError):
        replace(DEFAULT_CONFIG, joint_embedding='orthogonal_joint').validated()
    with pytest.raises(ValueError):
        replace(_joint(), enable_pilots=False).validated()


@pytest.mark.parametrize('seed',[2026,19])
def test_joint_clean_roundtrip(seed):
    host=make_demo_host(256,256)
    wm=make_demo_watermark(16)
    marked,key=embed('proposal',host,wm,seed=seed,method_params=_joint().to_dict())
    assert key.params['joint_embedding']=='orthogonal_joint'
    assert 0 < key.params['joint_overlap_blocks'] <= _joint().pilot_count
    assert key.params['integer_lattice_unresolved']==0
    assert key.params['integer_pilot_unresolved']==0
    assert np.array_equal(extract(marked,key),wm)


def test_ablation_keeps_joint_as_independently_testable_component():
    from scripts.proposal_structural_ablation import structural_variants
    from realtime_watermark.proposals.jilp_qim.config import JILPQIMConfig
    variants=structural_variants(_joint())
    labels={label for label, _, _ in variants}
    assert "without_joint_projection" in labels
    for label, params, _ in variants:
        config=JILPQIMConfig.from_mapping(params)
        if label=="without_joint_projection":
            assert config.joint_embedding=="sequential"
            assert config.hartley_support==48
