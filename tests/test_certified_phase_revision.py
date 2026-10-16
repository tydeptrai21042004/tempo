"""Proof-obligation checks for bounded QIM-phase candidate evaluation."""
import numpy as np
from realtime_watermark.proposals.jilp_qim import phase_registration as P
from realtime_watermark.proposals.jilp_qim.config import JILPQIMConfig,DEFAULT_CONFIG


def test_exact_branch_bound_never_discards_true_winner():
    rng=np.random.default_rng(23)
    field=rng.uniform(10,240,(48,48)).astype(np.float32)
    inds=np.arange(15,dtype=np.int32)
    bits=rng.integers(0,2,size=len(inds),dtype=np.uint8)
    basis=np.zeros((8,8),dtype=np.float32);basis[2,3]=1
    candidates=[(0,ang,0.,0,0) for ang in (-5.,-2.5,-1.,0.,1.,3.5,6.)]
    exact=[P.sparse_phase_score(field,inds,bits,basis,15.,.07,*c) for c in candidates]
    order=sorted(range(len(exact)),key=lambda i:exact[i],reverse=True)
    third=exact[order[2]]-1e-9
    for i,c in enumerate(candidates):
        bound=P.sparse_phase_score(field,inds,bits,basis,15.,.07,*c,score_floor=third)
        if bound==-2.:
            assert exact[i] < third
        else:
            assert np.isclose(bound,exact[i])


def test_opt_in_revisions_leave_original_config_unchanged():
    assert DEFAULT_CONFIG.phase_refinement=='off'
    assert DEFAULT_CONFIG.certified_phase_pruning is False
    assert DEFAULT_CONFIG.step>17.5
    alternative=JILPQIMConfig.from_mapping({**DEFAULT_CONFIG.to_dict(),
                                 'step':17.,'phase_refinement':'selective',
                                 'certified_phase_pruning':True})
    assert alternative.step==17. and alternative.phase_refinement=='selective'
