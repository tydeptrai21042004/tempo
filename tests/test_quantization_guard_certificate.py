import numpy as np
from scripts.quantization_guard_certificate import certify

def test_conditional_rounding_certificate_is_numerically_valid():
    r=certify(step=17.0)
    assert r['payload']['condition_holds']
    assert r['pilot']['condition_holds']
    assert r['payload']['guaranteed_residual_margin'] > 0
    assert r['pilot']['guaranteed_residual_margin'] > 0

def test_rounding_holder_inequality_with_random_rgb_blocks():
    from realtime_watermark.proposals.jilp_qim import engine as eng
    from realtime_watermark.proposals.jilp_qim.method import PAY10_BA,PAY10_BB
    eta=.07
    g=np.array([.299+eta,.587-.5*eta,.114-.5*eta])
    basis=.5*(PAY10_BA-PAY10_BB)
    coeff=basis[:,:,None]*g[None,None,:]
    lim=.5*np.sum(np.abs(coeff))
    rng=np.random.default_rng(51)
    for _ in range(50):
        x=rng.uniform(0,255,(8,8,3))
        err=np.rint(x)-x
        assert abs(np.sum(coeff*err))<=lim+1e-9
