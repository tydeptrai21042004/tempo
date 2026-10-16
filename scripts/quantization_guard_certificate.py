"""Analytical rounding-margin certificate for the *conditional* continuous model.

The returned inequality does not certify the released sequential uint8
embedding as a globally optimal joint integer projection. It certifies that
rounding an ALREADY JOINTLY GUARDED in-box continuous RGB block cannot switch
an intended carrier bit if rho > 0.5*||a||_1.
"""
import json
import numpy as np
from realtime_watermark.proposals.jilp_qim import engine as eng
from realtime_watermark.proposals.jilp_qim.config import DEFAULT_CONFIG
from realtime_watermark.proposals.jilp_qim.method import PAY10_BA, PAY10_BB


def certify(step=DEFAULT_CONFIG.step, pilot_step=DEFAULT_CONFIG.pilot_step,
            rho_frac=DEFAULT_CONFIG.rho_frac, pilot_rho_frac=DEFAULT_CONFIG.pilot_rho_frac,
            eta=DEFAULT_CONFIG.eta):
    g=np.array([.299+eta,.587-.5*eta,.114-.5*eta],dtype=float)
    p=.5*(PAY10_BA-PAY10_BB)
    s=.5*(eng.PILOT_BA-eng.PILOT_BB)
    p_bound=.5*np.sum(np.abs(p))*np.sum(np.abs(g))
    s_bound=.5*np.sum(np.abs(s))*np.sum(np.abs(g))
    return dict(payload=dict(rounding_error_bound=float(p_bound),guard_margin=step*rho_frac,
                             guaranteed_residual_margin=float(step*rho_frac-p_bound),
                             condition_holds=bool(step*rho_frac>p_bound)),
                pilot=dict(rounding_error_bound=float(s_bound),guard_margin=pilot_step*pilot_rho_frac,
                           guaranteed_residual_margin=float(pilot_step*pilot_rho_frac-s_bound),
                           condition_holds=bool(pilot_step*pilot_rho_frac>s_bound)),
                assumptions='Jointly feasible in-range continuous RGB block, componentwise nearest-integer rounding; no claim of integer global optimality or geometric invariance')

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--step',type=float,default=DEFAULT_CONFIG.step)
    args=ap.parse_args();print(json.dumps(certify(step=args.step),indent=2))
