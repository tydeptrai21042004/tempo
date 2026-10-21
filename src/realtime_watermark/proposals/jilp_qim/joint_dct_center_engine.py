"""Experimental joint QIM embedding for the original dense orthogonal DCT carriers.

This is a true two-constraint continuous minimum-energy RGB projection on
blocks shared by payload and pilot. It does *not* assert discrete global
optimality. Its final uint8 payload parity is closed using the same bounded
unit-coordinate correction rule as the existing center-QIM implementation.
The existing pilot parity closure is then applied to the final image.
Only selected by config.joint_embedding='orthogonal_joint_center'.
"""
from __future__ import annotations
import numpy as np
from numba import prange
from ...core import njit
from .engine import _field_rgb, _project_qim_center, _project_qim_margin


@njit(cache=True, fastmath=True, parallel=True)
def embed_dense_center_joint(out, indices, payload_bits, pilot_bits,
                             payload_ba, payload_bb, pilot_ba, pilot_bb,
                             step, pilot_step, pilot_rho, w_crop, eta):
    g0 = 0.299 + eta
    g1 = 0.587 - 0.5 * eta
    g2 = 0.114 - 0.5 * eta
    gg = g0*g0 + g1*g1 + g2*g2
    p0, p1, p2 = g0/gg, g1/gg, g2/gg
    count = len(indices)
    bc = w_crop // 8
    updates = np.zeros(count, dtype=np.int64)
    unresolved = np.zeros(count, dtype=np.int64)
    initial_bad = np.zeros(count, dtype=np.int64)
    initial_guard = np.zeros(count, dtype=np.int64)
    final_guard = np.zeros(count, dtype=np.int64)
    for k in prange(count):
        block = int(indices[k])
        r0, c0 = (block // bc) * 8, (block % bc) * 8
        cpa = 0.0
        cpb = 0.0
        csa = 0.0
        csb = 0.0
        for i in range(8):
            for j in range(8):
                f = _field_rgb(out[r0+i,c0+j,0], out[r0+i,c0+j,1], out[r0+i,c0+j,2], eta)
                cpa += f*payload_ba[i,j]
                cpb += f*payload_bb[i,j]
                csa += f*pilot_ba[i,j]
                csb += f*pilot_bb[i,j]
        vp = 0.5*(cpa-cpb)
        vs = 0.5*(csa-csb)
        dp = _project_qim_center(vp,step,int(payload_bits[k])) - vp
        ds = _project_qim_margin(vs,pilot_step,int(pilot_bits[k]),pilot_rho) - vs
        va = vp
        if abs(dp) > 1e-9 or abs(ds) > 1e-9:
            pa2 = 0.0
            pb2 = 0.0
            for i in range(8):
                for j in range(8):
                    # Unit-length DCT bases: 2*dp*a_p + 2*ds*a_s.
                    delta = dp*(payload_ba[i,j]-payload_bb[i,j]) + ds*(pilot_ba[i,j]-pilot_bb[i,j])
                    r, c = r0+i, c0+j
                    x0 = min(255.,max(0.,out[r,c,0]+delta*p0))
                    x1 = min(255.,max(0.,out[r,c,1]+delta*p1))
                    x2 = min(255.,max(0.,out[r,c,2]+delta*p2))
                    q0,q1,q2 = np.uint8(np.rint(x0)),np.uint8(np.rint(x1)),np.uint8(np.rint(x2))
                    out[r,c,0],out[r,c,1],out[r,c,2] = q0,q1,q2
                    f = _field_rgb(q0,q1,q2,eta)
                    pa2 += f*payload_ba[i,j]
                    pb2 += f*payload_bb[i,j]
            va = 0.5*(pa2-pb2)
        guard = 0.08*step
        qi = int(np.rint(va/step))
        if (qi & 1) != int(payload_bits[k]): initial_bad[k] = 1
        if 0.5*step-abs(va-qi*step) < guard: initial_guard[k] = 1
        for iteration in range(64):
            q = int(np.rint(va/step))
            if (q & 1) == int(payload_bits[k]) and 0.5*step-abs(va-q*step) >= guard:
                break
            target = _project_qim_center(va,step,int(payload_bits[k]))
            need = 1.0 if target > va else -1.0
            best_abs = 0.0
            best_r = -1
            best_c = -1
            best_ch = -1
            best_dir = 0
            best_slope = 0.0
            for i in range(8):
                for j in range(8):
                    kval = 0.5*(payload_ba[i,j]-payload_bb[i,j])
                    r, c = r0+i,c0+j
                    s0,s1,s2 = kval*g0,kval*g1,kval*g2
                    for ch in range(3):
                        slope = s0 if ch == 0 else (s1 if ch == 1 else s2)
                        d = 1 if need*slope > 0. else -1
                        if ((d > 0 and out[r,c,ch] < 255) or (d < 0 and out[r,c,ch] > 0)) and abs(slope) > best_abs:
                            best_abs=abs(slope)
                            best_r,best_c,best_ch=r,c,ch
                            best_dir,best_slope=d,slope
            if best_r < 0:
                break
            out[best_r,best_c,best_ch] = np.uint8(int(out[best_r,best_c,best_ch])+best_dir)
            va += best_slope*best_dir
            updates[k] += 1
        q = int(np.rint(va/step))
        if (q & 1) != int(payload_bits[k]): unresolved[k] = 1
        if 0.5*step-abs(va-q*step) < guard: final_guard[k] = 1
    return (int(np.sum(updates)),int(np.sum(unresolved)),int(np.sum(updates > 0)),
            int(np.sum(initial_bad)),int(np.sum(initial_guard)),int(np.sum(final_guard)))
