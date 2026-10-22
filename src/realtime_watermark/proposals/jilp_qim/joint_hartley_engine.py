"""Joint DCT-pilot / sparse h-Hartley QIM embedding on overlapping blocks.

Continuous optimization for a shared RGB block:
  minimize ||Delta X||_F^2
  subject to <g.Delta X, Q> = d_payload and <g.Delta X, P> = d_pilot.
P is the symmetric normalized DCT sum, Q the antisymmetric 48-tap carrier.
<P,Q> = 0, ||P||^2 = ||Q||^2 = 1/2. Hence the unique minimizer is
  Delta X[i,j,:] = (2*d_payload*Q[i,j] + 2*d_pilot*P[i,j])*g/||g||^2.
This is a *genuine joint continuous projection* on overlapping blocks, not
an assertion of globally minimum-distortion uint8 embedding. Quantization
and bounded parity closure follow the established proposal implementation.
"""
from __future__ import annotations
import numpy as np
from numba import prange
from ...core import njit
from .engine import _field_rgb, _project_qim_margin, _project_qim_center

@njit(cache=True, fastmath=True, parallel=True)
def embed_sparse_joint(out, indices, payload_bits, pilot_bits, ci, cj, coeff,
                       pilot_basis, payload_basis, rank_t, rank_ch, rank_s,
                       step, rho, pilot_step, pilot_rho, w_crop, eta, center_mode=False):
    """Fuse joint updates on shared pilot/payload blocks (no other block writes)."""
    n = len(indices)
    bc = w_crop // 8
    g0 = 0.299+eta; g1 = 0.587-0.5*eta; g2 = 0.114-0.5*eta
    gg = g0*g0+g1*g1+g2*g2
    p0=g0/gg; p1=g1/gg; p2=g2/gg
    corrections=np.zeros(n,dtype=np.int32)
    fails=np.zeros(n,dtype=np.int32)
    initial_parity=np.zeros(n,dtype=np.int32)
    initial_guard=np.zeros(n,dtype=np.int32)
    final_guard=np.zeros(n,dtype=np.int32)
    for k in prange(n):
        idx=int(indices[k]); r0=(idx//bc)*8; c0=(idx%bc)*8
        # Read original joint carrier coordinates before either perturbation.
        vs=0.0; vp=0.0
        for i in range(8):
            for j in range(8):
                f=_field_rgb(out[r0+i,c0+j,0],out[r0+i,c0+j,1],out[r0+i,c0+j,2],eta)
                vs+=f*pilot_basis[i,j]
        for t in range(len(ci)):
            vp+=_field_rgb(out[r0+ci[t],c0+cj[t],0],out[r0+ci[t],c0+cj[t],1],out[r0+ci[t],c0+cj[t],2],eta)*coeff[t]
        dp=(_project_qim_center(vp,step,int(payload_bits[k])) if center_mode else
            _project_qim_margin(vp,step,int(payload_bits[k]),rho))-vp
        ds=_project_qim_margin(vs,pilot_step,int(pilot_bits[k]),pilot_rho)-vs
        if abs(dp)>1e-9 or abs(ds)>1e-9:
            # Full 8x8 write required by the dense symmetric pilot, but only
            # on pilot/payload overlaps. Non-overlaps retain original kernels.
            for i in range(8):
                for j in range(8):
                    delta=2.0*(dp*payload_basis[i,j]+ds*pilot_basis[i,j])
                    r=r0+i; c=c0+j
                    x0=min(255.0,max(0.0,out[r,c,0]+delta*p0))
                    x1=min(255.0,max(0.0,out[r,c,1]+delta*p1))
                    x2=min(255.0,max(0.0,out[r,c,2]+delta*p2))
                    out[r,c,0]=np.uint8(np.rint(x0))
                    out[r,c,1]=np.uint8(np.rint(x1))
                    out[r,c,2]=np.uint8(np.rint(x2))
            va=0.0
            for t in range(len(ci)):
                va+=_field_rgb(out[r0+ci[t],c0+cj[t],0],out[r0+ci[t],c0+cj[t],1],out[r0+ci[t],c0+cj[t],2],eta)*coeff[t]
        else:
            va=vp
        guard=0.08*step
        qi=int(np.rint(va/step))
        initial_margin=.5*step-abs(va-qi*step)
        if (qi&1)!=int(payload_bits[k]): initial_parity[k]=1
        if initial_margin<guard: initial_guard[k]=1
        # Identical bounded payload-parity closure to the sparse kernel.
        for it in range(64):
            qcur=int(np.rint(va/step))
            margin=.5*step-abs(va-qcur*step)
            if (qcur&1)==int(payload_bits[k]) and margin>=guard: break
            wanted=_project_qim_margin(va,step,int(payload_bits[k]),guard)
            need=1.0 if wanted>va else -1.0
            found=False
            for j in range(len(rank_t)):
                t=rank_t[j]; ch=rank_ch[j]; slope=rank_s[j]
                r=r0+ci[t];c=c0+cj[t]
                d=1 if need*slope>0 else -1
                if (d>0 and out[r,c,ch]<255) or (d<0 and out[r,c,ch]>0):
                    out[r,c,ch]=np.uint8(int(out[r,c,ch])+d)
                    va+=slope*d
                    corrections[k]+=1
                    found=True
                    break
            if not found: break
        qc=int(np.rint(va/step))
        if .5*step-abs(va-qc*step)<guard: final_guard[k]=1
        if (qc&1)!=int(payload_bits[k]): fails[k]=1
    return (int(np.sum(corrections)),int(np.sum(fails)),
            int(np.sum(corrections>0)),int(np.sum(initial_parity)),
            int(np.sum(initial_guard)),int(np.sum(final_guard)))
