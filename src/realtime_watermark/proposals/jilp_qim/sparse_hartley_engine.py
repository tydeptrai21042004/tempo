"""Experimental sparse weighted Hartley--cosine convolution payload basis.

Uses a pair-symmetric best-M-term support projection of the convolutional
antisymmetric carrier. A sparse kernel is used for BOTH projection and writing.
The baseline 64-point loops are not reused (which would give no speedup).
"""
from functools import lru_cache
import numpy as np
from numba import prange
from ...core import njit,dct_basis
from .hartley_carrier import payload_pair
from .engine import _field_rgb, _project_qim_margin, _qim_margin_reliability

def _spectral_preserving_projection(q_convolution, selected_pairs):
    """Once-per-key constrained least squares in the antisymmetric 8x8 space.

    min ||Z-Qgamma||_F^2 + sum_{u,v} w_uv |DCT2(Z-Q0)[u,v]|^2,
    Z^T=-Z and support(Z) contained in selected mirrored pairs.

    w_uv=1/(1+u+v)^2. This fixed, data-independent weight makes low DCT
    modes more important without learning parameters or attack-specific choices.
    The unconstrained normalization is applied only after the solve.
    """
    d01 = dct_basis(8,0,1).astype(np.float64)
    d10 = dct_basis(8,1,0).astype(np.float64)
    q_original = (d01-d10)/np.sqrt(2.0)
    basis = np.zeros((64,len(selected_pairs)),dtype=np.float64)
    for k,(i,j) in enumerate(selected_pairs):
        basis[i*8+j,k] = 1/np.sqrt(2.0)
        basis[j*8+i,k] = -1/np.sqrt(2.0)
    # DCT matrix operates on vectorized carrier; performed only on a fixed
    # 8x8 kernel during setup and cached, never in the frame-processing loop.
    transform = np.stack([dct_basis(8,u,v).reshape(-1) for u in range(8) for v in range(8)])
    analysis = transform @ basis
    weights = np.array([1.0/(1+u+v)**2 for u in range(8) for v in range(8)],dtype=np.float64)
    lhs = np.eye(len(selected_pairs)) + analysis.T @ (weights[:,None]*analysis)
    rhs = basis.T @ q_convolution.ravel() + analysis.T @ (weights*(transform @ q_original.ravel()))
    amplitudes = np.linalg.solve(lhs,rhs)
    result = (basis@amplitudes).reshape(8,8)
    norm=float(np.linalg.norm(result))
    if norm<=1e-12:
        raise ValueError('degenerate frequency-preserving carrier')
    return result/norm


@lru_cache(maxsize=64)
def sparse_carrier(alpha: float, support: int, mode: str='sparse'):
    """Convolution-derived payload carrier and its exactly sparse coordinates.

    mode='sparse': original convolution best-M antisymmetric pair truncation.
    mode='frequency_preserving': data-free, weighted-DCT reference regularizer
    on the *same* support. Both branches use the same hot-loop JIT kernels.
    """
    support=int(support)
    if support < 2 or support > 56 or support % 2:
        raise ValueError('support must be even 2..56')
    if mode not in ('sparse','frequency_preserving'):
        raise ValueError('unknown Hartley carrier mode')
    ba,bb=payload_pair(float(alpha))
    q=(ba.astype(np.float64)-bb.astype(np.float64))/np.sqrt(2.0)
    q=(q-q.T)*0.5
    # Include a deterministic index tie-breaker for reproducibility.
    pairs=sorted([(i,j) for i in range(8) for j in range(i+1,8)],
                 key=lambda t:(-abs(float(q[t])),t[0],t[1]))
    chosen=pairs[:support//2]
    if mode=='frequency_preserving':
        s=_spectral_preserving_projection(q,chosen)
    else:
        s=np.zeros_like(q)
        for i,j in chosen:
            s[i,j]=q[i,j]
            s[j,i]=q[j,i]
        s/=float(np.linalg.norm(s))
    # Anti-symmetry is an exact subspace invariant of the finite support.
    if not np.allclose(s,-s.T,atol=1e-12):
        raise ArithmeticError('carrier lost antisymmetry')
    coeff=s/np.sqrt(2.0)
    coords=np.argwhere(coeff!=0)
    ci=np.ascontiguousarray(coords[:,0].astype(np.int32))
    cj=np.ascontiguousarray(coords[:,1].astype(np.int32))
    cc=np.ascontiguousarray(coeff[ci,cj].astype(np.float64))
    if len(cc)!=support:
        raise ArithmeticError('unexpected zero coefficient on sparse support')
    return ci,cj,cc


@lru_cache(maxsize=64)
def sparse_leverage(alpha, support, eta, mode: str='sparse'):
    ci,cj,cc=sparse_carrier(alpha,support,mode)
    gs=(0.299+eta,0.587-0.5*eta,0.114-0.5*eta)
    ranks=[(t,c,float(cc[t]*gs[c])) for t in range(len(cc)) for c in range(3)]
    ranks.sort(key=lambda z:-abs(z[2]))
    return (np.array([v[0] for v in ranks],dtype=np.int32),
            np.array([v[1] for v in ranks],dtype=np.int32),
            np.array([v[2] for v in ranks],dtype=np.float64))

@njit(cache=True,fastmath=True,parallel=True)
def embed_sparse(out,indices,bits,ci,cj,coeff,rank_t,rank_ch,rank_s,step,rho,w_crop,eta):
    n=len(indices); bc=w_crop//8
    g0=0.299+eta; g1=0.587-0.5*eta;g2=0.114-0.5*eta
    gg=g0*g0+g1*g1+g2*g2
    p0=g0/gg;p1=g1/gg;p2=g2/gg
    corrections=np.zeros(n,dtype=np.int32)
    fails=np.zeros(n,dtype=np.int32)
    initial_parity=np.zeros(n,dtype=np.int32)
    initial_guard=np.zeros(n,dtype=np.int32)
    final_guard=np.zeros(n,dtype=np.int32)
    for k in prange(n):
        idx=int(indices[k]);r0=(idx//bc)*8;c0=(idx%bc)*8
        v=0.0
        for t in range(len(ci)):
            r=r0+ci[t];c=c0+cj[t]
            f=_field_rgb(out[r,c,0],out[r,c,1],out[r,c,2],eta)
            v+=f*coeff[t]
        target=_project_qim_margin(v,step,int(bits[k]),rho)
        delta=target-v
        if abs(delta)>1e-9:
            va=0.0
            for t in range(len(ci)):
                r=r0+ci[t];c=c0+cj[t]
                d=2.0*delta*coeff[t]
                x0=min(255.0,max(0.0,out[r,c,0]+d*p0))
                x1=min(255.0,max(0.0,out[r,c,1]+d*p1))
                x2=min(255.0,max(0.0,out[r,c,2]+d*p2))
                q0=np.uint8(np.rint(x0));q1=np.uint8(np.rint(x1));q2=np.uint8(np.rint(x2))
                out[r,c,0]=q0;out[r,c,1]=q1;out[r,c,2]=q2
                va+=_field_rgb(q0,q1,q2,eta)*coeff[t]
        else:
            va=v
        guard=0.08*step
        qinitial=int(np.rint(va/step))
        initial_margin=0.5*step-abs(va-qinitial*step)
        if (qinitial&1)!=int(bits[k]): initial_parity[k]=1
        if initial_margin<guard: initial_guard[k]=1
        solved=False
        for it in range(64):
            qcur=int(np.rint(va/step))
            margin=0.5*step-abs(va-qcur*step)
            if (qcur&1)==int(bits[k]) and margin>=guard:
                solved=True;break
            desired=_project_qim_margin(va,step,int(bits[k]),guard)
            need=1.0 if desired>va else -1.0
            found=False
            for j in range(len(rank_t)):
                t=rank_t[j];ch=rank_ch[j];s=rank_s[j]
                r=r0+ci[t];c=c0+cj[t]
                d=1 if need*s>0 else -1
                if (d>0 and out[r,c,ch]<255) or (d<0 and out[r,c,ch]>0):
                    out[r,c,ch]=np.uint8(int(out[r,c,ch])+d)
                    va+=s*d
                    corrections[k]+=1
                    found=True;break
            if not found:break
        qcur=int(np.rint(va/step))
        margin=0.5*step-abs(va-qcur*step)
        if margin<guard: final_guard[k]=1
        if (qcur&1)!=int(bits[k]): fails[k]=1
    return (int(np.sum(corrections)),int(np.sum(fails)),
            int(np.sum(corrections>0)),int(np.sum(initial_parity)),
            int(np.sum(initial_guard)),int(np.sum(final_guard)))

@njit(cache=True,fastmath=True,parallel=True)
def extract_sparse(img,indices,ci,cj,coeff,step,w_crop,eta,off_y,off_x):
    n=len(indices);bc=(w_crop-off_x)//8
    bits=np.empty(n,dtype=np.uint8)
    for k in prange(n):
        idx=int(indices[k]);r0=off_y+(idx//bc)*8;c0=off_x+(idx%bc)*8
        v=0.0
        for t in range(len(ci)):
            r=r0+ci[t];c=c0+cj[t]
            v+=_field_rgb(img[r,c,0],img[r,c,1],img[r,c,2],eta)*coeff[t]
        bits[k]=np.uint8(int(np.rint(v/step))&1)
    return bits

@njit(cache=True,fastmath=True,parallel=True)
def extract_sparse_conf(img,indices,ci,cj,coeff,step,w_crop,eta,off_y,off_x):
    n=len(indices);bc=(w_crop-off_x)//8
    bits=np.empty(n,dtype=np.uint8);conf=np.empty(n,dtype=np.float32)
    for k in prange(n):
        idx=int(indices[k]);r0=off_y+(idx//bc)*8;c0=off_x+(idx%bc)*8
        v=0.0
        for t in range(len(ci)):
            r=r0+ci[t];c=c0+cj[t]
            v+=_field_rgb(img[r,c,0],img[r,c,1],img[r,c,2],eta)*coeff[t]
        bits[k]=np.uint8(int(np.rint(v/step))&1)
        conf[k]=np.float32(2.0*_qim_margin_reliability(v,step))
    return bits,conf
