"""Finite-grid carrier construction from weighted h-Hartley--cosine convolution.

This is an experimental, opt-in payload-carrier extension. It uses the paper's
h=1, gamma(omega)=cos(omega), even finite f[-1]=f[1]=1/4, f[0]=1/2.
For this f, C_gamma g[n]=.5(g[n-1]+g[n+1])+.25(g[n-2]+2g[n]+g[n+2]).
C_gamma/2 has unit DC gain. This uses zero extension at the 8x8 block boundary,
so the infinite-grid factorization identity is not an exact finite-block FFT
identity. No rank/invertibility claim about the original convolution is made.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np
from ...core import dct_basis


def weighted_hartley_cosine_conv1d(x: np.ndarray, axis: int) -> np.ndarray:
    """Normalized (1/2) C_gamma on h=1 grid; zero extension."""
    a=np.asarray(x,dtype=np.float64)
    if a.ndim != 2 or axis not in (0,1):
        raise ValueError('expected a 2-D array and axis 0 or 1')
    result=np.zeros_like(a)
    # Stencil obtained by evaluating the four-term convolution exactly.
    for lag, coeff in ((-2,0.125),(-1,0.25),(0,0.25),(1,0.25),(2,0.125)):
        if lag == 0:
            result += coeff*a
        elif axis == 0:
            if lag > 0: result[lag:,:] += coeff*a[:-lag,:]
            else: result[:lag,:] += coeff*a[-lag:,:]
        else:
            if lag > 0: result[:,lag:] += coeff*a[:,:-lag]
            else: result[:,:lag] += coeff*a[:,-lag:]
    return result


@lru_cache(maxsize=24)
def payload_pair(hartley_mix: float) -> tuple[np.ndarray, np.ndarray]:
    """Return DCT-carrier pair with preserved unit norm and pilot orthogonality.

    A payload direction Q changes by K_gamma Q; pilot direction P stays fixed.
    The projection onto P^perp plus normalization makes <P,Q>=0 exactly up
    to floating-point roundoff. Original two-basis ABI is maintained.
    """
    alpha=float(hartley_mix)
    if not np.isfinite(alpha) or not 0.0<=alpha<=1.0:
        raise ValueError('hartley_mix must lie in [0,1]')
    ba=dct_basis(8,0,1).astype(np.float64)
    bb=dct_basis(8,1,0).astype(np.float64)
    if alpha == 0.0:
        return ba.astype(np.float32),bb.astype(np.float32)
    p=(ba+bb)/np.sqrt(2.0)
    q=(ba-bb)/np.sqrt(2.0)
    # Two-dimensional, symmetric axis average of exactly the 1-D paper
    # operator. Conv is computed once for the fixed carrier, not each frame.
    filtered_q=0.5*(weighted_hartley_cosine_conv1d(q,0)+weighted_hartley_cosine_conv1d(q,1))
    new_q=(1-alpha)*q + alpha*filtered_q
    new_q -= (float(np.sum(new_q*p)) / float(np.sum(p*p))) * p
    new_q /= float(np.linalg.norm(new_q))
    p /= float(np.linalg.norm(p))
    a=(p+new_q)/np.sqrt(2.0)
    b=(p-new_q)/np.sqrt(2.0)
    return a.astype(np.float32),b.astype(np.float32)
