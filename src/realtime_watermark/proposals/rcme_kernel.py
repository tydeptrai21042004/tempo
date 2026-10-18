"""RCME kernel: global reliability allocation + exact TEMPO RGB arithmetic.

The allocation models *upper bounds*, not measured channel error rates.
No JPEG/geometric guarantees follow from the variance surrogate alone.
"""
from __future__ import annotations
import numpy as np
try:
    from numba import prange
except ImportError:  # pragma: no cover
    prange = range
from ..core import njit
from .jilp_qim.engine import _field_rgb, _project_qim_margin

@njit(cache=True, fastmath=True)
def _signed_clearances(host, indices, bits, ba, bb, step, w_crop, bs, eta):
    """Signed distance from the nearest requested-bit QIM cell boundary.

    Scalar minimum displacement at margin m is max(0, m - clearance).
    The selected parity-valid center is fixed over m in [0, step/2).
    """
    n = len(indices)
    values = np.empty(n, dtype=np.float64)
    bc = w_crop // bs
    for k in range(n):
        idx = int(indices[k]); r0 = idx // bc * bs; c0 = (idx % bc) * bs
        ca = 0.0; cb = 0.0
        for i in range(bs):
            for j in range(bs):
                rr = r0 + i; cc = c0 + j
                f = _field_rgb(host[rr,cc,0], host[rr,cc,1],host[rr,cc,2], eta)
                ca += f*ba[i,j]; cb += f*bb[i,j]
        v = 0.5*(ca-cb)
        qlow = int(bits[k]) + 2*int(np.floor(0.5*(v/step-int(bits[k]))))
        q = qlow if abs(v-qlow*step) <= abs(v-(qlow+2)*step) else qlow+2
        values[k] = 0.5*step-abs(v-q*step)
    return values

@njit(cache=True, fastmath=True)
def _margin_at_lambda(s, lam, mmin, mmax, eps):
    """Monotone quartic dual minimizer by safeguarded Newton iterations."""
    lo = max(mmin, min(mmax, s))
    if lam <= 0.0:
        return lo
    if (mmax-s)*(mmax-eps)**3 <= lam:
        return mmax
    if (lo-s)*(lo-eps)**3 >= lam:
        return lo
    hi = mmax
    m = 0.5*(lo+hi)
    for _ in range(6):
        d = m-eps
        residual = (m-s)*d*d*d-lam
        if residual > 0:
            hi = m
        else:
            lo = m
        derivative = d*d*(4*m-eps-3*s)
        proposal = m-residual/derivative if derivative > 1e-10 else 0.5*(lo+hi)
        if not (lo < proposal < hi):
            proposal = 0.5*(lo+hi)
        m = proposal
    return m

@njit(cache=True, fastmath=True)
def _allocate_margins(signed_clearance, step, reference_fraction, rounding_epsilon, iterations=8):
    """Solve the globally coupled continuous convex reliability-budget model.

    minimize sum_k max(0,m_k-s_k)^2
    such that sum_k 1/(m_k-eps)^2 <= L/(reference_margin-eps)^2.
    The constant channel second moment cancels from both sides.
    """
    n = len(signed_clearance)
    eps = rounding_epsilon
    mmin = max(eps + 0.08 * step, eps + 1e-6)
    mmax = 0.49*step
    mref = reference_fraction*step
    if not (mmin < mref < mmax):
        raise ValueError('invalid RCME reliability range')
    target = n/(mref-eps)**2
    # λ=0 minimizes only distortion. If it already satisfies the budget,
    # it is optimal with inactive reliability constraint.
    allocations = np.empty(n, dtype=np.float64)
    total = 0.0
    for k in range(n):
        mk = _margin_at_lambda(signed_clearance[k],0.0,mmin,mmax,eps)
        allocations[k]=mk
        total += 1.0/(mk-eps)**2
    if total <= target:
        return allocations
    # An analytic feasible dual upper bound avoids repeated doubling scans.
    # The derivative of the reliability constraint enables safeguarded
    # Newton updates of the ONE global multiplier.
    smin = np.min(signed_clearance)
    left = 0.0
    right = max(1.0, (mmax - smin)*(mmax - eps)**3)
    lam = 0.5*(left+right)
    for outer in range(iterations):
        total = 0.0
        gradient = 0.0
        for k in range(n):
            sk = signed_clearance[k]
            m = _margin_at_lambda(sk, lam, mmin, mmax, eps)
            d = m-eps
            total += 1.0/(d*d)
            if m > max(mmin,sk) + 1e-5 and m < mmax - 1e-5:
                gradient -= 2.0/(d*d*d*d*d*(4*m-eps-3*sk))
        difference = total - target
        if difference > 0.0:
            left = lam
        else:
            right = lam
        # A safeguarded Newton step converges much faster than bisection.
        proposal = lam - difference/gradient if gradient < -1e-16 else 0.5*(left+right)
        lower_safe = left + 0.04*(right-left)
        upper_safe = right - 0.04*(right-left)
        if not (lower_safe < proposal < upper_safe):
            proposal = 0.5*(left+right)
        lam = proposal
    # Use feasible upper dual bracket (right) for conservative budget.
    for k in range(n):
        allocations[k]=_margin_at_lambda(signed_clearance[k],right,mmin,mmax,eps)
    return allocations

@njit(cache=True, fastmath=True, parallel=True)
def _embed_rcme_blocks(out, indices, bits, ba, bb, step, rho_by_block, w_crop, block_size, eta, rounding_epsilon):
    """RCME payload: per-block globally allocated continuous QIM margins.

    The bounded uint8 correction is retained as a safety fallback; its
    heuristic nature is reported and never called globally optimal.
    """
    n = indices.shape[0]
    bs = block_size
    bc = w_crop // bs
    g0 = 0.299 + eta
    g1 = 0.587 - 0.5 * eta
    g2 = 0.114 - 0.5 * eta
    gg = g0 * g0 + g1 * g1 + g2 * g2
    p0 = g0 / gg
    p1 = g1 / gg
    p2 = g2 / gg
    corrections_per_block = np.zeros(n, dtype=np.int64)
    unresolved_per_block = np.zeros(n, dtype=np.int64)
    initial_parity_failures = np.zeros(n, dtype=np.int64)
    initial_guard_failures = np.zeros(n, dtype=np.int64)
    final_guard_failures = np.zeros(n, dtype=np.int64)

    for k in prange(n):
        block_corrections = 0
        idx = int(indices[k])
        r0 = (idx // bc) * bs
        c0 = (idx % bc) * bs
        ca = 0.0
        cb = 0.0
        for i in range(bs):
            for j in range(bs):
                rr = r0 + i
                cc = c0 + j
                f = _field_rgb(out[rr, cc, 0], out[rr, cc, 1], out[rr, cc, 2], eta)
                ca += f * ba[i, j]
                cb += f * bb[i, j]
        v = 0.5 * (ca - cb)
        target = _project_qim_margin(v, step, int(bits[k]), rho_by_block[k])
        delta = target - v
        # Track the exact carrier of the final bounded uint8 samples while they
        # are written. This preserves the original arithmetic order but removes
        # the separate post-write block rescan.
        if abs(delta) > 1e-9:
            ca2 = 0.0
            cb2 = 0.0
            for i in range(bs):
                for j in range(bs):
                    d = delta * (ba[i, j] - bb[i, j])
                    rr = r0 + i
                    cc = c0 + j
                    x0 = out[rr, cc, 0] + d * p0
                    x1 = out[rr, cc, 1] + d * p1
                    x2 = out[rr, cc, 2] + d * p2
                    if x0 < 0.0:
                        x0 = 0.0
                    elif x0 > 255.0:
                        x0 = 255.0
                    if x1 < 0.0:
                        x1 = 0.0
                    elif x1 > 255.0:
                        x1 = 255.0
                    if x2 < 0.0:
                        x2 = 0.0
                    elif x2 > 255.0:
                        x2 = 255.0
                    q0 = np.uint8(np.rint(x0))
                    q1 = np.uint8(np.rint(x1))
                    q2 = np.uint8(np.rint(x2))
                    out[rr, cc, 0] = q0
                    out[rr, cc, 1] = q1
                    out[rr, cc, 2] = q2
                    f2 = _field_rgb(q0, q1, q2, eta)
                    ca2 += f2 * ba[i, j]
                    cb2 += f2 * bb[i, j]
            va = 0.5 * (ca2 - cb2)
        else:
            va = v
        guard = max(0.08 * step, rho_by_block[k] - rounding_epsilon)
        initial_q = int(np.rint(va / step))
        initial_center = initial_q * step
        initial_margin = 0.5 * step - abs(va - initial_center)
        if (initial_q & 1) != int(bits[k]):
            initial_parity_failures[k] = 1
        if initial_margin < guard:
            initial_guard_failures[k] = 1
        solved = False
        for _it in range(64):
            qcur = int(np.rint(va / step))
            center = qcur * step
            margin = 0.5 * step - abs(va - center)
            if (qcur & 1) == int(bits[k]) and margin >= guard:
                solved = True
                break

            desired = _project_qim_margin(va, step, int(bits[k]), guard)
            need = 1.0 if desired > va else -1.0
            best_abs = 0.0
            best_i = -1
            best_j = -1
            best_c = -1
            best_dir = 0
            best_s = 0.0
            for i in range(bs):
                for j in range(bs):
                    kval = 0.5 * (ba[i, j] - bb[i, j])
                    rr = r0 + i
                    cc = c0 + j
                    s0 = kval * g0
                    s1 = kval * g1
                    s2 = kval * g2
                    dr = 1 if need * s0 > 0.0 else -1
                    if ((dr > 0 and out[rr, cc, 0] < 255) or (dr < 0 and out[rr, cc, 0] > 0)) and abs(s0) > best_abs:
                        best_abs = abs(s0); best_i = i; best_j = j; best_c = 0; best_dir = dr; best_s = s0
                    dr = 1 if need * s1 > 0.0 else -1
                    if ((dr > 0 and out[rr, cc, 1] < 255) or (dr < 0 and out[rr, cc, 1] > 0)) and abs(s1) > best_abs:
                        best_abs = abs(s1); best_i = i; best_j = j; best_c = 1; best_dir = dr; best_s = s1
                    dr = 1 if need * s2 > 0.0 else -1
                    if ((dr > 0 and out[rr, cc, 2] < 255) or (dr < 0 and out[rr, cc, 2] > 0)) and abs(s2) > best_abs:
                        best_abs = abs(s2); best_i = i; best_j = j; best_c = 2; best_dir = dr; best_s = s2
            if best_i < 0:
                break
            rr = r0 + best_i
            cc = c0 + best_j
            out[rr, cc, best_c] = np.uint8(int(out[rr, cc, best_c]) + best_dir)
            va += best_s * best_dir
            block_corrections += 1

        if not solved:
            qcur = int(np.rint(va / step))
            if (qcur & 1) != int(bits[k]):
                unresolved_per_block[k] = 1
        qend = int(np.rint(va / step))
        end_margin = 0.5 * step - abs(va - qend * step)
        if end_margin < guard:
            final_guard_failures[k] = 1
        corrections_per_block[k] = block_corrections
    return (
        int(np.sum(corrections_per_block)),
        int(np.sum(unresolved_per_block)),
        int(np.sum(corrections_per_block > 0)),
        int(np.sum(initial_parity_failures)),
        int(np.sum(initial_guard_failures)),
        int(np.sum(final_guard_failures)),
    )

