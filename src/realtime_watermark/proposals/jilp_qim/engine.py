from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:  # pragma: no cover
    cv2 = None
    CV2_AVAILABLE = False

from ...core import njit, dct_basis, make_block_indices

BLOCK_SIZE = 8
METHOD_ID = "proposal"

# Payload uses B01-B10. Pilots use B01+B10, represented as BA-BB with
# BB=-B10. These two low-frequency spatial directions are exactly orthogonal.
PILOT_BA = dct_basis(BLOCK_SIZE, 0, 1).astype(np.float32)
PILOT_BB = (-dct_basis(BLOCK_SIZE, 1, 0)).astype(np.float32)

@njit(cache=True, fastmath=True)
def _project_qim_margin(v: float, step: float, bit: int, rho: float) -> float:
    """Minimum-distance projection onto the QIM parity region with margin.

    Unlike center-QIM, this moves the carrier only when it is outside the safe
    decision interval. This is the minimum-distortion projection used by JILP-QIM.
    """
    q0 = int(np.rint(v / step))
    best = v
    bestd = 1e30
    for t in range(-5, 6):
        q = q0 + t
        if (q & 1) != bit:
            continue
        center = q * step
        lo = center - 0.5 * step + rho
        hi = center + 0.5 * step - rho
        if lo > hi:
            y = center
        elif v < lo:
            y = lo
        elif v > hi:
            y = hi
        else:
            y = v
        d = abs(y - v)
        if d < bestd:
            bestd = d
            best = y
    return best

@njit(cache=True, fastmath=True)
def _project_qim_center(v: float, step: float, bit: int) -> float:
    """Nearest center of the requested QIM parity coset."""
    q0 = int(np.rint(v / step))
    best = 0.0
    bestd = 1e30
    for t in range(-5, 6):
        q = q0 + t
        if (q & 1) != bit:
            continue
        y = q * step
        d = abs(y - v)
        if d < bestd:
            bestd = d
            best = y
    return best

@njit(cache=True, fastmath=True)
def _field_rgb(r: float, g: float, b: float, eta: float) -> float:
    y = 0.299 * r + 0.587 * g + 0.114 * b
    opp = r - 0.5 * g - 0.5 * b
    return y + eta * opp

@njit(cache=True, fastmath=True)
def _embed_one_carrier_md(host, out, indices, bits, ba, bb, step, rho, w_crop, block_size, eta):
    n = indices.shape[0]
    bs = block_size
    bc = w_crop // bs

    # Minimum-norm RGB direction for changing F = Y + eta*(R - 0.5G - 0.5B).
    g0 = 0.299 + eta
    g1 = 0.587 - 0.5 * eta
    g2 = 0.114 - 0.5 * eta
    gg = g0 * g0 + g1 * g1 + g2 * g2
    p0 = g0 / gg
    p1 = g1 / gg
    p2 = g2 / gg

    for k in range(n):
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
        target = _project_qim_margin(v, step, int(bits[k]), rho)
        delta = target - v
        if abs(delta) < 1e-9:
            continue
        for i in range(bs):
            for j in range(bs):
                d = delta * (ba[i, j] - bb[i, j])
                rr = r0 + i
                cc = c0 + j
                val0 = out[rr, cc, 0] + d * p0
                val1 = out[rr, cc, 1] + d * p1
                val2 = out[rr, cc, 2] + d * p2
                if val0 < 0.0:
                    val0 = 0.0
                elif val0 > 255.0:
                    val0 = 255.0
                if val1 < 0.0:
                    val1 = 0.0
                elif val1 > 255.0:
                    val1 = 255.0
                if val2 < 0.0:
                    val2 = 0.0
                elif val2 > 255.0:
                    val2 = 255.0
                out[rr, cc, 0] = np.uint8(np.rint(val0))
                out[rr, cc, 1] = np.uint8(np.rint(val1))
                out[rr, cc, 2] = np.uint8(np.rint(val2))

@njit(cache=True, fastmath=True)
def _embed_one_carrier_center(host, out, indices, bits, ba, bb, step, rho, w_crop, block_size, eta):
    n = indices.shape[0]
    bs = block_size
    bc = w_crop // bs

    # Minimum-norm RGB direction for changing F = Y + eta*(R - 0.5G - 0.5B).
    g0 = 0.299 + eta
    g1 = 0.587 - 0.5 * eta
    g2 = 0.114 - 0.5 * eta
    gg = g0 * g0 + g1 * g1 + g2 * g2
    p0 = g0 / gg
    p1 = g1 / gg
    p2 = g2 / gg

    for k in range(n):
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
        target = _project_qim_center(v, step, int(bits[k]))
        delta = target - v
        if abs(delta) < 1e-9:
            continue
        for i in range(bs):
            for j in range(bs):
                d = delta * (ba[i, j] - bb[i, j])
                rr = r0 + i
                cc = c0 + j
                val0 = out[rr, cc, 0] + d * p0
                val1 = out[rr, cc, 1] + d * p1
                val2 = out[rr, cc, 2] + d * p2
                if val0 < 0.0:
                    val0 = 0.0
                elif val0 > 255.0:
                    val0 = 255.0
                if val1 < 0.0:
                    val1 = 0.0
                elif val1 > 255.0:
                    val1 = 255.0
                if val2 < 0.0:
                    val2 = 0.0
                elif val2 > 255.0:
                    val2 = 255.0
                out[rr, cc, 0] = np.uint8(np.rint(val0))
                out[rr, cc, 1] = np.uint8(np.rint(val1))
                out[rr, cc, 2] = np.uint8(np.rint(val2))

@njit(cache=True, fastmath=True)
def _embed_one_carrier_jilp(out, indices, bits, ba, bb, step, rho, w_crop, block_size, eta):
    """One-pass bounded-integer lattice projection.

    The continuous minimum-norm update is immediately closed on the actual
    uint8 lattice inside each block.  This is part of the block solver; it does
    not extract the finished image and does not run a post-hoc repair pass.
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
    corrections = 0
    unresolved = 0

    for k in range(n):
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
        target = _project_qim_margin(v, step, int(bits[k]), rho)
        delta = target - v
        if abs(delta) > 1e-9:
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
                    out[rr, cc, 0] = np.uint8(np.rint(x0))
                    out[rr, cc, 1] = np.uint8(np.rint(x1))
                    out[rr, cc, 2] = np.uint8(np.rint(x2))

        # Close the constraint on the final bounded uint8 pixels.
        ca2 = 0.0
        cb2 = 0.0
        for i in range(bs):
            for j in range(bs):
                rr = r0 + i
                cc = c0 + j
                f = _field_rgb(out[rr, cc, 0], out[rr, cc, 1], out[rr, cc, 2], eta)
                ca2 += f * ba[i, j]
                cb2 += f * bb[i, j]
        va = 0.5 * (ca2 - cb2)
        guard = 0.08 * step
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
            corrections += 1

        if not solved:
            qcur = int(np.rint(va / step))
            if (qcur & 1) != int(bits[k]):
                unresolved += 1
    return corrections, unresolved

@njit(cache=True, fastmath=True)
def _embed_one_carrier_jilp_center(out, indices, bits, ba, bb, step, rho, w_crop, block_size, eta):
    """One-pass bounded-integer lattice projection.

    The continuous minimum-norm update is immediately closed on the actual
    uint8 lattice inside each block.  This is part of the block solver; it does
    not extract the finished image and does not run a post-hoc repair pass.
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
    corrections = 0
    unresolved = 0

    for k in range(n):
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
        target = _project_qim_center(v, step, int(bits[k]))
        delta = target - v
        if abs(delta) > 1e-9:
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
                    out[rr, cc, 0] = np.uint8(np.rint(x0))
                    out[rr, cc, 1] = np.uint8(np.rint(x1))
                    out[rr, cc, 2] = np.uint8(np.rint(x2))

        # Close the constraint on the final bounded uint8 pixels.
        ca2 = 0.0
        cb2 = 0.0
        for i in range(bs):
            for j in range(bs):
                rr = r0 + i
                cc = c0 + j
                f = _field_rgb(out[rr, cc, 0], out[rr, cc, 1], out[rr, cc, 2], eta)
                ca2 += f * ba[i, j]
                cb2 += f * bb[i, j]
        va = 0.5 * (ca2 - cb2)
        guard = 0.08 * step
        solved = False
        for _it in range(64):
            qcur = int(np.rint(va / step))
            center = qcur * step
            margin = 0.5 * step - abs(va - center)
            if (qcur & 1) == int(bits[k]) and margin >= guard:
                solved = True
                break

            desired = _project_qim_center(va, step, int(bits[k]))
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
            corrections += 1

        if not solved:
            qcur = int(np.rint(va / step))
            if (qcur & 1) != int(bits[k]):
                unresolved += 1
    return corrections, unresolved

@njit(cache=True, fastmath=True)
def _extract_one_carrier_bits(img, indices, ba, bb, step, w_crop, block_size, eta, off_y, off_x):
    n = indices.shape[0]
    bs = block_size
    bc = (w_crop - off_x) // bs
    out_bits = np.empty(n, dtype=np.uint8)
    for k in range(n):
        idx = int(indices[k])
        r0 = off_y + (idx // bc) * bs
        c0 = off_x + (idx % bc) * bs
        ca = 0.0
        cb = 0.0
        for i in range(bs):
            for j in range(bs):
                rr = r0 + i
                cc = c0 + j
                f = _field_rgb(img[rr, cc, 0], img[rr, cc, 1], img[rr, cc, 2], eta)
                ca += f * ba[i, j]
                cb += f * bb[i, j]
        v = 0.5 * (ca - cb)
        out_bits[k] = np.uint8(int(np.rint(v / step)) & 1)
    return out_bits

@njit(cache=True, fastmath=True)
def _extract_one_carrier_bits_shift(img, indices, ba, bb, step, w_crop, block_size, eta, off_y, off_x, sy, sx):
    n = indices.shape[0]
    bs = block_size
    bc = (w_crop - off_x) // bs
    h = img.shape[0]
    w = img.shape[1]
    out_bits = np.empty(n, dtype=np.uint8)
    for k in range(n):
        idx = int(indices[k])
        r0 = off_y + (idx // bc) * bs
        c0 = off_x + (idx % bc) * bs
        ca = 0.0
        cb = 0.0
        for i in range(bs):
            for j in range(bs):
                rr = r0 + i + sy
                cc = c0 + j + sx
                if rr < 0 or rr >= h or cc < 0 or cc >= w:
                    r = 0.0
                    g = 0.0
                    b = 0.0
                else:
                    r = img[rr, cc, 0]
                    g = img[rr, cc, 1]
                    b = img[rr, cc, 2]
                f = _field_rgb(r, g, b, eta)
                ca += f * ba[i, j]
                cb += f * bb[i, j]
        v = 0.5 * (ca - cb)
        out_bits[k] = np.uint8(int(np.rint(v / step)) & 1)
    return out_bits

@njit(cache=True, fastmath=True)
def _qim_margin_reliability(v: float, step: float) -> float:
    """QIM-center margin in [0, 0.5]; larger means farther from the boundary."""
    x = v / step
    q = np.rint(x)
    d = abs(x - q)
    r = 0.5 - d
    if r < 0.0:
        return 0.0
    return r

@njit(cache=True, fastmath=True)
def _best_integer_translation_kernel(img, indices, target_bits, ba, bb, step, w_crop, block_size, eta, radius):
    n = indices.shape[0]
    bs = block_size
    bc = w_crop // bs
    h = img.shape[0]
    w = img.shape[1]
    best_score = -1e9
    best_sy = 0
    best_sx = 0
    for sy in range(-radius, radius + 1):
        for sx in range(-radius, radius + 1):
            acc = 0.0
            for k in range(n):
                idx = int(indices[k])
                r0 = (idx // bc) * bs
                c0 = (idx % bc) * bs
                ca = 0.0
                cb = 0.0
                for i in range(bs):
                    for j in range(bs):
                        rr = r0 + i + sy
                        cc = c0 + j + sx
                        if rr < 0 or rr >= h or cc < 0 or cc >= w:
                            r = 0.0
                            g = 0.0
                            b = 0.0
                        else:
                            r = img[rr, cc, 0]
                            g = img[rr, cc, 1]
                            b = img[rr, cc, 2]
                        f = _field_rgb(r, g, b, eta)
                        ca += f * ba[i, j]
                        cb += f * bb[i, j]
                v = 0.5 * (ca - cb)
                bit = int(np.rint(v / step)) & 1
                if bit == int(target_bits[k]):
                    acc += 1.0
                else:
                    acc -= 1.0
            score = acc / n
            if score > best_score:
                best_score = score
                best_sy = sy
                best_sx = sx
    return best_score, best_sy, best_sx

def _crop_shape(shape: tuple[int, int, int] | tuple[int, int], block_size: int = BLOCK_SIZE) -> tuple[int, int]:
    h, w = int(shape[0]), int(shape[1])
    return h - (h % block_size), w - (w % block_size)

def _make_jilp_schedules(
    host_shape: tuple[int, int, int],
    watermark_shape: tuple[int, int],
    *,
    seed: int,
    pilot_count: int,
    block_size: int = BLOCK_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    h_crop, w_crop = _crop_shape(host_shape, block_size)
    capacity = (h_crop // block_size) * (w_crop // block_size)
    payload_len = int(watermark_shape[0] * watermark_shape[1])
    if payload_len > capacity:
        raise ValueError(f"capacity too small: payload={payload_len}, capacity={capacity}")
    payload_indices = make_block_indices(h_crop, w_crop, block_size, payload_len, 1, seed).astype(np.int32)
    rng = np.random.default_rng(int(seed) + 919)
    pilot_count = int(min(max(0, pilot_count), capacity))
    pilot_indices = rng.permutation(capacity)[:pilot_count].astype(np.int32)
    pilot_bits = (np.arange(pilot_count, dtype=np.uint8) & 1)
    if pilot_count > 1:
        pilot_bits = pilot_bits[np.random.default_rng(int(seed) + 1223).permutation(pilot_count)]
    return payload_indices, pilot_indices, pilot_bits.astype(np.uint8), capacity, h_crop, w_crop

def _normalize_key(key: Any) -> dict[str, Any]:
    """Return JILP-QIM parameters from a WatermarkKey or JSON dictionary."""
    if hasattr(key, "params"):
        params = dict(key.params)
        params.setdefault("method_id", getattr(key, "method_id"))
        params.setdefault("host_shape", tuple(getattr(key, "host_shape")))
        params.setdefault("watermark_shape", tuple(getattr(key, "watermark_shape")))
        params.setdefault("seed", int(getattr(key, "seed")))
        params.setdefault("step", float(getattr(key, "step")))
        params.setdefault("arnold_iter", int(getattr(key, "arnold_iter")))
        params.setdefault("arnold_period", int(getattr(key, "arnold_period")))
        return params
    if isinstance(key, dict):
        params = dict(key.get("params", {}))
        params.setdefault("method_id", key.get("method_id"))
        params.setdefault("host_shape", tuple(key.get("host_shape", params.get("host_shape"))))
        params.setdefault("watermark_shape", tuple(key.get("watermark_shape", params.get("watermark_shape"))))
        params.setdefault("seed", int(key.get("seed", params.get("seed", 2026))))
        params.setdefault("step", float(key.get("step", params.get("step", 12.0))))
        params.setdefault("arnold_iter", int(key.get("arnold_iter", params.get("arnold_iter", 7))))
        params.setdefault("arnold_period", int(key.get("arnold_period", params.get("arnold_period", 48))))
        return params
    raise TypeError("Unsupported JILP-QIM key type")

def _rotate_image(img: np.ndarray, deg: float) -> np.ndarray:
    if CV2_AVAILABLE:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(deg), 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return np.asarray(Image.fromarray(img).rotate(float(deg), resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(0, 0, 0)).convert("RGB"), dtype=np.uint8)

def _crop_inverse(img: np.ndarray, frac: float) -> np.ndarray:
    h, w = img.shape[:2]
    dx = int(round(w * frac / 2.0))
    dy = int(round(h * frac / 2.0))
    if dx <= 0 or dy <= 0:
        return img.copy()
    if CV2_AVAILABLE:
        small = cv2.resize(img, (w - 2 * dx, h - 2 * dy), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros_like(img)
        canvas[dy:h - dy, dx:w - dx] = small
        return canvas
    im = Image.fromarray(img)
    small = im.resize((w - 2 * dx, h - 2 * dy), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    canvas.paste(small, (dx, dy))
    return np.asarray(canvas.convert("RGB"), dtype=np.uint8)
