"""Sparse QIM-phase registration for the orthogonal DCT pilot carrier.

All candidate transforms are scored by the same QIM-phase functional, using
64 samples for each pilot rather than remapping the entire image.  Only the
winning transform is rendered.  This is a finite-grid registration method,
not a claim of geometric invariance.
"""
from __future__ import annotations

import math
import numpy as np

from ...core import njit
from . import engine as _eng


@njit(cache=True, fastmath=True)
def _sample_field_bilinear(field, x: float, y: float) -> float:
    """Bilinear sampling of a precomputed scalar field with zero exterior."""
    h, w = field.shape
    xi = int(math.floor(x))
    yi = int(math.floor(y))
    fx = x - xi
    fy = y - yi
    value = 0.0
    for di in range(2):
        yy = yi + di
        if yy < 0 or yy >= h:
            continue
        wy = (1.0 - fy) if di == 0 else fy
        for dj in range(2):
            xx = xi + dj
            if xx < 0 or xx >= w:
                continue
            wx = (1.0 - fx) if dj == 0 else fx
            value += wy * wx * field[yy, xx]
    return value


@njit(cache=True, fastmath=True)
def _integer_translation_phase_score(
    field, indices, pilot_bits, pilot_basis, step, shift_y, shift_x,
    block_size=8, score_floor=-2.0,
):
    """Exact lattice sampling of the same QIM phase score (no interpolation)."""
    n = len(indices)
    if n == 0:
        return -1.0
    h, w = field.shape
    bc = (w - w % block_size) // block_size
    acc = 0.0
    for k in range(n):
        idx = int(indices[k])
        r0 = (idx // bc) * block_size + shift_y
        c0 = (idx % bc) * block_size + shift_x
        v = 0.0
        for i in range(block_size):
            yy = r0 + i
            if yy < 0 or yy >= h:
                continue
            for j in range(block_size):
                xx = c0 + j
                if 0 <= xx < w:
                    v += float(pilot_basis[i, j]) * float(field[yy, xx])
        acc += (1.0 - 2.0 * float(pilot_bits[k])) * math.cos(math.pi * v / step)
        if score_floor > -1.0 and (k + 1) % 8 == 0:
            if (acc + (n - k - 1)) / n < score_floor - 1e-12:
                return -2.0
    return acc / n


@njit(cache=True, fastmath=True)
def sparse_phase_score(
    field, indices, pilot_bits, pilot_basis, step, eta,
    mode, angle_deg, crop_frac, shift_y, shift_x, block_size=8, score_floor=-2.0,
):
    """Compute mean signed QIM phase coherence for a candidate alignment.

    mode 0: rotation about the image center (includes identity/translation);
    mode 1: inverse center crop, implemented as direct coordinate sampling.
    All pilot bits, including saturated or boundary pilots, contribute to
    the same bounded score in [-1,1].
    """
    if mode == 0 and angle_deg == 0.0:
        return _integer_translation_phase_score(
            field, indices, pilot_bits, pilot_basis, step, shift_y, shift_x,
            block_size, score_floor,
        )
    n = len(indices)
    if n == 0:
        return -1.0
    h, w = field.shape
    bc = (w - w % block_size) // block_size
    cx = w / 2.0
    cy = h / 2.0
    rad = angle_deg * (math.pi / 180.0)
    c = math.cos(rad)
    s = math.sin(rad)
    dx = int(round(w * crop_frac / 2.0))
    dy = int(round(h * crop_frac / 2.0))
    sw = w - 2 * dx
    sh = h - 2 * dy
    acc = 0.0
    for k in range(n):
        idx = int(indices[k])
        r0 = (idx // bc) * block_size
        c0 = (idx % bc) * block_size
        v = 0.0
        for i in range(block_size):
            for j in range(block_size):
                xx = c0 + j + shift_x
                yy = r0 + i + shift_y
                if mode == 0:
                    # Destination-to-source mapping of an OpenCV rotation.
                    px = c * (xx - cx) - s * (yy - cy) + cx
                    py = s * (xx - cx) + c * (yy - cy) + cy
                    f = _sample_field_bilinear(field, px, py)
                else:
                    # cv2.resize pixel-center convention for crop restoration.
                    if xx < dx or xx >= w - dx or yy < dy or yy >= h - dy:
                        f = 0.0
                    else:
                        px = (xx - dx + 0.5) * (w / sw) - 0.5
                        py = (yy - dy + 0.5) * (h / sh) - 0.5
                        f = _sample_field_bilinear(field, px, py)
                v += pilot_basis[i, j] * f
        # At the required parity coset center: (-1)^bit cos(pi*v/step) = 1.
        acc += (1.0 - 2.0 * float(pilot_bits[k])) * math.cos(math.pi * v / step)
        # For the unevaluated signed cosine terms, the exact upper bound
        # is +1 each. Only reject a candidate if even that bound cannot
        # exceed the score of an existing fully evaluated candidate.
        if score_floor > -1.0 and (k + 1) % 8 == 0:
            if (acc + (n - k - 1)) / n < score_floor - 1e-12:
                return -2.0
    return acc / n


@njit(cache=True, fastmath=True)
def _identity_phase_rgb(img, indices, bits, basis, step, eta, block_size=8):
    """Same QIM phase functional on the raw image, no full-field allocation."""
    bc = (img.shape[1] - img.shape[1] % block_size)//block_size
    total = 0.0
    for k in range(len(indices)):
        idx = int(indices[k])
        r0 = (idx//bc)*block_size
        c0 = (idx%bc)*block_size
        v = 0.0
        for i in range(block_size):
            for j in range(block_size):
                v += basis[i,j]*_eng._field_rgb(
                    img[r0+i,c0+j,0],img[r0+i,c0+j,1],img[r0+i,c0+j,2],eta)
        total += (1.0-2.0*float(bits[k]))*math.cos(math.pi*v/step)
    return total/max(1,len(indices))


def render_registered(img, mode: int, angle: float, crop: float, sy: int, sx: int):
    """Apply the selected geometric transform only once to the whole image."""
    import cv2
    h, w = img.shape[:2]
    if mode == 0:
        if abs(angle) < 1e-12 and sy == 0 and sx == 0:
            return img
        rad = math.radians(angle)
        c, s = math.cos(rad), math.sin(rad)
        cx, cy = w / 2.0, h / 2.0
        mat = np.array([
            [c, -s, (1-c)*cx + s*cy + c*sx - s*sy],
            [s, c, -s*cx + (1-c)*cy + s*sx + c*sy],
        ], dtype=np.float32)
        return cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    dx = int(round(w * crop / 2.0))
    dy = int(round(h * crop / 2.0))
    if dx <= 0 or dy <= 0:
        return img
    scale_x = w / (w - 2*dx)
    scale_y = h / (h - 2*dy)
    mat = np.array([
        [scale_x, 0.0, (sx-dx+0.5)*scale_x - 0.5],
        [0.0, scale_y, (sy-dy+0.5)*scale_y - 0.5],
    ], dtype=np.float32)
    out = cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    # The inverse-crop map is zero outside the restored canvas.
    yy, xx = np.ogrid[:h, :w]
    valid = ((xx + sx >= dx) & (xx + sx < w - dx)
             & (yy + sy >= dy) & (yy + sy < h - dy))
    out[~valid] = 0
    return out


def register_sparse_phase(
    img, indices, bits, step: float, eta: float,
    *, quick_accept: float = 0.72, translation_radius: int = 8,
    phase_refinement: str = "off", certified_pruning: bool = False,
):
    """Finite-grid phase registration, preserving a single pilot fast path.

    One phase score is used for identity, rotations, crop restoration and
    translations.  A coarse-to-fine translation sweep is just a numerical
    optimization of that score, not a separate voting/acceptance heuristic.
    """
    img = np.asarray(img, dtype=np.uint8)
    indices = np.asarray(indices, dtype=np.int32)
    bits = np.asarray(bits, dtype=np.uint8)
    basis = (0.5 * (_eng.PILOT_BA - _eng.PILOT_BB)).astype(np.float32)
    identity = (0, 0.0, 0.0, 0, 0)
    raw_score = float(_identity_phase_rgb(img, indices, bits, basis, float(step), float(eta)))
    # Preserve a single fast-path decision; translate legacy accuracy threshold
    # into a conservative phase coherence threshold. The phase score itself is
    # the common mathematical objective for all candidates.
    certified_fast = raw_score >= max(0.80, 2.0 * float(quick_accept) - 1.0)
    if certified_fast:
        return img, {
            "path": "raw_fast_jilp", "candidate": "identity", "score": (raw_score+1)/2,
            "phase_score": raw_score, "raw_phase_score": raw_score,
            "pilot_valid": int(len(indices)), "shift_y": 0, "shift_x": 0,
            "registration": "sparse_qim_phase", "warp_count": 0,
        }

    # Compute the RGB-to-field projection only on the recovery path.
    # Linear projection commutes with bilinear interpolation.
    g0, g1, g2 = .299 + eta, .587 - .5*eta, .114 - .5*eta
    field = np.ascontiguousarray((g0 * img[:, :, 0].astype(np.float32)
                                + g1 * img[:, :, 1].astype(np.float32)
                                + g2 * img[:, :, 2].astype(np.float32)))
    def score(candidate, floor=-2.0):
        mode, deg, crop, sy, sx = candidate
        return float(sparse_phase_score(field, indices, bits, basis, float(step), float(eta),
                                        int(mode), float(deg), float(crop), int(sy), int(sx),
                                        score_floor=float(floor)))
    # Sparse angle grid addresses off-grid angles such as 2.5 degrees.
    rotations = [(0, float(a)*0.5, 0.0, 0, 0) for a in range(-12, 13)]
    crops = [(1, 0.0, frac, 0, 0) for frac in (0.10, 0.15, 0.25)]
    candidates = rotations + crops
    if certified_pruning:
        # Exact top-three ranking: a skipped candidate has a mathematical
        # upper bound below the current third-best *fully evaluated* score.
        # The original geometry-selection rule therefore remains unchanged.
        leaders = []
        for cand in candidates:
            floor = leaders[-1][0] if len(leaders) == 3 else -2.0
            value = score(cand, floor=floor)
            if value > -1.5:
                leaders.append((value, cand))
                leaders.sort(reverse=True, key=lambda item: item[0])
                del leaders[3:]
        scored = leaders
    else:
        scored = [(score(cand), cand) for cand in candidates]
    # Keep identity and the two best distinct geometry candidates so a small
    # translation does not spuriously turn into a false angle estimate.
    scored.sort(reverse=True, key=lambda item: item[0])
    geometries = [identity]
    for _, cand in scored:
        if cand not in geometries:
            geometries.append(cand)
        if len(geometries) == 3:
            break

    radius = max(0, int(translation_radius))
    # Coarse-to-fine optimization is performed on the SAME phase objective.
    # Use 24 pilots to rank the coarse translation lattice, then all pilots
    # for the winning neighborhood: no full-image warp during the search.
    short_n = min(24, len(indices))
    short_indices, short_bits = indices[:short_n], bits[:short_n]
    def coarse_score(candidate):
        mode, deg, crop, sy, sx = candidate
        return float(sparse_phase_score(field, short_indices, short_bits, basis,
                                        float(step), float(eta), int(mode),
                                        float(deg), float(crop), int(sy), int(sx)))

    stride = 4 if radius >= 4 else 1
    coarse = sorted(set([-radius, 0, radius] + list(range(-radius, radius + 1, stride))))
    ranked = []
    if certified_pruning:
        # Rank only the top three coarse hypotheses. For any unfinished
        # candidate, each remaining signed-cosine term is bounded by +1,
        # so pruning cannot remove an entry that would enter the top three.
        # Retain first-seen ties just as stable Python sorting does.
        for mode, deg, crop, _, _ in geometries:
            for sy in coarse:
                for sx in coarse:
                    cand = (mode, deg, crop, sy, sx)
                    floor = ranked[-1][0] if len(ranked) == 3 else -2.0
                    value = float(sparse_phase_score(
                        field, short_indices, short_bits, basis, float(step), float(eta),
                        int(mode), float(deg), float(crop), int(sy), int(sx),
                        score_floor=float(floor)))
                    if value < -1.5:
                        continue  # certified incapable of reaching the top three
                    ranked.append((value, cand))
                    ranked.sort(reverse=True, key=lambda item: item[0])
                    del ranked[3:]
    else:
        for mode, deg, crop, _, _ in geometries:
            for sy in coarse:
                for sx in coarse:
                    cand = (mode, deg, crop, sy, sx)
                    ranked.append((coarse_score(cand), cand))
        ranked.sort(reverse=True, key=lambda item: item[0])
    # Verify the top few coarse hypotheses against ALL pilots, so the reduced
    # search subset does not become a separate acceptance rule.
    verified = [(score(cand), cand) for _, cand in ranked[:min(3, len(ranked))]]
    verified.append((raw_score, identity))
    best = max(verified, key=lambda item: item[0])
    mode, deg, crop, sy, sx = best[1]
    for ty in range(max(-radius, sy-stride//2), min(radius, sy+stride//2)+1):
        for tx in range(max(-radius, sx-stride//2), min(radius, sx+stride//2)+1):
            cand = (mode, deg, crop, ty, tx)
            sc = score(cand)
            if sc > best[0]:
                best = (sc, cand)
    # Bounded continuous local refinement: retains the coarse geometry,
    # pilot count and phase objective. No additional pixel-domain warp until
    # the winning transform has been selected.
    if (phase_refinement == "selective" and best[1][0] == 0
            and abs(best[1][1]) >= 0.5 and best[1][3] == 0
            and best[1][4] == 0 and best[0] > 0.30):
        from scipy.optimize import minimize_scalar
        anchor = float(best[1][1])
        opt = minimize_scalar(
            lambda a: -score((0, float(a), 0.0, 0, 0)),
            bounds=(anchor - 0.32, anchor + 0.32), method="bounded",
            options={"maxiter": 10, "xatol": 0.035},
        )
        if opt.success and -float(opt.fun) > best[0] + 0.03:
            best = (-float(opt.fun), (0, float(opt.x), 0.0, 0, 0))
    winner_score, (mode, deg, crop, sy, sx) = best
    # The identity candidate is always eligible, so optimization cannot
    # select a lower phase score than the uncorrected image.
    if winner_score <= raw_score + 1e-9:
        mode, deg, crop, sy, sx = identity
        winner_score = raw_score
    aligned = render_registered(img, mode, deg, crop, sy, sx)
    return aligned, {
        "path": "wide_sync_jilp", "candidate": (f"rot_{deg:g}_shift_{sy}_{sx}" if mode == 0
                                                   else f"crop_{crop:g}_shift_{sy}_{sx}"),
        "score": (winner_score+1)/2, "phase_score": winner_score,
        "raw_phase_score": raw_score, "pilot_valid": int(len(indices)),
        "shift_y": int(sy), "shift_x": int(sx), "rotation": float(deg),
        "crop_fraction": float(crop), "registration": "sparse_qim_phase",
        "phase_refinement": str(phase_refinement),
        "certified_pruning": bool(certified_pruning),
        "warp_count": int((mode,deg,crop,sy,sx) != identity),
    }
