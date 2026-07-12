from __future__ import annotations

import numpy as np

from ..core import rgb_to_y


def _midband_coords(shape: tuple[int, int], payload_bits: int, repeat: int, seed: int, low: float = 0.08, high: float = 0.45) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    coords: list[tuple[int, int]] = []
    for u in range(h):
        for v in range(w):
            if u == 0 and v == 0:
                continue
            r = (u / max(1, h - 1) + v / max(1, w - 1)) / 2.0
            if low <= r <= high:
                coords.append((u, v))
    needed = int(payload_bits) * int(repeat)
    if len(coords) < needed:
        coords = [(u, v) for u in range(h) for v in range(w) if not (u == 0 and v == 0)]
    if len(coords) < needed:
        raise ValueError(f"Not enough mid-band coefficients: need {needed}, capacity {len(coords)}")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(coords))[:needed]
    return np.asarray([coords[i] for i in order], dtype=np.int32)


def _dwt_subvector_capacity(host_shape: tuple[int, int, int]) -> int:
    h, w = host_shape[:2]
    # 1-level Haar LL, zigzag into one vector, split into two half-length vectors.
    return max(1, (h // 2) * (w // 2) // 2)


def _dct_global_capacity(host_shape: tuple[int, int, int]) -> int:
    h, w = host_shape[:2]
    return max(1, int(0.20 * h * w))


def _repeat_from_capacity(payload_len: int, capacity: int, repeat: int | str, max_auto: int = 3) -> int:
    max_repeat = max(1, int(capacity) // max(1, int(payload_len)))
    if repeat == "auto":
        return max(1, min(int(max_auto), max_repeat))
    return max(1, min(int(repeat), max_repeat))


def _quadtree_leaf_blocks(y: np.ndarray, block_size: int, min_block_size: int = 8, var_threshold: float | None = None) -> list[tuple[int, int, int, float]]:
    """Variance-driven quadtree leaves for the Phadikar region baseline.

    Returns leaves as (row, col, size, score). This implementation deliberately
    uses only deterministic image features so the selected regions can be saved
    or regenerated without external state.
    """
    yy = np.asarray(y, dtype=np.float64)
    h, w = yy.shape
    n = min(h, w)
    # Start from the largest power-of-two square compatible with min_block_size.
    side = 1 << int(np.floor(np.log2(max(int(block_size), n))))
    side = min(side, h - (h % int(block_size)), w - (w % int(block_size)))
    if side < int(block_size):
        side = min(h, w)
    yy = yy[:side, :side]
    if var_threshold is None:
        # Adaptive threshold: split only visually non-flat regions.
        var_threshold = max(12.0, float(np.var(yy)) * 0.10)

    leaves: list[tuple[int, int, int, float]] = []

    def rec(r: int, c: int, size: int) -> None:
        patch = yy[r:r+size, c:c+size]
        gx = np.diff(patch, axis=1) if size > 1 else np.zeros_like(patch)
        gy = np.diff(patch, axis=0) if size > 1 else np.zeros_like(patch)
        score = float(np.var(patch) + 0.10 * (np.mean(np.abs(gx)) + np.mean(np.abs(gy))))
        if size > int(min_block_size) and score >= float(var_threshold):
            half = size // 2
            rec(r, c, half)
            rec(r, c + half, half)
            rec(r + half, c, half)
            rec(r + half, c + half, half)
        else:
            leaves.append((r, c, size, score))

    rec(0, 0, int(side))
    return leaves


def _quadtree_significant_block_indices_from_image(img: np.ndarray, block_size: int, payload_bits: int, repeat: int, seed: int) -> np.ndarray:
    """Select significant 8x8 DCT blocks using quadtree leaf regions.

    The original Phadikar-Maity-Verma method is region-based and uses quadtree
    segmentation. This function implements that algorithmic structure: split the
    image into quadtree leaves, score leaf regions, expand them to 8x8 DCT
    blocks, then select the most significant blocks with deterministic keyed tie
    breaking.
    """
    y = rgb_to_y(np.asarray(img, dtype=np.float64))
    bs = int(block_size)
    h = y.shape[0] - (y.shape[0] % bs)
    w = y.shape[1] - (y.shape[1] % bs)
    y = y[:h, :w]
    br, bc = h // bs, w // bs
    needed = int(payload_bits) * int(repeat)
    if needed > br * bc:
        raise ValueError(f"Not enough DCT blocks: need {needed}, capacity {br * bc}")

    # Pad/crop to the largest square for quadtree segmentation; non-square tails
    # are scored by the same block-energy formula below.
    leaves = _quadtree_leaf_blocks(y[:min(h, w), :min(h, w)], bs, min_block_size=bs)
    region_score = np.zeros((br, bc), dtype=np.float64)
    region_area = np.zeros((br, bc), dtype=np.float64)
    for r, c, size, score in leaves:
        r0, r1 = r // bs, min(br, (r + size + bs - 1) // bs)
        c0, c1 = c // bs, min(bc, (c + size + bs - 1) // bs)
        region_score[r0:r1, c0:c1] += float(score) * float(size * size)
        region_area[r0:r1, c0:c1] += float(size * size)
    region_area[region_area == 0.0] = 1.0
    region_score /= region_area

    rng = np.random.default_rng(int(seed))
    jitter = rng.random(br * bc) * 1e-9
    vals: list[tuple[float, int]] = []
    for rr in range(br):
        for cc in range(bc):
            idx = rr * bc + cc
            block = y[rr*bs:(rr+1)*bs, cc*bs:(cc+1)*bs]
            gx = np.diff(block, axis=1)
            gy = np.diff(block, axis=0)
            local = float(np.var(block) + 0.10 * (np.mean(np.abs(gx)) + np.mean(np.abs(gy))))
            sig = 0.65 * local + 0.35 * float(region_score[rr, cc]) + float(jitter[idx])
            vals.append((sig, idx))
    vals.sort(key=lambda x: (-x[0], x[1]))
    return np.asarray([idx for _, idx in vals[:needed]], dtype=np.int32)


def _region_block_indices_from_image(img: np.ndarray, block_size: int, payload_bits: int, repeat: int, seed: int) -> np.ndarray:
    # Backward-compatible name now uses the quadtree version.
    return _quadtree_significant_block_indices_from_image(img, block_size, payload_bits, repeat, seed)


__all__ = [
    "_midband_coords",
    "_dwt_subvector_capacity",
    "_dct_global_capacity",
    "_repeat_from_capacity",
    "_region_block_indices_from_image",
    "_quadtree_leaf_blocks",
    "_quadtree_significant_block_indices_from_image",
]
