from __future__ import annotations

import numpy as np

def _vote_bits(bits: np.ndarray, payload_len: int) -> np.ndarray:
    bit_pos = np.arange(len(bits), dtype=np.int32) % int(payload_len)
    votes0 = np.zeros(int(payload_len), dtype=np.int32)
    votes1 = np.zeros(int(payload_len), dtype=np.int32)
    np.add.at(votes0, bit_pos[np.asarray(bits) == 0], 1)
    np.add.at(votes1, bit_pos[np.asarray(bits) == 1], 1)
    return (votes1 >= votes0).astype(np.uint8)

def _embed_diff_pair(c1: float, c2: float, bit: int, margin: float) -> tuple[float, float]:
    diff = float(c1 - c2)
    target = float(margin) if int(bit) == 1 else -float(margin)
    if (int(bit) == 1 and diff >= target) or (int(bit) == 0 and diff <= target):
        return c1, c2
    delta = target - diff
    return c1 + delta / 2.0, c2 - delta / 2.0

def _embed_parity_scalar(value: float, step: float, bit: int) -> float:
    q0 = int(np.rint(value / step))
    if (q0 & 1) == int(bit):
        q = q0
    else:
        q_up = q0 + 1
        q_down = q0 - 1
        q = q_up if abs(q_up * step - value) <= abs(q_down * step - value) else q_down
    return float(max(0.0, q * step))

__all__ = ["_vote_bits", "_embed_diff_pair", "_embed_parity_scalar"]
