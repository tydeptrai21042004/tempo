from __future__ import annotations

import numpy as np

from ..core import rgb_to_y


def _apply_luminance_update(host: np.ndarray, y_new: np.ndarray) -> np.ndarray:
    """Minimum-norm RGB update that approximately realizes the requested luminance."""
    host_f = np.asarray(host, dtype=np.float64)
    y_old = rgb_to_y(host_f)
    dy = np.asarray(y_new, dtype=np.float64) - y_old
    g = np.array([0.299, 0.587, 0.114], dtype=np.float64)
    p = g / float(np.dot(g, g))
    out = host_f + dy[..., None] * p[None, None, :]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _rgb_to_ycbcr_float(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(rgb, dtype=np.float64)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return y, cb, cr


def _ycbcr_to_rgb_float(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    yy = np.asarray(y, dtype=np.float64)
    cbb = np.asarray(cb, dtype=np.float64) - 128.0
    crr = np.asarray(cr, dtype=np.float64) - 128.0
    r = yy + 1.402 * crr
    g = yy - 0.344136 * cbb - 0.714136 * crr
    b = yy + 1.772 * cbb
    return np.clip(np.rint(np.stack([r, g, b], axis=-1)), 0, 255).astype(np.uint8)


__all__ = ["rgb_to_y", "_apply_luminance_update", "_rgb_to_ycbcr_float", "_ycbcr_to_rgb_float"]
