from __future__ import annotations

import numpy as np

from .color import _rgb_to_ycbcr_float, _ycbcr_to_rgb_float
try:
    from scipy.fft import dctn as _dctn, idctn as _idctn
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scipy is required for the DCT baselines") from exc


def _dct2_batch(array: np.ndarray) -> np.ndarray:
    return _dctn(np.asarray(array, dtype=np.float64), axes=(-2, -1), norm="ortho")


def _idct2_batch(array: np.ndarray) -> np.ndarray:
    return _idctn(np.asarray(array, dtype=np.float64), axes=(-2, -1), norm="ortho")


def split_blocks_2d(array: np.ndarray, block_size: int = 8) -> tuple[np.ndarray, int, int]:
    """Split a 2-D array into row-major non-overlapping square blocks."""
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("split_blocks_2d expects a 2-D array")
    bs = int(block_size)
    if bs <= 0:
        raise ValueError("block_size must be positive")
    h, w = arr.shape
    h_crop = h - h % bs
    w_crop = w - w % bs
    if h_crop == 0 or w_crop == 0:
        raise ValueError("image is smaller than one block")
    blocks = (
        arr[:h_crop, :w_crop]
        .reshape(h_crop // bs, bs, w_crop // bs, bs)
        .transpose(0, 2, 1, 3)
        .reshape(-1, bs, bs)
    )
    return blocks, h_crop, w_crop


def merge_blocks_2d(blocks: np.ndarray, h_crop: int, w_crop: int, block_size: int = 8) -> np.ndarray:
    """Inverse of :func:`split_blocks_2d` for a cropped image region."""
    bs = int(block_size)
    arr = np.asarray(blocks, dtype=np.float64)
    expected = (int(h_crop) // bs) * (int(w_crop) // bs)
    if arr.shape != (expected, bs, bs):
        raise ValueError(f"block array shape {arr.shape} does not match {(expected, bs, bs)}")
    return (
        arr.reshape(int(h_crop) // bs, int(w_crop) // bs, bs, bs)
        .transpose(0, 2, 1, 3)
        .reshape(int(h_crop), int(w_crop))
    )



def gray_to_dct_blocks(gray: np.ndarray, block_size: int = 8):
    """Return orthonormal DCT blocks for a two-dimensional grayscale image."""
    image = np.asarray(gray, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError("gray_to_dct_blocks expects a 2-D grayscale image")
    blocks, h_crop, w_crop = split_blocks_2d(image, block_size)
    return _dct2_batch(blocks), image, h_crop, w_crop


def dct_blocks_to_gray(
    coeffs: np.ndarray,
    gray_original: np.ndarray,
    h_crop: int,
    w_crop: int,
    block_size: int = 8,
) -> np.ndarray:
    """Reconstruct uint8 grayscale data while preserving unprocessed borders."""
    gray_new = np.asarray(gray_original, dtype=np.float64).copy()
    spatial = _idct2_batch(np.asarray(coeffs, dtype=np.float64))
    gray_new[: int(h_crop), : int(w_crop)] = merge_blocks_2d(
        spatial, int(h_crop), int(w_crop), int(block_size)
    )
    return np.clip(np.rint(gray_new), 0, 255).astype(np.uint8)


def dct_blocks_from_gray(gray: np.ndarray, block_size: int = 8) -> tuple[np.ndarray, int, int]:
    image = np.asarray(gray, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError("dct_blocks_from_gray expects a 2-D grayscale image")
    blocks, h_crop, w_crop = split_blocks_2d(image, block_size)
    return _dct2_batch(blocks), h_crop, w_crop

def rgb_to_dct_blocks(host_rgb: np.ndarray, block_size: int = 8):
    """Return luminance DCT blocks and chroma planes needed for reconstruction."""
    host = np.asarray(host_rgb, dtype=np.uint8)
    y, cb, cr = _rgb_to_ycbcr_float(host)
    y_blocks, h_crop, w_crop = split_blocks_2d(y, block_size)
    coeffs = _dct2_batch(y_blocks)
    return coeffs, y, cb, cr, h_crop, w_crop


def dct_blocks_to_rgb(
    coeffs: np.ndarray,
    y_original: np.ndarray,
    cb: np.ndarray,
    cr: np.ndarray,
    h_crop: int,
    w_crop: int,
    block_size: int = 8,
) -> np.ndarray:
    """Reconstruct RGB while preserving unprocessed border pixels and chroma."""
    y_new = np.asarray(y_original, dtype=np.float64).copy()
    spatial = _idct2_batch(np.asarray(coeffs, dtype=np.float64))
    y_new[: int(h_crop), : int(w_crop)] = merge_blocks_2d(
        spatial, int(h_crop), int(w_crop), int(block_size)
    )
    return _ycbcr_to_rgb_float(y_new, cb, cr)


def dct_blocks_from_rgb(image_rgb: np.ndarray, block_size: int = 8) -> tuple[np.ndarray, int, int]:
    y, _cb, _cr = _rgb_to_ycbcr_float(np.asarray(image_rgb, dtype=np.uint8))
    blocks, h_crop, w_crop = split_blocks_2d(y, block_size)
    return _dct2_batch(blocks), h_crop, w_crop


__all__ = [
    "split_blocks_2d",
    "merge_blocks_2d",
    "gray_to_dct_blocks",
    "dct_blocks_to_gray",
    "dct_blocks_from_gray",
    "rgb_to_dct_blocks",
    "dct_blocks_to_rgb",
    "dct_blocks_from_rgb",
]
