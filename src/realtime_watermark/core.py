from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from io import BytesIO
from typing import Any, Callable
import csv
import json
import math
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageOps

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco

try:
    from skimage.metrics import structural_similarity as _skimage_ssim
    SKIMAGE_AVAILABLE = True
except Exception:  # pragma: no cover
    _skimage_ssim = None
    SKIMAGE_AVAILABLE = False


# -------------------------
# I/O and demo-data helpers
# -------------------------

def load_rgb(path: str | Path, mode: str = "full") -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if mode == "512":
        img = img.resize((512, 512), Image.Resampling.BICUBIC)
    elif mode in {"full", "native"}:
        pass
    else:
        raise ValueError("mode must be '512' or 'full'")
    return np.asarray(img, dtype=np.uint8)


def load_binary_watermark(path: str | Path | None, size: int = 64, threshold: int = 127) -> np.ndarray:
    """Load a non-degenerate binary watermark, preserving alpha-only logos.

    Some transparent PNG logos store all visible structure in the alpha channel
    while their RGB channels are uniformly black.  In that case converting
    directly to luminance silently produces an all-zero payload.  This loader
    selects the varying alpha channel only when RGB luminance is essentially
    flat and alpha is informative, then rejects one-symbol payloads.
    """
    if path is None:
        binary = make_demo_watermark(size)
    else:
        source = Image.open(path)
        gray_source = source.convert("L")
        if "A" in source.getbands():
            rgba = source.convert("RGBA")
            rgb_gray = np.asarray(rgba.convert("RGB").convert("L"), dtype=np.uint8)
            alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8)
            if int(np.ptp(rgb_gray)) <= 1 and int(np.ptp(alpha)) > 1:
                gray_source = Image.fromarray(alpha, mode="L")
        img = gray_source.resize((size, size), Image.Resampling.NEAREST)
        arr = np.asarray(img, dtype=np.uint8)
        binary = np.where(arr >= threshold, 255, 0).astype(np.uint8)
    if np.unique(binary).size < 2:
        raise ValueError(
            "Degenerate watermark payload: the binarized watermark contains only one bit value. "
            "Use a watermark containing both black and white pixels."
        )
    return binary


def watermark_statistics(watermark_binary: np.ndarray) -> dict[str, float | int]:
    bits = (np.asarray(watermark_binary) > 127).astype(np.uint8).ravel()
    ones = int(np.sum(bits))
    zeros = int(bits.size - ones)
    return {
        "payload_bits": int(bits.size),
        "zero_bits": zeros,
        "one_bits": ones,
        "one_fraction": float(ones / max(bits.size, 1)),
        "bit_balance": float(min(zeros, ones) / max(zeros, ones, 1)),
    }


def make_demo_host(width: int = 1920, height: int = 1080) -> np.ndarray:
    """Generate a deterministic RGB host when no image is supplied."""
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    r = 128 + 60 * np.sin(2 * np.pi * (x * 3 + y * 1.5)) + 40 * x
    g = 120 + 70 * np.cos(2 * np.pi * (x * 1.5 - y * 2.0)) + 35 * y
    b = 115 + 50 * np.sin(2 * np.pi * (x * 2.5 + y * 2.5)) + 25 * (1 - x)
    arr = np.stack([r, g, b], axis=-1)
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)


def make_demo_watermark(size: int = 64) -> np.ndarray:
    img = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)
    m = max(2, size // 12)
    d.rectangle([m, m, size - m - 1, size - m - 1], outline=255, width=max(1, size // 16))
    d.ellipse([size//4, size//4, 3*size//4, 3*size//4], outline=255, width=max(1, size // 16))
    d.line([m, size - m, size - m, m], fill=255, width=max(1, size // 16))
    return np.asarray(img, dtype=np.uint8)


def save_image(path: str | Path, arr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path)


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    # Use the union of keys so status/error rows can coexist with successful
    # benchmark rows without silently dropping diagnostic fields.
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key); fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# -------------------------
# Metrics
# -------------------------

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    af = np.asarray(a, dtype=np.float64)
    bf = np.asarray(b, dtype=np.float64)
    mse = float(np.mean((af - bf) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((255.0 * 255.0) / mse))


def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    """Standard local-window SSIM.

    Uses scikit-image when available. The dependency-free branch is kept only as
    a fallback for minimal environments and should not be used for paper tables.
    """
    aa = np.asarray(a, dtype=np.uint8)
    bb = np.asarray(b, dtype=np.uint8)
    if SKIMAGE_AVAILABLE:
        # Use multichannel SSIM for RGB inputs; it is the standard local-window
        # metric expected by image watermarking reviewers.
        if aa.ndim == 3 and aa.shape[-1] == 3:
            return float(_skimage_ssim(aa, bb, channel_axis=-1, data_range=255))
        return float(_skimage_ssim(aa, bb, data_range=255))

    # Fallback: global luminance approximation.
    af = rgb_to_y(np.asarray(a, dtype=np.float64))
    bf = rgb_to_y(np.asarray(b, dtype=np.float64))
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mux = af.mean()
    muy = bf.mean()
    vx = af.var()
    vy = bf.var()
    cov = ((af - mux) * (bf - muy)).mean()
    return float(((2*mux*muy + c1) * (2*cov + c2)) / ((mux*mux + muy*muy + c1) * (vx + vy + c2)))


def bit_accuracy(wm_true: np.ndarray, wm_pred: np.ndarray) -> float:
    a = (np.asarray(wm_true) > 127).astype(np.uint8).ravel()
    b = (np.asarray(wm_pred) > 127).astype(np.uint8).ravel()
    return float(np.mean(a == b))


def ber(wm_true: np.ndarray, wm_pred: np.ndarray) -> float:
    return 1.0 - bit_accuracy(wm_true, wm_pred)


def nc_binary(wm_true: np.ndarray, wm_pred: np.ndarray) -> float:
    """Normalized correlation for binary watermarks in bipolar form.

    Mapping bits to {-1,+1} removes the positive 0/1 cosine bias that gives
    random or constant-one outputs an apparent NC near 1/sqrt(2). For balanced
    binary payloads this is also exactly ``2 * bit_accuracy - 1``.
    """
    a = np.where(np.asarray(wm_true).ravel() > 127, 1.0, -1.0)
    b = np.where(np.asarray(wm_pred).ravel() > 127, 1.0, -1.0)
    if a.size != b.size:
        raise ValueError(f"watermark size mismatch: {a.size} != {b.size}")
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0.0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.clip(np.dot(a, b) / den, -1.0, 1.0))


def nc_binary_legacy_01(wm_true: np.ndarray, wm_pred: np.ndarray) -> float:
    """Deprecated uncentered 0/1 cosine metric retained for audit only."""
    a = (np.asarray(wm_true) > 127).astype(np.float64).ravel()
    b = (np.asarray(wm_pred) > 127).astype(np.float64).ravel()
    if a.size != b.size:
        raise ValueError(f"watermark size mismatch: {a.size} != {b.size}")
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0.0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.clip(np.dot(a, b) / den, -1.0, 1.0))


def rgb_to_y(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


# -------------------------
# Security/scrambling helpers
# -------------------------

@lru_cache(maxsize=32)
def arnold_period(n: int, max_iter: int = 8192) -> int:
    common = {16: 12, 32: 24, 64: 48, 128: 96, 256: 192}
    if int(n) in common:
        return common[int(n)]
    idx = np.arange(n * n, dtype=np.int32).reshape(n, n)
    cur = idx.copy()
    x, y = np.indices((n, n))
    rr = (x + y) % n
    cc = (x + 2 * y) % n
    for p in range(1, max_iter + 1):
        dst = np.empty_like(cur)
        dst[rr, cc] = cur
        cur = dst
        if np.array_equal(cur, idx):
            return p
    raise RuntimeError(f"Arnold period not found for n={n}")


def arnold_transform(mat01: np.ndarray, iterations: int) -> np.ndarray:
    src = np.asarray(mat01, dtype=np.uint8).copy()
    if src.ndim != 2 or src.shape[0] != src.shape[1]:
        raise ValueError("Arnold transform requires a square watermark")
    n = src.shape[0]
    x, y = np.indices((n, n))
    rr = (x + y) % n
    cc = (x + 2 * y) % n
    for _ in range(int(iterations)):
        dst = np.empty_like(src)
        dst[rr, cc] = src
        src = dst
    return src


def arnold_inverse(mat01: np.ndarray, iterations: int, period: int) -> np.ndarray:
    return arnold_transform(mat01, (int(period) - (int(iterations) % int(period))) % int(period))


def effective_repeat(host_shape: tuple[int, int, int], wm_shape: tuple[int, int], block_size: int, repeat: int | str) -> int:
    h, w = host_shape[:2]
    capacity = (h // block_size) * (w // block_size)
    payload = int(wm_shape[0] * wm_shape[1])
    max_repeat = max(1, capacity // payload)
    if repeat == "auto":
        # Large-host robust real-time mode, but gracefully falls back to 512 capacity.
        return min(3, max_repeat)
    return max(1, min(int(repeat), max_repeat))


def make_block_indices(h_crop: int, w_crop: int, block_size: int, payload_bits: int, repeat: int, seed: int) -> np.ndarray:
    br = h_crop // block_size
    bc = w_crop // block_size
    capacity = br * bc
    needed = int(payload_bits) * int(repeat)
    if needed > capacity:
        raise ValueError(f"Not enough blocks: need {needed}, capacity {capacity}")
    rng = np.random.default_rng(int(seed))
    return rng.permutation(capacity)[:needed].astype(np.int32)


def make_freq_indices(h: int, w: int, payload_bits: int, repeat: int, seed: int) -> np.ndarray:
    """Mid-frequency indices for a global DFT-magnitude baseline."""
    coords: list[tuple[int, int]] = []
    # positive quadrant only, avoid DC and very high frequencies
    u0, u1 = max(4, h // 32), max(8, h // 4)
    v0, v1 = max(4, w // 32), max(8, w // 4)
    for u in range(u0, u1):
        for v in range(v0, v1):
            # Use annulus-ish band to avoid too low or too high energy.
            ru = u / h
            rv = v / w
            r = math.sqrt(ru * ru + rv * rv)
            if 0.03 <= r <= 0.20:
                coords.append((u, v))
    needed = payload_bits * repeat
    if len(coords) < needed:
        raise ValueError(f"Not enough DFT coordinates: need {needed}, capacity {len(coords)}")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(coords))[:needed]
    return np.asarray([coords[i] for i in order], dtype=np.int32)


# -------------------------
# DCT-like bases
# -------------------------

@lru_cache(maxsize=128)
def dct_basis(block_size: int, u: int, v: int) -> np.ndarray:
    bs = int(block_size)
    x = np.arange(bs, dtype=np.float32)
    y = np.arange(bs, dtype=np.float32)
    au = np.sqrt(1.0 / bs) if u == 0 else np.sqrt(2.0 / bs)
    av = np.sqrt(1.0 / bs) if v == 0 else np.sqrt(2.0 / bs)
    bx = au * np.cos(np.pi * (2 * x + 1) * u / (2 * bs))
    by = av * np.cos(np.pi * (2 * y + 1) * v / (2 * bs))
    return np.outer(bx, by).astype(np.float32)


@lru_cache(maxsize=64)
def haar_ll_dct_basis(block_size: int, u: int, v: int) -> np.ndarray:
    """DCT basis on the Haar LL subband, expanded to the original block.

    This gives a fast approximation to a 1-level DWT + DCT baseline without
    calling a wavelet library for every block.
    """
    if block_size % 2 != 0:
        raise ValueError("haar_ll_dct_basis requires an even block size")
    half = block_size // 2
    low = dct_basis(half, u, v)
    # LL coefficient is the average of each 2x2 cell. Expand with 1/2 scaling
    # so the basis remains close to unit norm.
    return np.repeat(np.repeat(low, 2, axis=0), 2, axis=1).astype(np.float32) / 2.0


# -------------------------
# Attacks
# -------------------------

def attack_jpeg(img: np.ndarray, q: int) -> np.ndarray:
    """In-memory JPEG round trip safe for parallel benchmark workers."""
    buffer = BytesIO()
    Image.fromarray(np.asarray(img, dtype=np.uint8)).save(buffer, quality=int(q), format="JPEG")
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return np.asarray(decoded.convert("RGB"), dtype=np.uint8)


def attack_gaussian(img: np.ndarray, sigma: float, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = img.astype(np.float32) + rng.normal(0.0, float(sigma), img.shape)
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)


def attack_blur(img: np.ndarray, radius: float) -> np.ndarray:
    return np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=float(radius))).convert("RGB"), dtype=np.uint8)


def attack_median(img: np.ndarray, size: int = 3) -> np.ndarray:
    return np.asarray(Image.fromarray(img).filter(ImageFilter.MedianFilter(size=int(size))).convert("RGB"), dtype=np.uint8)


def attack_resize(img: np.ndarray, scale: float) -> np.ndarray:
    im = Image.fromarray(img)
    w, h = im.size
    small = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BICUBIC)
    rec = small.resize((w, h), Image.Resampling.BICUBIC)
    return np.asarray(rec.convert("RGB"), dtype=np.uint8)


def attack_salt_pepper(img: np.ndarray, amount: float = 0.005, seed: int = 321) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.asarray(img, dtype=np.uint8).copy()
    mask = rng.random(out.shape[:2]) < float(amount)
    salt = rng.random(out.shape[:2]) < 0.5
    out[mask & salt] = 255
    out[mask & ~salt] = 0
    return out


def attack_brightness_contrast(img: np.ndarray, brightness: float = 1.05, contrast: float = 1.05) -> np.ndarray:
    im = Image.fromarray(img)
    im = ImageEnhance.Brightness(im).enhance(float(brightness))
    im = ImageEnhance.Contrast(im).enhance(float(contrast))
    return np.asarray(im.convert("RGB"), dtype=np.uint8)


def attack_gamma(img: np.ndarray, gamma: float = 0.9) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    out = np.power(np.clip(arr, 0.0, 1.0), float(gamma)) * 255.0
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def attack_crop_restore(img: np.ndarray, crop_fraction: float = 0.10) -> np.ndarray:
    im = Image.fromarray(img)
    w, h = im.size
    dx = int(round(w * float(crop_fraction) / 2.0))
    dy = int(round(h * float(crop_fraction) / 2.0))
    cropped = im.crop((dx, dy, w - dx, h - dy))
    restored = cropped.resize((w, h), Image.Resampling.BICUBIC)
    return np.asarray(restored.convert("RGB"), dtype=np.uint8)


def attack_rotate(img: np.ndarray, degrees: float = 1.0) -> np.ndarray:
    im = Image.fromarray(img)
    rot = im.rotate(float(degrees), resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(0, 0, 0))
    return np.asarray(rot.convert("RGB"), dtype=np.uint8)


def attack_translate(img: np.ndarray, shift_x: int = 4, shift_y: int = 4) -> np.ndarray:
    arr = np.asarray(img, dtype=np.uint8)
    out = np.zeros_like(arr)
    sx = int(shift_x)
    sy = int(shift_y)
    h, w = arr.shape[:2]
    src_x0 = max(0, -sx)
    src_x1 = min(w, w - sx)
    src_y0 = max(0, -sy)
    src_y1 = min(h, h - sy)
    dst_x0 = max(0, sx)
    dst_x1 = dst_x0 + max(0, src_x1 - src_x0)
    dst_y0 = max(0, sy)
    dst_y1 = dst_y0 + max(0, src_y1 - src_y0)
    if src_x1 > src_x0 and src_y1 > src_y0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]
    return out


def attack_speckle(img: np.ndarray, sigma: float = 0.01, seed: int = 777) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = np.asarray(img, dtype=np.float32)
    noisy = arr + arr * rng.normal(0.0, float(sigma), arr.shape)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


def attack_poisson(img: np.ndarray, seed: int = 778) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = np.asarray(img, dtype=np.float32)
    vals = 2 ** np.ceil(np.log2(max(2, len(np.unique(arr)))))
    noisy = rng.poisson(arr * vals / 255.0) * (255.0 / vals)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


def attack_sharpen(img: np.ndarray, factor: float = 1.5) -> np.ndarray:
    im = Image.fromarray(img)
    return np.asarray(ImageEnhance.Sharpness(im).enhance(float(factor)).convert("RGB"), dtype=np.uint8)


def attack_equalize_y(img: np.ndarray) -> np.ndarray:
    im = Image.fromarray(img).convert("YCbCr")
    y, cb, cr = im.split()
    y2 = ImageOps.equalize(y)
    return np.asarray(Image.merge("YCbCr", (y2, cb, cr)).convert("RGB"), dtype=np.uint8)


def attack_saturation(img: np.ndarray, factor: float = 1.25) -> np.ndarray:
    im = Image.fromarray(img)
    return np.asarray(ImageEnhance.Color(im).enhance(float(factor)).convert("RGB"), dtype=np.uint8)


def attack_cutout(img: np.ndarray, fraction: float = 0.10) -> np.ndarray:
    out = np.asarray(img, dtype=np.uint8).copy()
    h, w = out.shape[:2]
    ch = max(1, int(round(h * float(fraction))))
    cw = max(1, int(round(w * float(fraction))))
    y0 = max(0, h // 2 - ch // 2)
    x0 = max(0, w // 2 - cw // 2)
    out[y0:y0+ch, x0:x0+cw] = 0
    return out


def attack_channel_scale(img: np.ndarray, r: float = 1.03, g: float = 0.98, b: float = 1.02) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32).copy()
    arr[..., 0] *= float(r)
    arr[..., 1] *= float(g)
    arr[..., 2] *= float(b)
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)


def attack_webp(img: np.ndarray, quality: int = 70) -> np.ndarray:
    """In-memory WebP round trip safe for parallel benchmark workers."""
    buffer = BytesIO()
    Image.fromarray(np.asarray(img, dtype=np.uint8)).save(buffer, quality=int(quality), format="WEBP")
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return np.asarray(decoded.convert("RGB"), dtype=np.uint8)



def attack_box_blur(img: np.ndarray, radius: float = 1.0) -> np.ndarray:
    return np.asarray(Image.fromarray(img).filter(ImageFilter.BoxBlur(radius=float(radius))).convert("RGB"), dtype=np.uint8)


def attack_motion_blur(img: np.ndarray, length: int = 7, direction: str = "horizontal") -> np.ndarray:
    """Deterministic linear motion blur using edge padding and no optional dependencies."""
    k = max(3, int(length))
    if k % 2 == 0:
        k += 1
    r = k // 2
    arr = np.asarray(img, dtype=np.float32)
    direction = str(direction).lower().strip()
    acc = np.zeros_like(arr, dtype=np.float32)
    if direction in {"vertical", "v"}:
        pad = np.pad(arr, ((r, r), (0, 0), (0, 0)), mode="edge")
        for i in range(k):
            acc += pad[i:i + arr.shape[0], :, :]
    elif direction in {"diagonal", "diag", "d"}:
        pad = np.pad(arr, ((r, r), (r, r), (0, 0)), mode="edge")
        for i in range(k):
            acc += pad[i:i + arr.shape[0], i:i + arr.shape[1], :]
    else:
        pad = np.pad(arr, ((0, 0), (r, r), (0, 0)), mode="edge")
        for i in range(k):
            acc += pad[:, i:i + arr.shape[1], :]
    return np.clip(np.rint(acc / float(k)), 0, 255).astype(np.uint8)


def attack_pixelate(img: np.ndarray, scale: float = 0.50) -> np.ndarray:
    im = Image.fromarray(img)
    w, h = im.size
    small = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.NEAREST)
    rec = small.resize((w, h), Image.Resampling.NEAREST)
    return np.asarray(rec.convert("RGB"), dtype=np.uint8)


def attack_aspect_resize(img: np.ndarray, scale_x: float = 0.75, scale_y: float = 1.25) -> np.ndarray:
    im = Image.fromarray(img)
    w, h = im.size
    warped = im.resize((max(1, int(w * scale_x)), max(1, int(h * scale_y))), Image.Resampling.BICUBIC)
    restored = warped.resize((w, h), Image.Resampling.BICUBIC)
    return np.asarray(restored.convert("RGB"), dtype=np.uint8)


def attack_posterize(img: np.ndarray, bits: int = 4) -> np.ndarray:
    im = Image.fromarray(img)
    return np.asarray(ImageOps.posterize(im, int(bits)).convert("RGB"), dtype=np.uint8)


def attack_quantize(img: np.ndarray, colors: int = 64) -> np.ndarray:
    im = Image.fromarray(img).convert("RGB")
    q = im.quantize(colors=max(2, int(colors))).convert("RGB")
    return np.asarray(q, dtype=np.uint8)


def attack_grayscale_y(img: np.ndarray) -> np.ndarray:
    im = Image.fromarray(img).convert("L").convert("RGB")
    return np.asarray(im, dtype=np.uint8)


def attack_hue_shift(img: np.ndarray, shift: int = 15) -> np.ndarray:
    hsv = np.asarray(Image.fromarray(img).convert("HSV"), dtype=np.uint8).copy()
    hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + int(shift)) % 256).astype(np.uint8)
    return np.asarray(Image.fromarray(hsv, mode="HSV").convert("RGB"), dtype=np.uint8)


def attack_color_jitter(img: np.ndarray, brightness: float = 0.95, contrast: float = 1.15, saturation: float = 1.25) -> np.ndarray:
    im = Image.fromarray(img)
    im = ImageEnhance.Brightness(im).enhance(float(brightness))
    im = ImageEnhance.Contrast(im).enhance(float(contrast))
    im = ImageEnhance.Color(im).enhance(float(saturation))
    return np.asarray(im.convert("RGB"), dtype=np.uint8)


def attack_stripe_dropout(img: np.ndarray, fraction: float = 0.10, orientation: str = "vertical") -> np.ndarray:
    out = np.asarray(img, dtype=np.uint8).copy()
    h, w = out.shape[:2]
    orientation = str(orientation).lower().strip()
    if orientation in {"horizontal", "h"}:
        stripe = max(1, int(round(h * float(fraction))))
        y0 = max(0, h // 2 - stripe // 2)
        out[y0:y0 + stripe, :] = 0
    else:
        stripe = max(1, int(round(w * float(fraction))))
        x0 = max(0, w // 2 - stripe // 2)
        out[:, x0:x0 + stripe] = 0
    return out


def attack_corner_crop_restore(img: np.ndarray, crop_fraction: float = 0.15) -> np.ndarray:
    """Crop from the top-left corner and resize back; harder than centered crop for synchronization."""
    im = Image.fromarray(img)
    w, h = im.size
    dx = int(round(w * float(crop_fraction)))
    dy = int(round(h * float(crop_fraction)))
    cropped = im.crop((dx, dy, w, h))
    restored = cropped.resize((w, h), Image.Resampling.BICUBIC)
    return np.asarray(restored.convert("RGB"), dtype=np.uint8)


def attack_affine_shear(img: np.ndarray, shear: float = 0.05) -> np.ndarray:
    im = Image.fromarray(img)
    w, h = im.size
    # x' = x + shear * y.  Negative offset keeps the center approximately aligned.
    matrix = (1.0, float(shear), -float(shear) * h / 2.0, 0.0, 1.0, 0.0)
    out = im.transform((w, h), Image.Transform.AFFINE, matrix, resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0))
    return np.asarray(out.convert("RGB"), dtype=np.uint8)


def attack_y_channel_gaussian(img: np.ndarray, sigma: float = 1.0, seed: int = 779) -> np.ndarray:
    """Add deterministic Gaussian noise to luminance only in YCbCr space."""
    rng = np.random.default_rng(seed)
    im = Image.fromarray(img).convert("YCbCr")
    y, cb, cr = im.split()
    yy = np.asarray(y, dtype=np.float32)
    yy = yy + rng.normal(0.0, float(sigma), yy.shape)
    y2 = Image.fromarray(np.clip(np.rint(yy), 0, 255).astype(np.uint8), mode="L")
    return np.asarray(Image.merge("YCbCr", (y2, cb, cr)).convert("RGB"), dtype=np.uint8)


def attack_chroma_blur(img: np.ndarray, radius: float = 1.0) -> np.ndarray:
    """Blur Cb/Cr channels while preserving the Y channel."""
    im = Image.fromarray(img).convert("YCbCr")
    y, cb, cr = im.split()
    cb2 = cb.filter(ImageFilter.GaussianBlur(radius=float(radius)))
    cr2 = cr.filter(ImageFilter.GaussianBlur(radius=float(radius)))
    return np.asarray(Image.merge("YCbCr", (y, cb2, cr2)).convert("RGB"), dtype=np.uint8)


def attack_morph_min(img: np.ndarray, size: int = 3) -> np.ndarray:
    """PIL MinFilter, equivalent to a light channel-wise erosion attack."""
    return np.asarray(Image.fromarray(img).filter(ImageFilter.MinFilter(size=int(size))).convert("RGB"), dtype=np.uint8)


def attack_morph_max(img: np.ndarray, size: int = 3) -> np.ndarray:
    """PIL MaxFilter, equivalent to a light channel-wise dilation attack."""
    return np.asarray(Image.fromarray(img).filter(ImageFilter.MaxFilter(size=int(size))).convert("RGB"), dtype=np.uint8)


def attack_checkerboard_dropout(img: np.ndarray, period: int = 16) -> np.ndarray:
    """Deterministic local missing-sample attack preserving image shape."""
    out = np.asarray(img, dtype=np.uint8).copy()
    h, w = out.shape[:2]
    yy, xx = np.indices((h, w))
    cell = max(2, int(period))
    mask = ((yy // cell + xx // cell) % 2) == 0
    # Replace dropped blocks by mid-gray instead of black to avoid a pure cutout duplicate.
    out[mask] = 128
    return out


def attack_line_dropout(img: np.ndarray, every: int = 16, orientation: str = "horizontal") -> np.ndarray:
    """Remove regularly spaced rows or columns, a scanner/display style attack."""
    out = np.asarray(img, dtype=np.uint8).copy()
    step = max(2, int(every))
    orientation = str(orientation).lower().strip()
    if orientation in {"vertical", "v"}:
        out[:, ::step] = 0
    else:
        out[::step, :] = 0
    return out



def attack_chroma_subsample_420(img: np.ndarray) -> np.ndarray:
    """Simulate 4:2:0 chroma subsampling while preserving luminance resolution."""
    pil = Image.fromarray(np.asarray(img, dtype=np.uint8), mode="RGB").convert("YCbCr")
    y, cb, cr = pil.split()
    w, h = pil.size
    small = (max(1, w // 2), max(1, h // 2))
    cb = cb.resize(small, Image.Resampling.BILINEAR).resize((w, h), Image.Resampling.BILINEAR)
    cr = cr.resize(small, Image.Resampling.BILINEAR).resize((w, h), Image.Resampling.BILINEAR)
    return np.asarray(Image.merge("YCbCr", (y, cb, cr)).convert("RGB"), dtype=np.uint8)


def attack_center_crop_resize(img: np.ndarray, keep_fraction: float = 0.90) -> np.ndarray:
    arr = np.asarray(img, dtype=np.uint8)
    h, w = arr.shape[:2]
    keep = min(1.0, max(0.1, float(keep_fraction)))
    ch, cw = max(1, int(round(h * keep))), max(1, int(round(w * keep)))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    crop = Image.fromarray(arr[y0:y0 + ch, x0:x0 + cw], mode="RGB")
    return np.asarray(crop.resize((w, h), Image.Resampling.BICUBIC), dtype=np.uint8)


def attack_perspective(img: np.ndarray, fraction: float = 0.02) -> np.ndarray:
    """Small deterministic keystone warp using OpenCV when available."""
    try:
        import cv2  # type: ignore
    except Exception:  # pragma: no cover
        return attack_affine_shear(img, float(fraction))
    arr = np.asarray(img, dtype=np.uint8)
    h, w = arr.shape[:2]
    dx, dy = float(w) * float(fraction), float(h) * float(fraction)
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32([[dx, dy], [w - 1 - dx, 0], [w - 1, h - 1 - dy], [0, h - 1]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(arr, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def attack_local_blur(img: np.ndarray, fraction: float = 0.30, radius: float = 2.0) -> np.ndarray:
    arr = np.asarray(img, dtype=np.uint8)
    h, w = arr.shape[:2]
    fh, fw = max(1, int(h * fraction)), max(1, int(w * fraction))
    y0, x0 = (h - fh) // 2, (w - fw) // 2
    out = arr.copy()
    patch = Image.fromarray(out[y0:y0 + fh, x0:x0 + fw], mode="RGB").filter(ImageFilter.GaussianBlur(radius=float(radius)))
    out[y0:y0 + fh, x0:x0 + fw] = np.asarray(patch, dtype=np.uint8)
    return out


def attack_overlay(img: np.ndarray, opacity: float = 0.18) -> np.ndarray:
    base = Image.fromarray(np.asarray(img, dtype=np.uint8), mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = base.size
    alpha = int(np.clip(float(opacity), 0.0, 1.0) * 255)
    draw.rectangle([w // 5, h // 3, 4 * w // 5, 2 * h // 3], fill=(32, 32, 32, alpha))
    draw.line([0, h - 1, w - 1, 0], fill=(255, 255, 255, alpha), width=max(1, min(w, h) // 80))
    return np.asarray(Image.alpha_composite(base, overlay).convert("RGB"), dtype=np.uint8)


def attack_jpeg_chain(img: np.ndarray, quality_first: int = 85, quality_second: int = 70) -> np.ndarray:
    return attack_jpeg(attack_jpeg(img, int(quality_first)), int(quality_second))


def attack_screen_capture(img: np.ndarray) -> np.ndarray:
    out = attack_chroma_subsample_420(img)
    out = attack_resize(out, 0.85)
    out = attack_gamma(out, 1.05)
    return attack_jpeg(out, 70)


def attack_print_scan(img: np.ndarray) -> np.ndarray:
    out = attack_rotate(img, 0.35)
    out = attack_blur(out, 0.45)
    out = attack_gaussian(out, 1.0, seed=881)
    return attack_brightness_contrast(out, 1.01, 0.98)


def attack_category(name: str) -> str:
    n = str(name).lower()
    if n == "clean": return "clean"
    if any(x in n for x in ("jpeg", "webp", "chroma420", "screen_capture")): return "compression"
    if any(x in n for x in ("rotate", "translate", "resize", "crop", "affine", "perspective", "pixelate", "aspect")): return "geometric"
    if any(x in n for x in ("blur", "median", "morph", "sharpen")): return "filtering"
    if any(x in n for x in ("gaussian", "speckle", "poisson", "saltpepper", "noise")): return "noise"
    if any(x in n for x in ("brightness", "contrast", "gamma", "saturation", "hue", "grayscale", "channel", "equalize", "posterize", "quantize")): return "photometric"
    if any(x in n for x in ("cutout", "dropout", "overlay", "line_")): return "occlusion"
    if "print_scan" in n: return "acquisition"
    return "combined"

def default_attacks(preset: str = "extended") -> list[tuple[str, Callable[[np.ndarray], np.ndarray]]]:
    """Reproducible attack suite.

    Presets:
      - smoke: clean + two light attacks for CI/testing.
      - standard: common attacks used for quick benchmark tables.
      - extended: full stress suite with hard compression, color, geometry, quantization, and combined attacks.
      - stress: alias of extended for explicit stress-test command lines.
    """
    attacks = [
        ("clean", lambda x: x),
        ("jpeg90", lambda x: attack_jpeg(x, 90)),
        ("jpeg70", lambda x: attack_jpeg(x, 70)),
        ("jpeg50", lambda x: attack_jpeg(x, 50)),
        ("jpeg30", lambda x: attack_jpeg(x, 30)),
        ("jpeg20", lambda x: attack_jpeg(x, 20)),
        ("jpeg10", lambda x: attack_jpeg(x, 10)),
        ("webp70", lambda x: attack_webp(x, 70)),
        ("webp50", lambda x: attack_webp(x, 50)),
        ("webp30", lambda x: attack_webp(x, 30)),
        ("gaussian_sigma1", lambda x: attack_gaussian(x, 1.0)),
        ("gaussian_sigma2", lambda x: attack_gaussian(x, 2.0)),
        ("gaussian_sigma5", lambda x: attack_gaussian(x, 5.0)),
        ("speckle_0_01", lambda x: attack_speckle(x, 0.01)),
        ("poisson", lambda x: attack_poisson(x)),
        ("saltpepper_0_005", lambda x: attack_salt_pepper(x, 0.005)),
        ("saltpepper_0_02", lambda x: attack_salt_pepper(x, 0.02)),
        ("blur_r0_5", lambda x: attack_blur(x, 0.5)),
        ("blur_r1_0", lambda x: attack_blur(x, 1.0)),
        ("box_blur1", lambda x: attack_box_blur(x, 1.0)),
        ("motion_blur7_h", lambda x: attack_motion_blur(x, 7, "horizontal")),
        ("motion_blur9_diag", lambda x: attack_motion_blur(x, 9, "diagonal")),
        ("median3", lambda x: attack_median(x, 3)),
        ("median5", lambda x: attack_median(x, 5)),
        ("resize075", lambda x: attack_resize(x, 0.75)),
        ("resize050", lambda x: attack_resize(x, 0.50)),
        ("resize025", lambda x: attack_resize(x, 0.25)),
        ("pixelate050", lambda x: attack_pixelate(x, 0.50)),
        ("aspect_resize_075_125", lambda x: attack_aspect_resize(x, 0.75, 1.25)),
        ("brightness105_contrast105", lambda x: attack_brightness_contrast(x, 1.05, 1.05)),
        ("brightness090_contrast110", lambda x: attack_brightness_contrast(x, 0.90, 1.10)),
        ("gamma090", lambda x: attack_gamma(x, 0.90)),
        ("gamma120", lambda x: attack_gamma(x, 1.20)),
        ("saturation125", lambda x: attack_saturation(x, 1.25)),
        ("color_jitter_hard", lambda x: attack_color_jitter(x, 0.95, 1.15, 1.25)),
        ("hue_shift15", lambda x: attack_hue_shift(x, 15)),
        ("grayscale_y", lambda x: attack_grayscale_y(x)),
        ("posterize4", lambda x: attack_posterize(x, 4)),
        ("quantize64", lambda x: attack_quantize(x, 64)),
        ("channel_scale", lambda x: attack_channel_scale(x, 1.03, 0.98, 1.02)),
        ("sharpen150", lambda x: attack_sharpen(x, 1.50)),
        ("equalize_y", lambda x: attack_equalize_y(x)),
        ("cutout10", lambda x: attack_cutout(x, 0.10)),
        ("cutout20", lambda x: attack_cutout(x, 0.20)),
        ("stripe_dropout10_v", lambda x: attack_stripe_dropout(x, 0.10, "vertical")),
        ("crop_restore10", lambda x: attack_crop_restore(x, 0.10)),
        ("crop_restore25", lambda x: attack_crop_restore(x, 0.25)),
        ("corner_crop_restore15", lambda x: attack_corner_crop_restore(x, 0.15)),
        ("affine_shear005", lambda x: attack_affine_shear(x, 0.05)),
        ("y_noise_sigma1", lambda x: attack_y_channel_gaussian(x, 1.0)),
        ("chroma_blur1", lambda x: attack_chroma_blur(x, 1.0)),
        ("morph_min3", lambda x: attack_morph_min(x, 3)),
        ("morph_max3", lambda x: attack_morph_max(x, 3)),
        ("checkerboard_dropout16", lambda x: attack_checkerboard_dropout(x, 16)),
        ("line_dropout16_h", lambda x: attack_line_dropout(x, 16, "horizontal")),
        ("line_dropout16_v", lambda x: attack_line_dropout(x, 16, "vertical")),
        ("rotate1", lambda x: attack_rotate(x, 1.0)),
        ("rotate3", lambda x: attack_rotate(x, 3.0)),
        ("rotate5", lambda x: attack_rotate(x, 5.0)),
        ("translate4_4", lambda x: attack_translate(x, 4, 4)),
        ("translate8_minus8", lambda x: attack_translate(x, 8, -8)),
        ("jpeg70_gaussian1", lambda x: attack_gaussian(attack_jpeg(x, 70), 1.0)),
        ("jpeg70_median3", lambda x: attack_median(attack_jpeg(x, 70), 3)),
        ("jpeg50_blur1", lambda x: attack_blur(attack_jpeg(x, 50), 1.0)),
        ("jpeg50_resize075", lambda x: attack_resize(attack_jpeg(x, 50), 0.75)),
        ("resize050_jpeg50", lambda x: attack_jpeg(attack_resize(x, 0.50), 50)),
        ("rotate1_jpeg70", lambda x: attack_jpeg(attack_rotate(x, 1.0), 70)),
        ("chroma420", lambda x: attack_chroma_subsample_420(x)),
        ("center_crop_resize90", lambda x: attack_center_crop_resize(x, 0.90)),
        ("perspective002", lambda x: attack_perspective(x, 0.02)),
        ("local_blur30", lambda x: attack_local_blur(x, 0.30, 2.0)),
        ("overlay18", lambda x: attack_overlay(x, 0.18)),
        ("jpeg85_jpeg70", lambda x: attack_jpeg_chain(x, 85, 70)),
        ("screen_capture_light", lambda x: attack_screen_capture(x)),
        ("print_scan_light", lambda x: attack_print_scan(x)),
    ]
    preset = str(preset).lower().strip()
    if preset in {"extended", "stress"}:
        return attacks
    smoke = {"clean", "jpeg90", "gaussian_sigma1"}
    if preset == "smoke":
        return [(n, f) for n, f in attacks if n in smoke]
    standard = {
        "clean", "jpeg90", "jpeg70", "jpeg50", "jpeg30", "webp70",
        "gaussian_sigma1", "gaussian_sigma2", "saltpepper_0_005",
        "blur_r0_5", "median3", "resize075",
        "brightness105_contrast105", "gamma090", "crop_restore10",
        "y_noise_sigma1", "chroma_blur1", "morph_min3",
        "rotate1", "translate4_4", "jpeg70_gaussian1",
        "chroma420", "center_crop_resize90", "jpeg85_jpeg70", "screen_capture_light",
    }
    if preset == "standard":
        return [(n, f) for n, f in attacks if n in standard]
    raise ValueError("preset must be 'smoke', 'standard', 'extended', or 'stress'")
