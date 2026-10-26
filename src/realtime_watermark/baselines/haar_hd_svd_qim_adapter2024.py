"""Independent 4096-bit *blind* Haar/Hessenberg/SVD-QIM comparison adapter.

This is NOT a reproduction of Ernawan/Sudrajat/Amrullah (2024) or of
Lakasek/Elhadi (2026). Their additive singular-value watermark extraction
uses original singular values (host side information); substituting blind QIM
is a material algorithmic change. This controlled adapter is therefore
explicitly named and cited only as an independent contemporary-transform
comparison, not as those papers' original benchmark.

One keyed 8x8 RGB block carries one bit. A single-level orthonormal Haar LL
block (4x4) is Hessenberg-reduced, then its principal singular value is QIM
modulated; the modified LL is reconstructed, and the 2x2 underlying pixels
receive its inverse-Haar shift equally in RGB. The decoder needs the key,
step, and size, not the original image or original host singular values.
"""
from __future__ import annotations

from typing import Any
import numpy as np
from scipy.linalg import hessenberg

from ..common.types import MethodRef, WatermarkKey

METHOD_ID = 'haar_hd_svd_qim_adapter2024'
DEFAULT_STEP = 16.0
METHOD_REF = MethodRef(
    id=METHOD_ID,
    display_name='Haar-HD-SVD blind QIM (independent 2024-inspired adapter)',
    paper=('Related methodological reference only: A. Sudrajat, F. Ernawan, A. Amrullah, '
           'Robust Image Watermarking Based on Lifting Wavelet Transform-Hessenberg '
           'Decomposition for Copyright Protection, IJCMS 5(3), 2024.'),
    url='https://doi.org/10.52866/ijcsm.2024.05.03.043',
    implementation_note=('INDEPENDENT, NOT PAPER-EXACT. 8x8 RGB block; Haar LL -> '
       'Hessenberg -> dominant-singular-value binary QIM. No original image or '
       'host singular values at detection. All 4096 bits use the same capacity '
       'as the TEMPO proposal on 512x512. Calibrate step only on development hosts.'),
    fully_blind=True, color_host=True, binary_watermark=True, real_time_target=False,
)


def _tiles_y(image: np.ndarray):
    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected H x W x 3 RGB image')
    h, w = rgb.shape[:2]
    hh, ww = h // 8 * 8, w // 8 * 8
    if hh == 0 or ww == 0:
        raise ValueError('At least one 8x8 block required')
    rgbf = rgb[:hh, :ww].astype(np.float64)
    y = 0.299 * rgbf[..., 0] + 0.587 * rgbf[..., 1] + 0.114 * rgbf[..., 2]
    blocks = y.reshape(hh//8, 8, ww//8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    # Separable orthonormal Haar LL coefficient: (a+b+c+d)/2.
    ll = blocks.reshape(-1, 4, 2, 4, 2).sum(axis=(2, 4)) * 0.5
    return rgb, ll, hh, ww


def _indices(capacity: int, bit_count: int, repeat: int, seed: int):
    if repeat < 1 or bit_count * repeat > capacity:
        raise ValueError(f'Insufficient 8x8 capacity: {capacity} blocks, {bit_count * repeat} scheduled bits')
    return np.random.default_rng(int(seed) + 54031).permutation(capacity)[:bit_count*repeat]


def _hess_svd(ll: np.ndarray):
    q = np.empty((len(ll), 4, 4), dtype=np.float64)
    h = np.empty_like(q)
    for i, a in enumerate(ll):
        h[i], q[i] = hessenberg(a, calc_q=True, check_finite=False)
    u, s, vt = np.linalg.svd(h, full_matrices=True)
    return q, u, s, vt


def _target(s: np.ndarray, bits: np.ndarray, step: float):
    if step <= 0 or not np.isfinite(step):
        raise ValueError('QIM step must be positive and finite')
    nearest = np.rint(s/step).astype(np.int64)
    wrong = nearest % 2 != bits.astype(np.int64)
    lo = nearest - 1
    hi = nearest + 1
    alternate = np.where(np.abs(hi * step - s) < np.abs(lo * step - s), hi, lo)
    chosen = np.where(wrong, alternate, nearest)
    # Leading singular values are nonnegative, including low-texture hosts.
    chosen = np.maximum(chosen, np.where(bits.astype(bool), 1, 0))
    return chosen.astype(float) * step


def prepare_embedding(host_rgb: np.ndarray, watermark_binary: np.ndarray, *,
    seed: int = 2026, repeat: int | str = 1, step: float | None = None,
    method_params: dict[str, Any] | None = None):
    del method_params
    rgb, ll, hh, ww = _tiles_y(host_rgb)
    wm = (np.asarray(watermark_binary) > 127).astype(np.uint8)
    nbits = int(wm.size)
    rep = max(1, len(ll) // nbits) if repeat == 'auto' else int(repeat)
    idx = _indices(len(ll), nbits, rep, seed)
    q, u, s, vt = _hess_svd(ll)
    return {'rgb': rgb, 'll': ll, 'hh':hh, 'ww':ww,
            'q':q, 'u':u, 's':s, 'vt':vt, 'indices':idx,
            'bits':np.tile(wm.ravel(),rep), 'wm_shape':wm.shape,
            'seed':int(seed),'repeat':rep, 'step':float(DEFAULT_STEP if step is None else step)}


def embed_prepared(host_rgb: np.ndarray, watermark_binary: np.ndarray,
    prepared: dict[str, Any], *, step: float | None = None,
    method_params: dict[str, Any] | None = None):
    del host_rgb, method_params
    p = prepared
    st = float(p['step'] if step is None else step)
    idx = p['indices']; bits = p['bits']
    sv = p['s'].copy()
    sv[idx, 0] = _target(sv[idx,0],bits,st)
    # Recompose Hessenberg H = U Sigma V^T, then inverse similarity.
    hmod = (p['u'] * sv[:,None,:]) @ p['vt']
    q = p['q']
    llmod = q @ hmod @ q.transpose(0,2,1)
    dll = llmod - p['ll']
    # Inverse LL-only orthonormal Haar update spreads each shift /2 to pixels.
    offsets = np.repeat(np.repeat(dll,2,axis=1),2,axis=2) * .5
    hh,ww = p['hh'],p['ww']
    view = offsets.reshape(hh//8,ww//8,8,8).transpose(0,2,1,3).reshape(hh,ww)
    raw = p['rgb'].astype(np.float64).copy()
    raw[:hh,:ww] += view[...,None]
    out=np.clip(np.rint(raw),0,255).astype(np.uint8)
    wm=np.asarray(watermark_binary)
    key = WatermarkKey(method_id=METHOD_ID,
        host_shape=tuple(p['rgb'].shape),watermark_shape=tuple(wm.shape),
        seed=p['seed'],repeat=p['repeat'],step=st,arnold_iter=0,arnold_period=1,threshold=127,
        params={'comparison_class':'independently_implemented_modern_transform_adapter',
                'paper_exact':False,'detector_class':'blind','capacity_bits':len(p['ll']),
                'embed_rule':'Haar LL -> Hessenberg -> largest singular value QIM',
                'original_host_side_information':False},
        fully_blind=True,side_information='key, step, watermark dimensions only')
    return out,key


def embed(host_rgb: np.ndarray, watermark_binary: np.ndarray, *, seed: int=2026,
    repeat: int | str='auto',step: float|None=None,
    method_params:dict[str,Any]|None=None):
    p=prepare_embedding(host_rgb,watermark_binary,seed=seed,repeat=repeat,step=step,method_params=method_params)
    return embed_prepared(host_rgb,watermark_binary,p,step=step,method_params=method_params)


def extract(image_rgb: np.ndarray,key: WatermarkKey|dict[str,Any]) -> np.ndarray:
    getter = key.get if isinstance(key,dict) else lambda name: getattr(key,name)
    _rgb,ll,_hh,_ww=_tiles_y(image_rgb)
    wmshape=tuple(getter('watermark_shape'))
    seed=int(getter('seed'));rep=int(getter('repeat')); step=float(getter('step'))
    indices=_indices(len(ll),int(np.prod(wmshape)),rep,seed)
    # Singular values are invariant under an orthogonal Hessenberg similarity.
    # Compute Hessenberg explicitly to match the method's declared transform.
    h=np.empty((len(ll),4,4),dtype=np.float64)
    for i,a in enumerate(ll):
        h[i]=hessenberg(a,calc_q=False,check_finite=False)
    sv=np.linalg.svd(h,compute_uv=False)[:,0]
    decisions=np.rint(sv[indices]/step).astype(np.int64)%2
    votes=decisions.reshape(rep,-1)
    b=(votes.sum(axis=0)*2>=rep).astype(np.uint8)
    return (b.reshape(wmshape)*255).astype(np.uint8)
