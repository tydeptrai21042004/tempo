from __future__ import annotations

"""Differential Energy Watermarking (DEW) baseline.

The paper-native idea is preserved: each label bit owns a keyed region of
8x8 luminance-DCT blocks, the region is split into two halves, and the bit is
encoded by selectively reducing high-frequency energy in one half.  The
``paper_zeroing`` mode removes the selected coefficients exactly; the default
``minimum_energy_removal`` mode is an explicitly disclosed decoded-frame
adapter that scales only as much energy as needed to create a requested
signed margin.
"""

from typing import Any

import numpy as np

from ..common.block_dct import dct_blocks_from_rgb, dct_blocks_to_rgb, rgb_to_dct_blocks
from ..common.types import MethodRef, WatermarkKey

METHOD_ID = "dew_langelaar2001"
BLOCK_SIZE = 8
DEFAULT_BLOCKS_PER_BIT = 16
DEFAULT_CUTOFF = 1
DEFAULT_MARGIN = 3000.0
PERMUTATION_STREAM_OFFSET = 36017

METHOD_REF = MethodRef(
    id=METHOD_ID,
    display_name="Differential Energy Watermarking (Langelaar--Lagendijk)",
    paper=(
        "G. C. Langelaar and R. L. Lagendijk, Optimal Differential Energy "
        "Watermarking of DCT Encoded Images and Video, IEEE Transactions on "
        "Image Processing 10(1), 2001."
    ),
    url="https://doi.org/10.1109/83.892451",
    implementation_note=(
        "Blind keyed-region DEW on 8x8 luminance-DCT blocks. Paper-zeroing is "
        "available; the default decoded-frame adapter uses minimum energy "
        "removal to reach a signed differential-energy margin."
    ),
    fully_blind=True,
    color_host=True,
    binary_watermark=True,
    real_time_target=True,
)


def zigzag_coordinates(n: int = BLOCK_SIZE) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for s in range(2 * n - 1):
        diagonal: list[tuple[int, int]] = []
        r0 = max(0, s - (n - 1))
        r1 = min(n - 1, s)
        for r in range(r0, r1 + 1):
            c = s - r
            diagonal.append((r, c))
        if s % 2 == 0:
            diagonal.reverse()
        coords.extend(diagonal)
    return coords


_ZIGZAG = zigzag_coordinates(BLOCK_SIZE)


def coefficient_mask(cutoff: int) -> np.ndarray:
    c = int(cutoff)
    if c < 1 or c >= BLOCK_SIZE * BLOCK_SIZE:
        raise ValueError("DEW cutoff must be in [1, 63]")
    mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)
    for u, v in _ZIGZAG[c:]:
        mask[u, v] = True
    return mask


def _block_groups(block_count: int, payload: int, blocks_per_bit: int, seed: int) -> np.ndarray:
    bpb = int(blocks_per_bit)
    if bpb < 2 or bpb % 2:
        raise ValueError("DEW blocks_per_bit must be an even integer >= 2")
    needed = int(payload) * bpb
    if needed > int(block_count):
        raise ValueError(
            f"DEW capacity too small: need {needed} DCT blocks for {payload} bits "
            f"at {bpb} blocks/bit, have {block_count}. Reduce payload or blocks_per_bit."
        )
    rng = np.random.default_rng(int(seed) + PERMUTATION_STREAM_OFFSET)
    return rng.permutation(int(block_count))[:needed].reshape(int(payload), bpb).astype(np.int32)


def _resolve_blocks_per_bit(block_count: int, payload: int, method_params: dict[str, Any]) -> int:
    requested = method_params.get("blocks_per_bit", "auto")
    if requested == "auto" or requested is None:
        available = int(block_count) // max(1, int(payload))
        bpb = min(DEFAULT_BLOCKS_PER_BIT, available)
        if bpb % 2:
            bpb -= 1
        if bpb < 2:
            raise ValueError(
                f"DEW needs at least two DCT blocks per bit; payload {payload} exceeds "
                f"the paper-style capacity {block_count // 2} bits for this host."
            )
        return bpb
    return int(requested)


def _region_energies(coeffs: np.ndarray, groups: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    half = groups.shape[1] // 2
    selected = np.asarray(coeffs[:, mask], dtype=np.float64)
    energy_per_block = np.sum(selected * selected, axis=1)
    ea = np.sum(energy_per_block[groups[:, :half]], axis=1)
    eb = np.sum(energy_per_block[groups[:, half:]], axis=1)
    return ea, eb


def _apply_minimum_energy_removal(
    coeffs: np.ndarray,
    groups: np.ndarray,
    mask: np.ndarray,
    bits: np.ndarray,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    out = np.asarray(coeffs, dtype=np.float64).copy()
    ea, eb = _region_energies(out, groups, mask)
    half = groups.shape[1] // 2
    scales = np.ones(bits.size, dtype=np.float64)
    for i, bit in enumerate(np.asarray(bits, dtype=np.uint8).ravel()):
        if int(bit) == 0:
            target_indices = groups[i, half:]
            target_energy = float(eb[i])
            other_energy = float(ea[i])
        else:
            target_indices = groups[i, :half]
            target_energy = float(ea[i])
            other_energy = float(eb[i])
        desired_target = max(0.0, other_energy - float(margin))
        if target_energy <= desired_target or target_energy <= 1e-15:
            scale = 1.0
        else:
            scale = float(np.sqrt(desired_target / target_energy))
            scale = min(1.0, max(0.0, scale))
            target = out[target_indices]
            target[:, mask] *= scale
            out[target_indices] = target
        scales[i] = scale
    return out, scales


def _apply_paper_zeroing(
    coeffs: np.ndarray,
    groups: np.ndarray,
    mask: np.ndarray,
    bits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    out = np.asarray(coeffs, dtype=np.float64).copy()
    half = groups.shape[1] // 2
    scales = np.ones(bits.size, dtype=np.float64)
    for i, bit in enumerate(np.asarray(bits, dtype=np.uint8).ravel()):
        target_indices = groups[i, half:] if int(bit) == 0 else groups[i, :half]
        target = out[target_indices]
        target[:, mask] = 0.0
        out[target_indices] = target
        scales[i] = 0.0
    return out, scales


def prepare_embedding(
    host_rgb: np.ndarray,
    watermark_binary: np.ndarray,
    *,
    seed: int = 2026,
    repeat: int | str = 1,
    step: float | None = None,
    method_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del repeat
    params = dict(method_params or {})
    host = np.asarray(host_rgb, dtype=np.uint8)
    wm = (np.asarray(watermark_binary) > 127).astype(np.uint8)
    coeffs, y, cb, cr, h_crop, w_crop = rgb_to_dct_blocks(host, BLOCK_SIZE)
    payload = int(wm.size)
    blocks_per_bit = _resolve_blocks_per_bit(coeffs.shape[0], payload, params)
    groups = _block_groups(coeffs.shape[0], payload, blocks_per_bit, int(seed))
    cutoff = int(params.get("cutoff", DEFAULT_CUTOFF))
    mask = coefficient_mask(cutoff)
    mode = str(params.get("embedding_mode", "minimum_energy_removal")).strip().lower()
    if mode not in {"minimum_energy_removal", "paper_zeroing"}:
        raise ValueError("DEW embedding_mode must be 'minimum_energy_removal' or 'paper_zeroing'")
    margin = float(DEFAULT_MARGIN if step is None else step)
    return {
        "coeffs": coeffs,
        "y": y,
        "cb": cb,
        "cr": cr,
        "h_crop": h_crop,
        "w_crop": w_crop,
        "groups": groups,
        "block_count": int(coeffs.shape[0]),
        "mask": mask,
        "bits": wm.ravel(),
        "blocks_per_bit": blocks_per_bit,
        "cutoff": cutoff,
        "embedding_mode": mode,
        "margin": margin,
        "seed": int(seed),
    }


def embed_prepared(
    host_rgb: np.ndarray,
    watermark_binary: np.ndarray,
    prepared: dict[str, Any],
    *,
    step: float | None = None,
    method_params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, WatermarkKey]:
    del method_params
    host = np.asarray(host_rgb, dtype=np.uint8)
    wm = np.asarray(watermark_binary, dtype=np.uint8)
    margin = float(prepared["margin"] if step is None else step)
    mode = str(prepared["embedding_mode"])
    if mode == "paper_zeroing":
        modified, scales = _apply_paper_zeroing(
            prepared["coeffs"], prepared["groups"], prepared["mask"], prepared["bits"]
        )
    else:
        modified, scales = _apply_minimum_energy_removal(
            prepared["coeffs"], prepared["groups"], prepared["mask"], prepared["bits"], margin
        )
    watermarked = dct_blocks_to_rgb(
        modified,
        prepared["y"], prepared["cb"], prepared["cr"],
        prepared["h_crop"], prepared["w_crop"], BLOCK_SIZE,
    )
    key = WatermarkKey(
        method_id=METHOD_ID,
        host_shape=tuple(int(x) for x in host.shape),
        watermark_shape=tuple(int(x) for x in wm.shape),
        seed=int(prepared["seed"]),
        repeat=1,
        step=margin,
        arnold_iter=0,
        arnold_period=1,
        threshold=127,
        params={
            "domain": "8x8 luminance DCT keyed differential-energy regions",
            "block_size": BLOCK_SIZE,
            "blocks_per_bit": int(prepared["blocks_per_bit"]),
            "cutoff": int(prepared["cutoff"]),
            "selected_coefficients": int(np.count_nonzero(prepared["mask"])),
            "embedding_mode": mode,
            "energy_margin": margin,
            "mean_target_scale": float(np.mean(scales)),
            "zeroed_or_scaled_region": "B for bit 0; A for bit 1",
            "block_permutation": "regenerated from seed",
            "capacity_bits": int(prepared["block_count"] // prepared["blocks_per_bit"]),
            "paper_equation": "D=E_A-E_B; bit 0 requires D>0 and bit 1 requires D<0",
        },
        schedule=[],
        fully_blind=True,
        side_information="seed, payload shape, blocks_per_bit, cutoff, energy margin, and embedding mode",
    )
    return watermarked, key


def embed(
    host_rgb: np.ndarray,
    watermark_binary: np.ndarray,
    *,
    seed: int = 2026,
    repeat: int | str = "auto",
    step: float | None = None,
    method_params: dict[str, Any] | None = None,
):
    prepared = prepare_embedding(
        host_rgb, watermark_binary, seed=seed, repeat=repeat, step=step, method_params=method_params
    )
    return embed_prepared(
        host_rgb, watermark_binary, prepared, step=step, method_params=method_params
    )


def extract(image_rgb: np.ndarray, key: WatermarkKey | dict[str, Any]) -> np.ndarray:
    if isinstance(key, dict):
        key = WatermarkKey(**key)
    coeffs, _h_crop, _w_crop = dct_blocks_from_rgb(image_rgb, int(key.params["block_size"]))
    payload = int(np.prod(key.watermark_shape))
    groups = _block_groups(
        coeffs.shape[0], payload, int(key.params["blocks_per_bit"]), int(key.seed)
    )
    mask = coefficient_mask(int(key.params["cutoff"]))
    ea, eb = _region_energies(coeffs, groups, mask)
    bits = (ea < eb).astype(np.uint8)
    return (bits.reshape(key.watermark_shape) * 255).astype(np.uint8)


def differential_energy(image_rgb: np.ndarray, key: WatermarkKey | dict[str, Any]) -> np.ndarray:
    """Return E_A-E_B for diagnostic and unit-test use."""
    if isinstance(key, dict):
        key = WatermarkKey(**key)
    coeffs, _h_crop, _w_crop = dct_blocks_from_rgb(image_rgb, int(key.params["block_size"]))
    payload = int(np.prod(key.watermark_shape))
    groups = _block_groups(coeffs.shape[0], payload, int(key.params["blocks_per_bit"]), int(key.seed))
    ea, eb = _region_energies(coeffs, groups, coefficient_mask(int(key.params["cutoff"])))
    return ea - eb


__all__ = [
    "METHOD_ID", "METHOD_REF", "BLOCK_SIZE", "DEFAULT_BLOCKS_PER_BIT",
    "DEFAULT_CUTOFF", "DEFAULT_MARGIN", "zigzag_coordinates", "coefficient_mask",
    "prepare_embedding", "embed_prepared", "embed", "extract", "differential_energy",
]
