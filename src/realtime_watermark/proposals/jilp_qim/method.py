from __future__ import annotations

from typing import Any
from functools import lru_cache

import os
import numpy as np

try:
    from numba import get_num_threads, prange, set_num_threads
except Exception:  # pragma: no cover
    prange = range
    def get_num_threads() -> int:
        return 1
    def set_num_threads(_n: int) -> None:
        return None

from ...common.types import MethodRef, WatermarkKey
from ...core import NUMBA_AVAILABLE, arnold_inverse, arnold_period, arnold_transform, dct_basis, make_block_indices, njit
from . import engine as _eng
from .engine import _field_rgb as _field_rgb_kernel, _qim_margin_reliability as _qim_margin_reliability_kernel
from .config import DEFAULT_CONFIG, JILPQIMConfig
from .sparse_hartley_engine import sparse_carrier,sparse_leverage,embed_sparse,extract_sparse,extract_sparse_conf
from .hartley_carrier import payload_pair

METHOD_ID = "proposal"
DEFAULT_REFERENCE_STEP = 14.875
DEFAULT_PILOT_STEP = 17.0
DEFAULT_PILOT_COUNT = 96
BLOCK_SIZE = _eng.BLOCK_SIZE


def _call_with_realtime_thread_budget(kernel, *args):
    """Apply a local bounded worker budget and restore the caller setting."""
    previous = int(get_num_threads())
    try:
        requested = int(os.environ.get("JILP_NUM_THREADS", "4"))
    except ValueError:
        requested = 4
    target = max(1, min(previous, requested))
    if target != previous:
        set_num_threads(target)
    try:
        return kernel(*args)
    finally:
        if target != previous:
            set_num_threads(previous)

# JILP-QIM uses one calibrated
# low-frequency differential carrier by default. A second carrier remains an
# explicit ablation option. The keyed schedule is payload-proportional; when a
# host has spare blocks, an optional candidate pool can be ranked without making
# extraction depend on the original host image.
PAY10_BA = dct_basis(BLOCK_SIZE, 0, 1).astype(np.float32)
PAY10_BB = dct_basis(BLOCK_SIZE, 1, 0).astype(np.float32)
PAY10_2_BA = dct_basis(BLOCK_SIZE, 0, 2).astype(np.float32)
PAY10_2_BB = dct_basis(BLOCK_SIZE, 2, 0).astype(np.float32)



@njit(cache=True, fastmath=True)
def _block_scores_selected_kernel(host, candidate_indices, ba1, bb1, ba2, bb2, eta, h_crop, w_crop, block_size):
    bs = int(block_size)
    bc = w_crop // bs
    n = int(candidate_indices.shape[0])
    scores = np.empty(n, dtype=np.float32)
    for k in range(n):
        idx = int(candidate_indices[k])
        bi = idx // bc
        bj = idx - bi * bc
        r0 = bi * bs
        c0 = bj * bs
        sf = 0.0
        ss = 0.0
        ca1 = 0.0
        cb1 = 0.0
        ca2 = 0.0
        cb2 = 0.0
        for i in range(bs):
            for j in range(bs):
                rr = r0 + i
                cc = c0 + j
                r = host[rr, cc, 0]
                g = host[rr, cc, 1]
                b = host[rr, cc, 2]
                f = 0.299 * r + 0.587 * g + 0.114 * b + eta * (r - 0.5 * g - 0.5 * b)
                sf += f
                ss += f * f
                ca1 += f * ba1[i, j]
                cb1 += f * bb1[i, j]
                ca2 += f * ba2[i, j]
                cb2 += f * bb2[i, j]
        mean = sf / (bs * bs)
        var = ss / (bs * bs) - mean * mean
        if var < 0.0:
            var = 0.0
        energy = ca1 * ca1 + cb1 * cb1 + ca2 * ca2 + cb2 * cb2
        scores[k] = np.float32(var + 0.015 * energy)
    return scores

@njit(cache=True, fastmath=True)
def _block_scores_kernel(host, ba1, bb1, ba2, bb2, eta, h_crop, w_crop, block_size):
    bs = int(block_size)
    br = h_crop // bs
    bc = w_crop // bs
    capacity = br * bc
    scores = np.empty(capacity, dtype=np.float32)
    for bi in range(br):
        for bj in range(bc):
            idx = bi * bc + bj
            r0 = bi * bs
            c0 = bj * bs
            sf = 0.0
            ss = 0.0
            ca1 = 0.0
            cb1 = 0.0
            ca2 = 0.0
            cb2 = 0.0
            for i in range(bs):
                for j in range(bs):
                    rr = r0 + i
                    cc = c0 + j
                    r = host[rr, cc, 0]
                    g = host[rr, cc, 1]
                    b = host[rr, cc, 2]
                    f = 0.299 * r + 0.587 * g + 0.114 * b + eta * (r - 0.5 * g - 0.5 * b)
                    sf += f
                    ss += f * f
                    ca1 += f * ba1[i, j]
                    cb1 += f * bb1[i, j]
                    ca2 += f * ba2[i, j]
                    cb2 += f * bb2[i, j]
            mean = sf / (bs * bs)
            var = ss / (bs * bs) - mean * mean
            if var < 0.0:
                var = 0.0
            energy = ca1 * ca1 + cb1 * cb1 + ca2 * ca2 + cb2 * cb2
            scores[idx] = np.float32(var + 0.015 * energy)
    return scores


METHOD_REF = MethodRef(
    id=METHOD_ID,
    display_name="Proposal: RT-JILP-QIM with deterministic block-parallel projection",
    paper="This work.",
    url="",
    implementation_note=(
        "Blind repair-free proposal using deterministic block-parallel bounded-integer projection, "
        "a primary low-frequency differential payload carrier, and an exactly orthogonal pilot "
        "carrier. The parallel schedule preserves the original per-block arithmetic and output."
    ),
)


def _build_key_params(
    host_shape: tuple[int, int, int],
    watermark_shape: tuple[int, int],
    *,
    seed: int = 2026,
    step: float = DEFAULT_REFERENCE_STEP,
    rho_frac: float = 0.45,
    pilot_count: int = DEFAULT_PILOT_COUNT,
    pilot_step: float | None = None,
    pilot_rho_frac: float = 0.49,
    eta: float = 0.07,
    candidate_pool_multiplier: float = 1.0,
    candidate_budget: int | None = None,
    use_second_carrier: bool = False,
    pilot_schedule: str = "stratified",
    use_integer_lattice_closure: bool = True,
    projection_mode: str = "minimum_distortion",
    enable_pilots: bool = True,
    enable_sync_search: bool = True,
    payload_schedule_mode: str = "random",
    field_mode: str = "opponent",
    temporal_vote_mode: str = "hard",
    scramble_payload: bool = True,
    phase_refinement: str = "off",
    certified_phase_pruning: bool = False,
) -> dict[str, Any]:
    period = arnold_period(int(watermark_shape[0]))
    primary_frac = 0.85 if bool(use_second_carrier) else 1.0
    step1 = float(step) * primary_frac
    step2 = float(step) * float(np.sqrt(max(1.0 - primary_frac * primary_frac, 0.0)))
    return {
        "method_family": "Deterministic block-parallel repair-free joint integer-lattice projection QIM",
        "method_id": METHOD_ID,
        "block_size": BLOCK_SIZE,
        "basis_kind": "dct",
        "realtime_execution": "residual-exact single-pass bounded-integer block-parallel projection and extraction",
        "parallel_equivalence": "same per-block arithmetic, QIM targets, schedules, and uint8 output",
        "residual_exact_integer_tracking": True,
        "schedule_decoupled_raster_execution": True,
        "key_compiled_execution_plan": True,
        "payload_carrier": "primary low-frequency difference carrier: (0,1)/(1,0)",
        "payload_carrier_1": "(0,1)/(1,0)",
        "payload_carrier_2": "(0,2)/(2,0)",
        "pilot_carrier": "orthogonal low-frequency sum: (0,1)+(1,0)",
        "field": "Y + eta*(R - 0.5G - 0.5B)" if str(field_mode) == "opponent" else "Y luminance only",
        "field_mode": str(field_mode),
        "eta_configured": float(eta),
        "eta": float(eta if str(field_mode) == "opponent" else 0.0),
        "payload_len": int(watermark_shape[0] * watermark_shape[1]),
        "seed": int(seed),
        "step": float(step),
        "step_reference": float(step),
        "step_split_primary_fraction": primary_frac,
        "step_carrier1": step1,
        "step_carrier2": step2,
        "rho_frac": float(rho_frac),
        "step_pilot": float(DEFAULT_PILOT_STEP if pilot_step is None else pilot_step),
        "pilot_rho_frac": float(pilot_rho_frac),
        "pilot_count_configured": int(pilot_count),
        "pilot_count": int(pilot_count if enable_pilots else 0),
        "arnold_iter": int(7 % period),
        "arnold_period": int(period),
        "schedule_kind": "payload-proportional keyed schedule with configurable pilot placement",
        "candidate_score": "block variance + normalized low-frequency DCT-carrier energy",
        "candidate_pool_multiplier": float(candidate_pool_multiplier),
        "candidate_budget": None if candidate_budget is None else int(candidate_budget),
        "host_size_policy": "unified host-size-free fixed payload schedule",
        "sync": "adaptive: fast raw extraction or pilot-gated robust rotation/crop/translation search",
        "sync_rotation_degrees": [-5, -3, -1, 1, 3, 5],
        "sync_crop_inverse_fractions": [0.10, 0.15, 0.25],
        "sync_translation_radius": 8,
        "sync_translation_subset": 96,
        "ecc": "none",
        "paper_alignment": "repository proposal; no external baseline-paper identity claim",
        "minimum_distortion_projection": bool(str(projection_mode) == "minimum_distortion"),
        "projection_mode": str(projection_mode),
        "bounded_integer_projection": bool(use_integer_lattice_closure),
        "exact_clean_decoding_by_construction": True,
        "strict_integer_feasibility": True,
        "dual_payload_carrier": bool(use_second_carrier),
        "pilot_schedule": str(pilot_schedule),
        "payload_schedule_mode": str(payload_schedule_mode),
        "pilots_enabled": bool(enable_pilots),
        "sync_search_enabled": bool(enable_sync_search),
        "temporal_vote_mode": str(temporal_vote_mode),
        "scramble_payload": bool(scramble_payload),
        "phase_refinement": str(phase_refinement),
        "certified_phase_pruning": bool(certified_phase_pruning),
        "structural_components": {
            "integer_lattice_closure": bool(use_integer_lattice_closure),
            "minimum_distortion_region": bool(str(projection_mode) == "minimum_distortion"),
            "orthogonal_pilots": bool(enable_pilots),
            "geometric_sync_search": bool(enable_sync_search),
            "content_adaptive_schedule": False,
            "opponent_color_field": bool(str(field_mode) == "opponent"),
            "confidence_weighted_temporal_vote": bool(str(temporal_vote_mode) == "confidence_weighted"),
            "payload_scrambling": bool(scramble_payload),
        },
        "energy_preserving_step_split": bool(use_second_carrier),
        "qim_margin_reliability_extraction": True,
        "content_adaptive_block_selection": False,
        "reduced_pilot_overhead": True,
        "clean_repair": "none",
        "clean_verification_iters": 0,
        "side_information": "schedule_regenerated_from_seed_no_original_host",
        "target_note": (
            "The released primary protocol uses a deterministic seed-regenerated random payload schedule. "
            "Host-ranked content selection is optional, explicitly stored in the key, and is not claimed "
            "as a primary contribution unless separately validated."
        ),
    }


def _field_for_scores(host: np.ndarray, eta: float, h_crop: int, w_crop: int) -> np.ndarray:
    arr = np.asarray(host[:h_crop, :w_crop], dtype=np.float32)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    return 0.299 * r + 0.587 * g + 0.114 * b + float(eta) * (r - 0.5 * g - 0.5 * b)


def _block_scores(host_rgb: np.ndarray, *, block_size: int = BLOCK_SIZE, eta: float = 0.10) -> tuple[np.ndarray, int, int, int]:
    host = np.asarray(host_rgb, dtype=np.uint8)
    h, w = host.shape[:2]
    h_crop = h - (h % block_size)
    w_crop = w - (w % block_size)
    capacity = (h_crop // block_size) * (w_crop // block_size)
    if capacity <= 0:
        return np.zeros(0, dtype=np.float32), h_crop, w_crop, capacity
    scores = _block_scores_kernel(
        host, PAY10_BA, PAY10_BB, PAY10_2_BA, PAY10_2_BB,
        float(eta), int(h_crop), int(w_crop), int(block_size)
    )
    return scores.astype(np.float32), h_crop, w_crop, capacity


def _block_scores_selected(
    host_rgb: np.ndarray,
    candidate_indices: np.ndarray,
    *,
    block_size: int = BLOCK_SIZE,
    eta: float = 0.07,
) -> tuple[np.ndarray, int, int, int]:
    host = np.asarray(host_rgb, dtype=np.uint8)
    h, w = host.shape[:2]
    h_crop = h - (h % block_size)
    w_crop = w - (w % block_size)
    capacity = (h_crop // block_size) * (w_crop // block_size)
    candidate_indices = np.asarray(candidate_indices, dtype=np.int32).reshape(-1)
    if capacity <= 0 or candidate_indices.size == 0:
        return np.zeros(0, dtype=np.float32), h_crop, w_crop, capacity
    valid = candidate_indices[(candidate_indices >= 0) & (candidate_indices < capacity)].astype(np.int32)
    if valid.size != candidate_indices.size:
        candidate_indices = valid
    scores = _block_scores_selected_kernel(
        host, candidate_indices, PAY10_BA, PAY10_BB, PAY10_2_BA, PAY10_2_BB,
        float(eta), int(h_crop), int(w_crop), int(block_size)
    )
    return scores.astype(np.float32), h_crop, w_crop, capacity


def _make_stratified_pilot_indices(
    h_crop: int,
    w_crop: int,
    *,
    pilot_count: int,
    seed: int,
    block_size: int = BLOCK_SIZE,
) -> np.ndarray:
    """Keyed spatially balanced pilot positions with deterministic jitter.

    Random pilots can cluster and leave corners or edges unsupported. This grid-jittered
    schedule preserves key dependence while covering the full frame, which improves the
    information available to crop/rotation/translation synchronization.
    """
    br = int(h_crop // block_size)
    bc = int(w_crop // block_size)
    capacity = br * bc
    count = int(min(max(0, pilot_count), capacity))
    if count <= 0:
        return np.zeros(0, dtype=np.int32)
    rng = np.random.default_rng(int(seed) + 919)
    # Match the grid aspect ratio to the block lattice.
    nr = max(1, int(np.floor(np.sqrt(count * br / max(bc, 1)))))
    nc = max(1, int(np.ceil(count / nr)))
    while nr * nc < count:
        if nr / max(nc, 1) < br / max(bc, 1):
            nr += 1
        else:
            nc += 1
    rows = np.linspace(0, br, nr + 1, dtype=np.int32)
    cols = np.linspace(0, bc, nc + 1, dtype=np.int32)
    chosen: list[int] = []
    for ir in range(nr):
        rlo, rhi = int(rows[ir]), int(rows[ir + 1])
        if rhi <= rlo:
            continue
        for ic in range(nc):
            clo, chi = int(cols[ic]), int(cols[ic + 1])
            if chi <= clo:
                continue
            # Jitter inside each cell, but every cell contributes at most one pilot.
            rr = int(rng.integers(rlo, rhi))
            cc = int(rng.integers(clo, chi))
            chosen.append(rr * bc + cc)
    # Keyed order prevents a public geometric pattern while retaining coverage.
    arr = np.asarray(chosen, dtype=np.int32)
    if arr.size > 0:
        arr = arr[rng.permutation(arr.size)]
    if arr.size < count:
        used = np.zeros(capacity, dtype=bool)
        used[arr] = True
        remaining = np.nonzero(~used)[0].astype(np.int32)
        extra = rng.choice(remaining, size=count - arr.size, replace=False).astype(np.int32)
        arr = np.concatenate([arr, extra])
    return arr[:count].astype(np.int32)


def _balanced_pilot_bits(count: int, seed: int) -> np.ndarray:
    """Return a keyed pilot sequence with nearly equal numbers of zeros and ones."""
    count = int(max(0, count))
    bits = np.arange(count, dtype=np.uint8) & 1
    rng = np.random.default_rng(int(seed) + 1223)
    if count > 1:
        bits = bits[rng.permutation(count)]
    return bits.astype(np.uint8)



@lru_cache(maxsize=256)
def _compiled_full_capacity_schedule(
    h_crop: int,
    w_crop: int,
    payload_len: int,
    seed: int,
    pilot_count: int,
    block_size: int,
    pilot_schedule: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compile the seed-defined online execution plan once per key geometry."""
    capacity = (h_crop // block_size) * (w_crop // block_size)
    payload_indices = make_block_indices(
        h_crop, w_crop, block_size, payload_len, 1, int(seed)
    ).astype(np.int32)
    if str(pilot_schedule).lower() == "stratified":
        pilot_indices = _make_stratified_pilot_indices(
            h_crop, w_crop, pilot_count=int(pilot_count), seed=int(seed), block_size=int(block_size)
        )
    else:
        rng_p = np.random.default_rng(int(seed) + 919)
        count = int(min(max(0, pilot_count), capacity))
        pilot_indices = rng_p.permutation(capacity)[:count].astype(np.int32)
    pilot_bits = _balanced_pilot_bits(len(pilot_indices), int(seed))
    # Treat cached arrays as immutable execution-plan data.
    payload_indices.setflags(write=False)
    pilot_indices.setflags(write=False)
    pilot_bits.setflags(write=False)
    return payload_indices, pilot_indices, pilot_bits


def _make_content_adaptive_schedules(
    host_rgb: np.ndarray,
    watermark_shape: tuple[int, int],
    *,
    seed: int,
    pilot_count: int,
    block_size: int = BLOCK_SIZE,
    eta: float = 0.07,
    candidate_pool_multiplier: float = 1.0,
    candidate_budget: int | None = None,
    pilot_schedule: str = "stratified",
    payload_schedule_mode: str = "random",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, int]:
    # If capacity equals payload length, every 8x8 block must carry one watermark
    # bit. There is no payload-selection freedom, so skip score computation to keep
    # Fixed-budget JILP-QIM schedule avoids host-capacity-dependent ranking by default.
    h_crop, w_crop = _eng._crop_shape(tuple(int(x) for x in np.asarray(host_rgb).shape), block_size)
    capacity = (h_crop // block_size) * (w_crop // block_size)
    payload_len = int(watermark_shape[0] * watermark_shape[1])
    if payload_len > capacity:
        raise ValueError(f"capacity too small: payload={payload_len}, capacity={capacity}")
    if capacity <= payload_len:
        payload_indices, pilot_indices, pilot_bits = _compiled_full_capacity_schedule(
            int(h_crop), int(w_crop), int(payload_len), int(seed), int(pilot_count),
            int(block_size), str(pilot_schedule).lower()
        )
        return payload_indices, pilot_indices, pilot_bits, capacity, h_crop, w_crop, payload_len

    rng = np.random.default_rng(int(seed))
    pool_size = int(min(capacity, max(payload_len, int(np.ceil(payload_len * float(candidate_pool_multiplier))))))
    if candidate_budget is not None:
        pool_size = int(min(capacity, max(payload_len, int(candidate_budget))))

    # Host-size-free unified policy: the expensive content score is computed only
    # when the single method is explicitly given a spare candidate pool. With the
    # default pool_size == payload_len, scoring would not change the selected
    # payload schedule; skip it to keep embedding complexity tied to payload size.
    if pool_size <= payload_len:
        payload_indices = rng.permutation(capacity)[:payload_len].astype(np.int32)
        pool = payload_indices
        pool_scores = np.zeros(pool.shape[0], dtype=np.float32)
        h_crop, w_crop = _eng._crop_shape(tuple(int(x) for x in np.asarray(host_rgb).shape), block_size)
    else:
        # Sample a fixed-size keyed candidate pool, score only that pool, then select
        # the strongest payload carriers. This avoids scanning every host block on
        # large video frames.
        if pool_size >= capacity:
            pool = rng.permutation(capacity).astype(np.int32)
        else:
            pool = rng.choice(capacity, size=pool_size, replace=False).astype(np.int32)
        if str(payload_schedule_mode).lower() == "random":
            payload_indices = pool[:payload_len].astype(np.int32)
        elif str(payload_schedule_mode).lower() == "content":
            pool_scores, h_crop, w_crop, capacity = _block_scores_selected(
                host_rgb, pool, block_size=block_size, eta=eta
            )
            local_order = np.argsort(pool_scores, kind="mergesort")[-payload_len:]
            selected_scores = pool_scores[local_order]
            payload_indices = pool[local_order]
            payload_indices = payload_indices[np.argsort(-selected_scores, kind="mergesort")].astype(np.int32)
        else:
            raise ValueError(f"Unknown payload_schedule_mode={payload_schedule_mode!r}")

    pilot_count = int(min(max(0, pilot_count), capacity))
    if str(pilot_schedule).lower() == "stratified":
        pilot_indices = _make_stratified_pilot_indices(
            h_crop, w_crop, pilot_count=pilot_count, seed=int(seed), block_size=int(block_size)
        )
    else:
        rng_p = np.random.default_rng(int(seed) + 919)
        pilot_indices = rng_p.permutation(capacity)[:pilot_count].astype(np.int32)
    pilot_bits = _balanced_pilot_bits(pilot_count, int(seed))
    return payload_indices, pilot_indices, pilot_bits, capacity, h_crop, w_crop, pool_size


def _indices_from_key(params: dict[str, Any], name: str) -> np.ndarray | None:
    if name not in params:
        return None
    arr = np.asarray(params.get(name), dtype=np.int32)
    return arr.reshape(-1)


def _bits_from_key(params: dict[str, Any], name: str) -> np.ndarray | None:
    if name not in params:
        return None
    arr = np.asarray(params.get(name), dtype=np.uint8)
    return arr.reshape(-1)


def _schedules_from_key(params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    bs = int(params.get("block_size", BLOCK_SIZE))
    host_shape = tuple(int(x) for x in params.get("host_shape"))
    wm_shape = tuple(int(x) for x in params.get("watermark_shape"))
    h_crop, w_crop = _eng._crop_shape(host_shape, bs)
    capacity = (h_crop // bs) * (w_crop // bs)
    payload_len = int(wm_shape[0] * wm_shape[1])
    payload_indices = _indices_from_key(params, "payload_indices")
    pilot_indices = _indices_from_key(params, "pilot_indices")
    pilot_bits = _bits_from_key(params, "pilot_bits")
    if payload_indices is not None and pilot_indices is not None and pilot_bits is not None:
        return (
            payload_indices.astype(np.int32), pilot_indices.astype(np.int32),
            pilot_bits.astype(np.uint8), capacity, h_crop, w_crop,
        )

    seed = int(params.get("seed", 2026))
    pilot_count = int(min(max(0, int(params.get("pilot_count", DEFAULT_PILOT_COUNT))), capacity))
    mode = str(params.get("payload_schedule_mode", "random")).lower()
    multiplier = float(params.get("candidate_pool_multiplier", 1.0))
    budget = params.get("candidate_budget")
    pool_size = int(min(capacity, max(payload_len, int(np.ceil(payload_len * multiplier)))))
    if budget is not None:
        pool_size = int(min(capacity, max(payload_len, int(budget))))

    if capacity <= payload_len:
        payload_indices, pilot_indices_cached, pilot_bits_cached = _compiled_full_capacity_schedule(
            int(h_crop), int(w_crop), int(payload_len), int(seed), int(pilot_count),
            int(bs), str(params.get("pilot_schedule", "random")).lower()
        )
        return payload_indices, pilot_indices_cached, pilot_bits_cached, capacity, h_crop, w_crop
    elif mode == "random":
        rng = np.random.default_rng(seed)
        # Mirror _make_content_adaptive_schedules exactly. NumPy's
        # permutation(...)[0:n] and choice(..., replace=False) are both valid
        # samples but do not generate the same sequence for a given seed.
        if pool_size <= payload_len:
            payload_indices = rng.permutation(capacity)[:payload_len].astype(np.int32)
        else:
            pool = (
                rng.permutation(capacity).astype(np.int32)
                if pool_size >= capacity
                else rng.choice(capacity, size=pool_size, replace=False).astype(np.int32)
            )
            payload_indices = pool[:payload_len].astype(np.int32)
    else:
        raise ValueError(
            "A host-ranked content schedule cannot be regenerated without explicit indices; key is incomplete."
        )

    if str(params.get("pilot_schedule", "random")).lower() == "stratified":
        pilot_indices = _make_stratified_pilot_indices(
            h_crop, w_crop, pilot_count=pilot_count, seed=seed, block_size=bs
        )
    else:
        rng_p = np.random.default_rng(seed + 919)
        pilot_indices = rng_p.permutation(capacity)[:pilot_count].astype(np.int32)
    pilot_bits = _balanced_pilot_bits(pilot_count, seed)
    return payload_indices, pilot_indices, pilot_bits, capacity, h_crop, w_crop


def _extract_payload_bits_raw(img: np.ndarray, key_params: dict[str, Any], off_y: int = 0, off_x: int = 0):
    img = np.asarray(img, dtype=np.uint8)
    bs = int(key_params.get("block_size", BLOCK_SIZE))
    h, w = img.shape[:2]
    h_crop = h - (h % bs)
    w_crop = w - (w % bs)
    br = (h_crop - int(off_y)) // bs
    bc = (w_crop - int(off_x)) // bs
    capacity = max(0, br * bc)
    wm_shape = tuple(int(x) for x in key_params.get("watermark_shape"))
    payload_len = int(wm_shape[0] * wm_shape[1])
    payload_indices, _pilot_indices, _pilot_bits, _cap, _hc, _wc = _schedules_from_key(key_params)
    valid = payload_indices < capacity
    idx = payload_indices[valid].astype(np.int32)
    ref_step = float(key_params.get("step", DEFAULT_REFERENCE_STEP))
    step1 = float(key_params.get("step_carrier1", ref_step))
    payload_ba, payload_bb = payload_pair(float(key_params.get("hartley_mix", 0.0)))
    if int(key_params.get('hartley_support',0)) > 0:
        ci,cj,cc=sparse_carrier(float(key_params.get('hartley_mix',0.0)),int(key_params['hartley_support']),str(key_params.get('hartley_carrier_mode','sparse')))
        bits=_call_with_realtime_thread_budget(
            extract_sparse,img,idx,ci,cj,cc,step1,int(w_crop),float(key_params.get('eta',0.07)),int(off_y),int(off_x))
    else:
        bits = _call_with_realtime_thread_budget(
            _eng._extract_one_carrier_bits_parallel,
            img, idx, payload_ba, payload_bb,
            step1, int(w_crop), bs, float(key_params.get("eta", 0.07)), int(off_y), int(off_x)
        )
    pos = np.arange(len(payload_indices), dtype=np.int32)[valid] % payload_len
    return bits, pos, valid


@njit(cache=True, fastmath=True)
def _extract_dual_carrier_bits_conf_kernel(img, indices, ba1, bb1, ba2, bb2, step1, step2, w_crop, block_size, eta, off_y, off_x):
    n = indices.shape[0]
    bs = block_size
    bc = (w_crop - off_x) // bs
    out_bits = np.empty(n, dtype=np.uint8)
    out_conf = np.empty(n, dtype=np.float32)
    for k in range(n):
        idx = int(indices[k])
        r0 = off_y + (idx // bc) * bs
        c0 = off_x + (idx % bc) * bs
        ca1 = 0.0
        cb1 = 0.0
        ca2 = 0.0
        cb2 = 0.0
        for i in range(bs):
            for j in range(bs):
                rr = r0 + i
                cc = c0 + j
                f = _field_rgb_kernel(img[rr, cc, 0], img[rr, cc, 1], img[rr, cc, 2], eta)
                ca1 += f * ba1[i, j]
                cb1 += f * bb1[i, j]
                ca2 += f * ba2[i, j]
                cb2 += f * bb2[i, j]
        v1 = 0.5 * (ca1 - cb1)
        v2 = 0.5 * (ca2 - cb2)
        b1 = int(np.rint(v1 / step1)) & 1
        b2 = int(np.rint(v2 / step2)) & 1
        r1 = _qim_margin_reliability_kernel(v1, step1)
        r2 = _qim_margin_reliability_kernel(v2, step2)
        if b1 == b2:
            out_bits[k] = np.uint8(b1)
            # Agreement between both carriers is much more reliable than a single carrier.
            out_conf[k] = np.float32(1.0 + r1 + r2)
        else:
            if r1 >= r2:
                out_bits[k] = np.uint8(b1)
            else:
                out_bits[k] = np.uint8(b2)
            # Disagreement is weak evidence; use only the margin gap so bad frames cannot dominate.
            gap = abs(r1 - r2)
            if gap < 1e-3:
                gap = 1e-3
            out_conf[k] = np.float32(gap)
    return out_bits, out_conf


@njit(cache=True, fastmath=True, parallel=True)
def _extract_one_carrier_bits_conf_kernel(img, indices, ba, bb, step, w_crop, block_size, eta, off_y, off_x):
    n = indices.shape[0]
    bs = block_size
    bc = (w_crop - off_x) // bs
    out_bits = np.empty(n, dtype=np.uint8)
    out_conf = np.empty(n, dtype=np.float32)
    for k in prange(n):
        idx = int(indices[k])
        r0 = off_y + (idx // bc) * bs
        c0 = off_x + (idx % bc) * bs
        ca = 0.0
        cb = 0.0
        for i in range(bs):
            for j in range(bs):
                rr = r0 + i
                cc = c0 + j
                f = _field_rgb_kernel(img[rr, cc, 0], img[rr, cc, 1], img[rr, cc, 2], eta)
                ca += f * ba[i, j]
                cb += f * bb[i, j]
        v = 0.5 * (ca - cb)
        out_bits[k] = np.uint8(int(np.rint(v / step)) & 1)
        # Normalize to [0,1], with 1 at a QIM center and 0 at a boundary.
        out_conf[k] = np.float32(2.0 * _qim_margin_reliability_kernel(v, step))
    return out_bits, out_conf


def _extract_payload_bits_confidence(img: np.ndarray, key_params: dict[str, Any], off_y: int = 0, off_x: int = 0):
    """Return scrambled payload bits plus QIM-margin confidence for temporal voting.

    This is used by the video pipeline only. It remains blind: the original host is
    not used, and all carrier indices come from the compact key.
    """
    img = np.asarray(img, dtype=np.uint8)
    bs = int(key_params.get("block_size", BLOCK_SIZE))
    h, w = img.shape[:2]
    h_crop = h - (h % bs)
    w_crop = w - (w % bs)
    br = (h_crop - int(off_y)) // bs
    bc = (w_crop - int(off_x)) // bs
    capacity = max(0, br * bc)
    wm_shape = tuple(int(x) for x in key_params.get("watermark_shape"))
    payload_len = int(wm_shape[0] * wm_shape[1])
    payload_indices, _pilot_indices, _pilot_bits, _cap, _hc, _wc = _schedules_from_key(key_params)
    valid = payload_indices < capacity
    idx = payload_indices[valid].astype(np.int32)
    ref_step = float(key_params.get("step", DEFAULT_REFERENCE_STEP))
    step1 = float(key_params.get("step_carrier1", ref_step))
    payload_ba, payload_bb = payload_pair(float(key_params.get("hartley_mix", 0.0)))
    if int(key_params.get('hartley_support',0))>0:
        ci,cj,cc=sparse_carrier(float(key_params.get('hartley_mix',0.0)),int(key_params['hartley_support']),str(key_params.get('hartley_carrier_mode','sparse')))
        bits,conf=_call_with_realtime_thread_budget(
            extract_sparse_conf,img,idx,ci,cj,cc,step1,int(w_crop),float(key_params.get('eta',0.07)),int(off_y),int(off_x))
    else:
        bits, conf = _call_with_realtime_thread_budget(
            _extract_one_carrier_bits_conf_kernel,
            img, idx, payload_ba, payload_bb,
            step1, int(w_crop), bs, float(key_params.get("eta", 0.07)), int(off_y), int(off_x)
        )
    pos = np.arange(len(payload_indices), dtype=np.int32)[valid] % payload_len
    return bits, pos, conf.astype(np.float32)


def _payload_from_candidate(img: np.ndarray, key_params: dict[str, Any], off_y: int = 0, off_x: int = 0):
    bits, pos, valid = _extract_payload_bits_raw(img, key_params, off_y, off_x)
    wm_shape = tuple(int(x) for x in key_params.get("watermark_shape"))
    payload_len = int(wm_shape[0] * wm_shape[1])
    if bits.shape[0] == payload_len and int(np.sum(valid)) == payload_len:
        rec_scr = bits.reshape(wm_shape)
    else:
        votes0 = np.zeros(payload_len, dtype=np.int16)
        votes1 = np.zeros(payload_len, dtype=np.int16)
        np.add.at(votes0, pos[bits == 0], 1)
        np.add.at(votes1, pos[bits == 1], 1)
        rec_scr = (votes1 > votes0).astype(np.uint8).reshape(wm_shape)
    rec = (
        arnold_inverse(rec_scr, int(key_params["arnold_iter"]), int(key_params["arnold_period"]))
        if bool(key_params.get("scramble_payload", True)) else rec_scr
    )
    return (rec * 255).astype(np.uint8), int(np.sum(valid))


def _payload_from_shift(img: np.ndarray, key_params: dict[str, Any], sy: int, sx: int, off_y: int = 0, off_x: int = 0):
    img = np.asarray(img, dtype=np.uint8)
    bs = int(key_params.get("block_size", BLOCK_SIZE))
    h, w = img.shape[:2]
    h_crop = h - (h % bs)
    w_crop = w - (w % bs)
    br = (h_crop - int(off_y)) // bs
    bc = (w_crop - int(off_x)) // bs
    capacity = max(0, br * bc)
    wm_shape = tuple(int(x) for x in key_params.get("watermark_shape"))
    payload_len = int(wm_shape[0] * wm_shape[1])
    payload_indices, _pilot_indices, _pilot_bits, _cap, _hc, _wc = _schedules_from_key(key_params)
    valid = payload_indices < capacity
    idx = payload_indices[valid].astype(np.int32)
    ref_step = float(key_params.get("step", DEFAULT_REFERENCE_STEP))
    step1 = float(key_params.get("step_carrier1", ref_step))
    payload_ba, payload_bb = payload_pair(float(key_params.get("hartley_mix", 0.0)))
    if int(key_params.get('hartley_support',0))>0:
        ci,cj,cc=sparse_carrier(float(key_params.get('hartley_mix',0.0)),int(key_params['hartley_support']),str(key_params.get('hartley_carrier_mode','sparse')))
        p_=(PAY10_BA.astype(np.float64)+PAY10_BB.astype(np.float64))/np.sqrt(2.0)
        q_=np.zeros((8,8),dtype=np.float64)
        q_[ci,cj]=cc*np.sqrt(2.0)
        payload_ba=((p_+q_)/np.sqrt(2.0)).astype(np.float32)
        payload_bb=((p_-q_)/np.sqrt(2.0)).astype(np.float32)
    bits = _call_with_realtime_thread_budget(
        _eng._extract_one_carrier_bits_shift,
        img, idx, payload_ba, payload_bb,
        step1, int(w_crop), bs, float(key_params.get("eta", 0.07)), int(off_y), int(off_x), int(sy), int(sx)
    )
    pos = np.arange(len(payload_indices), dtype=np.int32)[valid] % payload_len
    if bits.shape[0] == payload_len and int(np.sum(valid)) == payload_len:
        rec_scr = bits.reshape(wm_shape)
    else:
        votes0 = np.zeros(payload_len, dtype=np.int16)
        votes1 = np.zeros(payload_len, dtype=np.int16)
        np.add.at(votes0, pos[bits == 0], 1)
        np.add.at(votes1, pos[bits == 1], 1)
        rec_scr = (votes1 > votes0).astype(np.uint8).reshape(wm_shape)
    rec = (
        arnold_inverse(rec_scr, int(key_params["arnold_iter"]), int(key_params["arnold_period"]))
        if bool(key_params.get("scramble_payload", True)) else rec_scr
    )
    return (rec * 255).astype(np.uint8), int(np.sum(valid))


def _normalize_jilp_key(key: Any) -> dict[str, Any]:
    params = _eng._normalize_key(key)
    params.setdefault("step", DEFAULT_REFERENCE_STEP)
    params.setdefault("pilot_count", DEFAULT_PILOT_COUNT)
    params.setdefault("step_carrier1", float(params.get("step", DEFAULT_REFERENCE_STEP)))
    params.setdefault("step_carrier2", 0.0)
    return params


def _pilot_score_fast(img: np.ndarray, key_params: dict[str, Any], off_y: int = 0, off_x: int = 0) -> tuple[float, int]:
    bs = int(key_params.get("block_size", BLOCK_SIZE))
    h, w = img.shape[:2]
    h_crop = h - (h % bs)
    w_crop = w - (w % bs)
    br = (h_crop - int(off_y)) // bs
    bc = (w_crop - int(off_x)) // bs
    capacity = max(0, br * bc)
    _payload_indices, pilot_indices, pilot_bits, _cap, _hc, _wc = _schedules_from_key(key_params)
    valid = pilot_indices < capacity
    idx = pilot_indices[valid].astype(np.int32)
    bits = pilot_bits[valid].astype(np.uint8)
    if idx.size == 0:
        return 0.0, 0
    ebits = _eng._extract_one_carrier_bits(
        np.asarray(img, dtype=np.uint8), idx, _eng.PILOT_BA, _eng.PILOT_BB,
        float(key_params.get("step_pilot", key_params.get("step", DEFAULT_REFERENCE_STEP))),
        int(w_crop), bs, float(key_params.get("eta", 0.07)), int(off_y), int(off_x)
    )
    return float(np.mean(ebits == bits)), int(idx.size)


def _best_integer_translation_subset(img: np.ndarray, key_params: dict[str, Any], *, radius: int, subset: int) -> tuple[float, int, int, int]:
    bs = int(key_params.get("block_size", BLOCK_SIZE))
    h, w = img.shape[:2]
    h_crop = h - (h % bs)
    w_crop = w - (w % bs)
    capacity = max(0, (h_crop // bs) * (w_crop // bs))
    _payload_indices, pilot_indices, pilot_bits, _cap, _hc, _wc = _schedules_from_key(key_params)
    valid = pilot_indices < capacity
    idx = pilot_indices[valid].astype(np.int32)
    bits = pilot_bits[valid].astype(np.uint8)
    if idx.size == 0:
        return 0.0, 0, 0, 0
    n = min(int(subset), idx.size)
    idx = idx[:n]
    bits = bits[:n]
    sc, sy, sx = _eng._best_integer_translation_kernel(
        np.asarray(img, dtype=np.uint8), idx, bits, _eng.PILOT_BA, _eng.PILOT_BB,
        float(key_params.get("step_pilot", key_params.get("step", DEFAULT_REFERENCE_STEP))),
        int(w_crop), bs, float(key_params.get("eta", 0.07)), int(radius)
    )
    return float(sc), int(sy), int(sx), int(n)


def _best_candidate(img: np.ndarray, key_params: dict[str, Any], *, quick_accept: float, translation_radius: int, translation_subset: int):
    """One sparse phase objective replaces full-image candidate warping.

    The pilot-triggered fast path remains available; the registration search
    is an optimization over one phase functional, not several score scales.
    """
    from .phase_registration import register_sparse_phase

    _payload, pidx, pbits, _cap, _hc, _wc = _schedules_from_key(key_params)
    valid = pidx < ((img.shape[0] // BLOCK_SIZE) * (img.shape[1] // BLOCK_SIZE))
    pidx = pidx[valid][:max(1, int(translation_subset))]
    pbits = pbits[valid][:len(pidx)]
    if not len(pidx):
        return img, 0, 0, {"path": "raw_fast_jilp", "candidate": "no_pilots", "score": 0.0,
                             "pilot_valid": 0, "registration": "sparse_qim_phase"}
    aligned, meta = register_sparse_phase(
        img, pidx, pbits, float(key_params.get("step_pilot", key_params.get("step", DEFAULT_REFERENCE_STEP))),
        float(key_params.get("eta", 0.07)), quick_accept=quick_accept,
        translation_radius=translation_radius,
        phase_refinement=str(key_params.get("phase_refinement", "off")),
        certified_pruning=bool(key_params.get("certified_phase_pruning", False)),
    )
    # Translation is incorporated in the single rendered warp; do not apply it
    # a second time during payload decoding.
    return aligned, 0, 0, meta


def embed_jilp_qim(
    host_rgb: np.ndarray,
    watermark_binary: np.ndarray,
    *,
    seed: int = 2026,
    step: float | None = None,
    rho_frac: float = 0.45,
    pilot_count: int = DEFAULT_PILOT_COUNT,
    pilot_step: float | None = None,
    pilot_rho_frac: float = 0.49,
    eta: float = 0.07,
    verify_iters: int = 0,
    candidate_pool_multiplier: float = 1.0,
    candidate_budget: int | None = None,
    use_second_carrier: bool = False,
    pilot_schedule: str = "stratified",
    use_integer_lattice_closure: bool = True,
    projection_mode: str = "minimum_distortion",
    enable_pilots: bool = True,
    enable_sync_search: bool = True,
    payload_schedule_mode: str = "random",
    field_mode: str = "opponent",
    temporal_vote_mode: str = "hard",
    scramble_payload: bool = True,
    phase_refinement: str = "off",
    certified_phase_pruning: bool = False,
    hartley_mix: float = 0.0,
    hartley_support: int = 0,
    hartley_carrier_mode: str = "sparse",
) -> tuple[np.ndarray, dict[str, Any]]:
    if hartley_carrier_mode not in ("sparse", "frequency_preserving"):
        raise ValueError("invalid hartley_carrier_mode")
    if hartley_carrier_mode != "sparse" and (int(hartley_support) <= 0 or not float(hartley_mix) > 0):
        raise ValueError("frequency_preserving requires sparse Hartley payload carrier")
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba is required for JILP-QIM real-time kernels. Install numba.")
    payload_ba, payload_bb = payload_pair(float(hartley_mix))
    if int(hartley_support)>0:
        ci_,cj_,cc_=sparse_carrier(float(hartley_mix),int(hartley_support),hartley_carrier_mode)
        p_=(PAY10_BA.astype(np.float64)+PAY10_BB.astype(np.float64))/np.sqrt(2.0)
        q_=np.zeros((8,8),dtype=np.float64)
        q_[ci_,cj_]=cc_*np.sqrt(2.0)
        payload_ba=((p_+q_)/np.sqrt(2.0)).astype(np.float32)
        payload_bb=((p_-q_)/np.sqrt(2.0)).astype(np.float32)
    if int(verify_iters) != 0:
        raise ValueError("verify_iters is obsolete; JILP-QIM has no finished-image repair loop")
    if str(projection_mode) not in {"minimum_distortion", "center_qim"}:
        raise ValueError("projection_mode must be 'minimum_distortion' or 'center_qim'")
    if str(payload_schedule_mode) not in {"content", "random"}:
        raise ValueError("payload_schedule_mode must be 'content' or 'random'")
    if str(field_mode) not in {"opponent", "luminance"}:
        raise ValueError("field_mode must be 'opponent' or 'luminance'")
    if str(temporal_vote_mode) not in {"confidence_weighted", "hard"}:
        raise ValueError("temporal_vote_mode must be 'confidence_weighted' or 'hard'")
    host = np.asarray(host_rgb, dtype=np.uint8)
    wm = (np.asarray(watermark_binary) > 127).astype(np.uint8)
    used_step = float(DEFAULT_REFERENCE_STEP if step is None else step)
    params = _build_key_params(
        tuple(int(x) for x in host.shape), tuple(int(x) for x in wm.shape),
        seed=int(seed), step=used_step, rho_frac=float(rho_frac), pilot_count=int(pilot_count),
        pilot_step=pilot_step, pilot_rho_frac=float(pilot_rho_frac), eta=float(eta),
        candidate_pool_multiplier=float(candidate_pool_multiplier), candidate_budget=candidate_budget,
        use_second_carrier=bool(use_second_carrier), pilot_schedule=str(pilot_schedule),
        use_integer_lattice_closure=bool(use_integer_lattice_closure), projection_mode=str(projection_mode),
        enable_pilots=bool(enable_pilots), enable_sync_search=bool(enable_sync_search),
        payload_schedule_mode=str(payload_schedule_mode), field_mode=str(field_mode),
        temporal_vote_mode=str(temporal_vote_mode), scramble_payload=bool(scramble_payload),
        phase_refinement=str(phase_refinement),
        certified_phase_pruning=bool(certified_phase_pruning),
    )
    payload_indices, pilot_indices, pilot_bits, capacity, _h_crop, w_crop, candidate_pool_used = _make_content_adaptive_schedules(
        host, tuple(int(x) for x in wm.shape), seed=int(seed),
        pilot_count=int(pilot_count if enable_pilots else 0),
        block_size=BLOCK_SIZE, eta=float(eta if str(field_mode) == "opponent" else 0.0),
        candidate_pool_multiplier=float(candidate_pool_multiplier),
        candidate_budget=candidate_budget, pilot_schedule=str(pilot_schedule),
        payload_schedule_mode=str(payload_schedule_mode),
    )
    # Default schedules are fully determined by shape, seed, and pilot count.
    # Regenerating them keeps the key compact and avoids serializing thousands
    # of indices in the timed embedding path.  Explicit schedules are retained
    # only for genuinely host-adaptive or non-default pilot placement.
    actual_content_adaptation = bool(
        str(payload_schedule_mode).lower() == "content"
        and int(candidate_pool_used) > int(wm.size)
    )
    # Random payload schedules and both pilot schedules are deterministic from
    # shape/seed/config. Only a host-ranked content schedule needs explicit indices.
    explicit_schedule = actual_content_adaptation
    if explicit_schedule:
        params["payload_indices"] = payload_indices.astype(int).tolist()
        params["pilot_indices"] = pilot_indices.astype(int).tolist()
        params["pilot_bits"] = pilot_bits.astype(int).tolist()
        params["schedule_storage"] = "explicit_host_ranked"
        params["side_information"] = "explicit_host_ranked_payload_schedule_no_original_host"
    else:
        params["schedule_storage"] = "regenerate_from_seed"
        params["side_information"] = "schedule_regenerated_from_seed_no_original_host"
    if hartley_carrier_mode != "sparse":
        params["hartley_carrier_mode"] = hartley_carrier_mode
    if int(hartley_support)>0:
        params["hartley_support"] = int(hartley_support)
    if float(hartley_mix) > 0.0:
        params["hartley_mix"] = float(hartley_mix)
        params["payload_carrier"] = "orthogonalized weighted h-Hartley--cosine convolutional DCT carrier"
        params["carrier_convolution"] = "h=1, gamma=cos(w), f[0]=1/2, f[+/-1]=1/4, finite zero extension"
    params["capacity"] = int(capacity)
    params["adaptive_spare_capacity"] = int(max(0, capacity - int(wm.size)))
    params["candidate_pool_used"] = int(candidate_pool_used)
    params["content_adaptive_block_selection"] = actual_content_adaptation
    params["structural_components"]["content_adaptive_schedule"] = actual_content_adaptation

    bits = (
        arnold_transform(wm, int(params["arnold_iter"]))
        if bool(scramble_payload) else wm
    ).ravel().astype(np.uint8)
    out = host.copy()
    step1 = float(params["step_carrier1"])
    effective_eta = float(params["eta"])
    # Physical traversal is separated from the keyed logical bit schedule.
    # Processing blocks in raster order improves cache locality without changing
    # which bit is assigned to any block.
    execution_order = np.argsort(payload_indices, kind="stable")
    payload_indices_exec = payload_indices[execution_order]
    bits_exec = bits[execution_order]

    # Pilots use the exactly orthogonal low-frequency sum carrier.
    if bool(enable_pilots) and len(pilot_indices):
        _eng._embed_one_carrier_md(
            host, out, pilot_indices, pilot_bits, _eng.PILOT_BA, _eng.PILOT_BB,
            float(params["step_pilot"]), float(params["step_pilot"]) * float(params["pilot_rho_frac"]),
            int(w_crop), BLOCK_SIZE, effective_eta
        )

    if int(hartley_support)>0:
        if not bool(use_integer_lattice_closure) or str(projection_mode) != 'minimum_distortion':
            raise ValueError('sparse carrier requires minimum-distortion integer closure')
        ci,cj,cc=sparse_carrier(float(hartley_mix),int(hartley_support),hartley_carrier_mode)
        rt,rc,rs=sparse_leverage(float(hartley_mix),int(hartley_support),effective_eta,hartley_carrier_mode)
        (integer_updates,unresolved,corrected_blocks,initial_parity_failures,
         initial_guard_failures,final_guard_failures) = _call_with_realtime_thread_budget(
            embed_sparse,out,payload_indices_exec,bits_exec,ci,cj,cc,rt,rc,rs,
            step1,step1*float(rho_frac),int(w_crop),effective_eta)
    elif bool(use_integer_lattice_closure):
        kernel = (
            _eng._embed_one_carrier_jilp
            if str(projection_mode) == "minimum_distortion"
            else _eng._embed_one_carrier_jilp_center
        )
        (
            integer_updates, unresolved, corrected_blocks,
            initial_parity_failures, initial_guard_failures, final_guard_failures,
        ) = _call_with_realtime_thread_budget(
            kernel,
            out, payload_indices_exec, bits_exec, payload_ba, payload_bb, step1,
            step1 * float(rho_frac), int(w_crop), BLOCK_SIZE, effective_eta
        )
    else:
        kernel = (
            _eng._embed_one_carrier_md
            if str(projection_mode) == "minimum_distortion"
            else _eng._embed_one_carrier_center
        )
        kernel(
            host, out, payload_indices_exec, bits_exec, payload_ba, payload_bb, step1,
            step1 * float(rho_frac), int(w_crop), BLOCK_SIZE, effective_eta
        )
        integer_updates, unresolved = 0, 0
        corrected_blocks, initial_parity_failures, initial_guard_failures, final_guard_failures = 0, 0, 0, 0
    if bool(params.get("dual_payload_carrier", False)):
        raise ValueError("The repair-free JILP method uses one payload carrier; dual carrier is ablation-only and disabled.")
    # No finished-image extraction or post-hoc repair is performed.
    params["clean_verification_iters"] = 0
    params["clean_repaired_bits"] = 0
    params["integer_lattice_unit_updates"] = int(integer_updates)
    params["integer_lattice_unresolved"] = int(unresolved)
    params["integer_lattice_corrected_blocks"] = int(corrected_blocks)
    params["integer_lattice_initial_parity_failures"] = int(initial_parity_failures)
    params["integer_lattice_initial_guard_failures"] = int(initial_guard_failures)
    params["integer_lattice_payload_blocks"] = int(len(payload_indices_exec))
    params["integer_lattice_final_guard_failures"] = int(final_guard_failures)
    if bool(use_integer_lattice_closure) and int(unresolved) != 0:
        raise RuntimeError(
            f"JILP embedding is infeasible for {int(unresolved)} payload block(s); "
            "no non-exact watermarked image is returned."
        )
    if bool(use_integer_lattice_closure) and bool(enable_pilots) and len(pilot_indices):
        # Only final uint8 pilots are inspected, never a full-image repair pass.
        # On overlapping blocks, unit corrections must preserve payload parity.
        payload_lookup = np.full(capacity, -1, dtype=np.int16)
        payload_lookup[payload_indices] = bits.astype(np.int16)
        pilot_basis = 0.5 * (_eng.PILOT_BA - _eng.PILOT_BB)
        payload_basis = 0.5 * (payload_ba - payload_bb)
        pilot_updates, pilot_corrected, pilot_unresolved = _eng._close_pilot_integer_joint(
            out, pilot_indices, pilot_bits, payload_lookup, pilot_basis, payload_basis,
            float(params["step_pilot"]), step1, int(w_crop), BLOCK_SIZE, effective_eta,
        )
        params["integer_pilot_unit_updates"] = int(pilot_updates)
        params["integer_pilot_corrected_blocks"] = int(pilot_corrected)
        params["integer_pilot_unresolved"] = int(pilot_unresolved)
        if int(pilot_unresolved) != 0:
            raise RuntimeError(f"Pilot parity infeasible in {pilot_unresolved} uint8 block(s)")
    else:
        params["integer_pilot_unit_updates"] = 0
        params["integer_pilot_corrected_blocks"] = 0
        params["integer_pilot_unresolved"] = 0
    return out, params


def extract_jilp_qim(
    image_rgb: np.ndarray,
    key: Any,
    *,
    sync: bool = True,
    quick_accept: float = 0.72,
    translation_radius: int = 8,
    translation_subset: int = 96,
    return_meta: bool = False,
):
    key_params = _normalize_jilp_key(key)
    img = np.asarray(image_rgb, dtype=np.uint8)
    if (not sync) or (not bool(key_params.get("sync_search_enabled", True))) or int(key_params.get("pilot_count", 0)) <= 0:
        rec, used = _payload_from_candidate(img, key_params, 0, 0)
        meta = {"path": "raw_no_sync", "candidate": "raw", "payload_used": used, "extract_type": "fast_jilp_single"}
        return (rec, meta) if return_meta else rec

    best_img, sy, sx, meta = _best_candidate(
        img, key_params, quick_accept=float(quick_accept),
        translation_radius=int(translation_radius), translation_subset=int(translation_subset)
    )
    if sy != 0 or sx != 0:
        rec, used = _payload_from_shift(best_img, key_params, sy, sx, 0, 0)
    else:
        rec, used = _payload_from_candidate(best_img, key_params, 0, 0)
    meta = {**meta, "payload_used": used, "extract_type": "robust_jilp_single"}
    return (rec, meta) if return_meta else rec


def embed_proposal(
    host_rgb: np.ndarray,
    watermark_binary: np.ndarray,
    *,
    seed: int = 2026,
    repeat: int | str = "auto",
    step: float | None = None,
    method_params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, WatermarkKey]:
    del repeat  # the released proposal embeds one complete payload schedule
    host = np.asarray(host_rgb, dtype=np.uint8)
    wm = np.asarray(watermark_binary, dtype=np.uint8)
    config = JILPQIMConfig.from_mapping(method_params)
    if step is not None:
        config = config.with_step(float(step))
    watermarked, params = embed_jilp_qim(
        host,
        wm,
        seed=seed,
        step=config.step,
        rho_frac=config.rho_frac,
        pilot_count=config.pilot_count,
        pilot_step=config.pilot_step,
        pilot_rho_frac=config.pilot_rho_frac,
        eta=config.eta,
        candidate_pool_multiplier=config.candidate_pool_multiplier,
        candidate_budget=config.candidate_budget,
        use_second_carrier=config.use_second_carrier,
        pilot_schedule=config.pilot_schedule,
        use_integer_lattice_closure=config.use_integer_lattice_closure,
        projection_mode=config.projection_mode,
        enable_pilots=config.enable_pilots,
        enable_sync_search=config.enable_sync_search,
        payload_schedule_mode=config.payload_schedule_mode,
        field_mode=config.field_mode,
        temporal_vote_mode=config.temporal_vote_mode,
        scramble_payload=config.scramble_payload,
        phase_refinement=config.phase_refinement,
        certified_phase_pruning=config.certified_phase_pruning,
        hartley_mix=config.hartley_mix,
        hartley_support=config.hartley_support,
        hartley_carrier_mode=config.hartley_carrier_mode,
    )
    params["frozen_config"] = config.to_dict()
    params["config_hash_sha256"] = config.sha256()
    side_information = str(params.get("side_information", "schedule_regenerated_from_seed_no_original_host"))
    key = WatermarkKey(
        method_id=METHOD_ID,
        host_shape=tuple(int(x) for x in host.shape),
        watermark_shape=tuple(int(x) for x in wm.shape),
        seed=int(seed),
        repeat=1,
        step=float(params["step"]),
        arnold_iter=int(params["arnold_iter"]),
        arnold_period=int(params["arnold_period"]),
        threshold=127,
        params=params,
        schedule=[],
        fully_blind=True,
        side_information=side_information,
    )
    return watermarked, key


def extract_proposal(image_rgb: np.ndarray, key: WatermarkKey | dict[str, Any]) -> np.ndarray:
    return extract_jilp_qim(image_rgb, key, sync=True, return_meta=False)


def embed(
    host_rgb: np.ndarray,
    watermark_binary: np.ndarray,
    *,
    seed: int = 2026,
    repeat: int | str = "auto",
    step: float | None = None,
    method_params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, WatermarkKey]:
    return embed_proposal(
        host_rgb, watermark_binary, seed=seed, repeat=repeat, step=step,
        method_params=method_params,
    )


def extract(image_rgb: np.ndarray, key: WatermarkKey | dict[str, Any]) -> np.ndarray:
    return extract_proposal(image_rgb, key)


def warmup_jilp_qim() -> None:
    if not NUMBA_AVAILABLE:
        return
    host = np.zeros((64, 64, 3), dtype=np.uint8)
    wm = np.zeros((8, 8), dtype=np.uint8)
    out, params = embed_jilp_qim(host, wm, seed=1, pilot_count=16, verify_iters=0)
    key = {
        "method_id": METHOD_ID,
        "host_shape": host.shape,
        "watermark_shape": wm.shape,
        "seed": 1,
        "step": DEFAULT_REFERENCE_STEP,
        "arnold_iter": params["arnold_iter"],
        "arnold_period": params["arnold_period"],
        "params": params,
    }
    key_params = _normalize_jilp_key(key)
    _ = extract_jilp_qim(out, key, sync=False)
    # Video extraction uses the confidence-voting kernel directly. Warm it here
    # so the first measured video frame is not charged for Numba compilation.
    _ = _extract_payload_bits_confidence(out, key_params, 0, 0)

