from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..common.block_dct import dct_blocks_from_rgb, dct_blocks_to_rgb, rgb_to_dct_blocks
from ..common.embedding import _vote_bits
from ..common.types import MethodRef, WatermarkKey
from ..core import make_block_indices

METHOD_ID = "ca_qim_a2_mao2024"
BLOCK_SIZE = 8
COEFFICIENTS = ((2, 3), (3, 2))
DEFAULT_STEP = 24.0
LATTICE_ALPHA = 4
LATTICE_DIMENSION = 2
BITS_PER_DIMENSION = int(np.log2(LATTICE_ALPHA))
BITS_PER_VECTOR = LATTICE_DIMENSION * BITS_PER_DIMENSION
COSET_COUNT = LATTICE_ALPHA ** LATTICE_DIMENSION
A2_BASE = np.asarray([[1.0, 0.5], [0.0, np.sqrt(3.0) / 2.0]], dtype=np.float64)
COSET_REPRESENTATIVES = np.asarray(
    list(product(range(LATTICE_ALPHA), repeat=LATTICE_DIMENSION)), dtype=np.int64
)
NEIGHBOR_OFFSETS = np.asarray(list(product(range(-1, 2), repeat=2)), dtype=np.int64)

METHOD_REF = MethodRef(
    id=METHOD_ID,
    display_name="A2 CA-QIM, R=2 (Mao et al.)",
    paper=(
        "J. Mao, H. Tang, S. Lyu, Z. Zhou, and X. Cao, Content-Aware "
        "Quantization Index Modulation: Leveraging Data Statistics for Enhanced "
        "Image Watermarking, IEEE Transactions on Information Forensics and "
        "Security 19, 2024."
    ),
    url="https://doi.org/10.1109/TIFS.2023.3342612",
    implementation_note=(
        "Paper-aligned A2 specialization with code rate R=log2(alpha)=2, alpha=4, "
        "self-nested 4A2 coarse lattice, 16 cosets, a 16x16 content-dependent "
        "adjacency matrix, deterministic maximum-weight canonical labeling, and "
        "nearest-point CA-QIM embedding. One two-dimensional carrier transmits "
        "four bits. The benchmark retains its common balanced watermark payload."
    ),
    fully_blind=False,
)


def _capacity_vectors(host_shape: tuple[int, ...]) -> int:
    return (int(host_shape[0]) // BLOCK_SIZE) * (int(host_shape[1]) // BLOCK_SIZE)


def _lattice(step: float) -> tuple[np.ndarray, np.ndarray]:
    delta = float(step)
    if delta <= 0:
        raise ValueError("step must be positive")
    fine = delta * A2_BASE
    return fine, np.linalg.inv(fine)


def nearest_points_all_cosets(carriers: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the nearest point and squared distance in every 4A2 coset."""
    x = np.asarray(carriers, dtype=np.float64).reshape(-1, 2)
    fine, inv_fine = _lattice(step)
    coordinates = x @ inv_fine.T
    reps = COSET_REPRESENTATIVES
    centers = np.rint(
        (coordinates[:, None, :] - reps[None, :, :]) / float(LATTICE_ALPHA)
    ).astype(np.int64)
    lattice_coordinates = (
        reps[None, :, None, :]
        + LATTICE_ALPHA * (centers[:, :, None, :] + NEIGHBOR_OFFSETS[None, None, :, :])
    )
    points = lattice_coordinates @ fine.T
    distances = np.sum((points - x[:, None, None, :]) ** 2, axis=-1)
    best = np.argmin(distances, axis=-1)
    rows = np.arange(x.shape[0])[:, None]
    cosets = np.arange(COSET_COUNT)[None, :]
    nearest = points[rows, cosets, best]
    best_dist = distances[rows, cosets, best]
    return nearest, best_dist


def a2_decode_cosets(carriers: np.ndarray, step: float) -> np.ndarray:
    _points, distances = nearest_points_all_cosets(carriers, step)
    return np.argmin(distances, axis=1).astype(np.uint8)


def a2_quantize_to_cosets(carriers: np.ndarray, cosets: np.ndarray, step: float) -> np.ndarray:
    points, _distances = nearest_points_all_cosets(carriers, step)
    c = np.asarray(cosets, dtype=np.int64).reshape(-1)
    if points.shape[0] != c.size:
        raise ValueError("carrier and coset counts differ")
    if np.any((c < 0) | (c >= COSET_COUNT)):
        raise ValueError(f"coset indices must lie in [0,{COSET_COUNT - 1}]")
    return points[np.arange(c.size), c]


def bits_to_symbols(bits: np.ndarray) -> tuple[np.ndarray, int]:
    b = np.asarray(bits, dtype=np.uint8).ravel()
    padding = int((-b.size) % BITS_PER_VECTOR)
    if padding:
        b = np.concatenate([b, np.zeros(padding, dtype=np.uint8)])
    groups = b.reshape(-1, BITS_PER_VECTOR)
    weights = (1 << np.arange(BITS_PER_VECTOR, dtype=np.uint16))[None, :]
    return np.sum(groups.astype(np.uint16) * weights, axis=1).astype(np.uint8), padding


def symbols_to_bits(symbols: np.ndarray, bit_length: int) -> np.ndarray:
    s = np.asarray(symbols, dtype=np.uint16).ravel()
    bits = ((s[:, None] >> np.arange(BITS_PER_VECTOR, dtype=np.uint16)) & 1).astype(np.uint8)
    return bits.ravel()[: int(bit_length)]


def adjacency_matrix(carriers: np.ndarray, message_symbols: np.ndarray, step: float) -> np.ndarray:
    nearest = a2_decode_cosets(carriers, step).astype(np.int64)
    symbols = np.asarray(message_symbols, dtype=np.int64).ravel()
    if nearest.size != symbols.size:
        raise ValueError("carrier and message counts differ")
    W = np.zeros((COSET_COUNT, COSET_COUNT), dtype=np.int64)
    np.add.at(W, (nearest, symbols), 1)
    return W


def _maximum_assignment_weight(matrix: np.ndarray) -> int:
    W = np.asarray(matrix, dtype=np.int64)
    if W.size == 0:
        return 0
    rows, cols = linear_sum_assignment(-W)
    return int(W[rows, cols].sum())


def canonical_labeling(W: np.ndarray) -> np.ndarray:
    """Deterministic maximum-weight message-to-coset assignment.

    Hungarian optimization supplies the exact maximum weight. A short
    feasibility refinement then chooses the lexicographically smallest mapping
    among tied optima, avoiding solver-dependent tie choices in saved keys.
    """
    matrix = np.asarray(W, dtype=np.int64)
    if matrix.shape != (COSET_COUNT, COSET_COUNT):
        raise ValueError(f"A2 R=2 canonical labeling expects a {COSET_COUNT}x{COSET_COUNT} matrix")
    optimum = _maximum_assignment_weight(matrix)
    mapping = np.empty(COSET_COUNT, dtype=np.uint8)
    available = list(range(COSET_COUNT))
    accumulated = 0
    for message in range(COSET_COUNT):
        remaining_messages = list(range(message + 1, COSET_COUNT))
        chosen: int | None = None
        for coset in available:
            remaining_cosets = [x for x in available if x != coset]
            if remaining_messages:
                residual = matrix[np.ix_(remaining_cosets, remaining_messages)]
                possible = _maximum_assignment_weight(residual)
            else:
                possible = 0
            if accumulated + int(matrix[coset, message]) + possible == optimum:
                chosen = coset
                break
        if chosen is None:  # pragma: no cover - defensive against solver inconsistency
            raise RuntimeError("could not construct deterministic optimal canonical labeling")
        mapping[message] = chosen
        accumulated += int(matrix[chosen, message])
        available.remove(chosen)
    return mapping


def inverse_labeling(message_to_coset: np.ndarray) -> np.ndarray:
    mapping = np.asarray(message_to_coset, dtype=np.int64).ravel()
    if mapping.size != COSET_COUNT or sorted(mapping.tolist()) != list(range(COSET_COUNT)):
        raise ValueError(f"canonical labeling must be a permutation of {COSET_COUNT} cosets")
    inverse = np.empty(COSET_COUNT, dtype=np.uint8)
    inverse[mapping] = np.arange(COSET_COUNT, dtype=np.uint8)
    return inverse


def prepare_embedding(
    host_rgb: np.ndarray,
    watermark_binary: np.ndarray,
    *,
    seed: int = 2026,
    repeat: int | str = 1,
    step: float | None = None,
    method_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del method_params
    host = np.asarray(host_rgb, dtype=np.uint8)
    wm = (np.asarray(watermark_binary) > 127).astype(np.uint8)
    payload = int(wm.size)
    capacity_vectors = _capacity_vectors(host.shape)
    if isinstance(repeat, str):
        repeat_i = max(1, (BITS_PER_VECTOR * capacity_vectors) // payload) if repeat.lower() == "auto" else int(repeat)
    else:
        repeat_i = int(repeat)
    base_symbols, padding = bits_to_symbols(wm.ravel())
    symbols = np.tile(base_symbols, repeat_i).astype(np.uint8)
    if symbols.size > capacity_vectors:
        raise ValueError(f"capacity too small: need {symbols.size} vectors, have {capacity_vectors}")
    delta = float(DEFAULT_STEP if step is None else step)
    coeffs, y, cb, cr, h_crop, w_crop = rgb_to_dct_blocks(host, BLOCK_SIZE)
    indices = make_block_indices(h_crop, w_crop, BLOCK_SIZE, symbols.size, 1, int(seed)).astype(np.int32)
    carriers = np.stack([coeffs[indices, u, v] for u, v in COEFFICIENTS], axis=1)
    W = adjacency_matrix(carriers, symbols, delta)
    mapping = canonical_labeling(W)
    return {
        "coeffs": coeffs,
        "y": y,
        "cb": cb,
        "cr": cr,
        "h_crop": h_crop,
        "w_crop": w_crop,
        "indices": indices,
        "symbols": symbols,
        "padding": padding,
        "repeat": repeat_i,
        "step": delta,
        "seed": int(seed),
        "W": W,
        "mapping": mapping,
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
    delta = float(prepared["step"] if step is None else step)
    if not np.isclose(delta, float(prepared["step"]), atol=1e-12, rtol=0):
        raise ValueError("CA-QIM canonical labeling is step-dependent; prepare again for a changed step")
    coeffs = np.asarray(prepared["coeffs"], dtype=np.float64).copy()
    indices = np.asarray(prepared["indices"], dtype=np.int32)
    symbols = np.asarray(prepared["symbols"], dtype=np.uint8)
    mapping = np.asarray(prepared["mapping"], dtype=np.uint8)
    carriers = np.stack([coeffs[indices, u, v] for u, v in COEFFICIENTS], axis=1)
    target_cosets = mapping[symbols]
    quantized = a2_quantize_to_cosets(carriers, target_cosets, delta)
    for column, (u, v) in enumerate(COEFFICIENTS):
        coeffs[indices, u, v] = quantized[:, column]
    watermarked = dct_blocks_to_rgb(
        coeffs, prepared["y"], prepared["cb"], prepared["cr"],
        prepared["h_crop"], prepared["w_crop"], BLOCK_SIZE,
    )
    W = np.asarray(prepared["W"], dtype=np.int64)
    key = WatermarkKey(
        method_id=METHOD_ID,
        host_shape=tuple(int(x) for x in host.shape),
        watermark_shape=tuple(int(x) for x in wm.shape),
        seed=int(prepared["seed"]),
        repeat=int(prepared["repeat"]),
        step=delta,
        arnold_iter=0,
        arnold_period=1,
        threshold=127,
        params={
            "domain": "8x8 luminance DCT two-coefficient vectors",
            "block_size": BLOCK_SIZE,
            "coefficients": [list(x) for x in COEFFICIENTS],
            "fine_lattice": "A2",
            "coarse_lattice": "4A2",
            "lattice_alpha": LATTICE_ALPHA,
            "code_rate_bits_per_dimension": BITS_PER_DIMENSION,
            "bits_per_vector": BITS_PER_VECTOR,
            "coset_count": COSET_COUNT,
            "message_to_coset": mapping.astype(int).tolist(),
            "adjacency_shape": [COSET_COUNT, COSET_COUNT],
            "canonical_matching_weight": int(sum(W[int(mapping[m]), m] for m in range(COSET_COUNT))),
            "symbol_padding_bits": int(prepared["padding"]),
            "capacity_bits": BITS_PER_VECTOR * _capacity_vectors(host.shape),
            "schedule_storage": "regenerate_from_seed",
            "payload_distribution_disclosure": "common benchmark watermark; not forced to the paper's 0.9 zero prior",
            "paper_equations": "Neighbor(s)=argmin_i dist(s,Lambda_i); maximum-weight canonical labeling; s_w=Q_Lambda_gamma_i(s)",
        },
        schedule=[],
        fully_blind=False,
        side_information="semi-blind: seed, lattice scale, coefficient pair, image-dependent 16-entry canonical labeling, payload shape, repeat count",
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
    return embed_prepared(host_rgb, watermark_binary, prepared, step=step, method_params=method_params)


def extract(image_rgb: np.ndarray, key: WatermarkKey | dict[str, Any]) -> np.ndarray:
    if isinstance(key, dict):
        key = WatermarkKey(**key)
    coeffs, _h_crop, _w_crop = dct_blocks_from_rgb(image_rgb, int(key.params["block_size"]))
    payload = int(np.prod(key.watermark_shape))
    base_symbol_count = int(np.ceil(payload / float(BITS_PER_VECTOR)))
    repeat_i = int(key.repeat)
    symbol_count = base_symbol_count * repeat_i
    h, w, _c = key.host_shape
    indices = make_block_indices(
        int(h) - int(h) % BLOCK_SIZE,
        int(w) - int(w) % BLOCK_SIZE,
        BLOCK_SIZE,
        symbol_count,
        1,
        int(key.seed),
    ).astype(np.int32)
    carriers = np.stack([coeffs[indices, u, v] for u, v in COEFFICIENTS], axis=1)
    _nearest, coset_distances = nearest_points_all_cosets(carriers, float(key.step))
    mapping = np.asarray(key.params["message_to_coset"], dtype=np.int64)
    # Convert coset distances to message-symbol distances, then combine native
    # repeated transmissions with a soft minimum-distance decoder.  For
    # repeat=1 this is exactly nearest-coset canonical-label decoding.
    message_distances = coset_distances[:, mapping]
    combined = message_distances.reshape(repeat_i, base_symbol_count, COSET_COUNT).sum(axis=0)
    decoded_symbols = np.argmin(combined, axis=1).astype(np.uint8)
    raw = symbols_to_bits(decoded_symbols, payload)
    return (raw.reshape(key.watermark_shape) * 255).astype(np.uint8)


__all__ = [
    "METHOD_ID", "METHOD_REF", "BLOCK_SIZE", "COEFFICIENTS", "DEFAULT_STEP",
    "LATTICE_ALPHA", "BITS_PER_DIMENSION", "BITS_PER_VECTOR", "COSET_COUNT",
    "nearest_points_all_cosets", "a2_decode_cosets", "a2_quantize_to_cosets",
    "bits_to_symbols", "symbols_to_bits", "adjacency_matrix", "canonical_labeling",
    "inverse_labeling", "prepare_embedding", "embed_prepared", "embed", "extract",
]
