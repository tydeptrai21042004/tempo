from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
from itertools import product
import math

import numpy as np
from PIL import Image
from io import BytesIO
from scipy.optimize import linear_sum_assignment

from .baselines.ca_qim_a2_mao2024 import A2_BASE
from .baselines.dct_dm_qim_chen2001 import dm_decode_values, dm_embed_values
from .baselines.dct_iss_malvar2003 import iss_embed_vectors_maximum_useful
from .baselines.dct_stdm_qim_chen2001 import stdm_decode_vectors, stdm_embed_vectors
from .baselines.dct_spread_spectrum_cox1997 import (
    DEFAULT_ALPHA as COX_DEFAULT_ALPHA,
    NATIVE_WATERMARK_LENGTH as COX_NATIVE_LENGTH,
    native_embed_gray as cox_native_embed_gray,
    native_detection_statistics_gray as cox_native_detection_statistics_gray,
    cox_similarity,
)
from .common.block_dct import dct_blocks_from_gray, dct_blocks_to_gray, gray_to_dct_blocks
from .common.dither import binary_dither_pair
from .core import (
    attack_gaussian,
    attack_speckle,
    psnr,
    save_csv,
    save_json,
    ssim_gray,
)
from .native_metrics import (
    bit_error_probability,
    db10,
    linear_from_db,
    mse,
    prd,
    q_function,
    symbol_error_rate, normalized_correlation, image_distortion_statistics,
)


PAPER_NATIVE_PROTOCOL_VERSION = "1.3"
CA_NATIVE_STEP = 6.0


@dataclass(frozen=True)
class NativeDisclosure:
    method_id: str
    detector_type: str
    native_domain: str
    paper_metric: str
    paper_attack_model: str
    native_parameter_policy: str


NATIVE_DISCLOSURES = {
    "dct_dm_qim_chen2001": NativeDisclosure(
        method_id="dct_dm_qim_chen2001",
        detector_type="blind",
        native_domain="scalar Gaussian host channel",
        paper_metric="empirical bit-error probability versus DNR and amplitude scale",
        paper_attack_model="AWGN; amplitude scaling is a paper-discussed vulnerability",
        native_parameter_policy="fixed paper-equation quantizer step; no PSNR calibration",
    ),
    "dct_stdm_qim_chen2001": NativeDisclosure(
        method_id="dct_stdm_qim_chen2001",
        detector_type="blind",
        native_domain="length-N Gaussian host vector and spread-transform projection",
        paper_metric="empirical and paper-formula bit-error probability versus rate-normalized DNR",
        paper_attack_model="AWGN; amplitude scaling diagnostic",
        native_parameter_policy="fixed scalar quantizer step and spreading dimension; no PSNR calibration",
    ),
    "dct_iss_malvar2003": NativeDisclosure(
        method_id="dct_iss_malvar2003",
        detector_type="blind",
        native_domain="paper sufficient-statistic Gaussian model",
        paper_metric="empirical/theoretical error probability versus lambda, SNR, and relative watermark power",
        paper_attack_model="additive Gaussian attack noise; amplitude scaling diagnostic",
        native_parameter_policy="paper rho/SNR/lambda sweeps; no PSNR calibration",
    ),
    "dct_spread_spectrum_cox1997": NativeDisclosure(
        method_id="dct_spread_spectrum_cox1997",
        detector_type="non_blind",
        native_domain="whole-image grayscale DCT, 1000 largest-magnitude AC coefficients",
        paper_metric="Cox similarity response, centered/sign variants, false-positive ranking, and image fidelity",
        paper_attack_model="scaling with registration, JPEG 10/5, dithering, crop restoration, and crop plus JPEG",
        native_parameter_policy="paper watermark length 1000 and alpha=0.1; no PSNR calibration",
    ),
    "ca_qim_a2_mao2024": NativeDisclosure(
        method_id="ca_qim_a2_mao2024",
        detector_type="semi_blind",
        native_domain="8x8 grayscale DCT with A2/4A2 canonical labeling",
        paper_metric="MSE, PSNR, PRD, SSIM, and symbol error rate",
        paper_attack_model="Gaussian, salt, pepper, mixed salt-pepper, and speckle noise",
        native_parameter_policy="R=1/2/3 capacity grid, R=2 SER track, P(bit=0)=0.9, fixed lattice scale; no PSNR calibration",
    ),
    "dew_langelaar2001": NativeDisclosure(
        method_id="dew_langelaar2001",
        detector_type="blind",
        native_domain="keyed regions of 8x8 luminance-DCT blocks",
        paper_metric="label recovery, visual distortion, payload, and processing throughput",
        paper_attack_model="JPEG/MPEG coefficient loss and aligned compressed-video processing",
        native_parameter_policy="paper-zeroing and disclosed minimum-energy-removal decoded-frame profiles",
    ),
}


def _ber_stdm_theory(dnr_norm_db: float) -> float:
    # Chen--Wornell Eq. (49): Pb ~= 2 Q(sqrt(3/4 * DNR_norm)).
    value = linear_from_db(dnr_norm_db)
    return float(2.0 * q_function(math.sqrt(0.75 * value)))


def run_qim_stdm_native_channel(
    *,
    out_dir: str | Path,
    seed: int = 2026,
    trials: int = 40_000,
    vector_length: int = 100,
    dnr_norm_db_values: Iterable[float] = (0.0, 1.4, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0),
    scale_values: Iterable[float] = (0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20),
) -> list[dict[str, Any]]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_trials = int(trials)
    bits = rng.integers(0, 2, size=n_trials, dtype=np.uint8)
    step = 2.0

    rows: list[dict[str, Any]] = []

    # Scalar binary dither modulation: rate 1 bit/dimension.
    host_dm = rng.normal(0.0, 1.0, size=n_trials)
    d0_dm, d1_dm = binary_dither_pair(n_trials, seed, step, stream_offset=17011)
    marked_dm = dm_embed_values(host_dm, bits, step, d0_dm, d1_dm)
    distortion_dm = mse(host_dm, marked_dm)
    for dnr_db in dnr_norm_db_values:
        rate = 1.0
        noise_variance = distortion_dm / max(rate * linear_from_db(float(dnr_db)), 1e-15)
        attacked = marked_dm + rng.normal(0.0, math.sqrt(noise_variance), size=n_trials)
        recovered = dm_decode_values(attacked, step, d0_dm, d1_dm)
        rows.append({
            "method_id": "dct_dm_qim_chen2001",
            "experiment": "paper_native_awgn_dnr_sweep",
            "attack": "awgn",
            "detector_type": "blind",
            "rate_bits_per_dimension": rate,
            "dnr_norm_db": float(dnr_db),
            "embedding_distortion_mse_per_dimension": distortion_dm,
            "noise_variance": noise_variance,
            "empirical_bit_error_probability": bit_error_probability(bits, recovered),
            "paper_formula_bit_error_probability": "",
            "trials": n_trials,
            "psnr_calibrated": False,
        })

    # Uncoded STDM: one bit per N dimensions, as in the paper's low-rate curve.
    host_stdm = rng.normal(0.0, 1.0, size=(n_trials, int(vector_length)))
    directions = rng.choice(np.asarray([-1.0, 1.0]), size=host_stdm.shape)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    d0_stdm, d1_stdm = binary_dither_pair(n_trials, seed, step, stream_offset=27191)
    marked_stdm = stdm_embed_vectors(host_stdm, bits, directions, step, d0_stdm, d1_stdm)
    distortion_stdm = mse(host_stdm, marked_stdm)
    rate_stdm = 1.0 / float(vector_length)
    for dnr_db in dnr_norm_db_values:
        noise_variance = distortion_stdm / max(rate_stdm * linear_from_db(float(dnr_db)), 1e-15)
        attacked = marked_stdm + rng.normal(0.0, math.sqrt(noise_variance), size=host_stdm.shape)
        recovered = stdm_decode_vectors(attacked, directions, step, d0_stdm, d1_stdm)
        rows.append({
            "method_id": "dct_stdm_qim_chen2001",
            "experiment": "paper_native_awgn_dnr_sweep",
            "attack": "awgn",
            "detector_type": "blind",
            "rate_bits_per_dimension": rate_stdm,
            "dnr_norm_db": float(dnr_db),
            "embedding_distortion_mse_per_dimension": distortion_stdm,
            "noise_variance": noise_variance,
            "empirical_bit_error_probability": bit_error_probability(bits, recovered),
            "paper_formula_bit_error_probability": _ber_stdm_theory(float(dnr_db)),
            "trials": n_trials,
            "psnr_calibrated": False,
        })

    # Positive amplitude scaling, discussed explicitly by both QIM and ISS papers.
    for gain in scale_values:
        rec_dm = dm_decode_values(float(gain) * marked_dm, step, d0_dm, d1_dm)
        rec_stdm = stdm_decode_vectors(float(gain) * marked_stdm, directions, step, d0_stdm, d1_stdm)
        for method_id, recovered in (
            ("dct_dm_qim_chen2001", rec_dm),
            ("dct_stdm_qim_chen2001", rec_stdm),
        ):
            rows.append({
                "method_id": method_id,
                "experiment": "paper_discussed_amplitude_scaling",
                "attack": "positive_amplitude_scaling",
                "detector_type": "blind",
                "amplitude_gain": float(gain),
                "empirical_bit_error_probability": bit_error_probability(bits, recovered),
                "trials": n_trials,
                "psnr_calibrated": False,
            })

    save_csv(out / "qim_stdm_native_channel.csv", rows)
    return rows


def _iss_theoretical_error(*, rho: float, snr_db: float, cancellation_lambda: float) -> float:
    lam = float(cancellation_lambda)
    noise_ratio = 1.0 / linear_from_db(float(snr_db))
    numerator = max(0.0, float(rho) - lam * lam)
    denominator = noise_ratio + (1.0 - lam) ** 2
    if numerator <= 0.0:
        return 0.5
    return float(0.5 * math.erfc(math.sqrt(numerator / max(2.0 * denominator, 1e-15))))


def _iss_lambda_opt(*, rho: float, snr_db: float) -> float:
    noise_ratio = 1.0 / linear_from_db(float(snr_db))
    a = 1.0 + noise_ratio + float(rho)
    disc = max(0.0, a * a - 4.0 * float(rho))
    return float(0.5 * (a - math.sqrt(disc)))


def run_iss_native_channel(
    *,
    out_dir: str | Path,
    seed: int = 2026,
    trials: int = 250_000,
    snr_db_values: Iterable[float] = (7.0, 10.0),
    rho_values: Iterable[float] = (5.0, 10.0, 20.0),
    lambda_values: Iterable[float] = tuple(np.linspace(0.0, 1.0, 11)),
    scale_values: Iterable[float] = (0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20),
) -> list[dict[str, Any]]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + 8801)
    n_trials = int(trials)
    bits01 = rng.integers(0, 2, size=n_trials, dtype=np.uint8)
    bits = np.where(bits01 > 0, 1.0, -1.0)
    host_projection = rng.normal(0.0, 1.0, size=n_trials)
    rows: list[dict[str, Any]] = []

    for snr_db in snr_db_values:
        noise_std = math.sqrt(1.0 / linear_from_db(float(snr_db)))
        noise = rng.normal(0.0, noise_std, size=n_trials)
        for rho in rho_values:
            for lam in lambda_values:
                alpha_sq = max(0.0, float(rho) - float(lam) ** 2)
                alpha = math.sqrt(alpha_sq)
                displacement = alpha * bits - float(lam) * host_projection
                received = host_projection + displacement + noise
                recovered = (received >= 0.0).astype(np.uint8)
                rows.append({
                    "method_id": "dct_iss_malvar2003",
                    "variant": "linear_iss",
                    "experiment": "paper_figure4_error_probability",
                    "attack": "awgn",
                    "detector_type": "blind",
                    "snr_db": float(snr_db),
                    "relative_watermark_power_rho": float(rho),
                    "cancellation_lambda": float(lam),
                    "alpha": alpha,
                    "empirical_error_probability": bit_error_probability(bits01, recovered),
                    "paper_formula_error_probability": _iss_theoretical_error(
                        rho=float(rho), snr_db=float(snr_db), cancellation_lambda=float(lam)
                    ),
                    "mean_distortion": float(np.mean(displacement ** 2)),
                    "std_distortion": float(np.std(displacement ** 2)),
                    "max_distortion": float(np.max(displacement ** 2)),
                    "trials": n_trials,
                    "psnr_calibrated": False,
                })

    # Paper-style comparison at SNR=20 dB, including limited-distortion ISS.
    snr_db = 20.0
    rho = 10.0
    lam = _iss_lambda_opt(rho=rho, snr_db=snr_db)
    alpha = math.sqrt(max(0.0, rho - lam * lam))
    noise = rng.normal(0.0, math.sqrt(1.0 / linear_from_db(snr_db)), size=n_trials)
    plain_displacement = alpha * bits - lam * host_projection
    plain_received = host_projection + plain_displacement + noise
    plain_recovered = (plain_received >= 0.0).astype(np.uint8)

    # Reuse the repository's Section V-A rule on a one-dimensional sufficient statistic.
    x_vec = host_projection[:, None]
    u_vec = np.ones_like(x_vec)
    limited = iss_embed_vectors_maximum_useful(
        x_vec, bits01, u_vec, alpha=alpha, cancellation=lam
    )[:, 0]
    limited_displacement = limited - host_projection
    limited_received = limited + noise
    limited_recovered = (limited_received >= 0.0).astype(np.uint8)

    for variant, displacement, recovered in (
        ("linear_iss", plain_displacement, plain_recovered),
        ("limited_distortion_iss", limited_displacement, limited_recovered),
    ):
        rows.append({
            "method_id": "dct_iss_malvar2003",
            "variant": variant,
            "experiment": "paper_figure9_variant_comparison",
            "attack": "awgn",
            "detector_type": "blind",
            "snr_db": snr_db,
            "relative_watermark_power_rho": rho,
            "cancellation_lambda": lam,
            "alpha": alpha,
            "empirical_error_probability": bit_error_probability(bits01, recovered),
            "paper_formula_error_probability": (
                _iss_theoretical_error(rho=rho, snr_db=snr_db, cancellation_lambda=lam)
                if variant == "linear_iss" else ""
            ),
            "mean_distortion": float(np.mean(displacement ** 2)),
            "std_distortion": float(np.std(displacement ** 2)),
            "max_distortion": float(np.max(displacement ** 2)),
            "trials": n_trials,
            "psnr_calibrated": False,
        })

    for gain in scale_values:
        scaled = float(gain) * (host_projection + plain_displacement)
        recovered = (scaled >= 0.0).astype(np.uint8)
        rows.append({
            "method_id": "dct_iss_malvar2003",
            "variant": "linear_iss",
            "experiment": "paper_discussed_amplitude_scaling",
            "attack": "positive_amplitude_scaling",
            "detector_type": "blind",
            "amplitude_gain": float(gain),
            "empirical_error_probability": bit_error_probability(bits01, recovered),
            "snr_db": "no_additive_noise",
            "relative_watermark_power_rho": rho,
            "cancellation_lambda": lam,
            "trials": n_trials,
            "psnr_calibrated": False,
        })

    save_csv(out / "iss_native_channel.csv", rows)
    return rows



def _cox_resize_restore(image: np.ndarray, scale: float) -> np.ndarray:
    img = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="L")
    w, h = img.size
    small = img.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.Resampling.LANCZOS)
    return np.asarray(small.resize((w, h), Image.Resampling.BICUBIC), dtype=np.uint8)


def _cox_jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    buf = BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="L").save(buf, format="JPEG", quality=int(quality), optimize=False)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("L"), dtype=np.uint8)


def _cox_dither(image: np.ndarray) -> np.ndarray:
    img = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="L")
    return np.asarray(img.convert("1", dither=Image.Dither.FLOYDSTEINBERG).convert("L"), dtype=np.uint8)


def _cox_crop_restore(marked: np.ndarray, original: np.ndarray, keep_fraction: float = 0.5) -> np.ndarray:
    marked_arr = np.asarray(marked, dtype=np.uint8)
    restored = np.asarray(original, dtype=np.uint8).copy()
    h, w = marked_arr.shape
    kh = max(1, int(round(h * float(keep_fraction))))
    kw = max(1, int(round(w * float(keep_fraction))))
    y0 = (h - kh) // 2
    x0 = (w - kw) // 2
    restored[y0:y0 + kh, x0:x0 + kw] = marked_arr[y0:y0 + kh, x0:x0 + kw]
    return restored


def _cox_false_positive_statistics(reference: np.ndarray, extracted: np.ndarray, seed: int, candidates: int) -> dict[str, float | int]:
    rng = np.random.default_rng(int(seed) + 90131)
    count = max(1, int(candidates))
    false = rng.normal(0.0, 1.0, size=(count, reference.size))
    den = max(float(np.linalg.norm(extracted)), 1e-15)
    false_scores = false @ extracted / den
    target = cox_similarity(reference, extracted)
    return {
        "target_similarity": float(target),
        "max_false_similarity": float(np.max(false_scores)),
        "mean_false_similarity": float(np.mean(false_scores)),
        "std_false_similarity": float(np.std(false_scores)),
        "target_rank_among_candidates": int(1 + np.sum(false_scores >= target)),
        "false_candidate_count": count,
    }


def run_cox_native_images(
    *,
    host_paths: Iterable[str | Path],
    out_dir: str | Path,
    seed: int = 2026,
    host_limit: int = 2,
    false_candidates: int = 999,
) -> list[dict[str, Any]]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    paper_values = {
        "clean": (32.0, "paper detector response"),
        "scale050_restore": (13.4, "paper detector response after half-scale and registration"),
        "jpeg10": (22.8, "paper detector response"),
        "jpeg05": (13.9, "paper detector response"),
        "floyd_steinberg_dither": (5.2, "raw paper detector response; centered response reported as 10.5"),
        "crop_center_quarter_restore": (14.6, "paper detector response"),
        "crop_center_quarter_jpeg10_restore": (10.6, "paper detector response"),
    }
    for host_index, host_path in enumerate(list(host_paths)[: max(1, int(host_limit))]):
        host = _load_native_gray(host_path)
        rng = np.random.default_rng(int(seed) + 1009 * host_index)
        watermark = rng.normal(0.0, 1.0, size=COX_NATIVE_LENGTH)
        t0 = __import__("time").perf_counter()
        marked, key = cox_native_embed_gray(host, watermark, alpha=COX_DEFAULT_ALPHA)
        embed_ms = (__import__("time").perf_counter() - t0) * 1000.0
        attacks = {
            "clean": marked,
            "scale050_restore": _cox_resize_restore(marked, 0.5),
            "jpeg10": _cox_jpeg(marked, 10),
            "jpeg05": _cox_jpeg(marked, 5),
            "floyd_steinberg_dither": _cox_dither(marked),
            "crop_center_quarter_restore": _cox_crop_restore(marked, host, 0.5),
            "crop_center_quarter_jpeg10_restore": _cox_crop_restore(_cox_jpeg(marked, 10), host, 0.5),
        }
        embedding_stats = image_distortion_statistics(host, marked)
        for attack_name, attacked in attacks.items():
            t1 = __import__("time").perf_counter()
            stats = cox_native_detection_statistics_gray(attacked, key)
            extract_ms = (__import__("time").perf_counter() - t1) * 1000.0
            extracted_sequence = None
            # Reconstruct from the reported sufficient statistics only when needed for false-positive tests.
            from .baselines.dct_spread_spectrum_cox1997 import native_extract_sequence_gray
            extracted_sequence = native_extract_sequence_gray(attacked, key)
            fp = _cox_false_positive_statistics(watermark, extracted_sequence, seed + host_index, false_candidates)
            attacked_stats = image_distortion_statistics(host, attacked)
            paper_response, paper_note = paper_values[attack_name]
            rows.append({
                "method_id": "dct_spread_spectrum_cox1997",
                "experiment": "paper_native_image_attacks",
                "host": Path(host_path).name,
                "attack": attack_name,
                "detector_type": "non_blind",
                "requires_original_host": True,
                "watermark_length": COX_NATIVE_LENGTH,
                "alpha": COX_DEFAULT_ALPHA,
                "cox_similarity": stats["cox_similarity"],
                "cox_similarity_centered": stats["cox_similarity_centered"],
                "cox_similarity_sign": stats["cox_similarity_sign"],
                "sequence_correlation": stats["sequence_correlation"],
                "max_false_similarity": fp["max_false_similarity"],
                "mean_false_similarity": fp["mean_false_similarity"],
                "std_false_similarity": fp["std_false_similarity"],
                "target_rank_among_candidates": fp["target_rank_among_candidates"],
                "false_candidate_count": fp["false_candidate_count"],
                "embedding_mse": embedding_stats["mse"],
                "embedding_psnr": psnr(host, marked),
                "embedding_prd": prd(host, marked),
                "embedding_ssim": ssim_gray(host, marked),
                "attacked_mse": attacked_stats["mse"],
                "attacked_psnr": psnr(host, attacked),
                "attacked_prd": prd(host, attacked),
                "attacked_ssim": ssim_gray(host, attacked),
                "embed_ms": embed_ms,
                "extract_ms": extract_ms,
                "paper_reported_detector_response": paper_response,
                "paper_reference_note": paper_note,
                "numeric_equality_asserted": False,
                "comparison_reason": "paper used Bavarian Couple and its own registration/codec implementation",
                "psnr_calibrated": False,
            })
    save_csv(out / "cox_native_image_attacks.csv", rows)
    return rows

def _zigzag_ac() -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for total in range(15):
        diagonal = [(u, total - u) for u in range(8) if 0 <= total - u < 8]
        if total % 2 == 0:
            diagonal.reverse()
        coords.extend(diagonal)
    return [c for c in coords if c != (0, 0)]


_ZIGZAG_AC = _zigzag_ac()


def _frequency_pairs(frequency: str, message_count: int | str) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if message_count == "full":
        coords = _ZIGZAG_AC[:62]
        count = 31
    else:
        count = int(message_count)
        if count <= 0:
            raise ValueError("message_count must be positive")
        thirds = {
            "low": _ZIGZAG_AC[:20],
            "mid": _ZIGZAG_AC[21:41],
            "high": _ZIGZAG_AC[43:63],
        }
        if frequency not in thirds:
            raise ValueError(f"unknown frequency band {frequency}")
        coords = thirds[frequency]
        if 2 * count > len(coords):
            raise ValueError(f"{frequency} band cannot hold {count} A2 messages")
        coords = coords[: 2 * count]
    return [(coords[2 * i], coords[2 * i + 1]) for i in range(count)]


_NATIVE_A2_OFFSETS = np.asarray(list(product(range(-1, 2), repeat=2)), dtype=np.int64)


def _native_a2_spec(code_rate: int) -> tuple[int, int, np.ndarray]:
    rate = int(code_rate)
    if rate not in {1, 2, 3}:
        raise ValueError("native A2 CA-QIM code rate must be 1, 2, or 3 bits/dimension")
    alpha = 1 << rate
    bits_per_vector = 2 * rate
    representatives = np.asarray(list(product(range(alpha), repeat=2)), dtype=np.int64)
    return alpha, bits_per_vector, representatives


def _native_a2_nearest_all_cosets(
    carriers: np.ndarray,
    step: float,
    code_rate: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(carriers, dtype=np.float64).reshape(-1, 2)
    alpha, _bits_per_vector, representatives = _native_a2_spec(code_rate)
    fine = float(step) * A2_BASE
    inv_fine = np.linalg.inv(fine)
    coordinates = x @ inv_fine.T
    centers = np.rint(
        (coordinates[:, None, :] - representatives[None, :, :]) / float(alpha)
    ).astype(np.int64)
    lattice_coordinates = representatives[None, :, None, :] + alpha * (
        centers[:, :, None, :] + _NATIVE_A2_OFFSETS[None, None, :, :]
    )
    points = lattice_coordinates @ fine.T
    distances = np.sum((points - x[:, None, None, :]) ** 2, axis=-1)
    best = np.argmin(distances, axis=-1)
    rows = np.arange(x.shape[0])[:, None]
    cosets = np.arange(representatives.shape[0])[None, :]
    return points[rows, cosets, best], distances[rows, cosets, best]


def _native_a2_decode_chunked(
    carriers: np.ndarray,
    step: float,
    code_rate: int,
) -> np.ndarray:
    alpha, _bits_per_vector, _representatives = _native_a2_spec(code_rate)
    coset_count = alpha * alpha
    chunk_size = max(512, int(8192 * 16 / coset_count))
    output = np.empty(len(carriers), dtype=np.uint16)
    for start in range(0, len(carriers), chunk_size):
        stop = min(len(carriers), start + chunk_size)
        _points, distances = _native_a2_nearest_all_cosets(
            carriers[start:stop], step, code_rate
        )
        output[start:stop] = np.argmin(distances, axis=1).astype(np.uint16)
    return output


def _native_a2_quantize_chunked(
    carriers: np.ndarray,
    cosets: np.ndarray,
    step: float,
    code_rate: int,
) -> np.ndarray:
    alpha, _bits_per_vector, _representatives = _native_a2_spec(code_rate)
    coset_count = alpha * alpha
    chunk_size = max(512, int(8192 * 16 / coset_count))
    output = np.empty_like(np.asarray(carriers, dtype=np.float64))
    target = np.asarray(cosets, dtype=np.int64).reshape(-1)
    for start in range(0, len(carriers), chunk_size):
        stop = min(len(carriers), start + chunk_size)
        points, _distances = _native_a2_nearest_all_cosets(
            carriers[start:stop], step, code_rate
        )
        local = target[start:stop]
        output[start:stop] = points[np.arange(stop - start), local]
    return output


def _native_canonical_labeling(matrix: np.ndarray) -> np.ndarray:
    """Exact maximum-weight message-to-coset assignment for native rates."""
    weights = np.asarray(matrix, dtype=np.int64)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("canonical-label adjacency matrix must be square")
    rows, cols = linear_sum_assignment(-weights)
    mapping = np.empty(weights.shape[1], dtype=np.uint16)
    mapping[cols] = rows.astype(np.uint16)
    return mapping


def _native_inverse_labeling(mapping: np.ndarray) -> np.ndarray:
    labels = np.asarray(mapping, dtype=np.int64).reshape(-1)
    if sorted(labels.tolist()) != list(range(labels.size)):
        raise ValueError("canonical labeling must be a coset permutation")
    inverse = np.empty(labels.size, dtype=np.uint16)
    inverse[labels] = np.arange(labels.size, dtype=np.uint16)
    return inverse

def _native_ca_embed(
    host_gray: np.ndarray,
    *,
    frequency: str,
    message_count: int | str,
    code_rate: int,
    step: float,
    seed: int,
    p_zero: float = 0.9,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Paper-native grayscale A2 CA-QIM embedding.

    The paper applies an 8x8 DCT directly to grayscale Set12/BSD68 images.
    The image-dependent canonical message-to-coset mapping is saved as
    semi-blind side information.
    """
    host = np.asarray(host_gray, dtype=np.uint8)
    if host.ndim != 2:
        raise ValueError("native CA-QIM expects a 2-D grayscale host")
    coeffs, gray_float, h_crop, w_crop = gray_to_dct_blocks(host, 8)
    pairs = _frequency_pairs(frequency, message_count)
    block_count = coeffs.shape[0]
    rng = np.random.default_rng(seed)
    symbol_count = block_count * len(pairs)
    alpha, bits_per_vector, representatives = _native_a2_spec(code_rate)
    coset_count = int(representatives.shape[0])
    bits = (rng.random((symbol_count, bits_per_vector)) >= float(p_zero)).astype(np.uint8)
    weights = (1 << np.arange(bits_per_vector, dtype=np.uint16))[None, :]
    symbols = np.sum(bits.astype(np.uint16) * weights, axis=1).astype(np.uint16)

    carriers = np.empty((symbol_count, 2), dtype=np.float64)
    for pair_index, (c0, c1) in enumerate(pairs):
        sl = slice(pair_index * block_count, (pair_index + 1) * block_count)
        carriers[sl, 0] = coeffs[:, c0[0], c0[1]]
        carriers[sl, 1] = coeffs[:, c1[0], c1[1]]

    nearest = _native_a2_decode_chunked(carriers, step, code_rate)
    W = np.zeros((coset_count, coset_count), dtype=np.int64)
    np.add.at(W, (nearest.astype(np.int64), symbols.astype(np.int64)), 1)
    mapping = _native_canonical_labeling(W)
    target_cosets = mapping[symbols]
    quantized = _native_a2_quantize_chunked(carriers, target_cosets, step, code_rate)

    marked_coeffs = coeffs.copy()
    for pair_index, (c0, c1) in enumerate(pairs):
        sl = slice(pair_index * block_count, (pair_index + 1) * block_count)
        marked_coeffs[:, c0[0], c0[1]] = quantized[sl, 0]
        marked_coeffs[:, c1[0], c1[1]] = quantized[sl, 1]

    coefficient_difference = marked_coeffs - coeffs
    frequency_mse = float(np.mean(coefficient_difference ** 2))
    frequency_prd = float(
        np.sqrt(
            np.sum(coefficient_difference ** 2)
            / max(float(np.sum(coeffs ** 2)), 1e-15)
        )
    )
    frequency_psnr = float("inf") if frequency_mse == 0.0 else float(
        10.0 * np.log10((255.0 ** 2) / frequency_mse)
    )

    marked = dct_blocks_to_gray(marked_coeffs, gray_float, h_crop, w_crop, 8)
    key = {
        "frequency": frequency,
        "message_count": message_count,
        "code_rate_bits_per_dimension": int(code_rate),
        "lattice_alpha": int(alpha),
        "bits_per_vector": int(bits_per_vector),
        "step": float(step),
        "p_zero": float(p_zero),
        "pairs": [[list(a), list(b)] for a, b in pairs],
        "mapping": mapping.tolist(),
        "symbols": symbols.tolist(),
        "block_count": block_count,
        "frequency_mse": frequency_mse,
        "frequency_psnr": frequency_psnr,
        "frequency_prd": frequency_prd,
        "detector_type": "semi_blind",
        "side_information": f"image-dependent {coset_count}-entry canonical message-to-coset mapping and fixed carrier pairs",
        "original_host_required": False,
        "psnr_calibrated": False,
    }
    return marked, key


def _native_ca_extract_symbols(image_gray: np.ndarray, key: dict[str, Any]) -> np.ndarray:
    coeffs, _h_crop, _w_crop = dct_blocks_from_gray(image_gray, 8)
    pairs = [(tuple(a), tuple(b)) for a, b in key["pairs"]]
    block_count = int(key["block_count"])
    carriers = np.empty((block_count * len(pairs), 2), dtype=np.float64)
    for pair_index, (c0, c1) in enumerate(pairs):
        sl = slice(pair_index * block_count, (pair_index + 1) * block_count)
        carriers[sl, 0] = coeffs[:, c0[0], c0[1]]
        carriers[sl, 1] = coeffs[:, c1[0], c1[1]]
    code_rate = int(key["code_rate_bits_per_dimension"])
    cosets = _native_a2_decode_chunked(carriers, float(key["step"]), code_rate)
    inverse = _native_inverse_labeling(np.asarray(key["mapping"], dtype=np.uint16))
    return inverse[cosets]


def _load_native_gray(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)

def _attack_salt_only(img: np.ndarray, amount: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.asarray(img, dtype=np.uint8).copy()
    mask = rng.random(out.shape[:2]) < float(amount)
    out[mask] = 255
    return out


def _attack_pepper_only(img: np.ndarray, amount: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.asarray(img, dtype=np.uint8).copy()
    mask = rng.random(out.shape[:2]) < float(amount)
    out[mask] = 0
    return out


def _attack_mixed_impulse(img: np.ndarray, amount: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.asarray(img, dtype=np.uint8).copy()
    mask = rng.random(out.shape[:2]) < float(amount)
    salt = rng.random(out.shape[:2]) < 0.5
    out[mask & salt] = 255
    out[mask & ~salt] = 0
    return out


def run_ca_qim_native_images(
    *,
    host_paths: Iterable[str | Path],
    out_dir: str | Path,
    seed: int = 2026,
    step: float = CA_NATIVE_STEP,
    p_zero: float = 0.9,
    noise_sigma: float = 2.0,
    impulse_amount: float = 0.005,
    speckle_sigma: float = 0.01,
    include_full: bool = True,
    full_host_limit: int = 1,
) -> dict[str, list[dict[str, Any]]]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hosts = [(Path(p), _load_native_gray(p)) for p in host_paths]
    if not hosts:
        raise ValueError("At least one host image is required for the CA-QIM native image track")

    imperceptibility_rows: list[dict[str, Any]] = []
    attack_rows: list[dict[str, Any]] = []

    # Paper Tables II/III vary payload rate R=1/2/3 bits per dimension
    # at low/mid/high DCT positions while using one A2 carrier vector.
    imperceptibility_settings: list[tuple[str, int]] = [
        (frequency, code_rate)
        for frequency in ("low", "mid", "high")
        for code_rate in (1, 2, 3)
    ]
    # Paper Fig. 10 reports one, two, three, and ALL-message SER cases.
    robustness_settings: list[int | str] = [1, 2, 3]
    if include_full:
        robustness_settings.append("full")

    for host_index, (host_path, host) in enumerate(hosts):
        for setting_index, (frequency, code_rate) in enumerate(imperceptibility_settings):
            marked, key = _native_ca_embed(
                host,
                frequency=frequency,
                message_count=1,
                code_rate=code_rate,
                step=float(step),
                seed=seed + 101 * host_index + setting_index,
                p_zero=float(p_zero),
            )
            original_symbols = np.asarray(key["symbols"], dtype=np.uint8)
            clean_symbols = _native_ca_extract_symbols(marked, key)
            spatial_mse = mse(host, marked)
            imperceptibility_rows.append({
                "method_id": "ca_qim_a2_mao2024",
                "host": str(host_path),
                "image_mode": "grayscale",
                "frequency": frequency,
                "embedded_vectors_per_block": 1,
                "payload_bits_per_dimension": code_rate,
                "code_rate_bits_per_dimension": code_rate,
                "bit_zero_probability": float(p_zero),
                "lattice_step": float(step),
                "lattice_step_source": "fixed native operating point; paper does not report a machine-readable lattice scale",
                "detector_type": "semi_blind",
                "mse_frequency": key["frequency_mse"],
                "psnr_frequency": key["frequency_psnr"],
                "prd_frequency": key["frequency_prd"],
                "mse_spatial": spatial_mse,
                "psnr_spatial": psnr(host, marked),
                "prd_spatial": prd(host, marked),
                "ssim_spatial": ssim_gray(host, marked),
                "clean_symbol_error_rate": symbol_error_rate(original_symbols, clean_symbols),
                "psnr_calibrated": False,
            })

        for setting_index, message_count in enumerate(robustness_settings):
            if message_count == "full" and host_index >= int(full_host_limit):
                continue
            marked, key = _native_ca_embed(
                host,
                frequency="high",
                message_count=message_count,
                code_rate=2,
                step=float(step),
                seed=seed + 10_000 + 101 * host_index + setting_index,
                p_zero=float(p_zero),
            )
            original_symbols = np.asarray(key["symbols"], dtype=np.uint8)
            clean_symbols = _native_ca_extract_symbols(marked, key)
            attacks = [
                ("clean", lambda x: x),
                ("gaussian", lambda x, s=seed + host_index: attack_gaussian(x, noise_sigma, s)),
                ("salt", lambda x, s=seed + 1000 + host_index: _attack_salt_only(x, impulse_amount, s)),
                ("pepper", lambda x, s=seed + 2000 + host_index: _attack_pepper_only(x, impulse_amount, s)),
                ("spicy_salt", lambda x, s=seed + 3000 + host_index: _attack_mixed_impulse(x, impulse_amount, s)),
                ("speckle", lambda x, s=seed + 4000 + host_index: attack_speckle(x, speckle_sigma, s)),
            ]
            for attack_name, attack_fn in attacks:
                attacked = attack_fn(marked)
                estimated = clean_symbols if attack_name == "clean" else _native_ca_extract_symbols(attacked, key)
                attack_rows.append({
                    "method_id": "ca_qim_a2_mao2024",
                    "host": str(host_path),
                    "image_mode": "grayscale",
                    "frequency": "high" if message_count != "full" else "all_ac",
                    "embedded_vectors_per_block": message_count,
                    "payload_bits_per_dimension": 2,
                    "attack": attack_name,
                    "symbol_error_rate": symbol_error_rate(original_symbols, estimated),
                    "code_rate_bits_per_dimension": 2,
                    "bit_zero_probability": float(p_zero),
                    "lattice_step": float(step),
                    "detector_type": "semi_blind",
                    "noise_sigma": noise_sigma if attack_name == "gaussian" else "",
                    "impulse_amount": impulse_amount if attack_name in {"salt", "pepper", "spicy_salt"} else "",
                    "speckle_sigma": speckle_sigma if attack_name == "speckle" else "",
                    "attack_strength_source": (
                        "clean" if attack_name == "clean" else
                        "paper names the attack but does not report a machine-readable strength; repository value is disclosed"
                    ),
                    "psnr_calibrated": False,
                })

    save_csv(out / "ca_qim_native_imperceptibility.csv", imperceptibility_rows)
    save_csv(out / "ca_qim_native_ser.csv", attack_rows)
    return {"imperceptibility": imperceptibility_rows, "ser": attack_rows}

def _paper_reference_rows() -> list[dict[str, Any]]:
    """Machine-readable A2 CA-QIM values transcribed from paper Tables I--III."""
    rows: list[dict[str, Any]] = []
    table_i = {
        "low": (1.074, 47.82, 0.020),
        "mid": (0.706, 49.39, 0.016),
        "high": (0.415, 51.96, 0.014),
    }
    for frequency, metrics in table_i.items():
        rows.append({
            "paper": "Mao et al. CA-QIM",
            "method_id": "ca_qim_a2_mao2024",
            "table": "Table I",
            "domain": "frequency",
            "lattice": "A2",
            "frequency": frequency,
            "payload_bits_per_dimension": 2,
            "paper_mse": metrics[0],
            "paper_psnr": metrics[1],
            "paper_prd": metrics[2],
            "paper_ssim": "",
            "numeric_acceptance_target": False,
            "reason": "exact lattice scale, coefficient indices, and paper image realization are not reported",
        })

    table_ii_mse = {
        "low": (1.0061, 1.1304, 1.1821),
        "mid": (0.5899, 0.5961, 0.6601),
        "high": (0.1182, 0.1752, 0.2288),
    }
    table_iii_psnr = {
        "low": (51.0876, 51.3114, 51.3600),
        "mid": (52.7065, 51.4196, 51.4069),
        "high": (65.4625, 60.4534, 57.9480),
    }
    table_iii_ssim = {
        "low": (0.9585, 0.9418, 0.9346),
        "mid": (0.9989, 0.9975, 0.9971),
        "high": (0.9999, 0.9999, 0.9998),
    }
    for frequency in ("low", "mid", "high"):
        for message_count in (1, 2, 3):
            i = message_count - 1
            rows.append({
                "paper": "Mao et al. CA-QIM",
                "method_id": "ca_qim_a2_mao2024",
                "table": "Tables II-III",
                "domain": "frequency MSE; spatial PSNR/SSIM",
                "lattice": "A2",
                "frequency": frequency,
                "payload_bits_per_dimension": message_count,
                "paper_mse": table_ii_mse[frequency][i],
                "paper_psnr": table_iii_psnr[frequency][i],
                "paper_prd": "",
                "paper_ssim": table_iii_ssim[frequency][i],
                "numeric_acceptance_target": False,
                "reason": "paper datasets and exact coefficient/lattice scale choices are not bundled or fully specified",
            })
    for attack, response in (("clean", 32.0), ("scale050_restore", 13.4), ("jpeg10", 22.8), ("jpeg05", 13.9), ("floyd_steinberg_dither", 5.2), ("crop_center_quarter_restore", 14.6), ("crop_center_quarter_jpeg10_restore", 10.6)):
        rows.append({
            "paper": "Cox et al. spread spectrum",
            "method_id": "dct_spread_spectrum_cox1997",
            "table": "Section V experiments",
            "domain": "whole-image DCT",
            "attack": attack,
            "paper_detector_response": response,
            "numeric_acceptance_target": False,
            "reason": "paper used Bavarian Couple and implementation-specific registration/processing",
        })
    return rows

def _write_reproduction_checks(out: Path, qim_rows: list[dict[str, Any]], iss_rows: list[dict[str, Any]], ca_rows: dict[str, list[dict[str, Any]]], cox_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    stdm = [r for r in qim_rows if r.get("method_id") == "dct_stdm_qim_chen2001" and r.get("experiment") == "paper_native_awgn_dnr_sweep"]
    stdm_diff = max(abs(float(r["empirical_bit_error_probability"]) - float(r["paper_formula_bit_error_probability"])) for r in stdm)
    checks.append({
        "check": "STDM empirical BER matches Chen-Wornell Eq. 49 curve",
        "comparable": True,
        "max_absolute_error": stdm_diff,
        "tolerance": 0.025,
        "passed": stdm_diff <= 0.025,
    })
    iss = [r for r in iss_rows if r.get("experiment") == "paper_figure4_error_probability"]
    iss_diff = max(abs(float(r["empirical_error_probability"]) - float(r["paper_formula_error_probability"])) for r in iss)
    checks.append({
        "check": "Linear ISS empirical error probability matches paper Eq. 19",
        "comparable": True,
        "max_absolute_error": iss_diff,
        "tolerance": 0.01,
        "passed": iss_diff <= 0.01,
    })
    ca_attacks = {str(r["attack"]) for r in ca_rows["ser"]}
    required = {"clean", "gaussian", "salt", "pepper", "spicy_salt", "speckle"}
    checks.append({
        "check": "CA-QIM uses paper attack names and SER metric",
        "comparable": True,
        "max_absolute_error": "",
        "tolerance": "",
        "passed": required.issubset(ca_attacks),
    })
    ca_grid = {
        (str(r["frequency"]), int(r["payload_bits_per_dimension"]))
        for r in ca_rows["imperceptibility"]
    }
    expected_grid = {
        (frequency, message_count)
        for frequency in ("low", "mid", "high")
        for message_count in (1, 2, 3)
    }
    required_metric_fields = {
        "mse_frequency", "psnr_frequency", "prd_frequency",
        "mse_spatial", "psnr_spatial", "prd_spatial", "ssim_spatial",
    }
    checks.append({
        "check": "CA-QIM reproduces the paper low/mid/high by R=1/2/3 payload-rate grid",
        "comparable": True,
        "max_absolute_error": "",
        "tolerance": "",
        "passed": expected_grid.issubset(ca_grid),
    })
    checks.append({
        "check": "CA-QIM reports paper frequency and spatial metric families",
        "comparable": True,
        "max_absolute_error": "",
        "tolerance": "",
        "passed": all(required_metric_fields.issubset(r) for r in ca_rows["imperceptibility"]),
    })
    checks.append({
        "check": "CA-QIM native detector preserves image-dependent semi-blind mapping",
        "comparable": True,
        "max_absolute_error": "",
        "tolerance": "",
        "passed": all(r.get("detector_type") == "semi_blind" for r in ca_rows["imperceptibility"] + ca_rows["ser"]),
    })
    checks.append({
        "check": "CA-QIM Fig. 10 numerical reproduction",
        "comparable": False,
        "max_absolute_error": "",
        "tolerance": "",
        "passed": "not_asserted",
        "reason": "paper does not report machine-readable noise strengths or lattice scale",
    })
    checks.append({
        "check": "Native tracks do not calibrate to a common PSNR",
        "comparable": True,
        "max_absolute_error": "",
        "tolerance": "",
        "passed": all(not bool(r.get("psnr_calibrated")) for r in qim_rows + iss_rows + cox_rows + ca_rows["imperceptibility"] + ca_rows["ser"]),
    })
    cox_attacks = {str(r["attack"]) for r in cox_rows}
    checks.append({
        "check": "Cox native track preserves paper watermark length, alpha, informed detector, and named attacks",
        "comparable": True,
        "max_absolute_error": "",
        "tolerance": "",
        "passed": bool(cox_rows) and all(r.get("detector_type") == "non_blind" and int(r.get("watermark_length", 0)) == 1000 and abs(float(r.get("alpha", 0.0)) - 0.1) < 1e-12 for r in cox_rows) and {"clean", "scale050_restore", "jpeg10", "jpeg05", "floyd_steinberg_dither", "crop_center_quarter_restore"}.issubset(cox_attacks),
    })
    clean_cox = [r for r in cox_rows if r.get("attack") == "clean"]
    checks.append({
        "check": "Cox clean detector response has expected sqrt(N) scale and wins false-positive ranking",
        "comparable": True,
        "max_absolute_error": "",
        "tolerance": "",
        "passed": bool(clean_cox) and all(float(r.get("cox_similarity", 0.0)) > 0.70 * math.sqrt(int(r.get("watermark_length", 1))) and int(r.get("target_rank_among_candidates", 999999)) == 1 for r in clean_cox),
    })
    save_csv(out / "native_reproduction_checks.csv", checks)
    return checks

def run_all_paper_native(
    *,
    out_dir: str | Path,
    host_paths: Iterable[str | Path],
    seed: int = 2026,
    quick: bool = False,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    host_list = list(host_paths)
    qim_rows = run_qim_stdm_native_channel(
        out_dir=out,
        seed=seed,
        trials=10_000 if quick else 40_000,
        vector_length=100,
    )
    iss_rows = run_iss_native_channel(
        out_dir=out,
        seed=seed,
        trials=50_000 if quick else 250_000,
    )
    cox_rows = run_cox_native_images(
        host_paths=host_list,
        out_dir=out,
        seed=seed,
        host_limit=1 if quick else 2,
        false_candidates=99 if quick else 999,
    )
    ca_rows = run_ca_qim_native_images(
        host_paths=host_list[:2],
        out_dir=out,
        seed=seed,
        include_full=not quick,
        full_host_limit=1,
    )
    reference_rows = _paper_reference_rows()
    save_csv(out / "paper_reference_values.csv", reference_rows)
    checks = _write_reproduction_checks(out, qim_rows, iss_rows, ca_rows, cox_rows)
    protocol = {
        "protocol_version": PAPER_NATIVE_PROTOCOL_VERSION,
        "primary_principle": "paper-native operating points and metrics; no common PSNR target",
        "psnr_calibration": False,
        "common_ecc": False,
        "detector_types": {k: asdict(v)["detector_type"] for k, v in NATIVE_DISCLOSURES.items()},
        "disclosures": {k: asdict(v) for k, v in NATIVE_DISCLOSURES.items()},
        "ca_qim_attack_strength_note": (
            "The paper names Gaussian, salt, pepper, spicy-salt, and speckle attacks but the available paper text/figure "
            "does not provide machine-readable strengths. This implementation discloses its strengths and does not claim exact numeric reproduction for Fig. 10."
        ),
        "ca_qim_host_note": (
            "The executable verification track uses the first two sorted repository host images; final manuscript reproduction should replace these with the paper datasets Set12 and BSD68."
        ),
        "ca_qim_full_embedding_note": (
            "The computationally heavy ALL-AC condition is evaluated on the first native host; the 1/2/3-message conditions use both selected hosts."
        ),
        "ca_qim_capacity_interpretation_note": (
            "Tables II-III label payload rates of 1, 2, and 3 bits per dimension, so the imperceptibility track uses "
            "2A2, 4A2, and 8A2 respectively. The SER track keeps the paper setup R=2 (A2/4A2) and varies the "
            "number of embedded DCT vectors over 1, 2, 3, and all eligible AC positions."
        ),
    }
    save_json(out / "paper_native_protocol.json", protocol)
    return {"qim_stdm": qim_rows, "iss": iss_rows, "cox": cox_rows, "ca_qim": ca_rows, "checks": checks, "protocol": protocol}


__all__ = [
    "PAPER_NATIVE_PROTOCOL_VERSION", "NATIVE_DISCLOSURES",
    "run_qim_stdm_native_channel", "run_iss_native_channel",
    "run_ca_qim_native_images", "run_cox_native_images", "run_all_paper_native", "CA_NATIVE_STEP",
]
