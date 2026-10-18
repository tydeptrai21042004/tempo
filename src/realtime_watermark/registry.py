from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from .common.types import MethodRef, WatermarkKey
from .baselines import (
    dct_dm_qim_chen2001,
    dct_stdm_qim_chen2001,
    dct_iss_malvar2003,
    ca_qim_a2_mao2024,
    dct_spread_spectrum_cox1997,
    dew_langelaar2001,
)
from .proposals.jilp_qim import method as proposal_method
from .proposals import rcme_qim as rcme_method
from .paper_fidelity import fidelity_status

METHOD_MODULES = {
    dct_dm_qim_chen2001.METHOD_ID: dct_dm_qim_chen2001,
    dct_stdm_qim_chen2001.METHOD_ID: dct_stdm_qim_chen2001,
    dct_iss_malvar2003.METHOD_ID: dct_iss_malvar2003,
    ca_qim_a2_mao2024.METHOD_ID: ca_qim_a2_mao2024,
    dct_spread_spectrum_cox1997.METHOD_ID: dct_spread_spectrum_cox1997,
    dew_langelaar2001.METHOD_ID: dew_langelaar2001,
    proposal_method.METHOD_ID: proposal_method,
    rcme_method.METHOD_ID: rcme_method,
}

METHOD_ALIASES: dict[str, str] = {
    "dm_qim": dct_dm_qim_chen2001.METHOD_ID,
    "stdm_qim": dct_stdm_qim_chen2001.METHOD_ID,
    "stdm": dct_stdm_qim_chen2001.METHOD_ID,
    "iss": dct_iss_malvar2003.METHOD_ID,
    "ca_qim": ca_qim_a2_mao2024.METHOD_ID,
    "cox": dct_spread_spectrum_cox1997.METHOD_ID,
    "cox_ss": dct_spread_spectrum_cox1997.METHOD_ID,
    "dew": dew_langelaar2001.METHOD_ID,
    "differential_energy": dew_langelaar2001.METHOD_ID,
    "jilp_qim": proposal_method.METHOD_ID,
    "rcme": rcme_method.METHOD_ID,
}

BASELINE_METHODS = [
    dct_dm_qim_chen2001.METHOD_ID,
    dct_stdm_qim_chen2001.METHOD_ID,
    dct_iss_malvar2003.METHOD_ID,
    ca_qim_a2_mao2024.METHOD_ID,
    dct_spread_spectrum_cox1997.METHOD_ID,
    dew_langelaar2001.METHOD_ID,
]
PROPOSAL_METHOD = proposal_method.METHOD_ID
# Existing 4096-bit common image/video protocols are intentionally unchanged:
# paper-style DEW requires at least two DCT blocks per bit.  DEW is evaluated
# in the new 256-bit online stream pipeline and remains directly callable for
# lower-payload image experiments.
PAPER_NATIVE_METHODS: list[str] = [
    dct_dm_qim_chen2001.METHOD_ID,
    dct_stdm_qim_chen2001.METHOD_ID,
    dct_iss_malvar2003.METHOD_ID,
    ca_qim_a2_mao2024.METHOD_ID,
    dct_spread_spectrum_cox1997.METHOD_ID,
]
COMMON_IMAGE_METHODS = [*PAPER_NATIVE_METHODS, PROPOSAL_METHOD]
COMMON_VIDEO_METHODS = [
    dct_dm_qim_chen2001.METHOD_ID,
    dct_stdm_qim_chen2001.METHOD_ID,
    dct_iss_malvar2003.METHOD_ID,
    ca_qim_a2_mao2024.METHOD_ID,
    PROPOSAL_METHOD,
]
STREAM_BENCHMARK_METHODS = [
    dct_dm_qim_chen2001.METHOD_ID,
    dct_stdm_qim_chen2001.METHOD_ID,
    dct_iss_malvar2003.METHOD_ID,
    ca_qim_a2_mao2024.METHOD_ID,
    dew_langelaar2001.METHOD_ID,
    PROPOSAL_METHOD,
]
DIAGNOSTIC_METHODS: list[str] = [rcme_method.METHOD_ID]
IMAGE_BENCHMARK_METHODS = list(COMMON_IMAGE_METHODS)
ALL_METHODS = [*BASELINE_METHODS, PROPOSAL_METHOD]
ALL_PUBLIC_METHOD_IDS = sorted(set(ALL_METHODS) | set(METHOD_ALIASES) | set(DIAGNOSTIC_METHODS))


def normalize_method_id(method_id: str) -> str:
    return METHOD_ALIASES.get(str(method_id), str(method_id))


METHOD_REFS: dict[str, MethodRef] = {
    method_id: METHOD_MODULES[method_id].METHOD_REF for method_id in [*ALL_METHODS, *DIAGNOSTIC_METHODS]
}


def _call_with_optional_params(fn, *args, method_params: dict[str, Any] | None = None, **kwargs):
    try:
        return fn(*args, method_params=method_params, **kwargs)
    except TypeError as exc:
        if "method_params" not in str(exc):
            raise
        return fn(*args, **kwargs)


def embed(
    method_id: str,
    host_rgb: np.ndarray,
    watermark_binary: np.ndarray,
    *,
    seed: int = 2026,
    repeat: int | str = "auto",
    step: float | None = None,
    method_params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, WatermarkKey]:
    requested_id = str(method_id)
    canonical_id = normalize_method_id(requested_id)
    if canonical_id not in METHOD_MODULES:
        raise KeyError(requested_id)
    watermarked, key = _call_with_optional_params(
        METHOD_MODULES[canonical_id].embed,
        host_rgb,
        watermark_binary,
        seed=seed,
        repeat=repeat,
        step=step,
        method_params=method_params,
    )
    key.params.setdefault("canonical_method_id", canonical_id)
    key.params["requested_protocol_id"] = requested_id
    key.params.setdefault("fidelity", fidelity_status(canonical_id))
    return watermarked, key


def prepare_embedding(
    method_id: str,
    host_rgb: np.ndarray,
    watermark_binary: np.ndarray,
    *,
    seed: int = 2026,
    repeat: int | str = 1,
    step: float | None = None,
    method_params: dict[str, Any] | None = None,
):
    canonical_id = normalize_method_id(method_id)
    if canonical_id not in METHOD_MODULES:
        raise KeyError(method_id)
    module = METHOD_MODULES[canonical_id]
    fn = getattr(module, "prepare_embedding", None)
    if fn is None:
        return None
    return _call_with_optional_params(
        fn,
        host_rgb,
        watermark_binary,
        seed=seed,
        repeat=repeat,
        step=step,
        method_params=method_params,
    )


def embed_prepared(
    method_id: str,
    host_rgb: np.ndarray,
    watermark_binary: np.ndarray,
    prepared,
    *,
    seed: int = 2026,
    repeat: int | str = 1,
    step: float | None = None,
    method_params: dict[str, Any] | None = None,
):
    requested_id = str(method_id)
    canonical_id = normalize_method_id(requested_id)
    module = METHOD_MODULES[canonical_id]
    fn = getattr(module, "embed_prepared", None)
    if prepared is None or fn is None:
        watermarked, key = _call_with_optional_params(
            module.embed,
            host_rgb,
            watermark_binary,
            seed=seed,
            repeat=repeat,
            step=step,
            method_params=method_params,
        )
    else:
        watermarked, key = _call_with_optional_params(
            fn,
            host_rgb,
            watermark_binary,
            prepared,
            step=step,
            method_params=method_params,
        )
    key.params.setdefault("canonical_method_id", canonical_id)
    key.params["requested_protocol_id"] = requested_id
    key.params.setdefault("fidelity", fidelity_status(canonical_id))
    return watermarked, key


def extract(image_rgb: np.ndarray, key: WatermarkKey | dict[str, Any]) -> np.ndarray:
    method_id = key["method_id"] if isinstance(key, dict) else key.method_id
    canonical_id = normalize_method_id(method_id)
    if canonical_id not in METHOD_MODULES:
        raise KeyError(method_id)
    return METHOD_MODULES[canonical_id].extract(image_rgb, key)


def method_metadata() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_id in ALL_METHODS:
        row = asdict(METHOD_REFS[method_id])
        row["fidelity"] = fidelity_status(method_id)
        rows.append(row)
    return rows


__all__ = [
    "METHOD_MODULES", "BASELINE_METHODS", "PROPOSAL_METHOD", "PAPER_NATIVE_METHODS",
    "COMMON_IMAGE_METHODS", "COMMON_VIDEO_METHODS", "STREAM_BENCHMARK_METHODS", "DIAGNOSTIC_METHODS",
    "IMAGE_BENCHMARK_METHODS", "ALL_METHODS", "ALL_PUBLIC_METHOD_IDS", "METHOD_REFS",
    "METHOD_ALIASES", "normalize_method_id", "embed", "prepare_embedding",
    "embed_prepared", "extract", "method_metadata",
]
