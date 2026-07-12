from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from .common.keys import used_units_from_key
from .common.types import WatermarkKey
from .core import NUMBA_AVAILABLE
from .proposals.jilp_qim.method import warmup_jilp_qim as _warmup_jilp_qim
from .registry import (
    ALL_METHODS, METHOD_REFS, METHOD_ALIASES, normalize_method_id,
    embed as _registry_embed, extract as _registry_extract,
)

_NUMBA_WARMED_UP = False


def warmup_numba() -> None:
    global _NUMBA_WARMED_UP
    if _NUMBA_WARMED_UP or not NUMBA_AVAILABLE:
        return
    _warmup_jilp_qim()
    _NUMBA_WARMED_UP = True


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
    return _registry_embed(
        method_id, host_rgb, watermark_binary, seed=seed, repeat=repeat,
        step=step, method_params=method_params,
    )


def extract(image_rgb: np.ndarray, key: WatermarkKey | dict[str, Any]) -> np.ndarray:
    return _registry_extract(image_rgb, key)


def method_metadata() -> list[dict[str, Any]]:
    return [asdict(METHOD_REFS[m]) for m in ALL_METHODS]


__all__ = [
    "ALL_METHODS", "METHOD_REFS", "METHOD_ALIASES", "normalize_method_id",
    "embed", "extract", "method_metadata", "warmup_numba", "used_units_from_key",
]
