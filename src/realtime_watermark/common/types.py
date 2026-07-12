from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class MethodRef:
    id: str
    display_name: str
    paper: str
    url: str
    implementation_note: str
    fully_blind: bool = True
    color_host: bool = True
    binary_watermark: bool = True
    real_time_target: bool = True


@dataclass
class WatermarkKey:
    method_id: str
    host_shape: tuple[int, int, int]
    watermark_shape: tuple[int, int]
    seed: int
    repeat: int
    step: float
    arnold_iter: int
    arnold_period: int
    threshold: int
    params: dict[str, Any]
    # Kept for backward compatibility with old saved keys. New keys leave this
    # empty and regenerate the schedule from seed/shape/repeat at extraction time.
    schedule: list[Any] = field(default_factory=list)
    fully_blind: bool = True
    side_information: str = "compact_key_schedule_regenerated_from_seed"


class MethodPlugin(Protocol):
    METHOD_ID: str
    METHOD_REF: MethodRef

    def embed(self, host_rgb, watermark_binary, *, seed: int = 2026, repeat: int | str = "auto", step: float | None = None): ...
    def extract(self, image_rgb, key): ...
