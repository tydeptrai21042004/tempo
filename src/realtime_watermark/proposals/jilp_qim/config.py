from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class JILPQIMConfig:
    """Immutable experiment configuration for the proposed method."""

    schema_version: str = "2.0"
    step: float = 17.78404328715303
    rho_frac: float = 0.45
    pilot_count: int = 96
    pilot_step: float = 15.0
    pilot_rho_frac: float = 0.49
    eta: float = 0.07
    candidate_pool_multiplier: float = 1.0
    candidate_budget: int | None = None
    use_second_carrier: bool = False
    pilot_schedule: str = "stratified"
    use_integer_lattice_closure: bool = True
    projection_mode: str = "minimum_distortion"
    enable_pilots: bool = True
    enable_sync_search: bool = True
    payload_schedule_mode: str = "random"
    field_mode: str = "opponent"
    temporal_vote_mode: str = "hard"
    scramble_payload: bool = True

    def validated(self) -> "JILPQIMConfig":
        if self.step <= 0 or self.pilot_step <= 0:
            raise ValueError("payload and pilot steps must be positive")
        if self.pilot_count < 0:
            raise ValueError("pilot_count must be nonnegative")
        if self.candidate_pool_multiplier < 1.0:
            raise ValueError("candidate_pool_multiplier must be at least 1")
        if not (0.0 <= self.rho_frac < 0.5):
            raise ValueError("rho_frac must lie in [0,0.5)")
        if not (0.0 <= self.pilot_rho_frac < 0.5):
            raise ValueError("pilot_rho_frac must lie in [0,0.5)")
        if self.pilot_schedule not in {"random", "stratified"}:
            raise ValueError("pilot_schedule must be random or stratified")
        if self.payload_schedule_mode not in {"random", "content"}:
            raise ValueError("payload_schedule_mode must be random or content")
        if self.projection_mode not in {"minimum_distortion", "center_qim"}:
            raise ValueError("unsupported projection_mode")
        if self.field_mode not in {"opponent", "luminance"}:
            raise ValueError("unsupported field_mode")
        if self.temporal_vote_mode not in {"hard", "confidence_weighted"}:
            raise ValueError("unsupported temporal_vote_mode")
        if self.use_second_carrier:
            raise ValueError("the released repair-free proposal uses one payload carrier")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def with_step(self, step: float) -> "JILPQIMConfig":
        return replace(self, step=float(step)).validated()

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "JILPQIMConfig":
        if mapping is None:
            return cls().validated()
        raw = dict(mapping)
        # Accept the complete selection artifact as well as its parameters field.
        if isinstance(raw.get("parameters"), Mapping):
            raw = dict(raw["parameters"])
        allowed = {f.name for f in fields(cls)}
        values = {k: v for k, v in raw.items() if k in allowed}
        return cls(**values).validated()

    @classmethod
    def load(cls, path: str | Path) -> "JILPQIMConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_mapping(data)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps({"config_hash": self.sha256(), "parameters": self.to_dict()}, indent=2),
            encoding="utf-8",
        )


DEFAULT_CONFIG = JILPQIMConfig().validated()

__all__ = ["JILPQIMConfig", "DEFAULT_CONFIG"]
