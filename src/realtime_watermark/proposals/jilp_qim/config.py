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
    phase_refinement: str = "off"
    certified_phase_pruning: bool = False
    phase_fast_threshold: float | None = None  # experimental phase acceptance gate
    hartley_mix: float = 0.0  # experimental weighted-convolution payload carrier
    hartley_support: int = 0  # 0=original; even 2-56=compute-limited convolution carrier
    hartley_carrier_mode: str = "sparse"  # opt-in frequency-preserving carrier
    joint_embedding: str = "sequential"  # opt-in joint payload/pilot minimum-norm projection

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
        if self.phase_fast_threshold is not None and not (0.0 <= float(self.phase_fast_threshold) <= 1.0):
            raise ValueError("phase_fast_threshold must lie in [0,1]")
        if self.phase_refinement not in {"off", "selective"}:
            raise ValueError("phase_refinement must be off or selective")
        if not (0.0 <= self.hartley_mix <= 1.0):
            raise ValueError("hartley_mix must lie in [0,1]")
        if self.hartley_support and (self.hartley_support < 2 or self.hartley_support>56 or self.hartley_support%2):
            raise ValueError("hartley_support must be even 2-56")
        if self.hartley_carrier_mode not in {"sparse", "frequency_preserving"}:
            raise ValueError("invalid hartley_carrier_mode")
        if self.hartley_carrier_mode != "sparse" and (self.hartley_support == 0 or self.hartley_mix <= 0):
            raise ValueError("frequency_preserving requires positive hartley_mix and sparse support")
        if self.hartley_support > 0 and (
            not self.use_integer_lattice_closure or
            (self.projection_mode != "minimum_distortion" and
             not (self.projection_mode == "center_qim" and
                  self.joint_embedding == "orthogonal_joint_sparse_center"))
        ):
            raise ValueError("sparse Hartley carrier requires minimum-distortion integer closure or the sparse joint-center formulation")
        if self.joint_embedding not in {"sequential", "orthogonal_joint", "orthogonal_joint_center", "orthogonal_joint_sparse_center"}:
            raise ValueError("unsupported joint_embedding")
        if self.joint_embedding == "orthogonal_joint" and (
            self.hartley_support == 0 or not self.enable_pilots or
            self.projection_mode != "minimum_distortion" or not self.use_integer_lattice_closure
        ):
            raise ValueError("orthogonal_joint requires sparse Hartley carrier, pilots and integer closure")
        if self.joint_embedding == "orthogonal_joint_center" and (
            self.hartley_support != 0 or not self.enable_pilots or
            self.projection_mode != "center_qim" or not self.use_integer_lattice_closure
        ):
            raise ValueError("orthogonal_joint_center requires dense center-QIM, pilots and integer closure")
        if self.joint_embedding == "orthogonal_joint_sparse_center" and (
            self.hartley_support == 0 or not self.enable_pilots or
            self.projection_mode != "center_qim" or not self.use_integer_lattice_closure
        ):
            raise ValueError("orthogonal_joint_sparse_center requires sparse Hartley carrier, pilots, center-QIM and integer closure")
        if self.use_second_carrier:
            raise ValueError("the released repair-free proposal uses one payload carrier")
        return self

    def to_dict(self) -> dict[str, Any]:
        # Preserve the original frozen configuration hash and serialized key
        # when the two opt-in research extensions are disabled.
        values = asdict(self)
        if self.phase_refinement == "off":
            values.pop("phase_refinement")
        if not self.certified_phase_pruning:
            values.pop("certified_phase_pruning")
        if self.phase_fast_threshold is None:
            values.pop("phase_fast_threshold")
        if self.hartley_mix == 0.0:
            values.pop("hartley_mix")
        if self.hartley_support == 0:
            values.pop("hartley_support")
        if self.joint_embedding == "sequential":
            values.pop("joint_embedding")
        if self.hartley_carrier_mode == "sparse":
            values.pop("hartley_carrier_mode")
        return values

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
