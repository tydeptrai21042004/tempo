"""One strict, versioned RGB attack registry used by all common-image benchmarks.

No eval or arbitrary module imports; every operator resolves to a reviewed core
attack_ function. Existing historical ``stress`` runs are unchanged.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import inspect
from typing import Callable
import numpy as np

MANIFEST = Path(__file__).resolve().parents[2] / "configs/attacks/unified_109.json"


def _operators() -> dict[str, Callable]:
    from . import core
    return {name: getattr(core, name) for name in dir(core)
            if name.startswith("attack_") and callable(getattr(core, name))
            and name != "attack_category"}


def _validate_node(node, operators, depth=0):
    if depth > 5 or not isinstance(node, dict) or not set(node).issubset({"op", "args", "kwargs"}):
        raise ValueError("Invalid or overly nested attack operation")
    op = node.get("op")
    if op != "identity" and op not in operators:
        raise ValueError(f"Unknown attack operation: {op!r}")
    args = node.get("args")
    if not isinstance(args, list) or not args:
        raise ValueError("Attack requires an input image")
    for a in args:
        if isinstance(a, dict):
            _validate_node(a, operators, depth + 1)
        elif isinstance(a, (int, float, str)) and not isinstance(a, bool):
            if isinstance(a, str) and a != "$image":
                # Fixed string args are used for axis directions.
                if a not in {"horizontal", "vertical", "diagonal"}:
                    raise ValueError(f"Unknown argument token: {a!r}")
        else:
            raise ValueError("Invalid attack argument")
    if args[0] != "$image" and not isinstance(args[0], dict):
        raise ValueError("First argument must be the image or nested image transform")
    if any(a == "$image" for a in args[1:] if isinstance(a, str)):
        raise ValueError("Image token may only occur as first argument")
    kwargs = node.get("kwargs", {})
    if not isinstance(kwargs, dict):
        raise ValueError("Attack keyword arguments must be a mapping")
    for k, v in kwargs.items():
        if not isinstance(k, str) or not isinstance(v, (int, float, str)):
            raise ValueError("Unsupported attack keyword")
    if op != "identity":
        placeholders = [np.zeros((8, 8, 3), dtype=np.uint8)]
        for a in args[1:]:
            placeholders.append(np.zeros((8, 8, 3), dtype=np.uint8) if isinstance(a, dict) else a)
        inspect.signature(operators[op]).bind(*placeholders, **kwargs)


@lru_cache(maxsize=1)
def load_manifest() -> dict:
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "tempo-attacks-v1":
        raise ValueError("Unsupported attack manifest schema")
    entries = raw.get("attacks")
    if not isinstance(entries, list) or len(entries) != 110:
        raise ValueError("Expected exactly 1 clean + 109 attacked conditions")
    ops = _operators()
    seen = set()
    for i, e in enumerate(entries):
        attack_id = e.get("id")
        if not isinstance(attack_id, str) or not attack_id or attack_id in seen:
            raise ValueError(f"Duplicate/invalid attack ID at index {i}")
        seen.add(attack_id)
        _validate_node(e.get("pipeline"), ops)
    if entries[0]["id"] != "clean":
        raise ValueError("First condition must be clean")
    return raw


def attack_manifest_sha256() -> str:
    return hashlib.sha256(MANIFEST.read_bytes()).hexdigest()


def _evaluate(node: dict, img: np.ndarray, ops: dict) -> np.ndarray:
    args = [(_evaluate(a, img, ops) if isinstance(a, dict) else img if a == "$image" else a)
            for a in node["args"]]
    result = args[0] if node["op"] == "identity" else ops[node["op"]](*args, **node.get("kwargs", {}))
    result = np.asarray(result)
    if result.shape != img.shape or result.dtype != np.uint8:
        raise ValueError(f"Attack {node['op']} returned invalid RGB shape/dtype")
    return result


def unified_attacks() -> list[tuple[str, Callable]]:
    manifest = load_manifest()
    ops = _operators()
    def apply(node, img):
        original = np.asarray(img, dtype=np.uint8)
        if original.ndim != 3 or original.shape[-1] != 3:
            raise ValueError("Expected RGB uint8 input")
        return _evaluate(node, original, ops)
    return [(entry["id"], lambda x, node=entry["pipeline"]: apply(node, x))
            for entry in manifest["attacks"]]


def manifest_summary() -> dict:
    info = load_manifest()
    return {"protocol_id": info["protocol_id"], "sha256": attack_manifest_sha256(),
            "total_conditions": len(info["attacks"]),
            "attacked_conditions": len(info["attacks"]) - 1,
            "attack_ids": [entry["id"] for entry in info["attacks"]]}
