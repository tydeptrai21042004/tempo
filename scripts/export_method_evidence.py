#!/usr/bin/env python3
from __future__ import annotations

"""Export machine-readable fidelity and behavior-test evidence reports."""

import argparse
import ast
import csv
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realtime_watermark.paper_fidelity import PAPER_FIDELITY


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _behavior_tests(test_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(test_root.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                doc = ast.get_docstring(node) or ""
                rows.append({"file": str(path.relative_to(ROOT)), "test": node.name, "docstring": doc})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export baseline fidelity and behavior-test manifests.")
    parser.add_argument("--out-dir", default="reports/method_evidence")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    matrix: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for method_id, entry in PAPER_FIDELITY.items():
        matrix.append({
            "method_id": method_id,
            "native_method_id": entry["native_method_id"],
            "common_adapter_method_id": entry["common_adapter_method_id"],
            "native_status": entry["native_status"],
            "common_adapter_status": entry["common_adapter_status"],
            "paper_reproduction_eligible": entry["paper_reproduction_eligible"],
            "common_primary_eligible": entry["common_primary_eligible"],
            "blindness_tier": entry["blindness_tier"],
            "verified_capacity": entry["verified_capacity"],
            "verified_stage_count": len(entry["verified"]),
            "adapted_stage_count": len(entry["adapted"]),
            "unresolved_stage_count": len(entry["unresolved_core_stages"]),
            "evidence_files": ";".join(entry["evidence_files"]),
            "eligibility_reason": entry.get("eligibility_reason", ""),
        })
        for stage in entry["unresolved_core_stages"]:
            unresolved.append({
                "method_id": method_id,
                "stage": stage,
                "native_status": entry["native_status"],
                "blocks_paper_reproduction": True,
            })

    behavior = _behavior_tests(ROOT / "tests")
    manifest = {
        "scope": "algorithm behavior and reporting schemas; no PSNR, NC, BER, or FPS pass thresholds",
        "test_count": len(behavior),
        "tests": behavior,
        "method_count": len(PAPER_FIDELITY),
    }

    _write_csv(out / "baseline_evidence_matrix.csv", matrix)
    _write_csv(out / "unresolved_stage_report.csv", unresolved)
    (out / "behavior_test_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote evidence reports to {out.resolve()}")


if __name__ == "__main__":
    main()
