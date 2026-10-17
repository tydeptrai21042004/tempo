#!/usr/bin/env python3
"""Apply TEMPO's opt-in *minimal* mathematical refinement after normal tuning.

Only two choices change: the payload QIM projection becomes a nearest guarded
point with rho/step = 0.45, and sparse pilot scoring gains certified pruning.
All other tuned configuration fields are kept exactly as selected on the
DEVELOPMENT split. Original selection is preserved for an honest control.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from realtime_watermark.proposals.jilp_qim.config import JILPQIMConfig


def activate(selected: Path, control: Path, audit: Path) -> dict:
    if not selected.is_file():
        raise FileNotFoundError(selected)
    # Idempotence: only ever derive from the *original* selected configuration.
    if not control.is_file():
        control.parent.mkdir(parents=True, exist_ok=True)
        control.write_bytes(selected.read_bytes())
    original = JILPQIMConfig.load(control)
    refined = replace(
        original, projection_mode="minimum_distortion", rho_frac=0.45,
        certified_phase_pruning=True, phase_refinement="off",
    ).validated()
    refined.save(selected)
    report = {
        "status": "refined_proposal_selected",
        "original_selected_sha256": original.sha256(),
        "refined_selected_sha256": refined.sha256(),
        "original_config_path": str(control),
        "refined_config_path": str(selected),
        "changes": {
            "projection_mode": [original.projection_mode, refined.projection_mode],
            "rho_frac": [original.rho_frac, refined.rho_frac],
            "certified_phase_pruning": [original.certified_phase_pruning, refined.certified_phase_pruning],
            "phase_refinement": [original.phase_refinement, refined.phase_refinement],
        },
        "selection_note": "Development-selected parameters are otherwise unchanged; neither held-out test images nor video results influence this refinement.",
        "equivalence_note": "Phase pruning uses exact signed-cosine upper bounds; integer-shift scoring directly samples pixel centres.",
        "warning": "Improvements to PSNR, NC, or end-to-end FPS are empirical hypotheses until matched tests are run.",
    }
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-config", type=Path, required=True)
    parser.add_argument("--original-copy", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    result = activate(args.selected_config, args.original_copy, args.audit)
    print("TEMPO mathematical refinement enabled:")
    print(json.dumps(result["changes"], indent=2))
    print("Original SHA256:", result["original_selected_sha256"])
    print("Refined SHA256:", result["refined_selected_sha256"])


if __name__ == "__main__":
    main()
