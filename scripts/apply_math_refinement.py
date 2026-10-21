#!/usr/bin/env python3
"""Enable the minimal *synchronization-only* revision on a selected config.

No payload QIM strength, carrier, pilot number/placement, color model, or
integer closure is altered.  This avoids quietly comparing different
operating points when attributing the effect of mathematical phase search.
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
    if not control.is_file():
        control.parent.mkdir(parents=True, exist_ok=True)
        control.write_bytes(selected.read_bytes())
    original = JILPQIMConfig.load(control)
    refined = replace(
        original, certified_phase_pruning=True, phase_refinement="selective",
    ).validated()
    refined.save(selected)
    report = {
        "status": "refined_proposal_selected",
        "original_selected_sha256": original.sha256(),
        "refined_selected_sha256": refined.sha256(),
        "original_config_path": str(control),
        "refined_config_path": str(selected),
        "changes": {
            "certified_phase_pruning": [original.certified_phase_pruning, True],
            "phase_refinement": [original.phase_refinement, "selective"],
        },
        "selection_note": "All modulation and embedding parameters are preserved; holdout results must not be used for parameter selection.",
        "certification_scope": "Exact upper-bound pruning within each evaluated finite candidate search; local phase refinement is empirical and not globally optimal.",
        "warning": "Improvements in NC or FPS require paired end-to-end measurement; they are not mathematical guarantees.",
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
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
