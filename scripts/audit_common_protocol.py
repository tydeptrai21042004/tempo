#!/usr/bin/env python3
"""Validate versioned attacks, historical compatibility, and baseline disclosure.

This check is NOT proof of faithful paper reproduction, or completed experiments.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from realtime_watermark.attack_manifest import manifest_summary
from realtime_watermark.core import default_attacks
from realtime_watermark.registry import BASELINE_METHODS, PAPER_NATIVE_METHODS
from realtime_watermark.paper_fidelity import fidelity_status


def audit(preset: str = 'unified', compare_legacy: bool = True) -> dict:
    attacks = default_attacks(preset)
    names = [n for n, _ in attacks]
    if len(set(names)) != len(names) or not names or names[0] != 'clean':
        raise RuntimeError('Attack names must be unique; clean must be first')
    expected = manifest_summary() if preset == 'unified' else None
    if expected is not None and names != expected['attack_ids']:
        raise RuntimeError('Unified attack protocol does not match frozen manifest')
    rng = np.random.default_rng(20)
    sample = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
    # No silent fallbacks: execute every new attack on a real RGB sample before
    # costly watermarking experiments. Keep legacy sample comparisons strict.
    for name, fn in attacks:
        output = fn(sample.copy())
        if output.shape != sample.shape or output.dtype != np.uint8:
            raise RuntimeError(f'Invalid output from {name}')
        if name == 'clean' and not np.array_equal(sample, output):
            raise RuntimeError('Clean attack must be identity')
    legacy_match = None
    if preset == 'unified' and compare_legacy:
        old = default_attacks('stress')
        if names[:len(old)] != [name for name, _ in old]:
            raise RuntimeError('Original attack IDs/order changed')
        for (name, orig), (_, new) in zip(old, attacks):
            if not np.array_equal(orig(sample), new(sample)):
                raise RuntimeError(f'Unified manifest changed original attack semantics: {name}')
        legacy_match = True
    baselines = []
    for method in BASELINE_METHODS:
        detail = fidelity_status(method)
        baselines.append({
            'method_id': method,
            'blindness_tier': detail['blindness_tier'],
            'reproduction_eligible': detail.get('paper_reproduction_eligible', False),
            'primary_matched_psnr_eligible': detail['common_primary_eligible'],
            'adapter_status': detail['common_adapter_status'],
            'verified_capacity': detail['verified_capacity'],
            'source_evidence_file': detail['evidence_files'][0],
            'mathematical_reference_audit': 'requires independent source-level confirmation',
        })
    return {'protocol': preset,
            'attack_protocol': expected or {'attack_ids': names},
            'legacy_74_attacks_bit_equivalent_on_sample': legacy_match,
            'total_conditions': len(names),
            'attacked_conditions': len(names)-1,
            'baseline_disclosure': baselines,
            'primary_rank_detector_assumption': 'blind only',
            'method_code_changes': 'none in embedding/extraction',
            'status':'preflight_passed_not_full_benchmark'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--attack-preset', default='unified', choices=['unified','stress','standard','smoke'])
    args = p.parse_args()
    result = audit(args.attack_preset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('Attack protocol OK:', result['attacked_conditions'], 'non-clean conditions')

if __name__ == '__main__': main()
