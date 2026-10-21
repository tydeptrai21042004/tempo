"""Small mathematical refinement: equivalence and reproducibility checks."""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from scripts.apply_math_refinement import activate
from realtime_watermark.proposals.jilp_qim.config import DEFAULT_CONFIG, JILPQIMConfig
from realtime_watermark.proposals.jilp_qim import phase_registration as phase


def test_activation_keeps_every_other_tuned_parameter_and_is_idempotent(tmp_path):
    original = replace(DEFAULT_CONFIG, projection_mode="center_qim", rho_frac=0.35,
                       pilot_count=128, pilot_schedule="stratified",
                       field_mode="luminance", phase_refinement="off").validated()
    selected = tmp_path / "selected.json"
    control = tmp_path / "original.json"
    audit = tmp_path / "audit.json"
    original.save(selected)
    report = activate(selected, control, audit)
    new = JILPQIMConfig.load(selected)
    assert new.projection_mode == original.projection_mode
    assert new.rho_frac == original.rho_frac
    assert new.certified_phase_pruning is True
    assert new.phase_refinement == "selective"
    assert new.pilot_count == 128 and new.field_mode == "luminance"
    assert JILPQIMConfig.load(control) == original
    assert new == replace(original, certified_phase_pruning=True,
                          phase_refinement="selective")
    assert report["original_selected_sha256"] == original.sha256()
    assert report["refined_selected_sha256"] == new.sha256()
    assert json.loads(audit.read_text())["status"] == "refined_proposal_selected"
    activate(selected, control, audit)
    assert JILPQIMConfig.load(selected) == new
    assert JILPQIMConfig.load(control) == original


def test_integer_translation_scorer_equals_bilinear_reference_with_boundaries():
    rng = np.random.default_rng(173)
    field = rng.uniform(0, 255, (32, 40)).astype(np.float32)
    indices = np.array([0, 1, 4, 5, 9, 11, 14, 17, 19], dtype=np.int32)
    bits = rng.integers(0, 2, len(indices), dtype=np.uint8)
    basis = rng.normal(size=(8, 8)).astype(np.float32)
    step = 15.0
    for sy, sx in [(-8, 0), (0, -8), (8, 8), (0, 0), (3, -2), (-5, 7)]:
        actual = phase.sparse_phase_score(field, indices, bits, basis, step, 0.0,
                                           0, 0.0, 0.0, sy, sx)
        reference = 0.0
        blocks_per_row = field.shape[1] // 8
        for k, idx in enumerate(indices):
            y0, x0 = (int(idx) // blocks_per_row) * 8, (int(idx) % blocks_per_row) * 8
            coord = 0.0
            for i in range(8):
                for j in range(8):
                    coord += float(basis[i, j]) * phase._sample_field_bilinear(
                        field, float(x0 + j + sx), float(y0 + i + sy))
            reference += (1 - 2 * int(bits[k])) * np.cos(np.pi * coord / step)
        reference /= len(indices)
        assert np.isclose(actual, reference, rtol=0, atol=1e-5)


def test_certified_sparse_search_preserves_winner_and_score():
    rng = np.random.default_rng(2026)
    # Random visual content and pilot bits ensure recovery search is exercised.
    img = rng.integers(0, 256, (80, 96, 3), dtype=np.uint8)
    idx = np.arange(40, dtype=np.int32)
    bits = rng.integers(0, 2, len(idx), dtype=np.uint8)
    original, raw = phase.register_sparse_phase(
        img, idx, bits, 15.0, 0.0, quick_accept=2.0,
        translation_radius=4, certified_pruning=False)
    refined, fast = phase.register_sparse_phase(
        img, idx, bits, 15.0, 0.0, quick_accept=2.0,
        translation_radius=4, certified_pruning=True)
    assert raw["candidate"] == fast["candidate"]
    assert raw["shift_y"] == fast["shift_y"]
    assert raw["shift_x"] == fast["shift_x"]
    assert np.isclose(raw["phase_score"], fast["phase_score"], atol=1e-10)
    assert np.array_equal(original, refined)
