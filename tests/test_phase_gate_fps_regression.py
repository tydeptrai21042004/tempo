"""Gate regression: only the selected, validated phase mode uses 0.65."""

import numpy as np
import pytest

from realtime_watermark.proposals.jilp_qim import phase_registration as phase


@pytest.mark.parametrize(
    "refinement,pruning,quick_accept,expect_fast",
    [
        ("selective", True, 0.72, True),
        ("selective", False, 0.72, False),
        ("off", True, 0.72, False),
        ("off", False, 0.72, False),
        ("selective", True, 0.95, False),
    ],
)
def test_empirical_fast_gate_keeps_unmodified_modes(
    monkeypatch, refinement, pruning, quick_accept, expect_fast
):
    # 0.70 is strictly inside the newly accepted [0.65, 0.80) range.
    monkeypatch.setattr(phase, "_identity_phase_rgb", lambda *args: 0.70)

    class EntersRecovery(Exception):
        pass

    def fail_on_recovery(*args, **kwargs):
        raise EntersRecovery

    monkeypatch.setattr(phase, "sparse_phase_score", fail_on_recovery)
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    indices = np.array([0], dtype=np.int32)
    bits = np.array([1], dtype=np.uint8)
    kwargs = dict(
        quick_accept=quick_accept,
        phase_refinement=refinement,
        certified_pruning=pruning,
    )
    if not expect_fast:
        with pytest.raises(EntersRecovery):
            phase.register_sparse_phase(img, indices, bits, 15.0, 0.0, **kwargs)
    else:
        returned, meta = phase.register_sparse_phase(img, indices, bits, 15.0, 0.0, **kwargs)
        assert returned is img
        assert meta["path"] == "raw_fast_jilp"
        assert meta["phase_score"] == pytest.approx(0.70)
        assert meta["warp_count"] == 0
