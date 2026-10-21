# TEMPO — selective phase-gate FPS patch

Apply this ZIP **over the latest `tempo-main` repository root**, preserving the included paths. The archive contains only one modified production source file and one new regression test. No embedding, QIM, payload, pilot scheduling, or manuscript data files are changed.

## Source comparison

Original implementation (`src/realtime_watermark/proposals/jilp_qim/phase_registration.py`):

```python
certified_fast = raw_score >= max(0.80, 2.0 * float(quick_accept) - 1.0)
```

Patched implementation:

```python
phase_gate = 0.65 if (phase_refinement == "selective" and certified_pruning) else 0.80
certified_fast = raw_score >= max(phase_gate, 2.0 * float(quick_accept) - 1.0)
```

The 0.65 threshold applies **only** to the selected implementation with `phase_refinement="selective"` and `certified_phase_pruning=True`; every other configuration keeps the original 0.80 threshold. It is the same attack-independent QIM-phase objective and recovery algorithm, with a less conservative early-acceptance gate. This is **empirically calibrated**, not a mathematical proof of lossless recovery on arbitrary future attacks.

The selected 2026-10-21 experiment uses 4096-bit payloads, 128 pilots, center-QIM, selective refinement, and certified pruning. Its embedding and decoding formulas remain unchanged.

## Run verification

From the repository root:

```bash
python -m pip install -e '.[dev]'
PYTHONPATH=src:. python -m pytest -q
```

On the local CPU setup, 176 tests passed, including the five new gate-specific parameter tests. Paired local validation with 12 BMP hosts (Lena excluded), 2 original PNG watermarks, and 10 attack conditions per pair (240 cases), using the selected saved experiment configuration, found:

| Measurement | Original | Patched |
|---|---:|---:|
| Recovery path entered | 179 / 240 | 125 / 240 |
| Mean complete extraction duration | 3.703 ms | 2.724 ms |
| Median complete extraction duration | 3.518 ms | 2.987 ms |
| Recovered watermark bitmap differences | — | 0 of 240 |

Test environment and CPU load can affect timings. Pixel embedding/PSNR is unchanged by this extraction-only patch. The result is **not** evidence that end-to-end 720p JPEG70 streaming meets 30 or 60 FPS; that path is dominated by other components (JPEG, frame metrics, queueing). Do not claim a universally faster decoder or mathematically guaranteed equal outputs on untested attacks.

## Files in this patch

- `src/realtime_watermark/proposals/jilp_qim/phase_registration.py` — modified original file.
- `tests/test_phase_gate_fps_regression.py` — new unit regression tests, checking the selected branch and unchanged behavior for other modes.
