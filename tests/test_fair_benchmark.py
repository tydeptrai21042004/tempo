from __future__ import annotations

import json
import pytest

from realtime_watermark.fair_benchmark import calibrate_step_to_psnr, run_fair_compare
from realtime_watermark.proposals.jilp_qim.config import DEFAULT_CONFIG


def test_calibration_returns_positive_step(host128, wm16):
    step, observed = calibrate_step_to_psnr(
        "dct_dm_qim_chen2001", host128, wm16,
        target_psnr=45.0, seed=1, iterations=4,
    )
    assert step > 0
    assert observed > 0


def test_fair_protocol_rejects_identical_calibration_and_evaluation(tmp_path, host128, wm16):
    from PIL import Image
    host_path = tmp_path / "host.png"
    wm_path = tmp_path / "wm.png"
    Image.fromarray(host128).save(host_path)
    Image.fromarray(wm16).save(wm_path)
    with pytest.raises(ValueError, match="disjoint"):
        run_fair_compare(
            host_path=host_path, calibration_host_path=host_path,
            watermark_path=wm_path, out_dir=tmp_path / "out", mode="full",
            methods=["dct_dm_qim_chen2001"], wm_size=16,
            calibration_iterations=2, timing_repeats=1, timing_warmups=0,
            attack_preset="smoke",
        )


def test_small_fair_run_writes_consistent_config_hashes(tmp_path):
    result = run_fair_compare(
        host_path=None, watermark_path=None, out_dir=tmp_path,
        mode="512", methods=["proposal"], wm_size=16, seed=4,
        target_psnr=45.0, psnr_tolerance=3.0,
        calibration_iterations=3, timing_repeats=1, timing_warmups=0,
        attack_preset="smoke",
        method_params_by_method={"proposal": DEFAULT_CONFIG.to_dict()},
    )
    summary = result["summary"][0]
    hashes = {row["configuration_sha256"] for row in result["attacks"]}
    assert hashes == {summary["configuration_sha256"]}
    protocol = json.loads((tmp_path / "fair_protocol.json").read_text())
    assert protocol["method_params_by_method"]["proposal"] == DEFAULT_CONFIG.to_dict()
    assert "process_cumulative_peak_rss_mb" in summary
    assert "peak_process_rss_mb" not in summary
    assert "integer_lattice_unit_updates" in summary
    assert "recovery_invocation_rate" in summary
    assert all("extraction_path" in row for row in result["attacks"])
