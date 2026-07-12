from __future__ import annotations

from pathlib import Path

from realtime_watermark.all_metrics import metric_applicability_rows, run_all_methods_all_metrics
from realtime_watermark.registry import COMMON_IMAGE_METHODS


def test_metric_applicability_contains_every_generic_metric():
    rows = metric_applicability_rows(COMMON_IMAGE_METHODS)
    assert {r["method_id"] for r in rows} == set(COMMON_IMAGE_METHODS)
    for row in rows:
        assert all(row[name] for name in (
            "generic_BER", "generic_NC", "generic_SER_4bit", "generic_MSE",
            "generic_PSNR", "generic_PRD", "generic_SSIM", "generic_timing",
        ))
    cox = next(r for r in rows if r["method_id"] == "dct_spread_spectrum_cox1997")
    assert cox["detector_type"] == "non_blind"
    assert cox["requires_original_host"] is True


def test_quick_all_metrics_has_complete_schema(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    host = root / "data" / "user_dataset" / "host" / "lenna.bmp"
    result = run_all_methods_all_metrics(
        host_paths=[host], watermark_path=None, out_dir=tmp_path,
        methods=["dct_dm_qim_chen2001", "dct_spread_spectrum_cox1997"],
        seed=8, wm_size=8, attack_preset="smoke", mode="512",
    )
    assert len(result["rows"]) == 2 * 3
    required = {
        "ber", "empirical_error_probability", "nc_bipolar", "ser_4bit",
        "embedding_mse", "embedding_psnr", "embedding_prd", "embedding_ssim",
        "attacked_vs_host_mse", "attacked_vs_host_psnr", "attacked_vs_host_prd",
        "attacked_vs_host_ssim", "embed_ms", "extract_ms", "native_detection_metric",
        "native_detection_value", "detector_type", "key_bytes",
    }
    assert all(required <= set(row) for row in result["rows"])
    assert (tmp_path / "all_methods_all_metrics.csv").exists()
    assert (tmp_path / "metric_applicability.csv").exists()
    assert (tmp_path / "all_metrics_protocol.json").exists()
