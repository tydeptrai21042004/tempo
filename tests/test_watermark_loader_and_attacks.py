from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from realtime_watermark.core import (
    attack_category, default_attacks, load_binary_watermark, watermark_statistics,
)


def test_alpha_only_watermark_is_not_discarded(tmp_path: Path):
    rgba = np.zeros((16, 16, 4), dtype=np.uint8)
    rgba[..., 3] = 0
    rgba[4:12, 4:12, 3] = 255
    path = tmp_path / "alpha_logo.png"
    Image.fromarray(rgba, mode="RGBA").save(path)
    wm = load_binary_watermark(path, size=16)
    stats = watermark_statistics(wm)
    assert stats["zero_bits"] > 0
    assert stats["one_bits"] > 0


def test_degenerate_watermark_is_rejected(tmp_path: Path):
    path = tmp_path / "blank.png"
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8), mode="L").save(path)
    with pytest.raises(ValueError, match="Degenerate watermark"):
        load_binary_watermark(path, size=16)



@pytest.mark.parametrize("pixel", [0, 255])
def test_constant_recovered_watermark_is_scored_but_not_accepted_as_input(tmp_path: Path, pixel: int):
    from realtime_watermark.core import ber, nc_binary, psnr, ssim_gray

    recovered_path = tmp_path / "recovered.png"
    Image.fromarray(np.full((64, 64), pixel, dtype=np.uint8)).save(recovered_path)

    # The original watermark still rejects constant payloads.
    with pytest.raises(ValueError, match="Degenerate watermark"):
        load_binary_watermark(recovered_path, size=64)

    # The decoder's result must be measurable even when it is entirely wrong.
    recovered = load_binary_watermark(recovered_path, size=64, allow_degenerate=True)
    reference = np.tile([0, 255], 2048).astype(np.uint8).reshape(64, 64)
    assert np.unique(recovered).tolist() == [pixel]
    assert ber(reference, recovered) == 0.5
    assert np.isfinite(nc_binary(reference, recovered))
    assert np.isfinite(psnr(reference, recovered))
    assert np.isfinite(ssim_gray(reference, recovered))

def test_new_standard_attacks_are_present_and_shape_preserving(host128):
    attacks = dict(default_attacks("standard"))
    for name in ("chroma420", "center_crop_resize90", "jpeg85_jpeg70", "screen_capture_light"):
        assert name in attacks
        assert attacks[name](host128).shape == host128.shape
    assert attack_category("perspective002") == "geometric"
    assert attack_category("overlay18") == "occlusion"
