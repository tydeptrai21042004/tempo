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


def test_new_standard_attacks_are_present_and_shape_preserving(host128):
    attacks = dict(default_attacks("standard"))
    for name in ("chroma420", "center_crop_resize90", "jpeg85_jpeg70", "screen_capture_light"):
        assert name in attacks
        assert attacks[name](host128).shape == host128.shape
    assert attack_category("perspective002") == "geometric"
    assert attack_category("overlay18") == "occlusion"
