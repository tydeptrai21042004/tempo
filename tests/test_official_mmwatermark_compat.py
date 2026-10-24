"""Regression coverage for TEMPO's *external-provider* compatibility layer.

Upstream source is not edited, renamed, or replaced with local attacks.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from skimage.filters import gaussian


ROOT = Path(__file__).resolve().parents[1]


def _adapter():
    spec = importlib.util.spec_from_file_location(
        'tempo_official_mmwatermark_under_test',
        ROOT / 'scripts/run_official_mmwatermark_tempo.py',
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_legacy_multichannel_gaussian_works_with_new_skimage():
    mod = _adapter()
    upstream = SimpleNamespace(gaussian=gaussian)
    assert mod.adapt_skimage_gaussian(upstream) is True
    rgb = np.random.default_rng(15).random((16, 17, 3))
    actual = upstream.gaussian(rgb, sigma=1.5, multichannel=True)
    expected = gaussian(rgb, sigma=1.5, channel_axis=-1)
    np.testing.assert_allclose(actual, expected)
    assert mod.adapt_skimage_gaussian(upstream) is False  # wrapper is idempotent


def test_implicit_rgb_and_explicit_grayscale_gaussian_axes():
    mod = _adapter()
    upstream = SimpleNamespace(gaussian=gaussian)
    assert mod.adapt_skimage_gaussian(upstream)
    rgb = np.random.default_rng(2).random((12, 13, 3))
    np.testing.assert_allclose(
        upstream.gaussian(rgb, sigma=1.0),
        gaussian(rgb, sigma=1.0, channel_axis=-1),
    )
    gray = rgb[..., 0]
    np.testing.assert_allclose(
        upstream.gaussian(gray, sigma=1.0),
        gaussian(gray, sigma=1.0, channel_axis=None),
    )
    np.testing.assert_allclose(
        upstream.gaussian(rgb, sigma=1.0, multichannel=False),
        gaussian(rgb, sigma=1.0, channel_axis=None),
    )


def test_import_official_activates_module_local_adapter(tmp_path):
    mod = _adapter()
    upstream_path = tmp_path / 'image_perturb.py'
    upstream_path.write_text(
        'import numpy as np\n'
        'from skimage.filters import gaussian\n'
        'def gaussian_blur(x, severity):\n'
        '    return gaussian(np.asarray(x)/255.0, sigma=severity, multichannel=True)*255\n',
        encoding='utf-8',
    )
    official = mod.import_official(upstream_path, tmp_path)
    assert official._tempo_skimage_gaussian_compat is True
    img = np.random.default_rng(91).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    out = mod.call_official(official.gaussian_blur, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 2)
    assert out.shape == img.shape and out.dtype == np.uint8
    # Source bytes stay unchanged: only the loaded function binding is adapted.
    assert 'multichannel=True' in upstream_path.read_text()


def test_preflight_keeps_supported_conditions_and_records_failed_ones():
    mod = _adapter()
    sample = np.zeros((16, 16, 3), dtype=np.uint8)

    def gaussian_blur(img, severity):
        return img

    def broken_attack(img, severity):
        if severity == 1:
            raise TypeError('upstream incompatibility')
        return img[:5, :5]

    namespace = SimpleNamespace(gaussian_blur=gaussian_blur,
                                broken_attack=broken_attack)
    supported, unsupported = mod.preflight_official(
        namespace, ['gaussian_blur', 'broken_attack'], [1, 2],
        sample, 'sample.bmp', 'wm.png', seed=2026,
    )
    assert supported == [('gaussian_blur', 1), ('gaussian_blur', 2)]
    assert len(unsupported) == 2
    assert unsupported[0]['attack'] == 'broken_attack'
    assert unsupported[0]['severity'] == 1
    assert 'upstream incompatibility' in unsupported[0]['reason']
    assert 'changed geometry' in unsupported[1]['reason']


def test_upstream_rotate_img_is_discovered(tmp_path):
    mod = _adapter()
    path = tmp_path / 'image_perturb.py'
    path.write_text('def gaussian_noise(x, severity): return x\n'
                    'def rotate_img(x, severity=5): return x\n'
                    'def unrelated(x, severity): return x\n', encoding='utf8')
    assert mod.discover_names(path) == ['gaussian_noise','rotate_img']


def test_upstream_skimage_random_noise_is_repeatably_seeded(tmp_path):
    mod = _adapter()
    path = tmp_path / 'image_perturb.py'
    path.write_text('import skimage as sk\n'
                    'def impulse_noise(x, severity):\n'
                    '    return sk.util.random_noise(x/255.0, mode="s&p", amount=0.2)*255\n',
                    encoding='utf8')
    official = mod.import_official(path, tmp_path)
    assert official._tempo_random_noise_seed_adapter
    x = np.full((24, 24, 3), 127, dtype=np.uint8)
    bgr = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    a = mod.call_official(official.impulse_noise, bgr, 1, seed=2026)
    b = mod.call_official(official.impulse_noise, bgr, 1, seed=2026)
    c = mod.call_official(official.impulse_noise, bgr, 1, seed=2027)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert path.read_text().startswith('import skimage as sk')
