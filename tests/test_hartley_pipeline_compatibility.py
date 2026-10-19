"""Search and ablation must respect sparse Hartley kernel constraints."""
from dataclasses import replace

import pytest

from realtime_watermark.proposals.jilp_qim.config import DEFAULT_CONFIG
from scripts.proposal_structural_ablation import structural_variants


def _hartley():
    return replace(
        DEFAULT_CONFIG, hartley_mix=0.6, hartley_support=48,
        hartley_carrier_mode="frequency_preserving",
    ).validated()


def test_hartley_requires_supported_projection():
    config = _hartley()
    assert config.use_integer_lattice_closure
    assert config.projection_mode == "minimum_distortion"
    with pytest.raises(ValueError, match="minimum-distortion integer closure"):
        replace(config, projection_mode="center_qim").validated()
    with pytest.raises(ValueError, match="minimum-distortion integer closure"):
        replace(config, use_integer_lattice_closure=False).validated()


def test_hartley_ablation_has_valid_without_carrier_control():
    variants = structural_variants(_hartley(), include_security=True)
    labels = {label for label, _, _ in variants}
    assert "center_qim_instead_of_minimum_distortion" not in labels
    assert "without_hartley_convolution" in labels
    for label, parameters, component in variants:
        assert component is None or isinstance(component, str)
        from realtime_watermark.proposals.jilp_qim.config import JILPQIMConfig
        config = JILPQIMConfig.from_mapping(parameters)
        if label == "without_hartley_convolution":
            assert config.hartley_support == 0
            assert config.hartley_mix == 0.0


def test_original_dense_projection_ablation_is_preserved():
    variants = structural_variants(DEFAULT_CONFIG)
    labels = {label for label, _, _ in variants}
    assert "center_qim_instead_of_minimum_distortion" in labels
    assert "without_hartley_convolution" not in labels
