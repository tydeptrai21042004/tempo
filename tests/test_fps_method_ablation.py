"""Regression tests for isolated TEMPO method / FPS ablations."""
from dataclasses import replace
import numpy as np
import pytest
from realtime_watermark.core import make_demo_host,make_demo_watermark,bit_accuracy
from realtime_watermark.proposals.jilp_qim.config import JILPQIMConfig,DEFAULT_CONFIG
from realtime_watermark.proposals.jilp_qim.method import embed_proposal,extract_jilp_qim
from realtime_watermark.proposals.jilp_qim import phase_registration as registration
from scripts.proposal_fps_ablation import variants_for,VARIANTS


def test_phase_gate_omitted_from_legacy_serialization():
    cfg=DEFAULT_CONFIG
    assert cfg.phase_fast_threshold is None
    assert 'phase_fast_threshold' not in cfg.to_dict()
    assert JILPQIMConfig.from_mapping(cfg.to_dict()).sha256()==cfg.sha256()
    override=replace(cfg,phase_fast_threshold=.65).validated()
    assert override.to_dict()['phase_fast_threshold']==.65
    assert override.sha256()!=cfg.sha256()


def test_phase_gate_limits():
    for value in (-.01,1.01):
        with pytest.raises(ValueError):replace(DEFAULT_CONFIG,phase_fast_threshold=value).validated()


def test_phase_threshold_overrides_default(monkeypatch):
    img=np.zeros((32,32,3),dtype=np.uint8)
    ids=np.array([0,1,2],dtype=np.int32);bits=np.array([0,1,0],dtype=np.uint8)
    monkeypatch.setattr(registration,'_identity_phase_rgb',lambda *x:.70)
    out,meta=registration.register_sparse_phase(img,ids,bits,15.,.0,
        phase_refinement='selective',certified_pruning=True,phase_fast_threshold=.65)
    assert meta['path']=='raw_fast_jilp'
    def recover_called(*args,**kwargs):
        raise RuntimeError('geometric search reached')
    monkeypatch.setattr(registration,'sparse_phase_score',recover_called)
    with pytest.raises(RuntimeError,match='geometric search reached'):
        registration.register_sparse_phase(img,ids,bits,15.,.0,
            phase_refinement='selective',certified_pruning=True,phase_fast_threshold=.80)


def test_only_center_dense_joint_is_newly_allowed():
    base=replace(DEFAULT_CONFIG,projection_mode='center_qim',pilot_count=16)
    assert replace(base,joint_embedding='orthogonal_joint_center').validated()
    with pytest.raises(ValueError):replace(base,joint_embedding='orthogonal_joint_center',use_integer_lattice_closure=False).validated()
    with pytest.raises(ValueError):replace(base,joint_embedding='orthogonal_joint_center',projection_mode='minimum_distortion').validated()


def test_all_declared_ablations_are_valid_on_saved_baseline():
    from pathlib import Path
    base=JILPQIMConfig.load(Path(__file__).parents[1]/'configs/frozen_selected_20261021.json')
    variants=variants_for(base,list(VARIANTS))
    assert len(variants)>=20
    assert len({label for label,_,_ in variants})==len(variants)


def test_dense_joint_clean_roundtrip_and_overlap():
    host=make_demo_host(128,128)
    watermark=make_demo_watermark(16)
    config=replace(DEFAULT_CONFIG,projection_mode='center_qim',field_mode='luminance',pilot_count=32,step=17.784,
                   pilot_step=15.,joint_embedding='orthogonal_joint_center',phase_refinement='selective',certified_phase_pruning=True)
    marked,key=embed_proposal(host,watermark,seed=2026,method_params=config.to_dict())
    recovered=extract_jilp_qim(marked,key)
    assert bit_accuracy(watermark,recovered)==1.0
    assert key.params['joint_overlap_blocks']==32
    assert key.params['integer_lattice_unresolved']==0
    assert key.params['integer_pilot_unresolved']==0


def test_verified_development_holdout_separation():
    from pathlib import Path
    from scripts.proposal_fps_ablation import get_hosts
    root=Path(__file__).parents[1]
    hosts=root/'data/user_dataset/host'
    manifest=root/'configs/holdout_split_manifest_20261021.json'
    if not hosts.is_dir():
        pytest.skip('Bundled image inputs are optional')
    development=get_hosts(str(hosts),0,'development',str(manifest))
    holdout=get_hosts(str(hosts),0,'holdout',str(manifest))
    assert len(development)==3
    assert len(holdout)==9
    assert not ({x[2] for x in development} & {x[2] for x in holdout})
