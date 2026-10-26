"""Paper-equation reproduction tests for Sudrajat et al. 2024 (DOI 10.52866/ijcsm.2024.05.03.043)."""
from __future__ import annotations
import numpy as np
import pytest
from scripts.run_recent_svd_paper_structure import embed,decode,haar_forward,haar_inverse, METHOD,REF


def test_haar_three_levels_is_invertible():
    rng=np.random.default_rng(2026)
    img=rng.uniform(0,255,(128,128))
    ll,detail=haar_forward(img,levels=3)
    assert ll.shape==(16,16)
    assert np.max(np.abs(haar_inverse(ll,detail)-img))<1e-10


def test_paper_equations_2024_informed_and_both_decoder_rules():
    rng=np.random.default_rng(2026)
    host=rng.integers(20,235,size=(128,128),dtype=np.uint8)
    watermark=(rng.random((16,16))>.5).astype(np.uint8)*255
    marked,key=embed(host,watermark,alpha=.2,levels=3)
    assert marked.shape==host.shape and marked.dtype==np.uint8
    assert key['detector_class']=='informed'
    assert key['paper_equation_reproduction'] and not key['authors_code_reproduced']
    assert key['side_information']=='original_host_singular_values'
    assert key['reference_doi']==REF
    assert '2024_lwt_hd_svd' in METHOD
    a=decode(marked,key,'paper2024_eq22')
    b=decode(marked,key,'paper2023_watermark_vectors')
    assert a.shape==watermark.shape and b.shape==watermark.shape
    assert a.dtype==np.uint8 and b.dtype==np.uint8


def test_paper_2024_rejects_invalid_alpha_and_wrong_capacity():
    host=np.zeros((128,128),dtype=np.uint8)
    wm=np.zeros((16,16),dtype=np.uint8)
    for alpha in (0.,-1.,float('nan')):
        with pytest.raises(ValueError):embed(host,wm,alpha,levels=3)
    with pytest.raises(ValueError,match='matching dimensions'):
        embed(host,np.zeros((64,64),dtype=np.uint8),.2,levels=3)
