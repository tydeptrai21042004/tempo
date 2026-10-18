"""Opt-in, pilot-free reliability-constrained minimum-energy QIM.

RCME is an experimental candidate, not a validated PSNR/NC/FPS improvement.
It preserves the TEMPO payload channel and blind extractor, with a coupled
convex allocation of decoding margins. Decoder sees a fixed QIM step only.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
from typing import Any
import numpy as np

from ..common.types import WatermarkKey, MethodRef
from ..core import arnold_transform, arnold_period
from .jilp_qim import method as original
from .rcme_kernel import _signed_clearances, _allocate_margins, _embed_rcme_blocks

METHOD_ID = 'rcme_qim'
METHOD_REF = MethodRef(
    id=METHOD_ID,
    display_name='RCME-QIM (pilot-free reliability-budgeted minimum-energy)',
    paper='Research prototype extending TEMPO (not validated for publication).',
    url='',
    implementation_note='One-carrier global convex margin allocation, original uint8 RGB embedding direction, direct blind QIM extraction; geometric registration intentionally omitted.',
)

@dataclass(frozen=True)
class RCMEConfig:
    step: float = 17.78404328715303
    eta: float = 0.07
    reference_margin_fraction: float = 0.45
    # No pilot or attack-specific parameters; same decoder as original.
    def __post_init__(self):
        if not np.isfinite(self.step) or self.step <= 0:
            raise ValueError('step must be positive and finite')
        if not np.isfinite(self.eta):
            raise ValueError('eta must be finite')
        if not (0.0 < self.reference_margin_fraction < 0.49):
            raise ValueError('reference_margin_fraction must be in (0,0.49)')

    @classmethod
    def from_mapping(cls, data: dict[str,Any] | None):
        source = dict(data or {})
        if isinstance(source.get('parameters'), dict):
            source = dict(source['parameters'])
        allowed = {f.name for f in fields(cls)}
        unexpected = set(source)-allowed
        if unexpected:
            raise ValueError(f'Unsupported RCME parameters: {sorted(unexpected)}')
        return cls(**source)

def embed(host_rgb, watermark_binary, *, seed=2026, repeat='auto', step=None, method_params=None):
    del repeat
    config = RCMEConfig.from_mapping(method_params)
    if step is not None:
        config = replace(config, step=float(step))
    host = np.asarray(host_rgb, dtype=np.uint8)
    wm = (np.asarray(watermark_binary)>127).astype(np.uint8)
    if host.ndim != 3 or host.shape[2] != 3 or wm.ndim!=2 or wm.shape[0]!=wm.shape[1]:
        raise ValueError('Expected HxWx3 RGB uint8 host and square 2D watermark')
    n=wm.size
    payload_indices,_,_,capacity,_,w_crop,_=original._make_content_adaptive_schedules(
        host, wm.shape, seed=int(seed), pilot_count=0, block_size=original.BLOCK_SIZE,
        eta=config.eta, candidate_pool_multiplier=1.0, candidate_budget=None,
        pilot_schedule='random', payload_schedule_mode='random',
    )
    if n > capacity:
        raise ValueError('Not enough 8x8 blocks for watermark payload')
    period=arnold_period(int(wm.shape[0]))
    arnold_iter=int(7 % period)
    bits=arnold_transform(wm,arnold_iter).ravel().astype(np.uint8)
    order=np.argsort(payload_indices, kind='stable')
    indices=np.ascontiguousarray(payload_indices[order].astype(np.int32))
    bits_exec=np.ascontiguousarray(bits[order].astype(np.uint8))
    a=0.5*(original.PAY10_BA.astype(float)-original.PAY10_BB.astype(float))
    g=np.array([0.299+config.eta,0.587-0.5*config.eta,0.114-0.5*config.eta])
    rounding_epsilon=0.5*np.sum(np.abs(a))*np.sum(np.abs(g))
    max_margin=0.49*config.step
    if not rounding_epsilon+0.08*config.step < config.reference_margin_fraction*config.step < max_margin:
        raise ValueError('reference margin too close to finite-precision limit for this step/field')
    clearances=_signed_clearances(host,indices,bits_exec,original.PAY10_BA,original.PAY10_BB,
                                   float(config.step),int(w_crop),original.BLOCK_SIZE,float(config.eta))
    margins=_allocate_margins(clearances,float(config.step),float(config.reference_margin_fraction),float(rounding_epsilon))
    result=host.copy()
    stats=original._call_with_realtime_thread_budget(
        _embed_rcme_blocks,result,indices,bits_exec,
        original.PAY10_BA,original.PAY10_BB,float(config.step),margins,
        int(w_crop),original.BLOCK_SIZE,float(config.eta),float(rounding_epsilon),
    )
    integer_updates,unresolved,corrected_blocks,initial_parity_failures,initial_guard_failures,final_guard_failures=stats
    if unresolved:
        raise RuntimeError(f'RCME QIM parity unresolved on {unresolved} blocks; output rejected')
    original_params=original._build_key_params(
        tuple(host.shape),tuple(wm.shape),seed=int(seed),step=float(config.step),
        rho_frac=float(config.reference_margin_fraction),pilot_count=0,pilot_step=17.,
        pilot_rho_frac=0.49,eta=float(config.eta),candidate_pool_multiplier=1.0,
        candidate_budget=None,use_second_carrier=False,pilot_schedule='random',
        use_integer_lattice_closure=True,projection_mode='minimum_distortion',
        enable_pilots=False,enable_sync_search=False,payload_schedule_mode='random',
        field_mode='opponent',temporal_vote_mode='hard',scramble_payload=True,
    )
    original_params.update({
        'method_id': METHOD_ID, 'method_family':'reliability-constrained minimum-energy QIM',
        'host_shape':list(host.shape), 'watermark_shape':list(wm.shape),
        'capacity':int(capacity),'sync':'off; geometric attacks unsupported',
        'sync_search_enabled':False,'pilot_count':0,'pilots_enabled':False,
        'reliability_allocation':'globally-coupled convex dual; homoscedastic second-moment surrogate',
        'reliability_budget_mean':float(1/(config.reference_margin_fraction*config.step-rounding_epsilon)**2),
        'achieved_budget_mean':float(np.mean(1/(margins-rounding_epsilon)**2)),
        'finite_precision_rounding_bound':float(rounding_epsilon),
        'allocated_margin_min':float(np.min(margins)),
        'allocated_margin_mean':float(np.mean(margins)),
        'allocated_margin_max':float(np.max(margins)),
        'integer_lattice_unit_updates':int(integer_updates),
        'integer_lattice_unresolved':int(unresolved),
        'integer_lattice_corrected_blocks':int(corrected_blocks),
        'integer_lattice_initial_parity_failures':int(initial_parity_failures),
        'integer_lattice_initial_guard_failures':int(initial_guard_failures),
        'integer_lattice_final_guard_failures':int(final_guard_failures),
        'reliability_assumption':'Homoscedastic carrier noise second moments; NOT a JPEG bound',
        'claim_status':'candidate only; neither real-time nor NC superiority assumed',
        'schedule_storage':'regenerate_from_seed',
        'side_information':'key, shape and fixed step; no original host or margin map',
    })
    cfg_data=asdict(config)
    original_params['rcme_config']=cfg_data
    original_params['rcme_config_sha256']=hashlib.sha256(json.dumps(cfg_data,sort_keys=True).encode()).hexdigest()
    key=WatermarkKey(
        method_id=METHOD_ID,host_shape=tuple(host.shape),watermark_shape=tuple(wm.shape),
        seed=int(seed),repeat=1,step=float(config.step),arnold_iter=arnold_iter,
        arnold_period=int(period),threshold=127,params=original_params,schedule=[],
        fully_blind=True,side_information='No original host and no per-block margin side information',
    )
    return result,key

def extract(image_rgb,key):
    # Original TEMPO direct blind decoder; keys share compatible carrier and schedule.
    return original.extract_jilp_qim(np.asarray(image_rgb,dtype=np.uint8),key,sync=False)
