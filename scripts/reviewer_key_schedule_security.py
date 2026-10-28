#!/usr/bin/env python3
"""Empirical keyed-block-schedule overlap and sampled key-collision experiment.

This is descriptive for the specified sampled seeds, not proof of cryptographic
key security or a claim of 256-bit effective key entropy.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from reviewer_expansion_common import *
from realtime_watermark.proposals.jilp_qim.method import _schedules_from_key


def main(a):
    init_cpu()
    hosts,marks=dataset(a)
    cfg=config(a.config)
    out=Path(a.out_dir)
    rows=[]
    rng=np.random.default_rng(a.seed)
    key_samples=8 if a.quick else a.samples
    # Use the same known image dimensions as the evaluated host; no claim that
    # changing seed alone changes other secret key material not implemented.
    for hp in hosts:
        original=load_host(hp)
        for wi,wp in enumerate(marks):
            wm=mark(wp)
            marked,key,_=encode(original,wm,cfg,case_seed(a.seed,wi))
            ref={**key.params,'host_shape':tuple(original.shape),'watermark_shape':tuple(wm.shape)}
            base_payload,base_pilot,base_bits,capacity,_,_=_schedules_from_key(ref)
            seen={(tuple(base_payload.tolist()),tuple(base_pilot.tolist()))}
            for j in range(key_samples):
                candidate=int(rng.integers(0,2**31-1))
                while candidate==int(ref['seed']):candidate=int(rng.integers(0,2**31-1))
                changed={**ref,'seed':candidate}
                p,pi,bits,_,_,_=_schedules_from_key(changed)
                signature=(tuple(p.tolist()),tuple(pi.tolist()))
                collision=signature in seen
                seen.add(signature)
                rows.append({'host':hp.name,'watermark':wp.name,'candidate_index':j,
                    'sampled_seed':candidate,'source_seed':int(ref['seed']),
                    'payload_schedule_exact_collision':bool(np.array_equal(p,base_payload)),
                    'pilot_schedule_exact_collision':bool(np.array_equal(pi,base_pilot)),
                    'combined_schedule_duplicate_in_sample':collision,
                    'payload_position_set_overlap_fraction':float(len(set(p)&set(base_payload))/len(base_payload)),
                    'pilot_position_set_overlap_fraction':float(len(set(pi)&set(base_pilot))/max(1,len(base_pilot))),
                    'pilot_sequence_bit_similarity':float(np.mean(bits==base_bits)) if len(bits)==len(base_bits) and len(bits) else None,
                    'host_block_capacity':int(capacity),'payload_bits':int(wm.size),
                    'pilot_count':int(len(base_pilot))})
    write_rows(out/'sampled_key_schedule_overlap.csv',rows)
    write_json(out/'manifest.json',{'status':'complete','seed_samples_per_pair':key_samples,
        'cases':len(rows),'config_sha256':cfg.sha256(),
        'observations_only':True,
        'limitations':['The effective secret entropy is determined by key generation and exposure, not the hash output size.',
            'At full capacity every block is in the payload position set, so set overlap is 1 regardless of secret ordering.',
            'This is not an exhaustive key-space or cryptographic proof.',
            'The random seeds sampled here use a finite public PRNG for reproducibility.'],**info()})

if __name__=='__main__':
    p=base_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument('--samples',type=int,default=256)
    main(p.parse_args())
