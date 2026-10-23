#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for item in (SRC, ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from realtime_watermark.fair_benchmark import calibrate_step_on_hosts, make_independent_calibration_host
from realtime_watermark.proposals.jilp_qim.config import DEFAULT_CONFIG, JILPQIMConfig
from scripts.proposal_study_common import load_watermark, run_variants


def structural_variants(config: JILPQIMConfig, include_security: bool = False):
    variants = [("full_primary_method", config.to_dict(), None)]
    if config.joint_embedding == "orthogonal_joint":
        variants.append((
            "without_joint_projection",
            replace(config, joint_embedding="sequential").to_dict(),
            "joint_dct_hartley_minimum_norm_projection",
        ))
    if config.hartley_support > 0:
        # A center-QIM sparse-carrier kernel does not exist. Compare against
        # the dense original carrier instead of reporting an invalid ablation.
        variants.append((
            "without_hartley_convolution",
            replace(
                config, hartley_mix=0.0, hartley_support=0,
                hartley_carrier_mode="sparse", joint_embedding="sequential",
            ).to_dict(),
            "hartley_convolutional_carrier",
        ))
    else:
        variants.append((
            "center_qim_instead_of_minimum_distortion",
            replace(config, projection_mode="center_qim").to_dict(),
            "minimum_distortion_projection",
        ))
    variants.extend([
        (
            "without_orthogonal_pilots",
            replace(config, enable_pilots=False, enable_sync_search=False, pilot_count=0,
                    joint_embedding="sequential").to_dict(),
            "orthogonal_pilots",
        ),
        (
            "without_geometric_sync_search",
            replace(config, enable_sync_search=False).to_dict(),
            "geometric_sync_search",
        ),
        (
            "host_ranked_content_schedule",
            replace(config, payload_schedule_mode="content", candidate_pool_multiplier=1.75).to_dict(),
            "schedule_policy_study",
        ),
        (
            "luminance_only_field",
            replace(config, field_mode="luminance").to_dict(),
            "opponent_color_field",
        ),
    ])
    if include_security:
        variants.append((
            "without_payload_scrambling",
            replace(config, scramble_payload=False).to_dict(),
            "payload_scrambling_security_only",
        ))
    return variants


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Image structural ablation. Temporal voting is intentionally excluded and tested on video."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--out-dir", default="reports/proposal_structural_ablation")
    parser.add_argument("--host", default=None)
    parser.add_argument("--host-dir", default=None)
    parser.add_argument("--max-hosts", type=int, default=None)
    parser.add_argument("--watermark", default=None)
    parser.add_argument("--host-size", type=int, default=768)
    parser.add_argument("--wm-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--seed-stride", type=int, default=101)
    parser.add_argument("--attack-preset", choices=["smoke", "standard", "extended", "stress", "unified"], default="standard")
    parser.add_argument("--target-psnr", type=float, default=45.0)
    parser.add_argument("--calibration-host-count", type=int, default=3)
    parser.add_argument("--calibration-iterations", type=int, default=8)
    parser.add_argument("--include-security", action="store_true")
    parser.add_argument("--no-calibrate", action="store_true", help="Use native variant strengths without forcing a target PSNR.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = JILPQIMConfig.load(args.config) if args.config else DEFAULT_CONFIG
    raw_variants = structural_variants(config, args.include_security)
    watermark = load_watermark(args.watermark, args.wm_size)
    calibration_hosts = []
    variants = []
    calibration_details = []
    if args.no_calibrate:
        for label, params, changed_component in raw_variants:
            variant_config = JILPQIMConfig.from_mapping(params)
            variants.append((label, variant_config.to_dict(), changed_component))
            calibration_details.append({
                "variant": label,
                "changed_component": changed_component or "none",
                "calibrated_step": None,
                "calibration_median_psnr": None,
                "configuration_sha256": variant_config.sha256(),
            })
    else:
        calibration_hosts = [
            make_independent_calibration_host(
                args.host_size, args.host_size, seed=31013 + 991 * index
            )
            for index in range(max(1, int(args.calibration_host_count)))
        ]
        for label, params, changed_component in raw_variants:
            variant_config = JILPQIMConfig.from_mapping(params)
            calibrated_step, observed_psnr = calibrate_step_on_hosts(
                "proposal", calibration_hosts, watermark,
                target_psnr=float(args.target_psnr), seed=int(args.seed),
                iterations=int(args.calibration_iterations),
                method_params=variant_config.to_dict(),
            )
            calibrated_config = variant_config.with_step(calibrated_step)
            variants.append((label, calibrated_config.to_dict(), changed_component))
            calibration_details.append({
                "variant": label,
                "changed_component": changed_component or "none",
                "calibrated_step": calibrated_step,
                "calibration_median_psnr": observed_psnr,
                "configuration_sha256": calibrated_config.sha256(),
            })
    run_variants(
        variants=variants,
        out_dir=args.out_dir,
        prefix="structural_ablation",
        host=args.host,
        host_dir=args.host_dir,
        max_hosts=args.max_hosts,
        watermark=args.watermark,
        host_size=args.host_size,
        wm_size=args.wm_size,
        seed=args.seed,
        seed_stride=args.seed_stride,
        attack_preset=args.attack_preset,
        manifest={
            "study": "image_structure_only_ablation",
            "base_configuration_sha256": config.sha256(),
            "base_configuration": config.to_dict(),
            "matched_distortion_target_psnr": None if args.no_calibrate else float(args.target_psnr),
            "forced_fixed_psnr": not args.no_calibrate,
            "calibration_host_count": len(calibration_hosts),
            "calibration_hosts_are_disjoint_synthetic_controls": True,
            "calibration_details": calibration_details,
            "temporal_vote_excluded": True,
            "integer_closure_ablation_excluded": "previous data did not activate a measurable difference; not claimed separately",
            "center_qim_ablation_status": (
                "not_applicable_to_sparse_hartley_kernel" if config.hartley_support > 0 else "included"
            ),
            "without_hartley_convolution_control": bool(config.hartley_support > 0),
            "schedule_variant_is_a_policy_study": True,
            "variants": [
                {"variant": label, "changed_component": changed, "parameters": params}
                for label, params, changed in variants
            ],
        },
    )


if __name__ == "__main__":
    main()
