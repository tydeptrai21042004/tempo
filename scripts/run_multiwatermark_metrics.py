#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from realtime_watermark.all_metrics import run_all_methods_all_metrics
from realtime_watermark.core import save_csv, save_json
from realtime_watermark.proposals.jilp_qim.config import JILPQIMConfig
from realtime_watermark.registry import COMMON_IMAGE_METHODS


def _safe(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all retained image methods on all selected BMP hosts and both PNG watermarks."
    )
    parser.add_argument("--host-dir", required=True)
    parser.add_argument("--watermark-dir", required=True)
    parser.add_argument("--proposal-config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--wm-size", type=int, default=64)
    parser.add_argument(
        "--attack-preset",
        choices=["smoke", "standard", "extended", "stress", "unified"],
        default="stress",
    )
    parser.add_argument("--host-limit", type=int, default=0)
    args = parser.parse_args()

    host_dir = Path(args.host_dir)
    watermark_dir = Path(args.watermark_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    hosts = sorted(path for path in host_dir.glob("*.bmp") if path.is_file())
    watermarks = sorted(path for path in watermark_dir.glob("*.png") if path.is_file())
    if args.host_limit > 0:
        hosts = hosts[: args.host_limit]
    if not hosts:
        raise RuntimeError("No selected BMP hosts were found.")
    if len(watermarks) != 2:
        raise RuntimeError(f"Expected exactly two PNG watermarks; found {len(watermarks)}.")

    proposal = JILPQIMConfig.load(args.proposal_config)
    all_rows: list[dict[str, Any]] = []
    applicability: list[dict[str, Any]] = []

    for watermark_index, watermark in enumerate(watermarks, start=1):
        watermark_out = out / f"watermark_{watermark_index:02d}_{_safe(watermark.stem)}"
        result = run_all_methods_all_metrics(
            host_paths=hosts,
            watermark_path=watermark,
            out_dir=watermark_out,
            methods=COMMON_IMAGE_METHODS,
            seed=args.seed + 1000003 * (watermark_index - 1),
            wm_size=args.wm_size,
            attack_preset=args.attack_preset,
            mode="512",
            method_params_by_method={"proposal": proposal.to_dict()},
            repeat_by_method={"proposal": 1},
            watermark_label=watermark.name,
        )
        for row in result["rows"]:
            row = dict(row)
            row["watermark_index"] = watermark_index
            row["watermark_name"] = watermark.name
            row["watermark_path"] = str(watermark)
            if row["method_id"] == "proposal":
                actual = str(row.get("configuration_sha256", ""))
                if actual != proposal.sha256():
                    raise RuntimeError(
                        "Proposal configuration mismatch in all-metrics run: "
                        f"expected={proposal.sha256()}, actual={actual}"
                    )
            all_rows.append(row)
        if not applicability:
            applicability = list(result["applicability"])

    save_csv(out / "all_watermarks_all_methods_all_metrics.csv", all_rows)

    summary: list[dict[str, Any]] = []
    attack_summary: list[dict[str, Any]] = []
    for watermark_index, watermark in enumerate(watermarks, start=1):
        for method_id in COMMON_IMAGE_METHODS:
            group = [
                row
                for row in all_rows
                if row["watermark_index"] == watermark_index and row["method_id"] == method_id
            ]
            clean = [row for row in group if row["attack"] == "clean"]
            non_clean = [row for row in group if row["attack"] != "clean"]
            if not clean:
                continue
            import numpy as np

            recovery_rows = [row for row in non_clean if row.get("extraction_path") in {"wide_sync_jilp", "local_sync_jilp"}]
            fast_rows = [row for row in non_clean if row.get("extraction_path") == "raw_fast_jilp"]
            summary.append(
                {
                    "watermark_index": watermark_index,
                    "watermark_name": watermark.name,
                    "method_id": method_id,
                    "detector_type": group[0]["detector_type"],
                    "host_count": len({row["host"] for row in group}),
                    "attack_count": len({row["attack"] for row in non_clean}),
                    "mean_embedding_psnr": float(np.mean([row["embedding_psnr"] for row in clean])),
                    "mean_embedding_ssim": float(np.mean([row["embedding_ssim"] for row in clean])),
                    "mean_clean_bit_accuracy": float(np.mean([row["clean_bit_accuracy"] for row in clean])),
                    "mean_attack_bit_accuracy": float(np.mean([row["bit_accuracy"] for row in non_clean])),
                    "mean_attack_ber": float(np.mean([row["ber"] for row in non_clean])),
                    "mean_attack_nc": float(np.mean([row["nc_bipolar"] for row in non_clean])),
                    "mean_attack_ser_4bit": float(np.mean([row["ser_4bit"] for row in non_clean])),
                    "global_worst_nc": float(np.min([row["nc_bipolar"] for row in non_clean])),
                    "mean_embed_ms": float(np.mean([row["embed_ms"] for row in clean])),
                    "mean_clean_extract_ms": float(np.mean([row["clean_extract_ms"] for row in clean])),
                    "mean_integer_lattice_unit_updates": (float(np.mean([row.get("integer_lattice_unit_updates", 0) for row in clean])) if method_id == "proposal" else ""),
                    "total_integer_lattice_unresolved": (int(sum(int(row.get("integer_lattice_unresolved", 0)) for row in clean)) if method_id == "proposal" else ""),
                    "mean_integer_lattice_corrected_blocks": (float(np.mean([row.get("integer_lattice_corrected_blocks", 0) for row in clean])) if method_id == "proposal" else ""),
                    "mean_integer_lattice_initial_parity_failures": (float(np.mean([row.get("integer_lattice_initial_parity_failures", 0) for row in clean])) if method_id == "proposal" else ""),
                    "mean_integer_lattice_initial_guard_failures": (float(np.mean([row.get("integer_lattice_initial_guard_failures", 0) for row in clean])) if method_id == "proposal" else ""),
                    "mean_integer_lattice_payload_blocks": (float(np.mean([row.get("integer_lattice_payload_blocks", 0) for row in clean])) if method_id == "proposal" else ""),
                    "mean_pilot_lattice_unit_updates": (float(np.mean([float(row["pilot_lattice_unit_updates"]) for row in clean])) if method_id=="proposal" else ""),
                    "total_pilot_lattice_unresolved": (int(sum(int(row["pilot_lattice_unresolved"]) for row in clean)) if method_id=="proposal" else ""),
                    "total_pilot_overlap_unresolved": (int(sum(int(row["pilot_overlap_unresolved"]) for row in clean))
                                                        if all(row.get("pilot_overlap_unresolved", "") != "" for row in clean) else ""),
                    "recovery_invocation_rate": float(len(recovery_rows) / max(len(non_clean), 1)),
                    "fast_path_extract_ms_median": float(np.median([row["extract_ms"] for row in fast_rows])) if fast_rows else float("nan"),
                    "recovery_path_extract_ms_median": float(np.median([row["extract_ms"] for row in recovery_rows])) if recovery_rows else float("nan"),
                    "recovery_path_extract_ms_p95": float(np.percentile([row["extract_ms"] for row in recovery_rows], 95)) if recovery_rows else float("nan"),
                    "proposal_configuration_sha256": proposal.sha256() if method_id == "proposal" else "",
                }
            )
            for attack_name in sorted({row["attack"] for row in group}):
                attack_group = [row for row in group if row["attack"] == attack_name]
                attack_summary.append(
                    {
                        "watermark_index": watermark_index,
                        "watermark_name": watermark.name,
                        "method_id": method_id,
                        "attack": attack_name,
                        "host_count": len(attack_group),
                        "mean_bit_accuracy": float(np.mean([row["bit_accuracy"] for row in attack_group])),
                        "mean_ber": float(np.mean([row["ber"] for row in attack_group])),
                        "mean_nc": float(np.mean([row["nc_bipolar"] for row in attack_group])),
                        "mean_ser_4bit": float(np.mean([row["ser_4bit"] for row in attack_group])),
                        "mean_extract_ms": float(np.mean([row["extract_ms"] for row in attack_group])),
                        "recovery_invocation_rate": float(np.mean([row.get("extraction_path") in {"wide_sync_jilp", "local_sync_jilp"} for row in attack_group])),
                        "mean_pilot_score": float(np.nanmean([float(row.get("pilot_score", float("nan"))) for row in attack_group])) if any(str(row.get("pilot_score", "")) not in {"", "nan"} for row in attack_group) else float("nan"),
                    }
                )

    save_csv(out / "summary_by_method_watermark.csv", summary)
    save_csv(out / "summary_by_attack.csv", attack_summary)
    save_csv(out / "metric_applicability.csv", applicability)
    save_json(
        out / "all_watermarks_protocol.json",
        {
            "host_rule": {
                "extension": ".bmp",
                "excluded_name_substring": "lena",
                "host_count": len(hosts),
                "hosts": [str(path) for path in hosts],
            },
            "watermark_rule": {
                "extension": ".png",
                "required_count": 2,
                "watermark_count": len(watermarks),
                "watermarks": [str(path) for path in watermarks],
            },
            "methods": list(COMMON_IMAGE_METHODS),
            "attack_preset": args.attack_preset,
            "forced_common_psnr": False,
            "common_ecc": False,
            "proposal_configuration_sha256": proposal.sha256(),
            "proposal_configuration_passed_explicitly": True,
        },
    )
    print(f"Image benchmark complete: {out}")
    print(f"Hosts: {len(hosts)}; watermarks: {len(watermarks)}; rows: {len(all_rows)}")


if __name__ == "__main__":
    main()
