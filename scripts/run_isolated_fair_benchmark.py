#!/usr/bin/env python3
from __future__ import annotations

"""Run each image method in a fresh process and merge fair reports."""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fresh-process fair timing for each requested method.")
    p.add_argument("--methods", nargs="+", default=["dct_dm_qim_chen2001", "dct_stdm_qim_chen2001", "dct_iss_malvar2003", "ca_qim_a2_mao2024", "dct_spread_spectrum_cox1997", "proposal"])
    p.add_argument("--host", default=None)
    p.add_argument("--watermark", default=None)
    p.add_argument("--mode", choices=["512", "full"], default="512")
    p.add_argument("--wm-size", type=int, default=64)
    p.add_argument("--target-psnr", type=float, default=45.0)
    p.add_argument("--timing-repeats", type=int, default=30)
    p.add_argument("--timing-warmups", type=int, default=5)
    p.add_argument("--attack-preset", choices=["smoke", "standard", "extended", "stress", "unified"], default="standard")
    p.add_argument("--out-dir", default="results/fair_isolated")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--proposal-config", default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    out = Path(args.out_dir)
    work = out / "per_method"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(args.threads)

    for method in args.methods:
        method_out = work / method
        cmd = [
            sys.executable, str(ROOT / "main.py"), "--fair", "--mode", args.mode,
            "--methods", method, "--wm-size", str(args.wm_size),
            "--target-psnr", str(args.target_psnr),
            "--timing-repeats", str(args.timing_repeats),
            "--timing-warmups", str(args.timing_warmups),
            "--attack-preset", args.attack_preset, "--out", str(method_out),
        ]
        if args.host:
            cmd += ["--host", args.host]
        if args.watermark:
            cmd += ["--watermark", args.watermark]
        if args.proposal_config:
            cmd += ["--proposal-config", args.proposal_config]
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)

    summary: list[dict[str, str]] = []
    timing: list[dict[str, str]] = []
    attacks: list[dict[str, str]] = []
    protocols: dict[str, object] = {}
    for method in args.methods:
        method_out = work / method
        summary += _read_csv(method_out / "fair_summary_all.csv")
        timing += _read_csv(method_out / "fair_timing_distribution.csv")
        attacks += _read_csv(method_out / "fair_per_attack_all.csv")
        p = method_out / "fair_protocol.json"
        if p.exists():
            protocols[method] = json.loads(p.read_text(encoding="utf-8"))

    _write_csv(out / "fair_summary_all.csv", summary)
    _write_csv(out / "fair_summary_primary.csv", [r for r in summary if str(r.get("primary_rank_eligible", "")).lower() == "true"])
    _write_csv(out / "fair_timing_distribution.csv", timing)
    _write_csv(out / "fair_per_attack_all.csv", attacks)
    (out / "fair_protocol.json").write_text(json.dumps({
        "protocol": "fresh-process fair benchmark",
        "schema_version": "2.0",
        "thread_count": args.threads,
        "timing_warmups": args.timing_warmups,
        "timing_repeats": args.timing_repeats,
        "methods": args.methods,
        "per_method_protocols": protocols,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
