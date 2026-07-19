#!/usr/bin/env python3
from __future__ import annotations

"""Aggregate repeated-seed CSV outputs with bootstrap confidence intervals."""

import argparse
import csv
import glob
from pathlib import Path
from typing import Any

import numpy as np


def _read(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                item = dict(row)
                item["source_file"] = str(path)
                rows.append(item)
    return rows


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap_ci(values: np.ndarray, *, samples: int, seed: int) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(values, size=(max(100, int(samples)), values.size), replace=True)
    means = np.mean(draws, axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Combine CSV results from several seeds and report mean, standard deviation, and 95% bootstrap CI."
    )
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--group-by", default="method_id")
    parser.add_argument(
        "--metrics",
        default="mean_attack_nc,evaluation_watermarked_psnr,total_embed_ms_median,extract_fast_ms_median,recovery_invocation_rate",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = [Path(path) for path in sorted(glob.glob(args.input_glob, recursive=True))]
    if not paths:
        raise FileNotFoundError(f"No CSV files matched {args.input_glob!r}")
    rows = _read(paths)
    group_fields = [field.strip() for field in args.group_by.split(",") if field.strip()]
    metric_fields = [field.strip() for field in args.metrics.split(",") if field.strip()]
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(field, "") for field in group_fields), []).append(row)

    output: list[dict[str, Any]] = []
    for group_index, (key, group) in enumerate(sorted(grouped.items())):
        item: dict[str, Any] = {field: value for field, value in zip(group_fields, key)}
        item["observations"] = len(group)
        item["source_files"] = len({row["source_file"] for row in group})
        for metric in metric_fields:
            values: list[float] = []
            for row in group:
                raw = row.get(metric, "")
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    values.append(value)
            array = np.asarray(values, dtype=np.float64)
            low, high = _bootstrap_ci(
                array,
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed + group_index * 1009 + len(metric),
            )
            item[f"{metric}_n"] = int(array.size)
            item[f"{metric}_mean"] = float(np.mean(array)) if array.size else float("nan")
            item[f"{metric}_std"] = float(np.std(array, ddof=1)) if array.size > 1 else 0.0 if array.size == 1 else float("nan")
            item[f"{metric}_ci95_low"] = low
            item[f"{metric}_ci95_high"] = high
        output.append(item)
    _write(Path(args.out), output)
    print(f"Aggregated {len(rows)} rows from {len(paths)} files into {args.out}")


if __name__ == "__main__":
    main()
