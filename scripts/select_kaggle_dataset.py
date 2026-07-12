#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def _safe_copy_name(prefix: str, index: int, source: Path) -> str:
    return f"{prefix}_{index:03d}_{source.name}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select all BMP hosts except Lena and exactly two PNG watermarks."
    )
    parser.add_argument("--source-data", default="data/user_dataset")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--exclude-host-substring", action="append", default=None, help="Case-insensitive host-name substring to exclude; repeatable. Defaults to lena and lenna.")
    parser.add_argument("--required-watermarks", type=int, default=2)
    args = parser.parse_args()

    source_data = Path(args.source_data).resolve()
    source_host = source_data / "host"
    source_watermark = source_data / "watermark"
    output = Path(args.output).resolve()
    manifest_dir = Path(args.manifest_dir).resolve()
    selected_host = output / "host"
    selected_watermark = output / "watermark"

    if not source_host.is_dir():
        raise FileNotFoundError(source_host)
    if not source_watermark.is_dir():
        raise FileNotFoundError(source_watermark)

    excluded_values = args.exclude_host_substring or ["lena", "lenna"]
    excluded = tuple(value.casefold() for value in excluded_values)
    hosts = sorted(
        path
        for path in source_host.rglob("*")
        if path.is_file()
        and path.suffix.casefold() == ".bmp"
        and not any(value in path.name.casefold() for value in excluded)
    )
    watermarks = sorted(
        path
        for path in source_watermark.rglob("*")
        if path.is_file() and path.suffix.casefold() == ".png"
    )

    if not hosts:
        raise RuntimeError("No BMP host image remains after applying the exclusion rule.")
    if len(watermarks) != args.required_watermarks:
        listed = "\n".join(str(path) for path in watermarks)
        raise RuntimeError(
            f"Expected exactly {args.required_watermarks} PNG watermarks, "
            f"but found {len(watermarks)}:\n{listed}"
        )

    shutil.rmtree(output, ignore_errors=True)
    selected_host.mkdir(parents=True, exist_ok=True)
    selected_watermark.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    host_rows: list[dict[str, object]] = []
    watermark_rows: list[dict[str, object]] = []

    for index, source in enumerate(hosts, start=1):
        name = _safe_copy_name("host", index, source)
        destination = selected_host / name
        shutil.copy2(source, destination)
        host_rows.append(
            {
                "host_index": index,
                "selected_name": name,
                "selected_path": str(destination),
                "source_path": str(source),
                "source_relative_path": str(source.relative_to(source_host)),
            }
        )

    for index, source in enumerate(watermarks, start=1):
        name = _safe_copy_name("watermark", index, source)
        destination = selected_watermark / name
        shutil.copy2(source, destination)
        watermark_rows.append(
            {
                "watermark_index": index,
                "selected_name": name,
                "selected_path": str(destination),
                "source_path": str(source),
                "source_relative_path": str(source.relative_to(source_watermark)),
            }
        )

    for filename, rows in (
        ("selected_hosts.csv", host_rows),
        ("selected_watermarks.csv", watermark_rows),
    ):
        with (manifest_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    manifest = {
        "host_selection": {
            "source_directory": str(source_host),
            "extension": ".bmp",
            "excluded_filename_substrings": list(excluded_values),
            "case_sensitive": False,
            "count": len(host_rows),
            "files": host_rows,
        },
        "watermark_selection": {
            "source_directory": str(source_watermark),
            "extension": ".png",
            "required_count": args.required_watermarks,
            "count": len(watermark_rows),
            "files": watermark_rows,
        },
    }
    (manifest_dir / "selected_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"Selected hosts: {len(host_rows)}")
    for row in host_rows:
        print(f"HOST: {row['source_relative_path']}")
    print(f"Selected watermarks: {len(watermark_rows)}")
    for row in watermark_rows:
        print(f"WATERMARK: {row['source_relative_path']}")


if __name__ == "__main__":
    main()
