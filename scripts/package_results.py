#!/usr/bin/env python3
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

REPORT_EXTENSIONS = {".csv", ".json", ".txt", ".log", ".rst", ".xml"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package reproducible result reports without large media files.")
    parser.add_argument("--results-dir", default="results/complete")
    parser.add_argument("--output", default="reports/watermark_reports.zip")
    parser.add_argument("--csv-output", default="reports/watermark_csv.zip")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.results_dir).resolve()
    if not root.is_dir():
        raise SystemExit(f"Results directory does not exist: {root}")

    output = Path(args.output).resolve()
    csv_output = Path(args.csv_output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.resolve() in {output, csv_output}:
                continue
            if path.is_file() and path.suffix.lower() in REPORT_EXTENSIONS:
                archive.write(path, path.relative_to(root).as_posix())

    with zipfile.ZipFile(csv_output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*.csv")):
            if path.resolve() in {output, csv_output}:
                continue
            archive.write(path, path.relative_to(root).as_posix())

    print(f"Report ZIP: {output}")
    print(f"CSV ZIP: {csv_output}")


if __name__ == "__main__":
    main()
