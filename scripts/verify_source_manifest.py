#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the repository SHA-256 manifest.")
    parser.add_argument("--root", default=None)
    parser.add_argument("--manifest", default="SOURCE_MANIFEST.sha256")
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    manifest = root / args.manifest
    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    verified: list[str] = []
    missing: list[str] = []
    mismatched: list[str] = []

    for line_number, raw_line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            expected, relative = line.split(maxsplit=1)
        except ValueError as exc:
            raise RuntimeError(f"Malformed manifest line {line_number}: {raw_line!r}") from exc
        relative = relative.lstrip("*")
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual == expected:
            verified.append(relative)
        else:
            mismatched.append(f"{relative}\texpected={expected}\tactual={actual}")

    if args.report_dir:
        report = Path(args.report_dir)
        report.mkdir(parents=True, exist_ok=True)
        (report / "source_manifest_verified.txt").write_text("\n".join(verified) + ("\n" if verified else ""), encoding="utf-8")
        (report / "source_manifest_missing.txt").write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
        (report / "source_manifest_mismatched.txt").write_text("\n".join(mismatched) + ("\n" if mismatched else ""), encoding="utf-8")

    print(f"Manifest verified: {len(verified)}")
    print(f"Manifest missing: {len(missing)}")
    print(f"Manifest mismatched: {len(mismatched)}")
    for item in missing:
        print(f"MISSING: {item}")
    for item in mismatched:
        print(f"MISMATCH: {item}")

    if mismatched or (missing and not args.allow_missing):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
