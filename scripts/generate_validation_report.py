#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PACKAGES = ["numpy", "Pillow", "numba", "scikit-image", "scipy", "opencv-python-headless", "pytest"]


def package_versions() -> dict[str, str]:
    result = {}
    for package in PACKAGES:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "source-archive-without-git-metadata"


def junit_counts(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        key: sum(int(float(suite.attrib.get(key, "0"))) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--junit", required=True)
    parser.add_argument("--out-dir", default="validation/current")
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--command", default="python -m pytest -q")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = junit_counts(Path(args.junit))
    passed = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    report = {
        "schema_version": "2.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit": git_commit(project),
        "command": args.command,
        "exit_code": args.exit_code,
        "status": "passed" if args.exit_code == 0 and counts["failures"] == 0 and counts["errors"] == 0 else "failed",
        "counts": {**counts, "passed": passed},
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": package_versions(),
    }
    (out / "test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "Current validation report", "=========================", "",
        f":Status: {report['status']}", f":Tests: {counts['tests']}",
        f":Passed: {passed}", f":Failures: {counts['failures']}",
        f":Errors: {counts['errors']}", f":Skipped: {counts['skipped']}",
        f":Exit code: {args.exit_code}", f":Commit/source: {report['commit']}",
        f":Command: {args.command}", f":Generated: {report['timestamp_utc']}", "",
        "Dependencies", "------------", "",
    ] + [f"* {name}=={value}" for name, value in report["dependencies"].items()]
    (out / "test_report.rst").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
