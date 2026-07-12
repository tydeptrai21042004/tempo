#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BAD_DIRS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", "external", "dtcwt", "reports", "results", "validation"}
BAD_SUFFIXES = {".pyc", ".pyo", ".nbc", ".nbi", ".tmp", ".log", ".doc", ".docx", ".md", ".rst"}
OBSOLETE = {
    ''.join(parts)
    for parts in (
        ('dwt_dct_', 'benoraira2015'),
        ('dft_dct_', 'hamidi2018'),
        ('dct_qim_', 'phadikar2011'),
        ('camd_qim_', 'mao2024'),
        ('fqschur_', 'qiu2025'),
        ('qwt_qsvd_', 'zhang2022'),
        ('dtcwt_svd_', 'wang2022'),
    )
}

bad: list[str] = []
for path in ROOT.rglob("*"):
    relative = path.relative_to(ROOT)
    if any(part in BAD_DIRS for part in relative.parts):
        bad.append(str(relative))
    elif path.is_file() and path.suffix.lower() in BAD_SUFFIXES:
        bad.append(str(relative))

python_text = "\n".join(
    path.read_text(encoding="utf-8", errors="ignore")
    for path in ROOT.rglob("*.py")
)
for token in sorted(OBSOLETE):
    if token in python_text:
        bad.append(f"obsolete Python identifier: {token}")

if bad:
    print("Package cleanliness check failed:")
    for item in bad[:200]:
        print(f"  {item}")
    raise SystemExit(1)
print("OK: clean six-method source package.")
