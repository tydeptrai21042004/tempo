from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OBSOLETE = tuple(
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
)


def test_no_document_files_or_generated_result_directories():
    forbidden_suffixes = {".md", ".rst", ".doc", ".docx"}
    # Source-controlled experiment protocols are intentional, not generated documents.
    allowed_protocols = {"TEMPO_METHOD_REVISION_README.md", "VIDEO_EVIDENCE_README.md"}
    documents = [
        p for p in ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in forbidden_suffixes
        and p.name not in allowed_protocols
        and ".pytest_cache" not in p.parts and ".venv" not in p.parts
    ]
    generated_dirs = [ROOT / name for name in ("reports", "results", "validation")]
    assert not documents
    assert not [p for p in generated_dirs if p.exists()]


def test_external_and_vendored_dtcwt_are_absent():
    assert not (ROOT / "external").exists()
    assert not (ROOT / "src" / "dtcwt").exists()


def test_obsolete_baseline_ids_are_absent_from_python_source():
    text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in ROOT.rglob("*.py"))
    for token in OBSOLETE:
        assert token not in text


def test_all_python_files_compile():
    files = [str(p) for p in ROOT.rglob("*.py") if ".venv" not in p.parts]
    subprocess.run([sys.executable, "-m", "py_compile", *files], check=True)
