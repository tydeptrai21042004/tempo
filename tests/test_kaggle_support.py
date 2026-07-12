from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_kaggle_shell_scripts_parse():
    scripts = [
        ROOT / "kaggle.sh",
        ROOT / "kaggle_full_benchmark_with_ablation.sh",
        ROOT / "scripts" / "kaggle_setup.sh",
        ROOT / "scripts" / "kaggle_pipeline.sh",
    ]
    for script in scripts:
        assert script.is_file()
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_kaggle_dataset_selector_uses_all_bmp_except_lena_and_two_png(tmp_path):
    source = tmp_path / "data"
    host = source / "host"
    watermark = source / "watermark"
    host.mkdir(parents=True)
    watermark.mkdir(parents=True)
    for name in ("a.bmp", "B.BMP", "lena.bmp", "lenna.bmp", "other.jpg"):
        (host / name).write_bytes(b"x")
    for name in ("one.png", "two.PNG"):
        (watermark / name).write_bytes(b"x")

    output = tmp_path / "selected"
    manifest_dir = tmp_path / "reports"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "select_kaggle_dataset.py"),
            "--source-data",
            str(source),
            "--output",
            str(output),
            "--manifest-dir",
            str(manifest_dir),
        ],
        check=True,
    )
    selected_hosts = sorted(path.name for path in (output / "host").iterdir())
    selected_watermarks = sorted(path.name for path in (output / "watermark").iterdir())
    assert len(selected_hosts) == 2
    assert all("lena" not in name.lower() for name in selected_hosts)
    assert all("lenna" not in name.lower() for name in selected_hosts)
    assert len(selected_watermarks) == 2
    manifest = json.loads((manifest_dir / "selected_dataset_manifest.json").read_text())
    assert manifest["host_selection"]["count"] == 2
    assert manifest["watermark_selection"]["count"] == 2


def test_run_script_exposes_kaggle_commands():
    result = subprocess.run(
        ["bash", str(ROOT / "run.sh"), "help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "kaggle-setup" in result.stdout
    assert "kaggle-quick" in result.stdout
    assert "kaggle-full" in result.stdout
