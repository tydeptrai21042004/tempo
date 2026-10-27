"""No-network regression tests for genuine GitHub upstream transport retries."""
import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace
import pytest


def load_module():
    src = Path(__file__).resolve().parents[1] / 'scripts' / 'run_official_mmwatermark_tempo.py'
    spec = importlib.util.spec_from_file_location('tempo_github_retry_test', src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_upstream_clone_retries_and_removes_partial_directory(tmp_path, monkeypatch):
    mod = load_module()
    path = tmp_path / 'official'
    calls = []
    monkeypatch.setenv('TEMPO_GIT_CLONE_TIMEOUT_S', '45')
    monkeypatch.setenv('TEMPO_GIT_CLONE_RETRIES', '2')
    monkeypatch.setenv('TEMPO_GIT_CLONE_RETRY_PAUSE_S', '0')
    def fake_run(command, *, check, timeout, env):
        calls.append(command)
        if len(calls) == 1:
            path.mkdir()
            (path / 'incomplete-fetch').write_text('partial clone')
            raise subprocess.TimeoutExpired(command, timeout)
        assert not path.exists(), 'Retry must start with a clean destination'
        (path / '.git').mkdir(parents=True)
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(mod.subprocess,'run',fake_run)
    mod.clone_upstream_with_retry(mod.OFFICIAL_URL, path)
    assert len(calls) == 2
    assert (path / '.git').is_dir()
    assert 'http.version=HTTP/1.1' in calls[0]
    assert not (path / 'incomplete-fetch').exists()


def test_upstream_clone_reports_terminal_failure_without_leaving_partial(tmp_path, monkeypatch):
    mod = load_module()
    path = tmp_path / 'official'
    monkeypatch.setenv('TEMPO_GIT_CLONE_TIMEOUT_S', '45')
    monkeypatch.setenv('TEMPO_GIT_CLONE_RETRIES', '2')
    monkeypatch.setenv('TEMPO_GIT_CLONE_RETRY_PAUSE_S', '0')
    calls = []
    def fake_run(command, *, check, timeout, env):
        calls.append(command)
        path.mkdir()
        raise subprocess.TimeoutExpired(command, timeout)
    monkeypatch.setattr(mod.subprocess,'run',fake_run)
    with pytest.raises(RuntimeError,match='after 2 attempts'):
        mod.clone_upstream_with_retry(mod.COMPANION_URL, path)
    assert len(calls) == 2
    assert not path.exists()


def test_upstream_clone_does_not_delete_existing_checkout(tmp_path):
    mod = load_module()
    path = tmp_path / 'upstream'
    path.mkdir()
    (path / 'existing.txt').write_text('preserve')
    with pytest.raises(RuntimeError,match='Refusing to replace'):
        mod.clone_upstream_with_retry(mod.OFFICIAL_URL, path)
    assert (path / 'existing.txt').read_text() == 'preserve'
