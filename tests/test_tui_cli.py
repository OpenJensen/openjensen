"""Noninteractive CLI stays independent of the optional terminal UI."""

import json
import subprocess
import sys

import httpx
from typer.testing import CliRunner
from vla_platform import cli, cli_client

runner = CliRunner()


def mock_api(monkeypatch, handler):
    from vla_platform.tui_client import ApiClient

    monkeypatch.setattr(
        cli_client,
        "client",
        lambda: ApiClient("http://127.0.0.1", transport=httpx.MockTransport(handler)),
    )


def test_help_version_and_no_textual_import():
    assert runner.invoke(cli.app, ["--help"]).exit_code == 0
    assert runner.invoke(cli.app, ["tui", "--help"]).exit_code == 0
    assert runner.invoke(cli.app, ["--version"]).stdout.strip() == "0.1.0"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys;import vla_platform.cli;assert 'textual' not in sys.modules",
        ],
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_json_stdout_and_transport_error_stderr(monkeypatch):
    mock_api(monkeypatch, lambda request: httpx.Response(200, json=[{"id": "p"}]))
    result = runner.invoke(cli.app, ["projects", "list"])
    assert result.exit_code == 0 and json.loads(result.stdout) == [{"id": "p"}]
    assert not result.stderr

    def offline(*a, **kw):
        raise httpx.ConnectError("fixture offline")

    mock_api(monkeypatch, offline)
    result = runner.invoke(cli.app, ["projects", "list"])
    assert result.exit_code == 1 and not result.stdout
    assert "Cannot reach" in result.stderr


def test_bad_json_and_recipe_are_actionable_without_tracebacks(tmp_path, monkeypatch):
    mock_api(monkeypatch, lambda request: httpx.Response(200, text="not json"))
    result = runner.invoke(cli.app, ["jobs", "list", "p"])
    assert result.exit_code == 1 and "invalid JSON" in result.stderr
    for raw in ("{", "[]"):
        path = tmp_path / "recipe.json"
        path.write_text(raw)
        result = runner.invoke(cli.app, ["policy", "submit", "p", str(path)])
        assert result.exit_code == 2 and "Cannot read recipe" in result.output
    result = runner.invoke(cli.app, ["augmentation", "submit", "p", str(tmp_path / "missing")])
    assert result.exit_code == 2 and "Cannot read recipe" in result.output


def test_intake_snapshot_flag_is_exact_and_local_only(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "call", lambda *args: calls.append(args))
    result = runner.invoke(
        cli.app, ["inspect", "p", "--path", "/dataset", "--snapshot-for-training"]
    )
    assert result.exit_code == 0
    assert calls == [
        (
            "POST",
            "/projects/p/intakes",
            {"source": "local", "path": "/dataset", "snapshot_for_training": True},
        )
    ]
    result = runner.invoke(cli.app, ["inspect", "p", "--repo-id", "o/d", "--snapshot-for-training"])
    assert result.exit_code == 2 and len(calls) == 1


def test_tui_rejects_noninteractive_output():
    result = runner.invoke(cli.app, ["tui"])
    assert result.exit_code == 2
    assert "interactive terminal" in result.stderr and not result.stdout


def test_missing_optional_extra_reports_installation_without_traceback():
    code = """
import builtins, sys
from vla_platform import cli
sys.stdin.isatty = lambda: True
sys.stdout.isatty = lambda: True
original = builtins.__import__
def absent(name, *args, **kwargs):
    if name == 'textual':
        raise ModuleNotFoundError('Optional package absent', name='textual')
    return original(name, *args, **kwargs)
builtins.__import__ = absent
cli.app(['tui'])
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 2 and not result.stdout
    assert "--extra tui" in result.stderr and "Traceback" not in result.stderr


def test_recipe_bound_and_fifo_reject_before_http(tmp_path, monkeypatch):
    import os

    calls = []
    monkeypatch.setattr(cli, "call", lambda *args: calls.append(args))
    large = tmp_path / "large.json"
    large.write_bytes(b" " * (1024 * 1024 + 1))
    result = runner.invoke(cli.app, ["policy", "submit", "p", str(large)])
    assert result.exit_code == 2 and "1 MiB" in result.output
    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "pipe"
        os.mkfifo(fifo)
        result = subprocess.run(
            [sys.executable, "-m", "vla_platform.cli", "policy", "submit", "p", str(fifo)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == 2 and "regular JSON" in result.stderr
    assert not calls
