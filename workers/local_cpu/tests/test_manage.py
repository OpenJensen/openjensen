from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("local_cpu_manage", ROOT / "manage.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    act = repo / "workers/act_optimizer"
    own = repo / "workers/local_cpu"
    act.mkdir(parents=True)
    own.mkdir()
    for name in ("pyproject.toml", "uv.lock"):
        (act / name).write_text(name)
    for name in ("reader-macos-arm64.lock", "reader-linux-x86_64.lock", "probe.py"):
        (own / name).write_text(name)
    monkeypatch.setattr(m, "REPO", repo)
    monkeypatch.setattr(m, "HERE", own)
    monkeypatch.setattr(m, "platform_key", lambda: "macos-arm64")
    return tmp_path, repo


def test_plan_and_install_without_execute_never_spawn_or_write(fixture, monkeypatch, capsys):
    root, repo = fixture
    monkeypatch.setattr(m.subprocess, "Popen", lambda *a, **k: pytest.fail("process created"))
    monkeypatch.setattr(
        m.subprocess, "check_output", lambda *a, **k: pytest.fail("process created")
    )
    for action in ("plan", "install"):
        assert (
            m.main(
                [
                    action,
                    "--root",
                    str(root / "new"),
                    "--python",
                    "/bin/python3.12",
                    "--uv",
                    "/bin/uv",
                ]
            )
            == 0
        )
        plan = json.loads(capsys.readouterr().out)
        assert plan["network_required"] is True
        assert len(plan["commands"]) == 6
        assert not (root / "new").exists()
        assert "--locked" in plan["commands"][0]
        assert "--require-hashes" in plan["commands"][2]
        assert "+cpu" not in str(plan)  # mac chooses native CPU wheel through ACT lock


@pytest.mark.parametrize(
    "system,machine", [("Windows", "AMD64"), ("Linux", "aarch64"), ("Darwin", "x86_64")]
)
def test_unverified_platform_rejected(system, machine):
    with pytest.raises(ValueError, match="Supported setup"):
        m.platform_key(system, machine)


@pytest.mark.parametrize(
    "system,machine,key", [("Darwin", "arm64", "macos-arm64"), ("Linux", "x86_64", "linux-x86_64")]
)
def test_explicit_supported_platforms(system, machine, key):
    assert m.platform_key(system, machine) == key


def test_unsafe_output_paths_and_symlinks_rejected(fixture):
    root, _ = fixture
    with pytest.raises(ValueError):
        m.absolute("relative")
    with pytest.raises(ValueError):
        m.absolute("/a/../b")
    target = root / "target"
    target.mkdir()
    link = root / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        m.setup_plan(link / "new", Path("/python"), Path("/uv"))


def test_no_overwrite_owner_only_atomic_json(tmp_path):
    out = tmp_path / "result.json"
    m.write_new(out, {"ready": True})
    before = out.read_bytes()
    assert out.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        m.write_new(out, {"ready": False})
    assert out.read_bytes() == before
    assert not list(tmp_path.glob(".local-cpu-*"))


def test_no_live_config_secret_environment_forwarding(tmp_path, monkeypatch):
    for key in [
        "FIREBIRD_RUNTIME_CONFIG",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "OPENROUTER_API_KEY",
        "UV_INDEX_URL",
        "PYTHONPATH",
        "PIP_INDEX_URL",
    ]:
        monkeypatch.setenv(key, "sentinel-private-value")
    env = m.environment(tmp_path, offline=True)
    assert "sentinel-private-value" not in str(env)
    assert env["UV_PYTHON_DOWNLOADS"] == "never"
    assert env["UV_OFFLINE"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == ""


@pytest.mark.parametrize("exists", [True, False])
def test_existing_or_missing_parent_rejected_before_tools(fixture, monkeypatch, exists):
    root, _ = fixture
    dest = root / "new"
    if exists:
        dest.mkdir()
    else:
        dest = root / "missing" / "new"
    monkeypatch.setattr(m.subprocess, "check_output", lambda *a, **k: pytest.fail("tool invoked"))
    with pytest.raises(ValueError, match="must be new"):
        m.install(dest, Path("/python"), Path("/uv"))


@pytest.fixture
def configured(fixture, monkeypatch):
    root, repo = fixture
    installation = root / "installed"
    installation.mkdir()
    m.write_new(installation / "installation.json", {"installation_root": str(installation)})
    monkeypatch.setattr(m, "verify", lambda *a, **k: {"verified": True})
    return root, repo, installation


def test_new_config_preserves_base_bytes_all_fields_and_sources(configured):
    root, repo, installation = configured
    base = root / "old.json"
    original = {
        "runtimes": [{"id": "original", "env": {"PRIVATE": "unchanged"}, "custom": "preserved"}],
        "sources": [{"id": "weights", "path": "/original"}],
    }
    base.write_text(json.dumps(original))
    before = base.read_bytes()
    result = m.config(installation, root / "new.json", base)
    written = json.loads((root / "new.json").read_text())
    assert base.read_bytes() == before
    assert written["runtimes"][0] == original["runtimes"][0]
    assert written["sources"] == original["sources"]
    assert len(written["runtimes"]) == 3
    assert result["activated"] is False and "PRIVATE" not in str(result)
    replay = written["runtimes"][1]
    assert replay["native_replay_only"] is True
    assert replay["native_replay_root"] == str(repo / "workers/isaac_sim")
    assert replay["native_replay_dataset_python"] == str(installation / "dataset-reader/bin/python")


@pytest.mark.parametrize(
    "value",
    [
        {"runtimes": [{"id": "local-act-replay-cpu"}]},
        {"runtimes": [{"id": "x"}, {"id": "x"}]},
        {"runtimes": {}},
        {"runtimes": [1]},
        {"unknown": True},
    ],
)
def test_registry_collisions_bad_shapes_rejected_without_output(configured, value):
    root, repo, installation = configured
    base = root / "base.json"
    base.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        m.config(installation, root / "new.json", base)
    assert not (root / "new.json").exists()


def test_config_never_overwrites_base(configured):
    root, repo, installation = configured
    base = root / "base.json"
    base.write_text("{}")
    before = base.read_bytes()
    with pytest.raises(FileExistsError):
        m.config(installation, base, base)
    assert base.read_bytes() == before


def test_config_without_verified_installation_cannot_be_written(fixture):
    root, repo = fixture
    with pytest.raises(ValueError):
        m.config(root / "missing", root / "new.json", None)
    assert not (root / "new.json").exists()


def test_duplicate_json_rejected(tmp_path):
    p = tmp_path / "base.json"
    p.write_text('{"runtimes":[],"runtimes":[]}')
    with pytest.raises(ValueError, match="Duplicate"):
        m.read_json(p)


def test_actual_timeout_reaps_owned_group_and_restores_signal_handlers(tmp_path):
    import os
    import signal
    import subprocess

    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    pidfile = tmp_path / "pid"
    source = (
        "import os,time,pathlib;pathlib.Path("
        + repr(str(pidfile))
        + ").write_text(str(os.getpid()));time.sleep(30)"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        m.run(
            [sys.executable, "-c", source],
            env=m.environment(tmp_path, offline=True),
            log=tmp_path / "run.log",
            timeout=0.3,
        )
    pid = int(pidfile.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert all(signal.getsignal(s) == h for s, h in previous.items())


def test_failed_install_retains_plan_but_no_success_or_config(fixture, monkeypatch):
    root, repo = fixture
    python = root / "python"
    uv = root / "uv"
    python.touch()
    uv.touch()
    monkeypatch.setattr(
        m.subprocess,
        "check_output",
        lambda cmd, **kw: "uv 0.12.19 (fixture)" if cmd[0] == str(uv) else "3.12",
    )

    def fail(*a, **k):
        raise ValueError("generated installer failure")

    monkeypatch.setattr(m, "run", fail)
    with pytest.raises(ValueError, match="generated"):
        m.install(root / "new", python, uv)
    assert (root / "new/installation-plan.json").is_file()
    assert not (root / "new/installation.json").exists()
    assert not (root / "new/runtimes.json").exists()


def test_successful_mock_install_checks_both_runtimes_before_receipt(fixture, monkeypatch):
    root, repo = fixture
    python = root / "python"
    uv = root / "uv"
    python.touch()
    uv.touch()
    monkeypatch.setattr(
        m.subprocess,
        "check_output",
        lambda cmd, **kw: "uv 0.12.19 (fixture)" if cmd[0] == str(uv) else "3.12",
    )
    commands = []
    monkeypatch.setattr(m, "run", lambda cmd, **kw: commands.append(cmd))
    monkeypatch.setattr(m, "verify", lambda root: {"verified": {"model": True, "reader": True}})
    result = m.install(root / "new", python, uv)
    assert result["verified"] == {"model": True, "reader": True}
    assert len(commands) == 6
    assert (root / "new/installation.json").is_file()
    assert (repo / "workers/act_optimizer/uv.lock").read_text() == "uv.lock"


def test_generated_registry_matches_actual_core_contract(configured):
    from vla_platform.lifecycle.runtime import RuntimeCatalog

    root, repo, installation = configured
    output = root / "registry.json"
    m.config(installation, output, None)
    registry = RuntimeCatalog.load(output)
    assert len(registry.runtimes) == 2
    assert registry.runtimes[0].native_replay_only
    assert registry.runtimes[1].native_distillation_only
    assert all(item.provider == "local" and item.device == "cpu" for item in registry.runtimes)


def test_hashed_locks_cover_exact_closure_without_gpu_packages():
    import re

    for key in ("macos-arm64", "linux-x86_64"):
        raw = (ROOT / f"reader-{key}.lock").read_text()
        rows = [line for line in raw.splitlines() if line and not line[0].isspace()]
        assert len(rows) == 62
        assert all(
            "==" in line or " @ https://download-r2.pytorch.org/whl/cpu/" in line for line in rows
        )
        assert not re.search(r"^(nvidia|cuda|triton)", raw, re.M)
        assert raw.count("--hash=sha256:") >= len(rows)
        assert not any(word in raw for word in ("/private/", "file://", "--extra-index-url"))
        if key.startswith("linux"):
            assert "torch-2.11.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64" in raw
            assert "torchvision-0.26.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64" in raw


def test_registry_growth_between_stat_and_read_is_bounded(tmp_path, monkeypatch):
    target = tmp_path / "growing.json"
    target.write_text("{}")
    original = Path.open
    grown = False

    def opening(path, *args, **kwargs):
        nonlocal grown
        if path == target and not grown:
            grown = True
            with original(target, "w") as stream:
                stream.write(json.dumps({"growing": "x" * (1024 * 1024)}))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opening)
    with pytest.raises(ValueError, match="grew"):
        m.read_json(target)


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "1e309"])
def test_nonfinite_base_registry_rejected(tmp_path, value):
    target = tmp_path / "bad.json"
    target.write_text('{"runtimes": [], "x": ' + value + "}")
    with pytest.raises(ValueError, match="Nonfinite"):
        m.read_json(target)


def test_probe_output_is_read_with_fixed_limit(tmp_path, monkeypatch):
    from types import SimpleNamespace

    def excessive(command, **kwargs):
        kwargs["stdout"].write(b"x" * (64 * 1024 + 1))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(m.subprocess, "run", excessive)
    with pytest.raises(ValueError, match="response too large"):
        m.verify(tmp_path, model_python=Path(sys.executable), reader_python=Path(sys.executable))


def test_non_utf8_base_registry_rejected(tmp_path):
    target = tmp_path / "bad.json"
    target.write_bytes('{"runtimes": []}'.encode("utf-16"))
    with pytest.raises(UnicodeError):
        m.read_json(target)


def test_source_requirement_has_archive_filename_for_uv(fixture):
    from urllib.parse import urlsplit

    root, _ = fixture
    plan = m.setup_plan(root / "new", root / "python", root / "uv")
    requirement = plan["commands"][3][-1]
    name, url = requirement.split(" @ ", 1)
    parsed = urlsplit(url)
    assert name == "lerobot"
    assert parsed.scheme == "https" and parsed.netloc == "github.com"
    # uv rejects the old codeload /tar.gz/<commit> endpoint before fetching it.
    assert parsed.path.endswith("/" + m.UPSTREAM + ".tar.gz")
    assert parsed.fragment == "sha256=" + m.UPSTREAM_SHA
