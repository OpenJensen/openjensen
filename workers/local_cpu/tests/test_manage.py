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
    assert len(written["runtimes"]) == 4
    assert result["runtime_count"] == 4
    assert result["activated"] is False and "PRIVATE" not in str(result)
    replay = written["runtimes"][1]
    assert replay["native_replay_only"] is True
    assert replay["native_replay_root"] == str(repo / "workers/isaac_sim")
    assert replay["native_replay_dataset_python"] == str(installation / "dataset-reader/bin/python")


@pytest.mark.parametrize(
    "value",
    [
        {"runtimes": [{"id": "local-act-replay-cpu"}]},
        {"runtimes": [{"id": "local-act-distillation-cpu"}]},
        {"runtimes": [{"id": "local-act-quantization-cpu"}]},
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
    before = base.read_bytes()
    with pytest.raises(ValueError):
        m.config(installation, root / "new.json", base)
    assert not (root / "new.json").exists()
    assert base.read_bytes() == before


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


def test_actual_timeout_reaps_owned_group_and_restores_signal_handlers(tmp_path, monkeypatch):
    import os
    import signal
    import subprocess

    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    # Observe the real process at creation, without assuming its Python startup
    # writes a receipt before the deliberately short timeout expires under load.
    processes = []
    popen = subprocess.Popen

    def capture(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(m.subprocess, "Popen", capture)
    with pytest.raises(subprocess.TimeoutExpired):
        m.run(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            env=m.environment(tmp_path, offline=True),
            log=tmp_path / "run.log",
            timeout=0.3,
        )
    assert len(processes) == 1
    assert processes[0].returncode is not None
    with pytest.raises(ProcessLookupError):
        os.kill(processes[0].pid, 0)
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
    assert len(registry.runtimes) == 3
    assert registry.runtimes[0].native_replay_only
    assert registry.runtimes[1].native_distillation_only
    quantization = registry.runtimes[2]
    assert quantization.id == "local-act-quantization-cpu"
    assert quantization.native_quantization_only
    assert quantization.native_quantization_python == str(installation / "act-model/bin/python")
    assert quantization.native_quantization_root == str(repo / "workers/firebird_quant")
    assert not quantization.native_replay_only and not quantization.native_distillation_only
    raw = json.loads(output.read_text())["runtimes"][2]
    assert not any("dataset_python" in key for key in raw)
    assert not any("native_replay" in key or "native_distillation" in key for key in raw)
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


@pytest.fixture
def shared_uv_cache(tmp_path):
    cache = tmp_path / "previous-attempt/cache"
    cache.mkdir(parents=True)
    (cache / "CACHEDIR.TAG").write_bytes(b"Signature: 8a477f597d28d172789f06886806bc55")
    (cache / "wheels-v6").mkdir()
    (cache / "wheels-v6/generated-wheel-record").write_bytes(b"existing generated cache fixture")
    return cache


def test_explicit_cache_plan_preserves_hashes_and_never_spawns_or_writes(
    fixture, shared_uv_cache, monkeypatch, capsys
):
    root, _ = fixture
    before = {p: p.read_bytes() for p in shared_uv_cache.rglob("*") if p.is_file()}
    monkeypatch.setattr(m.subprocess, "Popen", lambda *a, **kw: pytest.fail("process created"))
    monkeypatch.setattr(
        m.subprocess, "check_output", lambda *a, **kw: pytest.fail("process created")
    )
    default = m.setup_plan(root / "new", Path("/python"), Path("/uv"))
    for action in ("plan", "install"):
        assert (
            m.main(
                [
                    action,
                    "--root",
                    str(root / "new"),
                    "--python",
                    "/python",
                    "--uv",
                    "/uv",
                    "--cache",
                    str(shared_uv_cache),
                ]
            )
            == 0
        )
        plan = json.loads(capsys.readouterr().out)
        assert plan["commands"] == default["commands"]
        assert plan["inputs"] == default["inputs"]
        assert plan["cache"] == {
            "path": str(shared_uv_cache),
            "shared": True,
            "scope": "uv_distribution_artifacts_only",
            "link_mode": "copy",
        }
        assert not (root / "new").exists()
        assert before == {p: p.read_bytes() for p in shared_uv_cache.rglob("*") if p.is_file()}
    assert default["cache"]["path"] == str(root / "new/cache")
    assert default["cache"]["shared"] is False


def test_shared_cache_only_affects_uv_not_runtime_imports_or_model_storage(
    fixture, shared_uv_cache, monkeypatch
):
    root, _ = fixture
    for key in ("UV_CACHE_DIR", "UV_LINK_MODE", "PYTHONPATH", "HF_HOME", "HF_HUB_CACHE"):
        monkeypatch.setenv(key, "untrusted-inherited-value")
    env = m.environment(root / "new", offline=False, cache=shared_uv_cache)
    assert env["UV_CACHE_DIR"] == str(shared_uv_cache)
    assert env["UV_LINK_MODE"] == "copy"
    assert env["UV_PROJECT_ENVIRONMENT"] == str(root / "new/act-model")
    assert env["HF_HOME"] == str(root / "new/cache/huggingface")
    assert "PYTHONPATH" not in env and "untrusted-inherited-value" not in str(env)
    assert "UV_LINK_MODE" not in m.environment(root / "new", offline=False)


@pytest.mark.parametrize("kind", ["same", "inside", "ancestor"])
def test_shared_cache_cannot_overlap_new_environment(fixture, shared_uv_cache, kind):
    root, _ = fixture
    destination = {
        "same": shared_uv_cache,
        "inside": shared_uv_cache.parent,
        "ancestor": shared_uv_cache / "new",
    }[kind]
    with pytest.raises(ValueError, match="disjoint"):
        m.setup_plan(destination, root / "python", root / "uv", cache=shared_uv_cache)


@pytest.mark.parametrize("kind", ["missing", "file", "no-marker", "bad-marker", "large-marker"])
def test_shared_cache_rejects_unrecognized_directories(fixture, kind):
    root, _ = fixture
    cache = root / "cache"
    if kind == "file":
        cache.write_text("not a directory")
    elif kind != "missing":
        cache.mkdir()
        if kind == "bad-marker":
            (cache / "CACHEDIR.TAG").write_text("not a uv cache")
        elif kind == "large-marker":
            (cache / "CACHEDIR.TAG").write_bytes(
                b"Signature: 8a477f597d28d172789f06886806bc55" + b"x" * 1024
            )
    with pytest.raises(ValueError):
        m.setup_plan(root / "new", root / "python", root / "uv", cache=cache)
    assert not (root / "new").exists()


@pytest.mark.parametrize(
    "kind", ["relative", "traversal", "leaf-link", "ancestor-link", "marker-link"]
)
def test_shared_cache_rejects_ambiguous_or_linked_paths(fixture, shared_uv_cache, kind):
    root, _ = fixture
    cache = shared_uv_cache
    if kind == "relative":
        cache = Path("relative/cache")
    elif kind == "traversal":
        cache = root / "previous-attempt/../previous-attempt/cache"
    elif kind == "leaf-link":
        cache = root / "cache-link"
        cache.symlink_to(shared_uv_cache, target_is_directory=True)
    elif kind == "ancestor-link":
        (root / "parent-link").symlink_to(shared_uv_cache.parent, target_is_directory=True)
        cache = root / "parent-link/cache"
    else:
        tag = shared_uv_cache / "CACHEDIR.TAG"
        original = tag.read_bytes()
        tag.unlink()
        (root / "outside-marker").write_bytes(original)
        tag.symlink_to(root / "outside-marker")
    with pytest.raises(ValueError):
        m.setup_plan(root / "new", root / "python", root / "uv", cache=cache)


@pytest.mark.parametrize("level", ["root", "ancestor"])
def test_shared_cache_does_not_admit_installed_environments(fixture, shared_uv_cache, level):
    root, _ = fixture
    target = shared_uv_cache if level == "root" else shared_uv_cache.parent
    (target / "pyvenv.cfg").write_text("generated installed environment fixture")
    with pytest.raises(ValueError, match="installed Python environment"):
        m.setup_plan(root / "new", root / "python", root / "uv", cache=shared_uv_cache)


def test_mock_install_reuses_cache_but_creates_separate_environments(
    fixture, shared_uv_cache, monkeypatch
):
    root, _ = fixture
    python, uv = root / "python", root / "uv"
    python.touch()
    uv.touch()
    original = {p: p.read_bytes() for p in shared_uv_cache.rglob("*") if p.is_file()}
    monkeypatch.setattr(
        m.subprocess,
        "check_output",
        lambda cmd, **kw: "uv 0.12.19 (fixture)" if cmd[0] == str(uv) else "3.12",
    )
    commands = []
    monkeypatch.setattr(m, "run", lambda cmd, **kw: commands.append((cmd, kw["env"])))
    verified = []
    monkeypatch.setattr(
        m,
        "verify",
        lambda target: verified.append(target) or {"verified": {"model": True, "reader": True}},
    )
    result = m.install(root / "new", python, uv, cache=shared_uv_cache)
    assert len(commands) == 6 and verified == [root / "new"]
    assert all(
        env["UV_CACHE_DIR"] == str(shared_uv_cache) and env["UV_LINK_MODE"] == "copy"
        for _, env in commands
    )
    assert commands[0][0][-1] == str(python)
    assert commands[1][0][-1] == str(root / "new/dataset-reader")
    assert "--locked" in commands[0][0] and "--require-hashes" in commands[2][0]
    assert original == {p: p.read_bytes() for p in shared_uv_cache.rglob("*") if p.is_file()}
    assert result["cache"]["path"] == str(shared_uv_cache)
    assert json.loads((root / "new/installation.json").read_text())["cache"] == result["cache"]
    with pytest.raises(ValueError, match="must be new"):
        m.install(root / "new", python, uv, cache=shared_uv_cache)


@pytest.fixture
def partial_installation(fixture):
    root, repo = fixture
    python, uv = root / "python", root / "uv"
    python.touch()
    uv.touch()
    target = root / "partial"
    plan = m.setup_plan(target, python, uv)
    target.mkdir()
    m.write_new(target / "installation-plan.json", plan)
    (target / "install.log").write_bytes(b"preserved original verification failure\n")
    (target / "act-project").mkdir()
    for name in ("pyproject.toml", "uv.lock"):
        (target / "act-project" / name).write_bytes(
            (repo / "workers/act_optimizer" / name).read_bytes()
        )
    return target, python, uv, plan


def mock_completion_tools(monkeypatch, uv, *, failure=None):
    calls = []
    monkeypatch.setattr(
        m.subprocess,
        "check_output",
        lambda cmd, **kw: "uv 0.12.19 (fixture)" if cmd[0] == str(uv) else "3.12",
    )

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if failure == "dependencies":
            raise ValueError("generated dependency failure")

    def verify(root):
        if failure == "probe":
            raise ValueError("generated probe failure")
        calls.append(("verify", root))
        return {"verified": {"model": True, "reader": True}}

    monkeypatch.setattr(m, "run", run)
    monkeypatch.setattr(m, "verify", verify)
    return calls


def test_completion_without_execute_never_spawns_writes_or_accepts(
    partial_installation, monkeypatch, capsys
):
    target, python, uv, _ = partial_installation
    before = {p: p.read_bytes() for p in target.rglob("*") if p.is_file()}
    monkeypatch.setattr(m.subprocess, "Popen", lambda *a, **kw: pytest.fail("process created"))
    monkeypatch.setattr(
        m.subprocess, "check_output", lambda *a, **kw: pytest.fail("process created")
    )
    assert (
        m.main(["complete", "--root", str(target), "--python", str(python), "--uv", str(uv)]) == 0
    )
    planned = json.loads(capsys.readouterr().out)
    assert planned["network_required"] is False
    assert len(planned["commands"]) == 2 and len(planned["probes"]) == 2
    assert before == {p: p.read_bytes() for p in target.rglob("*") if p.is_file()}
    assert not (target / "installation.json").exists()


@pytest.mark.parametrize(
    "change",
    [
        "command",
        "root",
        "repo",
        "platform",
        "hash",
        "schema-bool",
        "extra",
        "copied-lock",
        "copied-project",
        "current-source",
        "python",
        "uv",
    ],
)
def test_completion_rejects_changed_plan_source_or_copied_inputs_before_tools(
    partial_installation, fixture, monkeypatch, change
):
    target, python, uv, plan = partial_installation
    _, repo = fixture
    if change == "command":
        plan["commands"][0] = ["/untrusted/command"]
    elif change == "root":
        plan["installation_root"] = str(target.parent / "other")
    elif change == "repo":
        plan["repository"] = "/other-checkout"
    elif change == "platform":
        plan["platform"] = "linux-x86_64"
    elif change == "hash":
        plan["inputs"]["workers/act_optimizer/uv.lock"] = "0" * 64
    elif change == "schema-bool":
        plan["schema_version"] = True
    elif change == "extra":
        plan["unrecognized"] = "ignored?"
    elif change == "copied-lock":
        (target / "act-project/uv.lock").write_text("changed")
    elif change == "copied-project":
        (target / "act-project/pyproject.toml").write_text("changed")
    elif change == "current-source":
        (repo / "workers/local_cpu/probe.py").write_text("changed")
    elif change == "python":
        python = python.with_name("other-python")
    else:
        uv = uv.with_name("other-uv")
    (target / "installation-plan.json").write_text(json.dumps(plan))
    monkeypatch.setattr(m.subprocess, "check_output", lambda *a, **kw: pytest.fail("tool invoked"))
    with pytest.raises(ValueError):
        m.complete(target, python, uv)
    assert not list(target.glob("completion-*.log"))
    assert not (target / "installation.json").exists()


def test_completion_rejects_existing_receipt_and_linked_copy(partial_installation, monkeypatch):
    target, python, uv, _ = partial_installation
    copied = target / "act-project/uv.lock"
    copied.unlink()
    copied.symlink_to(target.parent / "outside")
    monkeypatch.setattr(m.subprocess, "check_output", lambda *a, **kw: pytest.fail("tool invoked"))
    with pytest.raises(ValueError, match="symlink"):
        m.complete(target, python, uv)
    (target / "installation.json").write_text('{"prior": true}')
    with pytest.raises(FileExistsError):
        m.complete(target, python, uv)
    assert (target / "installation.json").read_text() == '{"prior": true}'


@pytest.mark.parametrize("failure", ["dependencies", "probe"])
def test_completion_failure_preserves_failure_and_publishes_nothing(
    partial_installation, monkeypatch, failure
):
    target, python, uv, _ = partial_installation
    before = (target / "install.log").read_bytes()
    mock_completion_tools(monkeypatch, uv, failure=failure)
    with pytest.raises(ValueError, match="generated"):
        m.complete(target, python, uv)
    assert (target / "install.log").read_bytes() == before
    assert len(list(target.glob("completion-*.log"))) == 1
    assert not (target / "installation.json").exists()


def test_completion_only_checks_offline_before_publishing_new_receipt(
    partial_installation, monkeypatch
):
    target, python, uv, plan = partial_installation
    before = (target / "install.log").read_bytes()
    calls = mock_completion_tools(monkeypatch, uv)
    result = m.complete(target, python, uv)
    assert len(calls) == 3 and calls[-1] == ("verify", target)
    for index, folder in enumerate(("act-model", "dataset-reader")):
        command, options = calls[index]
        assert command == [
            str(uv),
            "--no-config",
            "--no-python-downloads",
            "pip",
            "check",
            "--python",
            str(target / folder / "bin/python"),
        ]
        assert options["env"]["UV_OFFLINE"] == options["env"]["HF_HUB_OFFLINE"] == "1"
        assert options["log"].parent == target
    assert result["inputs"] == plan["inputs"] and result["cache"] == plan["cache"]
    assert result["completion"]["packages_installed_or_repaired"] is False
    assert result["completion"]["mode"] == "verified_existing_partial_installation"
    assert result["completion"]["installation_plan_sha256"] == m.digest(
        target / "installation-plan.json"
    )
    assert (target / "installation.json").stat().st_mode & 0o777 == 0o600
    assert (target / "install.log").read_bytes() == before
    with pytest.raises(FileExistsError):
        m.complete(target, python, uv)


def test_completion_rechecks_preserved_inputs_after_fresh_probes(partial_installation, monkeypatch):
    target, python, uv, plan = partial_installation
    mock_completion_tools(monkeypatch, uv)

    def changing_probe(root):
        plan["platform"] = "changed-during-verification"
        (root / "installation-plan.json").write_text(json.dumps(plan))
        return {"verified": True}

    monkeypatch.setattr(m, "verify", changing_probe)
    with pytest.raises(ValueError, match="Preserved installation plan"):
        m.complete(target, python, uv)
    assert not (target / "installation.json").exists()


def test_completion_requires_original_shared_cache_choice(
    partial_installation, shared_uv_cache, monkeypatch
):
    target, python, uv, _ = partial_installation
    shared_plan = m.setup_plan(target, python, uv, cache=shared_uv_cache)
    (target / "installation-plan.json").write_text(json.dumps(shared_plan))
    monkeypatch.setattr(m.subprocess, "check_output", lambda *a, **kw: pytest.fail("tool invoked"))
    with pytest.raises(ValueError, match="Preserved installation plan"):
        m.complete(target, python, uv)
    calls = mock_completion_tools(monkeypatch, uv)
    result = m.complete(target, python, uv, cache=shared_uv_cache)
    assert result["cache"] == shared_plan["cache"]
    assert all(options["env"]["UV_CACHE_DIR"] == str(shared_uv_cache) for _, options in calls[:2])


def test_failed_directory_sync_rolls_back_only_owned_publication(tmp_path, monkeypatch):
    import os
    import stat

    target = tmp_path / "receipt.json"
    fsync = os.fsync

    def fail_directory(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("generated directory sync failure")
        fsync(fd)

    monkeypatch.setattr(m.os, "fsync", fail_directory)
    with pytest.raises(OSError, match="generated"):
        m.write_new(target, {"ready": True})
    assert not target.exists()
    assert not list(tmp_path.glob(".local-cpu-*"))


@pytest.mark.parametrize("replacement", ["regular", "symlink"])
def test_failed_publication_does_not_remove_observed_replacement(
    tmp_path, monkeypatch, replacement
):
    import os
    import stat

    target = tmp_path / "receipt.json"
    unrelated = tmp_path / "other.json"
    unrelated.write_text("unrelated concurrent receipt")
    fsync = os.fsync

    def replace_then_fail(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            if replacement == "regular":
                os.replace(unrelated, target)
            else:
                target.unlink()
                target.symlink_to(unrelated)
            raise OSError("generated sync failure after replacement")
        fsync(fd)

    monkeypatch.setattr(m.os, "fsync", replace_then_fail)
    with pytest.raises(OSError, match="generated"):
        m.write_new(target, {"ready": True})
    assert target.read_text() == "unrelated concurrent receipt"
    assert target.is_symlink() is (replacement == "symlink")


def test_completion_directory_sync_failure_leaves_no_success_receipt(
    partial_installation, monkeypatch
):
    import os
    import stat

    target, python, uv, _ = partial_installation
    before = (target / "install.log").read_bytes()
    mock_completion_tools(monkeypatch, uv)
    fsync = os.fsync

    def fail_directory(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("generated completion sync failure")
        fsync(fd)

    monkeypatch.setattr(m.os, "fsync", fail_directory)
    with pytest.raises(OSError, match="generated"):
        m.complete(target, python, uv)
    assert not (target / "installation.json").exists()
    assert (target / "install.log").read_bytes() == before
    assert len(list(target.glob("completion-*.log"))) == 1


def test_interruption_after_real_link_does_not_leave_accepted_receipt(tmp_path, monkeypatch):
    target = tmp_path / "receipt.json"
    link = m.os.link

    def linked_then_interrupted(source, destination):
        link(source, destination)
        raise KeyboardInterrupt

    monkeypatch.setattr(m.os, "link", linked_then_interrupted)
    with pytest.raises(KeyboardInterrupt):
        m.write_new(target, {"ready": True})
    assert not target.exists()
    assert not list(tmp_path.glob(".local-cpu-*"))


@pytest.mark.parametrize(
    "stdout,complete",
    [
        (b"", False),
        (b'{"partial":', False),
        (b'{"done":true}', True),
        (b"[]", False),
        (b"x" * 65537, False),
    ],
)
def test_timeout_retains_bounded_stderr_and_output_progress(
    tmp_path, monkeypatch, stdout, complete
):
    def timed_out(command, **kwargs):
        assert kwargs["timeout"] == 60
        kwargs["stdout"].write(stdout)
        kwargs["stderr"].write(b"unretained-prefix" + b"x" * 3000 + b"last import marker")
        raise m.subprocess.TimeoutExpired(command, 60)

    monkeypatch.setattr(m.subprocess, "run", timed_out)
    with pytest.raises(ValueError) as caught:
        m.verify(tmp_path, model_python=Path(sys.executable), reader_python=Path(sys.executable))
    message = str(caught.value)
    expected = "a complete JSON object was captured" if complete else "no complete JSON object"
    assert expected in message and "60-second deadline" in message and "not accepted" in message
    assert "last import marker" in message and "unretained-prefix" not in message
    assert len(message) < 2250


@pytest.mark.parametrize("complete", [True, False])
def test_real_probe_timeout_keeps_output_and_reaps_child(tmp_path, monkeypatch, complete):
    import os
    import subprocess
    import time

    original_run, original_popen = subprocess.run, subprocess.Popen
    children = []
    stdout_value = '{"done":true}' if complete else '{"partial":'
    code = (
        "import sys,time;"
        f"sys.stdout.write({stdout_value!r});sys.stdout.flush();"
        "sys.stderr.write('generated last import marker');sys.stderr.flush();"
        "time.sleep(60)"
    )

    def spawned(command, **kwargs):
        child = original_popen(command, **kwargs)
        children.append(child)
        # Handshake waits for actual output, not a guessed startup delay.
        deadline = time.monotonic() + 5
        while os.fstat(kwargs["stderr"].fileno()).st_size == 0:
            if child.poll() is not None or time.monotonic() > deadline:
                child.kill()
                child.wait(timeout=2)
                pytest.fail("generated timeout child failed its output handshake")
            time.sleep(0.01)
        return child

    def short_observer(command, **kwargs):
        assert kwargs["timeout"] == 60
        return original_run([sys.executable, "-I", "-u", "-c", code], **dict(kwargs, timeout=0.1))

    monkeypatch.setattr(m.subprocess, "Popen", spawned)
    monkeypatch.setattr(m.subprocess, "run", short_observer)
    with pytest.raises(ValueError) as caught:
        m.verify(tmp_path, model_python=Path(sys.executable), reader_python=Path(sys.executable))
    assert "generated last import marker" in str(caught.value)
    expected = "a complete JSON object was captured" if complete else "no complete JSON object"
    assert expected in str(caught.value)
    assert len(children) == 1 and children[0].poll() is not None
    with pytest.raises(ProcessLookupError):
        os.kill(children[0].pid, 0)
