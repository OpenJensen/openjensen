"""Normal-interpreter sidecar proofs; no freezer/native GUI or ML acceptance."""

import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import pytest

SIDE = pathlib.Path(__file__).resolve().parents[1]
ROOT = SIDE.parents[2]
sys.path.insert(0, str(SIDE))


def module(name):
    spec = importlib.util.spec_from_file_location(name, SIDE / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def resource_tree(path):
    web = path / "web"
    web.mkdir(parents=True)
    (web / "index.html").write_text("<title>OPEN JENSEN fixture</title>")
    files = {
        "index.html": {
            "bytes": (web / "index.html").stat().st_size,
            "sha256": hashlib.sha256((web / "index.html").read_bytes()).hexdigest(),
        }
    }
    manifest = {"schema_version": 1, "build_id": "a" * 40, "app_version": "0.1.0", "files": files}
    (path / "resources.json").write_text(json.dumps(manifest))
    return manifest


def test_source_command_is_unchanged_and_frozen_has_no_generic_python_dispatch(monkeypatch):
    from vla_platform.frozen_commands import intake_command

    monkeypatch.delattr(sys, "frozen", raising=False)
    assert intake_command(pathlib.Path("request.json"), pathlib.Path("result.json")) == [
        sys.executable,
        "-m",
        "vla_platform.datasets.worker",
        "request.json",
        "result.json",
    ]
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert intake_command(pathlib.Path("/job/request.json"), pathlib.Path("/job/result.json")) == [
        sys.executable,
        "intake-worker",
        "/job/request.json",
        "/job/result.json",
    ]


def test_immutable_static_inventory(tmp_path):
    resources = module("sidecar_resources")
    manifest = resource_tree(tmp_path)
    result = resources.load_resources(tmp_path)
    assert result.web == tmp_path / "web"
    assert result.build_id == manifest["build_id"]
    (tmp_path / "web/index.html").write_text("changed")
    with pytest.raises(ValueError):
        resources.load_resources(tmp_path)


@pytest.mark.parametrize(
    "fault",
    [
        "extra",
        "missing",
        "symlink",
        "traversal",
        "boolean-size",
        "duplicate",
        "nonfinite",
        "not-object",
    ],
)
def test_invalid_resource_tree_is_rejected(tmp_path, fault):
    resources = module("sidecar_resources")
    manifest = resource_tree(tmp_path)
    if fault == "extra":
        (tmp_path / "web/extra.js").write_text("extra")
    elif fault == "missing":
        (tmp_path / "web/index.html").unlink()
    elif fault == "symlink":
        (tmp_path / "web/index.html").unlink()
        (tmp_path / "outside").write_text("<title>OPEN JENSEN fixture</title>")
        (tmp_path / "web/index.html").symlink_to(tmp_path / "outside")
    elif fault == "traversal":
        manifest["files"]["../escape"] = manifest["files"]["index.html"]
    elif fault == "boolean-size":
        manifest["files"]["index.html"]["bytes"] = True
    elif fault == "not-object":
        manifest = []
    (tmp_path / "resources.json").write_text(json.dumps(manifest))
    if fault == "duplicate":
        (tmp_path / "resources.json").write_text('{"schema_version":1,"schema_version":1}')
    elif fault == "nonfinite":
        (tmp_path / "resources.json").write_text('{"schema_version":1e309}')
    with pytest.raises(ValueError):
        resources.load_resources(tmp_path)


@pytest.mark.parametrize(
    "value",
    [
        {"schema_version": True, "command": "start", "nonce": "a" * 32},
        {"schema_version": 1, "command": "exec", "nonce": "a" * 32},
        {"schema_version": 1, "command": "start", "nonce": "wrong"},
        {"schema_version": 1, "command": "start", "nonce": "a" * 32, "args": ["-c", "1"]},
    ],
)
def test_fixed_control_protocol_rejects_arbitrary_commands(value):
    entry = module("entrypoint")
    with pytest.raises(ValueError):
        entry.control_record(json.dumps(value).encode(), "start")


def test_no_generic_interpreter_or_module_execution(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "packages/core/src"), PYTHONDONTWRITEBYTECODE="1")
    for args in [
        ["-c", "print('executed')"],
        ["-m", "os"],
        ["serve", "--data-dir", "relative", "--resources", str(tmp_path)],
    ]:
        result = subprocess.run(
            [sys.executable, str(SIDE / "entrypoint.py"), *args],
            env=env,
            capture_output=True,
            timeout=15,
        )
        assert result.returncode != 0
        assert b"executed" not in result.stdout


def test_build_resources_preserve_source_and_refuse_overwrite(tmp_path):
    builder = module("prepare_resources")
    source = tmp_path / "source"
    resource_tree(source)
    before = (source / "web/index.html").read_bytes()
    output = tmp_path / "bundled"
    builder.prepare(source / "web", output, "a" * 40)
    assert module("sidecar_resources").load_resources(output).build_id == "a" * 40
    with pytest.raises(ValueError):
        builder.prepare(source / "web", output, "a" * 40)
    assert (source / "web/index.html").read_bytes() == before
    assert (output / "web/index.html").read_bytes() == before


def test_builder_refuses_symlink_and_nested_output_before_creating_output(tmp_path):
    builder = module("prepare_resources")
    source = tmp_path / "source"
    resource_tree(source)
    with pytest.raises(ValueError):
        builder.prepare(source / "web", source / "web/nested", "a" * 40)
    assert not (source / "web/nested").exists()
    (source / "web/link").symlink_to(source / "web/index.html")
    with pytest.raises(ValueError):
        builder.prepare(source / "web", tmp_path / "out", "a" * 40)
    assert not (tmp_path / "out").exists()


def test_inherited_optional_environment_is_not_activated(monkeypatch):
    entry = module("entrypoint")
    monkeypatch.setattr(os, "environ", dict(os.environ))
    for name in (
        "FIREBIRD_TEACHING_URL",
        "FIREBIRD_TEACHING_CONTROL_TOKEN_FILE",
        "FIREBIRD_DECISION_PYTHON",
        "FIREBIRD_RUNTIME_CONFIG",
        "GEMINI_API_KEY",
    ):
        monkeypatch.setenv(name, "not-a-real-secret-or-runtime")
    monkeypatch.setenv("PATH", "/test/platform/path")
    entry.clear_optional_environment()
    assert not any(name.startswith("FIREBIRD_") for name in os.environ)
    assert "GEMINI_API_KEY" not in os.environ
    assert os.environ["PATH"] == "/test/platform/path"
