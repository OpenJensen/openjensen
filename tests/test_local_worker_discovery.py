"""Fixed local discovery checks readiness without installing packages or running models."""

import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from vla_platform import local_worker_discovery as discovery
from vla_platform.lifecycle.runtime import RuntimeCatalog


def ready_response(**gpu):
    return json.dumps(
        {
            "schema_version": 1,
            "status": "ready",
            "reason": None,
            "gpu": {
                "name": "NVIDIA RTX 3070",
                "memory_mib": 8192,
                "uuid": "GPU-1234-abcd",
                "index": 0,
                **gpu,
            },
        }
    ).encode()


def unavailable_response(reason):
    return json.dumps(
        {"schema_version": 1, "status": "unavailable", "reason": reason, "gpu": None}
    ).encode()


@pytest.fixture
def local(tmp_path, monkeypatch):
    root = tmp_path / "workers/smolvla_qlora"
    python = root / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("installed environment placeholder")
    python.chmod(0o700)
    catalog = RuntimeCatalog()

    class Registry:
        issues = []
        writes = 0

        def register(self, runtime):
            existing = catalog.runtime(runtime.id)
            if existing:
                return existing
            self.writes += 1
            catalog.runtimes.append(runtime)
            return runtime

    lifecycle = SimpleNamespace(catalog=catalog, local_workers=Registry())
    service = discovery.LocalWorkerDiscoveryService(lifecycle)
    monkeypatch.setattr(discovery, "_worker_root", lambda: root)
    monkeypatch.setattr(discovery.platform, "system", lambda: "Linux")
    monkeypatch.setattr(discovery.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(discovery, "_hardware", lambda env: None)
    monkeypatch.setattr(discovery, "_run_bounded", lambda *args, **kwargs: ready_response())
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return service, root


def test_check_is_read_only_and_public_response_has_no_execution_paths(local):
    service, root = local
    before = {p: p.stat().st_mtime_ns for p in root.rglob("*")}
    result = asyncio.run(service.check())
    assert result.status == "ready"
    assert result.candidates[0].status == "ready"
    assert result.candidates[0].training_model_ids == ["smolvla"]
    assert result.candidates[0].gpu_memory_mib == 8192
    assert service.lifecycle.local_workers.writes == 0
    assert before == {p: p.stat().st_mtime_ns for p in root.rglob("*")}
    assert str(root) not in result.model_dump_json()
    assert "GPU-1234-abcd" not in result.model_dump_json()
    assert service.last_result == result


def test_add_rechecks_and_registers_once_without_enabling_compute(local, monkeypatch):
    service, root = local
    calls = []
    monkeypatch.setattr(
        discovery, "_run_bounded", lambda *args, **kwargs: calls.append(args) or ready_response()
    )
    checked = asyncio.run(service.check())
    identifier = checked.candidates[0].id
    runtime = asyncio.run(service.add(identifier))
    repeated = asyncio.run(service.add(identifier))
    assert len(calls) == 3
    assert runtime == repeated
    assert service.lifecycle.local_workers.writes == 1
    assert runtime.training_only and runtime.device == "cuda"
    assert runtime.training_root == str(root)
    assert runtime.env == {"CUDA_VISIBLE_DEVICES": "GPU-1234-abcd"}
    assert service.last_result.candidates[0].status == "registered"
    assert service.last_result.candidates[0].runtime_id == runtime.id


@pytest.mark.parametrize("reason", ["missing_dependencies", "cuda_unavailable", "unsupported_gpu"])
def test_installed_but_unready_worker_cannot_be_added(local, monkeypatch, reason):
    service, _ = local
    monkeypatch.setattr(
        discovery, "_run_bounded", lambda *args, **kwargs: unavailable_response(reason)
    )
    result = asyncio.run(service.check())
    assert result.status == "unavailable"
    assert result.candidates[0].reason == discovery.REASONS[reason]
    with pytest.raises(ValueError, match="Check this machine again"):
        asyncio.run(service.add(result.candidates[0].id))
    assert service.lifecycle.local_workers.writes == 0


def test_no_gpu_or_environment_is_not_ready(local):
    service, root = local
    (root / ".venv/bin/python").unlink()
    result = asyncio.run(service.check())
    assert result.status == "unavailable"
    assert result.candidates[0].gpu_name is None
    assert result.candidates[0].status == "setup_required"


def test_hardware_alone_does_not_advertise_training(local, monkeypatch):
    service, root = local
    (root / ".venv/bin/python").unlink()
    monkeypatch.setattr(
        discovery, "_hardware", lambda env: ("0", "GPU-1234", "NVIDIA RTX 3070", 8192)
    )
    result = asyncio.run(service.check())
    assert result.status == "unavailable"
    assert result.candidates[0].gpu_name == "NVIDIA RTX 3070"
    assert result.candidates[0].status == "setup_required"


@pytest.mark.parametrize("raw", [b"invalid/private/secret", b"{}", ready_response(memory_mib=True)])
def test_invalid_probe_response_is_safe(local, monkeypatch, raw):
    service, root = local
    monkeypatch.setattr(discovery, "_run_bounded", lambda *args, **kwargs: raw)
    result = asyncio.run(service.check())
    assert result.status == "error"
    assert "invalid readiness response" in result.candidates[0].reason
    assert "private/secret" not in result.model_dump_json()
    assert str(root) not in result.model_dump_json()


def test_timeout_is_public_safe_failure(local, monkeypatch):
    service, _ = local

    def timeout(*args, **kwargs):
        raise discovery._ProbeError("The local worker check timed out.")

    monkeypatch.setattr(discovery, "_run_bounded", timeout)
    result = asyncio.run(service.check())
    assert result.status == "error"
    assert "timed out" in result.candidates[0].reason


def test_unknown_expired_or_changed_candidate_is_rejected(local, monkeypatch):
    service, _ = local
    with pytest.raises(ValueError, match="Check this machine again"):
        asyncio.run(service.add("../../arbitrary-python"))
    identifier = asyncio.run(service.check()).candidates[0].id
    monkeypatch.setattr(discovery, "CANDIDATE_TTL", -1)
    with pytest.raises(ValueError, match="Check this machine again"):
        asyncio.run(service.add(identifier))
    monkeypatch.setattr(discovery, "CANDIDATE_TTL", 300)
    monkeypatch.setattr(
        discovery, "_run_bounded", lambda *args, **kwargs: unavailable_response("cuda_unavailable")
    )
    with pytest.raises(ValueError, match="changed or is unavailable"):
        asyncio.run(service.add(identifier))
    assert service.lifecycle.local_workers.writes == 0


def test_changed_visible_gpu_subset_requires_new_check(local, monkeypatch):
    service, _ = local
    identifier = asyncio.run(service.check()).candidates[0].id
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(ValueError, match="visible GPU selection changed"):
        asyncio.run(service.add(identifier))
    assert service.lifecycle.local_workers.writes == 0


def test_missing_sources_and_frozen_install_are_graceful(local, monkeypatch):
    service, _ = local
    monkeypatch.setattr(discovery, "_worker_root", lambda: None)
    result = asyncio.run(service.check())
    assert result.status == "unavailable"
    assert "sources" in result.candidates[0].reason


def test_probe_environment_omits_secrets_and_respects_device_subset(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "private-hf-token")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/private/key.json")
    monkeypatch.setenv("PYTHONPATH", "/private/python")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    env = discovery._environment()
    assert not any(
        key in env for key in ("HF_TOKEN", "GOOGLE_APPLICATION_CREDENTIALS", "PYTHONPATH")
    )
    assert env["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert env["HF_HUB_OFFLINE"] == "1"


def test_subprocess_output_and_time_are_bounded():
    started = time.monotonic()
    with pytest.raises(discovery._ProbeError, match="timed out"):
        discovery._run_bounded(
            [sys.executable, "-I", "-c", "import time; time.sleep(10)"],
            timeout=0.1,
            env={},
        )
    assert time.monotonic() - started < 3
    with pytest.raises(discovery._ProbeError, match="invalid readiness response"):
        discovery._run_bounded(
            [sys.executable, "-I", "-c", "import sys; sys.stdout.write('a' * 100000)"],
            timeout=3,
            env={},
        )


def test_real_missing_worker_probe_stays_offline():
    script = Path(discovery.__file__).with_name("local_worker_probe.py")
    result = discovery._parse_probe(
        discovery._run_bounded(
            [sys.executable, "-I", str(script), "/nonexistent"],
            timeout=3,
            env=discovery._environment(),
        )
    )
    assert result["status"] == "unavailable"
    assert result["reason"] == "incompatible_environment"


def test_fixed_probe_uses_worker_interpreter_without_resolving_symlink(local, monkeypatch):
    service, root = local
    python = root / ".venv/bin/python"
    python.unlink()
    python.symlink_to(sys.executable)
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return ready_response()

    monkeypatch.setattr(discovery, "_run_bounded", run)
    asyncio.run(service.check())
    assert calls[0][0] == [
        str(python),
        "-I",
        str(Path(discovery.__file__).with_name("local_worker_probe.py")),
        str(root),
    ]
    assert calls[0][1]["timeout"] == discovery.PROBE_TIMEOUT


@pytest.mark.parametrize("visibility,expected", [(None, "0"), ("2,0", "2"), ("", None)])
def test_nvidia_hardware_query_honors_visible_subset(monkeypatch, visibility, expected):
    monkeypatch.setattr(discovery.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return b"0, GPU-first, First NVIDIA GPU, 8192\n2, GPU-second, Second GPU, 16384\n"

    monkeypatch.setattr(discovery, "_run_bounded", run)
    result = discovery._hardware({} if visibility is None else {"CUDA_VISIBLE_DEVICES": visibility})
    assert (result[0] if result else None) == expected
    assert calls == [
        [
            "/usr/bin/nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    ]


def test_frozen_application_never_discovers_relative_worker_sources(monkeypatch):
    monkeypatch.setattr(discovery.sys, "frozen", True, raising=False)
    assert discovery._worker_root() is None


def test_worker_probe_syntax_remains_compatible_with_python_311():
    import ast

    script = Path(discovery.__file__).with_name("local_worker_probe.py")
    ast.parse(script.read_text(), feature_version=(3, 11))


def test_probe_rejects_duplicate_fields():
    raw = (
        b'{"schema_version":1,"status":"ready","status":"unavailable",'
        b'"reason":"cuda_unavailable","gpu":null}'
    )
    with pytest.raises(discovery._ProbeError):
        discovery._parse_probe(raw)


@pytest.mark.parametrize("visibility", ["", "-1"])
def test_explicitly_hidden_cuda_devices_are_never_registered(local, monkeypatch, visibility):
    service, _ = local
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visibility)
    result = asyncio.run(service.check())
    assert result.status == "unavailable"
    assert result.candidates[0].status == "setup_required"
    assert "hidden" in result.candidates[0].reason


def test_operator_worker_inheriting_visible_gpu_is_reused_without_registry_write(
    local, monkeypatch
):
    from vla_platform.lifecycle.runtime import Runtime

    service, root = local
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    operator = Runtime(
        id="operator-smolvla",
        label="Existing operator GPU",
        device="cuda",
        training_only=True,
        training_root=str(root),
        training_python=str(root / ".venv/bin/python"),
    )
    service.lifecycle.catalog.runtimes.append(operator)
    before = service.lifecycle.catalog.model_dump_json()
    checked = asyncio.run(service.check())
    assert checked.candidates[0].status == "registered"
    assert checked.candidates[0].runtime_id == operator.id
    added = asyncio.run(service.add(checked.candidates[0].id))
    assert added.id == operator.id
    assert added.env == {}
    assert service.lifecycle.local_workers.writes == 0
    assert service.lifecycle.catalog.model_dump_json() == before
    assert service.last_result.candidates[0].runtime_id == operator.id


@pytest.mark.parametrize(
    "environment",
    [
        {"CUDA_VISIBLE_DEVICES": "GPU-other"},
        {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"},
        {"CUDA_VISIBLE_DEVICES": "2"},
    ],
)
def test_operator_gpu_overrides_are_not_guessed(local, monkeypatch, environment):
    from vla_platform.lifecycle.runtime import Runtime

    service, root = local
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    operator = Runtime(
        id="operator-smolvla",
        label="Operator GPU",
        device="cuda",
        training_only=True,
        training_root=str(root),
        training_python=str(root / ".venv/bin/python"),
        env=environment,
    )
    service.lifecycle.catalog.runtimes.append(operator)
    checked = asyncio.run(service.check())
    assert checked.candidates[0].status == "ready"
    assert checked.candidates[0].runtime_id is None


def test_successful_probe_also_cleans_up_its_process_group(monkeypatch):
    import signal

    killed = []
    original = discovery.os.killpg

    def kill_group(pid, sig):
        killed.append((pid, sig))
        return original(pid, sig)

    monkeypatch.setattr(discovery.os, "killpg", kill_group)
    result = discovery._run_bounded([sys.executable, "-I", "-c", "print('{}')"], timeout=3, env={})
    assert result == b"{}\n"
    assert len(killed) == 1
    assert killed[0][1] == signal.SIGKILL


def test_missing_gpu_uuid_cannot_persist_a_numeric_device_target(local, monkeypatch):
    service, _ = local
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setattr(
        discovery, "_run_bounded", lambda *args, **kwargs: ready_response(uuid=None)
    )
    result = asyncio.run(service.check())
    candidate = result.candidates[0]
    assert result.status == "unavailable"
    assert candidate.status == "setup_required"
    assert candidate.runtime_id is None
    assert "stable GPU identity" in candidate.reason
    assert candidate.gpu_name == "NVIDIA RTX 3070"
    with pytest.raises(ValueError, match="Check this machine again"):
        asyncio.run(service.add(candidate.id))
    assert service.lifecycle.local_workers.writes == 0
    assert not service._candidates


def test_gpu_uuid_disappearing_before_add_never_registers_numeric_fallback(local, monkeypatch):
    service, _ = local
    identifier = asyncio.run(service.check()).candidates[0].id
    monkeypatch.setattr(
        discovery, "_run_bounded", lambda *args, **kwargs: ready_response(uuid=None)
    )
    with pytest.raises(ValueError, match="changed or is unavailable"):
        asyncio.run(service.add(identifier))
    assert service.lifecycle.local_workers.writes == 0
    assert service.last_result.candidates[0].status == "setup_required"


@pytest.mark.parametrize(
    "raw,expected",
    [
        (
            "d18657c5-f98c-5caa-ae46-72f0ec25315f",
            "GPU-d18657c5-f98c-5caa-ae46-72f0ec25315f",
        ),
        (
            "GPU-D18657C5-F98C-5CAA-AE46-72F0EC25315F",
            "GPU-d18657c5-f98c-5caa-ae46-72f0ec25315f",
        ),
        (None, None),
        ("0", None),
        ("not-a-gpu-identifier", None),
        ("00000000-0000-0000-0000-000000000000", None),
    ],
)
def test_torch_271_device_uuid_uses_stable_cuda_selector(raw, expected):
    from vla_platform.local_worker_probe import gpu_uuid

    assert gpu_uuid(raw) == expected


def test_probe_entrypoint_keeps_dependency_diagnostics_out_of_json_stdout():
    script = Path(discovery.__file__).with_name("local_worker_probe.py")
    fixture = """
import importlib.metadata
import runpy
import sys
sys.version_info = (3, 11)
sys.prefix = '/isolated-environment'
sys.base_prefix = '/base-python'
def noisy_dependency(name):
    print('dependency diagnostic with /private/path and private-token')
    raise importlib.metadata.PackageNotFoundError(name)
importlib.metadata.distribution = noisy_dependency
sys.argv = [sys.argv[1], '/unused-worker-root']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    output = discovery._run_bounded(
        [sys.executable, "-I", "-c", fixture, str(script)], timeout=3, env={}
    )
    result = discovery._parse_probe(output)
    assert result["status"] == "unavailable"
    assert result["reason"] == "missing_dependencies"
    assert b"private" not in output
    assert output.count(b"\n") == 1


@pytest.mark.parametrize("action", ["check", "add"])
@pytest.mark.parametrize("fails", [False, True])
def test_repeated_cancellation_retains_scan_lock_until_thread_finishes(
    local, monkeypatch, action, fails
):
    import threading

    service, _ = local

    async def exercise():
        initial = await service.check()
        candidate_id = initial.candidates[0].id
        response = service.last_result
        candidates = service._candidates.copy()
        loop = asyncio.get_running_loop()
        started, second_started = asyncio.Event(), asyncio.Event()
        release = threading.Event()
        guard = threading.Lock()
        count = active = peak = 0

        def scan():
            nonlocal count, active, peak
            with guard:
                count += 1
                number = count
                active += 1
                peak = max(peak, active)
            loop.call_soon_threadsafe((started if number == 1 else second_started).set)
            try:
                if number == 1:
                    if not release.wait(3):
                        raise AssertionError("Fixture release did not arrive")
                    if fails:
                        raise RuntimeError("Generated scan failure after cancellation")
                return response, candidates
            finally:
                with guard:
                    active -= 1

        monkeypatch.setattr(service, "_discover", scan)
        original = asyncio.create_task(
            service.check() if action == "check" else service.add(candidate_id)
        )
        successor = None
        try:
            await asyncio.wait_for(started.wait(), 1)
            original.cancel()
            await asyncio.sleep(0)
            original.cancel()
            await asyncio.sleep(0)
            successor = asyncio.create_task(service.check())
            await asyncio.sleep(0.03)
            assert not original.done(), "Cancelled caller released its still-running scan"
            assert service._lock.locked()
            assert not second_started.is_set()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(original, 1)
            await asyncio.wait_for(successor, 1)
            assert peak == 1 and active == 0
            assert service.lifecycle.local_workers.writes == 0
        finally:
            release.set()
            await asyncio.gather(
                original, *([successor] if successor else []), return_exceptions=True
            )

    asyncio.run(exercise())
