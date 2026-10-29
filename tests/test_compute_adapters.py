"""Portable scheduler task compilation; no provisioning command is executed."""

import json

import pytest
from vla_platform.compute.contracts import bind_runtime, verify_binding
from vla_platform.compute.providers import AdapterRegistry
from vla_platform.lifecycle import sky_runner
from vla_platform.lifecycle.runtime import Runtime


@pytest.fixture
def generic_target():
    return {
        "infrastructure": "ssh",
        "location": "gpu-pool",
        "resources": {
            "cpus": 4,
            "memory_gib": 16,
            "accelerator": "L4",
            "gpu_count": 1,
            "disk_gib": 100,
        },
        "accelerator": "L4",
        "gpu_count": 1,
        "workspace": "default",
        "sky_api_endpoint": "http://127.0.0.1:46580",
        "idle_minutes": 10,
    }


@pytest.mark.parametrize(
    "infrastructure,location,partition,selector",
    [
        ("ssh", "gpu-pool", None, "ssh/gpu-pool"),
        ("slurm", "research-cluster", "gpu", "slurm/research-cluster/gpu"),
        ("gcp", "us-central1", None, "gcp/us-central1"),
        ("runpod", None, None, "runpod"),
        ("nebius", "eu-north1", None, "nebius/eu-north1"),
    ],
)
def test_resource_compilation_is_infrastructure_independent(
    generic_target, infrastructure, location, partition, selector
):
    result = sky_runner.target_resources(
        {
            **generic_target,
            "infrastructure": infrastructure,
            "location": location,
            "partition": partition,
        }
    )
    assert result == {
        "infra": selector,
        "accelerators": "L4:1",
        "cpus": "4+",
        "memory": "16+",
        "disk_size": 100,
        "use_spot": False,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("infrastructure", "unregistered"),
        ("location", "pool; command"),
        ("location", "../pool"),
        ("location", "pool/other"),
        ("location", ""),
        ("partition", "gpu;command"),
        ("account", "user\ncommand"),
        ("qos", "$(command)"),
        ("resources", {}),
    ],
)
def test_bad_destinations_cannot_be_compiled(generic_target, field, value):
    with pytest.raises(ValueError):
        sky_runner.target_resources({**generic_target, field: value})


@pytest.mark.parametrize(
    "name,value",
    [
        ("cpus", True),
        ("cpus", 0),
        ("memory_gib", 0),
        ("memory_gib", "4+"),
        ("disk_gib", 1),
        ("gpu_count", True),
        ("gpu_count", 2),
        ("accelerator", "L4;command"),
    ],
)
def test_invalid_resources_never_become_scheduler_requests(generic_target, name, value):
    changed = {**generic_target, "resources": {**generic_target["resources"], name: value}}
    with pytest.raises(ValueError):
        sky_runner.target_resources(changed)


@pytest.mark.parametrize("infrastructure", ["ssh", "gcp", "runpod", "nebius"])
def test_slurm_only_fields_are_rejected_on_other_infrastructure(generic_target, infrastructure):
    with pytest.raises(ValueError):
        sky_runner.target_resources(
            {**generic_target, "infrastructure": infrastructure, "partition": "gpu"}
        )


@pytest.fixture
def worker_source(tmp_path, monkeypatch):
    source = tmp_path / "worker"
    package = source / "src/jensen_vla"
    package.mkdir(parents=True)
    (package / "application.py").write_text("# compiler fixture only\n")
    (source / "pyproject.toml").write_text("[project]\nname='fixture'\nversion='1'\n")
    (source / "requirements-smolvla-linux.txt").write_text("setuptools==80.10.2\n")
    monkeypatch.setattr(sky_runner, "worker_root", lambda: source)
    return source


def portable_payload():
    return {
        "schema_version": 1,
        "operation": "policy.finetune",
        "job_id": "portable-job",
        "parameters": {
            "timeout_seconds": 7200,
            "training_method": "lora",
            "training": {"steps": 1},
        },
        "runtime": {"env": {"SECRET": "not-transferred"}},
        "dataset": {"source": "huggingface", "repo_id": "fixture/dataset", "revision": "a" * 40},
        "artifact": None,
        "source": None,
    }


@pytest.mark.parametrize("infrastructure", ["ssh", "slurm", "runpod", "nebius", "gcp"])
def test_portable_task_uses_one_worker_wire_protocol(
    worker_source, generic_target, tmp_path, infrastructure
):
    destination = {
        **generic_target,
        "infrastructure": infrastructure,
        "location": "cluster" if infrastructure == "slurm" else generic_target["location"],
    }
    if infrastructure == "slurm":
        destination.update(partition="gpu", account="research", qos="normal")
    payload = portable_payload()
    task_path, state = sky_runner.prepare(payload, tmp_path / infrastructure, destination)
    task = json.loads(task_path.read_text())
    wire = json.loads((task_path.parent / "sky-bundle/request.json").read_text())
    assert task["resources"] == sky_runner.target_resources(destination)
    assert task["run"] == ".venv/bin/python bootstrap.py"
    assert wire["job_id"] == payload["job_id"] and wire["operation"] == payload["operation"]
    assert "not-transferred" not in json.dumps(wire)
    assert state["phase"] == "prepared" and state["target"]["infrastructure"] == infrastructure
    if infrastructure in {"ssh", "slurm"}:
        assert "sudo apt-get" not in task["setup"]
    if infrastructure == "slurm":
        assert "slurm" in task["config"]


def test_accepted_runtime_binding_cannot_be_redirected():
    runtime = Runtime(
        id="cpu-worker",
        label="Configured worker",
        python="python",
        worker_root="worker",
        vendor="vendor",
        build="build",
    )
    binding = bind_runtime(runtime)
    verify_binding(binding, runtime.model_copy(update={"label": "Friendly new name"}))
    with pytest.raises(ValueError, match="changed"):
        verify_binding(binding, runtime.model_copy(update={"python": "different-python"}))
    assert binding.model_dump().get("env") is None
    assert "worker_root" not in binding.model_dump()
    verify_binding(None, runtime)


def test_duplicate_adapter_registration_is_rejected():
    registry = AdapterRegistry()
    registry.register("fixture", object())
    with pytest.raises(ValueError):
        registry.register("fixture", object())
