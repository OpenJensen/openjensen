"""Installed alpha and release worker layouts keep explicit operator configuration usable."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError
from vla_platform.lifecycle.runtime import (
    Runtime,
    RuntimeCatalog,
    command,
    native_distillation_ready,
    native_quantization_ready,
    native_replay_ready,
)
from vla_platform.local_worker_registry import LocalWorkerRegistry
from vla_platform.worker_paths import (
    canonical_module,
    worker_module,
    worker_package,
    worker_project,
)


@pytest.mark.parametrize(
    ("directory", "module"),
    [
        ("smolvla_qlora/src", "firebird_vla.application"),
        ("act_optimizer/src", "firebird_act.application"),
        ("policy_distillation/src", "firebird_distill.application"),
        ("decision/src", "firebird_decision.__main__"),
        ("teaching", "firebird_teaching.managed"),
        ("jensen_quant/src", "firebird_quant.native_application"),
    ],
)
def test_alpha_import_alias_resolves_new_source_without_loading_model_dependencies(
    directory, module
):
    root = Path(__file__).resolve().parents[1] / "workers" / directory
    code = (
        "import importlib.util, sys\n"
        f"sys.path.insert(0, {str(root)!r})\n"
        f"spec = importlib.util.find_spec({module!r})\n"
        "assert spec is not None\n"
        "assert '/jensen_' in spec.origin\n"
        "assert 'torch' not in sys.modules\n"
    )
    subprocess.run(
        [sys.executable, "-I", "-B", "-c", code], check=True, timeout=5, capture_output=True
    )


def source(root, package, names, layout="src"):
    directory = root / layout / package if layout else root / package
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_text("# Isolated contract fixture; no model imports.\n")
    return directory


def trainer(root, module):
    return Runtime(
        id="managed-local-layout",
        label="Local worker",
        training_only=True,
        training_python=sys.executable,
        training_root=str(root),
        training_module=module,
        device="cuda",
        env={"CUDA_VISIBLE_DEVICES": "0"},
    )


@pytest.mark.parametrize("installed", ["jensen_vla", "firebird_vla"])
@pytest.mark.parametrize("configured", ["jensen_vla.application", "firebird_vla.application"])
def test_training_command_loads_installed_layout_without_mutating_configuration(
    tmp_path, installed, configured
):
    root = tmp_path / "worker"
    package = source(root, installed, ["__init__.py", "application.py"])
    (package / "application.py").write_text(
        f"from pathlib import Path\nimport sys\nPath(sys.argv[2]).write_text({installed!r})\n"
    )
    runtime = trainer(root, configured)
    original = runtime.model_dump_json()
    result = tmp_path / "result"
    argv, cwd, env = command(
        runtime,
        tmp_path / "request",
        result,
        tmp_path,
        "owned-fixture",
        training=True,
        operation="policy.finetune",
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(argv, cwd=cwd, env=env, check=True, timeout=5, capture_output=True)
    assert result.read_text() == installed
    assert argv[2] == installed + ".application"
    assert runtime.model_dump_json() == original


def test_saved_legacy_registry_is_loaded_and_equivalent_registration_does_not_rewrite_it(tmp_path):
    runtime = trainer(tmp_path / "worker", "firebird_vla.application")
    path = tmp_path / "local-workers.json"
    original = json.dumps({"version": 1, "runtimes": [runtime.model_dump()]})
    path.write_text(original)
    registry = LocalWorkerRegistry(tmp_path, RuntimeCatalog())
    assert registry.issues == []
    assert registry.merge().runtimes[0].training_module == "firebird_vla.application"
    equivalent = trainer(tmp_path / "worker", "jensen_vla.application")
    assert registry.register(equivalent).training_module == "firebird_vla.application"
    assert len(registry.merge().runtimes) == 1
    assert path.read_text() == original


@pytest.mark.parametrize("module", ["firebird_vla.application", "jensen_vla.application"])
def test_alias_cannot_bypass_bundled_trainer_model_validation(tmp_path, module):
    value = trainer(tmp_path / "worker", module).model_dump()
    value["training_model_ids"] = ["act"]
    with pytest.raises(ValidationError, match="only SmolVLA"):
        Runtime.model_validate(value)


@pytest.mark.parametrize("prefix", ["jensen", "firebird"])
def test_native_preflight_accepts_complete_installed_alpha_or_release_source(tmp_path, prefix):
    workers = tmp_path / "workers"
    act = workers / "act_optimizer"
    source(
        act,
        prefix + "_act",
        ["__init__.py", "application.py", "bundle.py", "probe.py", "control_schema.py"],
    )
    source(
        workers / "smolvla_qlora",
        prefix + "_vla",
        ["local_dataset.py", "control_contract.py", "control_schema.py"],
    )
    distill = workers / "policy_distillation"
    source(
        distill,
        prefix + "_distill",
        [
            "__init__.py",
            "application.py",
            "prepare.py",
            "runtime.py",
            "contracts.py",
            "provenance.py",
        ],
    )
    quant = workers / (prefix + "_quant")
    source(
        quant,
        prefix + "_quant",
        [
            "__init__.py",
            "native_application.py",
            "native_package.py",
            "native_probe.py",
            "native_consumer.py",
            "codec.py",
            "model.py",
            "state.py",
        ],
    )
    replay = workers / "isaac_sim"
    rollout = replay / "sim_worker/rollout"
    rollout.mkdir(parents=True)
    for name in [
        "native_replay.py",
        "native_replay_prepare.py",
        "native_replay_contracts.py",
        "server.py",
    ]:
        (rollout / name).write_text("# contract fixture\n")
    common = {"id": "local", "label": "Local", "device": "cpu"}
    assert native_distillation_ready(
        Runtime(
            **common,
            native_distillation_only=True,
            native_distillation_python=sys.executable,
            native_distillation_dataset_python=sys.executable,
            native_distillation_root=str(distill),
        )
    )
    assert native_quantization_ready(
        Runtime(
            **common,
            native_quantization_only=True,
            native_quantization_python=sys.executable,
            native_quantization_root=str(quant),
        )
    )
    assert native_replay_ready(
        Runtime(
            **common,
            native_replay_only=True,
            native_replay_python=sys.executable,
            native_replay_dataset_python=sys.executable,
            native_replay_root=str(replay),
        )
    )


def test_partial_or_missing_source_produces_upgrade_message_without_copying_files(tmp_path):
    source(tmp_path, "firebird_vla", ["__init__.py"])
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    with pytest.raises(ValueError, match="Upgrade that worker source tree"):
        worker_module(tmp_path, "firebird_vla.application")
    assert before == sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))


def test_source_resolver_preserves_custom_modules_and_prefers_release_package(tmp_path):
    source(tmp_path, "firebird_teaching", ["managed.py"], layout="")
    assert (
        worker_module(tmp_path, "jensen_teaching.managed", layout="") == "firebird_teaching.managed"
    )
    source(tmp_path, "jensen_teaching", ["managed.py"], layout="")
    assert worker_package(tmp_path, "firebird_teaching", layout="").name == "jensen_teaching"
    assert (
        worker_module(tmp_path, "firebird_teaching.managed", layout="") == "jensen_teaching.managed"
    )
    assert worker_module(tmp_path, "custom.application") == "custom.application"
    assert canonical_module("firebird_vla.lerobot_application") == "jensen_vla.lerobot_application"
    (tmp_path / "firebird_quant").mkdir()
    assert worker_project(tmp_path, "jensen_quant").name == "firebird_quant"
    (tmp_path / "jensen_quant").mkdir()
    assert worker_project(tmp_path, "firebird_quant").name == "jensen_quant"
