"""Packed CPU mode preserves owned two-task submission; no cloud or model work."""

import copy
import json
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import test_rollout as baseline
import yaml

HERE = Path(__file__).resolve().parents[1]
launcher = baseline.launcher


@pytest.fixture
def case(monkeypatch):
    """Reuse the existing no-cloud rollout fixture and its owned cleanup."""
    value = baseline.RolloutLaunchTests()
    value.setUp()
    monkeypatch.setattr(
        launcher,
        "_source_paths",
        lambda: {
            "~/firebird-quant-src": value.worker.parent / "firebird_quant/src",
            "~/firebird-act-src": value.worker.parent / "act_optimizer/src",
        },
    )
    try:
        yield value
    finally:
        value.doCleanups()


def cpu_documents(case):
    documents = copy.deepcopy(case.documents)
    template = list(yaml.safe_load_all((HERE / "rollout.packed-cpu.example.yaml").read_text()))
    policy = documents[2]
    policy.update(setup=template[2]["setup"], run=template[2]["run"])
    policy["resources"] = copy.deepcopy(template[2]["resources"])
    policy["envs"].update(
        POLICY_RUNTIME="packed-act-cpu",
        POLICY_DEVICE="cpu",
        POLICY_SETUP_TIMEOUT="2400",
        POLICY_JOB_TIMEOUT="4500",
    )
    remote = launcher._ROOT / "remote"
    remote.mkdir()
    policy["file_mounts"]["~/sim-control"] = str(remote)
    for mount, source in launcher._source_paths().items():
        source.mkdir(parents=True)
        policy["file_mounts"][mount] = str(source)
    return documents


def packed_info(case):
    # Inspect the established float fixture before replacing only its storage
    # markers. The packed validator itself is covered by test_packed_checkpoint.
    info = asdict(launcher._inspect(case.checkpoint))
    info["model_format"] = "firebird_quant"
    (case.checkpoint / "model.safetensors").rename(case.checkpoint / "model.fbq")
    (case.checkpoint / "encoding.json").write_text("{}")
    return SimpleNamespace(**info)


@pytest.mark.parametrize("mode", [launcher._Mode.READINESS, launcher._Mode.EXPERIMENTAL])
def test_cpu_profile_requires_explicit_mode_and_preserves_owned_resources(case, mode):
    documents = cpu_documents(case)
    tasks = launcher._tasks(documents, mode, "packed-act-cpu")
    assert set(tasks) == {"isaac", "vla"}
    assert tasks["isaac"]["resources"]["accelerators"] == "L4:1"
    assert tasks["vla"]["resources"]["instance_type"] == "n2-standard-8"
    assert tasks["vla"]["resources"]["use_spot"] is False
    assert "accelerators" not in tasks["vla"]["resources"]
    assert not case.dispatch.called
    with pytest.raises(ValueError):
        launcher._tasks(documents, mode)  # No implicit switch from default CUDA.
    with pytest.raises(ValueError, match="explicit readiness"):
        launcher._tasks(documents, launcher._Mode.ROLLOUT, "packed-act-cpu")


@pytest.mark.parametrize("accelerator", [None, "", {}, "H100:1", "L4:1"])
def test_even_null_or_empty_accelerator_claim_is_refused(case, accelerator):
    documents = cpu_documents(case)
    documents[2]["resources"]["accelerators"] = accelerator
    with pytest.raises(ValueError, match="omit accelerators"):
        launcher._tasks(documents, launcher._Mode.EXPERIMENTAL, "packed-act-cpu")
    assert not case.dispatch.called


@pytest.mark.parametrize(
    "change",
    [
        "machine",
        "spot",
        "region",
        "nodes",
        "public-port",
        "ordered",
        "extra-mount",
        "wrong-source",
        "setup",
        "run",
        "device",
        "runtime",
        "deadline",
        "pythonpath",
        "model-format",
    ],
)
def test_cpu_template_cannot_override_resources_sources_or_execution(case, change):
    documents = cpu_documents(case)
    task = documents[2]
    if change == "machine":
        task["resources"]["instance_type"] = "a3-highgpu-1g"
    elif change == "spot":
        task["resources"]["use_spot"] = True
    elif change == "region":
        task["resources"]["infra"] = "gcp/us-east4"
    elif change == "nodes":
        task["num_nodes"] = True
    elif change == "public-port":
        task["resources"]["ports"] = [8080]
    elif change == "ordered":
        task["resources"]["ordered"] = [{"instance_type": "n2-standard-8"}]
    elif change == "extra-mount":
        task["file_mounts"]["~/secret"] = "/operator/config"
    elif change == "wrong-source":
        task["file_mounts"]["~/firebird-quant-src"] = str(case.worker)
    elif change in {"setup", "run"}:
        task[change] += "\necho altered\n"
    elif change == "device":
        task["envs"]["POLICY_DEVICE"] = "cuda"
    elif change == "runtime":
        task["envs"]["POLICY_RUNTIME"] = "lerobot-cuda"
    elif change == "deadline":
        task["envs"]["POLICY_JOB_TIMEOUT"] = "72000"
    elif change == "model-format":
        task["envs"]["POLICY_MODEL_FORMAT"] = "safetensors"
    else:
        task["envs"]["PYTHONPATH"] = "/operator/plugins"
    with pytest.raises(ValueError):
        launcher._tasks(documents, launcher._Mode.EXPERIMENTAL, "packed-act-cpu")
    assert not case.dispatch.called


@pytest.mark.parametrize(
    "field,value",
    [
        ("POLICY_RUNTIME", "packed-act-cpu"),
        ("POLICY_DEVICE", "cpu"),
        ("POLICY_MODEL_FORMAT", "firebird_quant"),
        ("POLICY_MODEL_FORMAT", ["safetensors"]),
    ],
)
def test_cuda_template_cannot_disagree_with_selected_runtime(case, field, value):
    documents = copy.deepcopy(case.documents)
    documents[2]["envs"][field] = value
    with pytest.raises(ValueError, match="requested CUDA lane"):
        launcher._tasks(documents, launcher._Mode.EXPERIMENTAL)
    assert not case.dispatch.called


@pytest.mark.parametrize("kind", ["direct", "ancestor"])
def test_yaml_checkpoint_links_reach_strict_inspector_without_resolution(case, kind):
    documents = cpu_documents(case)
    link = case.worker.parent / "checkpoint-link"
    if kind == "direct":
        link.symlink_to(case.checkpoint, target_is_directory=True)
        selected = link
    else:
        link.symlink_to(case.checkpoint.parent, target_is_directory=True)
        selected = link / case.checkpoint.name
    documents[2]["file_mounts"]["~/vla-checkpoint"] = str(selected)
    inspected = []

    def strict_inspector(path):
        inspected.append(path)
        # The real packed inspector rejects lexical links before opening files.
        # This boundary test ensures the launcher does not erase that evidence.
        if path != selected.absolute():
            raise AssertionError("Launcher erased the checkpoint's lexical path")
        raise ValueError("Strict packed inspector refused a link")

    with patch("sim_worker.rollout.checkpoint.inspect_checkpoint", side_effect=strict_inspector):
        with pytest.raises(ValueError, match="refused a link"):
            launcher._checkpoint(documents[2], case.manifest, "packed-act-cpu")
        case.task.write_text(yaml.safe_dump_all(documents))
        with pytest.raises(ValueError, match="refused a link"):
            with launcher._select_checkpoint(
                case.task, selected, launcher._Mode.EXPERIMENTAL, None, "packed-act-cpu"
            ):
                pytest.fail("Linked checkpoint reached a generated launch task")
    assert inspected == [selected.absolute(), selected.absolute()]
    assert not case.dispatch.called


@pytest.mark.parametrize("mode", [launcher._Mode.READINESS, launcher._Mode.EXPERIMENTAL])
@pytest.mark.parametrize("mismatch", ["packed-cuda", "float-cpu", "smol-cpu", "unattested-cpu"])
def test_wrong_storage_or_family_fails_before_submission(case, mode, mismatch):
    documents = case.documents if mismatch == "packed-cuda" else cpu_documents(case)
    policy_runtime = "lerobot-cuda" if mismatch == "packed-cuda" else "packed-act-cpu"
    info = packed_info(case) if mismatch != "float-cpu" else launcher._inspect(case.checkpoint)
    if mismatch == "smol-cpu":
        info.policy_type = "smolvla"
    if mismatch == "unattested-cpu":
        del info.model_format
    case.task.write_text(yaml.safe_dump_all(documents))
    args = SimpleNamespace(mode=mode, policy_runtime=policy_runtime)
    with (
        patch.object(launcher, "_inspect", return_value=info),
        patch.object(launcher, "_launch") as launch,
    ):
        with pytest.raises(ValueError, match="format differs|ACT only"):
            launcher._submit(args, case.task)
        launch.assert_not_called()


def test_cpu_prepare_binds_identity_and_disables_both_task_restarts(case):
    documents = cpu_documents(case)
    info = packed_info(case)
    case.task.write_text(yaml.safe_dump_all(documents))
    with patch.object(launcher, "_inspect", return_value=info):
        prepared = launcher._prepare(case.task, launcher._Mode.EXPERIMENTAL, "packed-act-cpu")
    assert prepared[0]["primary_tasks"] == ["isaac"]
    assert prepared[0]["name"].startswith("isaac-act-test-")
    for task in prepared[1:]:
        assert task["resources"]["job_recovery"] == {"max_restarts_on_errors": 0}
        assert task["resources"]["labels"]["rollout-role"] == task["name"]
    env = prepared[2]["envs"]
    assert env["POLICY_RUNTIME"] == "packed-act-cpu"
    assert env["POLICY_DEVICE"] == "cpu"
    assert env["POLICY_MODEL_FORMAT"] == "firebird_quant"
    assert env["ROLLOUT_ID"] == prepared[1]["envs"]["ROLLOUT_ID"]


def test_cpu_checkpoint_selection_binds_exact_model_and_retains_source_mounts(case):
    documents = cpu_documents(case)
    info = packed_info(case)
    case.task.write_text(yaml.safe_dump_all(documents))
    original = case.task.read_bytes()
    with (
        patch.object(launcher, "_inspect", return_value=info),
        launcher._select_checkpoint(
            case.task, case.checkpoint, launcher._Mode.EXPERIMENTAL, 2, "packed-act-cpu"
        ) as selected,
    ):
        prepared = launcher._prepare(selected, launcher._Mode.EXPERIMENTAL, "packed-act-cpu")
        policy = prepared[2]
        assert policy["envs"]["MODEL_ID"] == info.model_id
        assert policy["envs"]["POLICY_ACTION_STEPS"] == "2"
        assert policy["file_mounts"]["~/vla-checkpoint"] == str(case.checkpoint)
        assert set(launcher._source_paths()) <= policy["file_mounts"].keys()
        snapshot = selected
    assert case.task.read_bytes() == original
    assert not snapshot.exists()


def test_cpu_context_is_saved_before_one_launch_and_cannot_be_reused(case):
    documents = cpu_documents(case)
    info = packed_info(case)
    case.task.write_text(yaml.safe_dump_all(documents))
    destination = case.receipt_dir.parent / "cpu-receipts"
    args = SimpleNamespace(
        mode=launcher._Mode.EXPERIMENTAL,
        policy_runtime="packed-act-cpu",
        validate_only=False,
        yes=True,
        detach_run=True,
        receipt_dir=destination,
        expected_model_id=case.model_id,
        expected_results_prefix="gs://results/rollouts",
    )

    def submitted(prepared, path, mode, options, receipt_dir):
        receipt = json.loads((receipt_dir / "launch-context.json").read_text())
        assert receipt["policy_runtime"] == "packed-act-cpu"
        assert receipt["policy_device"] == "cpu"
        assert receipt["model_format"] == "firebird_quant"
        assert receipt["model_id"] == case.model_id
        assert receipt["group_name"] == prepared[0]["name"]
        dispatched = list(yaml.safe_load_all(Path(path).read_text()))
        assert dispatched == prepared
        assert dispatched[2]["envs"]["POLICY_MODEL_FORMAT"] == "firebird_quant"
        return 0

    with (
        patch.object(launcher, "_inspect", return_value=info),
        patch.object(launcher, "_launch", side_effect=submitted) as launch,
    ):
        assert launcher._submit(args, case.task) == 0
        with pytest.raises(FileExistsError):
            launcher._submit(args, case.task)
        assert launch.call_count == 1


def test_default_cuda_keeps_existing_resources_and_command(case):
    expected = copy.deepcopy(case.documents[2])
    prepared = case._prepare(launcher._Mode.EXPERIMENTAL)
    assert prepared[2]["run"] == expected["run"]
    assert prepared[2]["setup"] == expected["setup"]
    assert prepared[2]["resources"]["accelerators"] == "H100:1"
    assert prepared[2]["resources"]["instance_type"] == "a3-highgpu-1g"
    assert prepared[2]["envs"]["POLICY_RUNTIME"] == "lerobot-cuda"
    assert prepared[2]["envs"]["POLICY_DEVICE"] == "cuda"
    assert prepared[2]["envs"]["POLICY_MODEL_FORMAT"] == "safetensors"


def test_cpu_wheel_pins_match_existing_lock_and_commands_are_cpu_only():
    requirements = (HERE / "remote/policy-cpu.requirements.txt").read_text()
    lock = (HERE.parent / "act_optimizer/uv.lock").read_text()
    for line in requirements.splitlines():
        if " @ " in line:
            url, digest = line.split(" @ ")[1].split("#sha256=")
            assert url in lock
            assert "sha256:" + digest in lock
            assert "%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl" in url
    setup = (HERE / "remote/policy_cpu_setup.sh").read_text()
    run = (HERE / "remote/policy_cpu_run.sh").read_text()
    assert "runtime_versions()" in setup
    assert "torch.version.cuda is None" in setup
    assert "--device cpu" in run
    assert "--device cuda" not in run
    assert "torch.set_num_threads(1)" in run
    assert "torch.set_num_interop_threads(1)" in run


@pytest.mark.parametrize("selection", ["directory", "archive", "omitted"])
@pytest.mark.parametrize("runtime", ["lerobot-cuda", "packed-act-cpu"])
def test_cli_keeps_selected_input_separate_from_library_source_paths(
    case, monkeypatch, selection, runtime
):
    selected = case.worker.parent / "operator-selected-input"
    args = ["launch", str(case.task), "--experimental", "--policy-runtime", runtime]
    if selection != "omitted":
        args.extend(
            [
                "--checkpoint-archive" if selection == "archive" else "--checkpoint",
                selected.name,
            ]
        )
    monkeypatch.chdir(case.worker.parent)
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(sys, "path", list(sys.path))
    resolved, chosen, submitted = [], [], []

    @contextmanager
    def resolve(source, *, archive):
        resolved.append((source, archive))
        yield SimpleNamespace(directory=case.checkpoint)

    def choose(task, checkpoint, mode, steps, policy_runtime):
        chosen.append((task, checkpoint, mode, steps, policy_runtime))
        return nullcontext(case.task)

    def submit(arguments, task):
        submitted.append((arguments.policy_runtime, task))
        return 0

    monkeypatch.setitem(
        sys.modules,
        "sim_worker.rollout.checkpoint_package",
        SimpleNamespace(resolve_checkpoint=resolve),
    )
    monkeypatch.setattr(launcher, "_select_checkpoint", choose)
    monkeypatch.setattr(launcher, "_submit", submit)
    assert launcher._main() == 0
    assert resolved == ([] if selection == "omitted" else [(selected, selection == "archive")])
    assert chosen == [
        (
            case.task,
            None if selection == "omitted" else case.checkpoint,
            launcher._Mode.EXPERIMENTAL,
            None,
            runtime,
        )
    ]
    assert submitted == [(runtime, case.task)]
    assert not case.dispatch.called
