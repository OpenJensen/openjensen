"""Unpaid cloud runner tests: all provisioning commands are replaced with doubles."""

import asyncio
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest
from vla_platform.lifecycle import sky_bootstrap, sky_runner


@pytest.fixture
def payload():
    return {
        "schema_version": 1,
        "operation": "policy.finetune",
        "job_id": "training-job",
        "parameters": {"timeout_seconds": 7200, "training_method": "qlora"},
        "runtime": {"env": {"SECRET": "never-upload"}, "worker_root": "/private/source"},
        "dataset": {"source": "huggingface", "repo_id": "fixture/robot", "revision": "a" * 40},
        "artifact": None,
        "source": None,
    }


@pytest.fixture
def target():
    return {
        "project_id": "robotics-demo",
        "region": "us-central1",
        "accelerator": "A100",
        "gpu_count": 1,
        "disk_size_gb": 200,
        "idle_minutes": 10,
        "instance_type": "a2-highgpu-1g",
        "workspace": "firebird-gcp-fixture",
        "sky_api_endpoint": "http://127.0.0.1:46580",
    }


@pytest.fixture(autouse=True)
def isolated_worker(tmp_path, monkeypatch):
    worker = tmp_path / "worker-src"
    (worker / "src" / "firebird_vla").mkdir(parents=True)
    (worker / "src" / "firebird_vla" / "application.py").write_text("# fixture only\n")
    (worker / "pyproject.toml").write_text("[project]\nname = 'fixture'\nversion = '1'\n")
    (worker / "requirements-smolvla-linux.txt").write_text("setuptools==80.10.2\n")
    monkeypatch.setattr(sky_runner, "worker_root", lambda: worker)
    monkeypatch.setattr(sky_runner, "executable", lambda: "/fixture/sky")

    async def storage(sky, target):
        return "gs://firebird-fixture"

    monkeypatch.setattr(sky_runner, "ensure_cloud_storage", storage)

    async def verify(sky, project, *, endpoint):
        assert sky == "/fixture/sky" and project == "robotics-demo"
        assert endpoint == "http://127.0.0.1:46580"
        return {"workspace": "firebird-gcp-fixture", "sky_api_endpoint": endpoint}

    monkeypatch.setattr(
        sky_runner.cloud_compute_catalog, "verify_sky_target", verify, raising=False
    )


def test_task_pins_reviewed_resources_and_syncs_only_worker(payload, target, tmp_path):
    stage = tmp_path / "jobs" / payload["job_id"] / "operation"
    task_path, state = sky_runner.prepare(payload, stage, target)
    task = json.loads(task_path.read_text())
    assert task["resources"] == {
        "infra": "gcp/us-central1",
        "accelerators": "A100:1",
        "cpus": "2+",
        "memory": "4+",
        "disk_size": 200,
        "use_spot": False,
        "instance_type": "a2-highgpu-1g",
    }
    assert json.loads((stage / "sky-config.yaml").read_text()) == {
        "active_workspace": "firebird-gcp-fixture",
        "api_server": {"endpoint": "http://127.0.0.1:46580"},
    }
    assert state["target"]["project_id"] == "robotics-demo"
    assert state["phase"] == "prepared"
    assert "never-upload" not in (stage / "sky-bundle" / "request.json").read_text()
    assert task["run"] == ".venv/bin/python bootstrap.py"
    assert "python 3.11" in task["setup"]
    assert Path(task["workdir"]).is_relative_to(stage)
    assert not (stage / "sky-bundle" / ".git").exists()
    with pytest.raises(ValueError, match="already has"):
        sky_runner.prepare(payload, stage, target)


@pytest.mark.parametrize(
    "field,value",
    [
        ("project_id", "wrong/../project"),
        ("region", "us; echo bad"),
        ("accelerator", "A100;bad"),
        ("gpu_count", 8),
        ("gpu_count", True),
        ("disk_size_gb", 10000),
        ("idle_minutes", 0),
        ("instance_type", "a2;bad"),
    ],
)
def test_invalid_targets_never_reach_dispatch(payload, target, tmp_path, field, value):
    with pytest.raises(ValueError):
        sky_runner.prepare(payload, tmp_path / "stage", {**target, field: value})


def test_resume_inputs_are_copied_and_rebased(payload, target, tmp_path):
    checkpoint = tmp_path / "original-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "recipe.json").write_text("{}")
    payload["resume_checkpoint"] = str(checkpoint)
    stage = tmp_path / "stage"
    sky_runner.prepare(payload, stage, target)
    assert (stage / "sky-bundle" / "inputs" / "resume" / "recipe.json").is_file()
    remote = json.loads((stage / "sky-bundle" / "request.json").read_text())
    assert remote["resume_checkpoint"] == "inputs/resume"
    assert payload["resume_checkpoint"] == str(checkpoint)


@pytest.mark.parametrize("source", ["remote-artifact", "local-resume"])
@pytest.mark.parametrize(
    "caller_recipe", [None, {"steps": 200}, {"model_id": "lerobot/smolvla_base"}]
)
def test_resume_selects_native_worker_from_checkpoint_lineage(
    payload, target, tmp_path, source, caller_recipe
):
    from vla_platform.lifecycle.native_profiles import LEROBOT_REVISION

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    recipe = {"model_id": "code://lerobot/act", "model_revision": LEROBOT_REVISION}
    payload["parameters"].update(training=caller_recipe, training_method="full")
    if source == "local-resume":
        (checkpoint / "recipe.json").write_text(json.dumps(recipe))
        payload["resume_checkpoint"] = str(checkpoint)
    else:
        (checkpoint / "remote.json").write_text("{}")
        payload["artifact"] = {
            "path": str(checkpoint),
            "format": "training_checkpoint",
            "metadata": {
                "base_model": {"repository": recipe["model_id"], "revision": LEROBOT_REVISION}
            },
        }
    stage = tmp_path / "stage"
    task_path, _ = sky_runner.prepare(payload, stage, target)
    task = json.loads(task_path.read_text())
    dispatch = json.loads((stage / "sky-bundle/dispatch.json").read_text())
    assert dispatch["worker_module"] == "firebird_vla.lerobot_application"
    assert "--python 3.12" in task["setup"]
    assert LEROBOT_REVISION in task["setup"]
    assert "--index-url https://download.pytorch.org/whl/cu128" in task["setup"]
    assert "--constraint native-cu128-constraints.txt" in task["setup"]
    assert (stage / "sky-bundle/native-cu128-constraints.txt").read_text() == (
        "torch==2.11.0+cu128\ntorchvision==0.26.0+cu128\n"
    )


def test_refuses_input_symlinks(payload, target, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "secret").symlink_to(tmp_path / "outside")
    payload["resume_checkpoint"] = str(checkpoint)
    with pytest.raises(ValueError, match="symlinks"):
        sky_runner.prepare(payload, tmp_path / "stage", target)


@pytest.mark.parametrize("resume", [False, True])
def test_psi_dispatch_uses_its_isolated_environment(payload, target, tmp_path, monkeypatch, resume):
    from vla_platform.lifecycle.training_catalog import TRAINING_MODEL_BY_ID

    worker = Path(__file__).resolve().parents[1] / "workers/smolvla_qlora"
    monkeypatch.setattr(sky_runner, "worker_root", lambda: worker)
    model = TRAINING_MODEL_BY_ID["psi0"]
    recipe = {"model_id": model.model_id, "model_revision": model.model_revision}
    payload["parameters"].update(training_method="full", training=None if resume else recipe)
    if resume:
        artifact = tmp_path / "descriptor"
        artifact.mkdir()
        (artifact / "remote.json").write_text("{}")
        payload["artifact"] = {
            "path": str(artifact),
            "metadata": {
                "base_model": {"repository": model.model_id, "revision": model.model_revision}
            },
        }
    stage = tmp_path / "stage"
    task_path, _ = sky_runner.prepare(payload, stage, target)
    task = json.loads(task_path.read_text())
    dispatch = json.loads((stage / "sky-bundle/dispatch.json").read_text())
    assert dispatch["worker_module"] == "firebird_vla.psi_application"
    assert task["envs"]["FIREBIRD_PSI_ROOT"] == "psi"
    assert task["resources"]["memory"] == "32+"
    assert "--python 3.11" in task["setup"]
    assert "--project psi --frozen --group psi --inexact" in task["setup"]
    assert "-r worker/requirements-smolvla-linux.txt" not in task["setup"]


def _download_bundle(home, cluster, entries):
    log_dir = home / "sky_logs" / cluster / "1-firebird-training"
    log_dir.mkdir(parents=True)
    path = log_dir / "firebird-output.tar"
    with tarfile.open(path, "w") as archive:
        for name, data in entries.items():
            entry = tarfile.TarInfo(name)
            if data is None:
                entry.type = tarfile.SYMTYPE
                entry.linkname = "/tmp/escape"
                archive.addfile(entry)
            else:
                entry.size = len(data)
                archive.addfile(entry, io.BytesIO(data))
    content = path.read_bytes()
    path.unlink()
    part = log_dir / "firebird-output.part-00000"
    part.write_bytes(content)
    (log_dir / "firebird-output.json").write_text(
        json.dumps(
            {
                "sha256": hashlib.sha256(content).hexdigest(),
                "size": len(content),
                "parts": [{"name": part.name, "size": len(content)}],
            }
        )
    )
    return part


def test_download_rebases_artifact_without_replacing_app_state(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "request.json").write_text("original")
    _download_bundle(
        tmp_path,
        "fixture",
        {
            "result.json": json.dumps({"artifact": {"path": "bundle"}}).encode(),
            "bundle/manifest.json": b"{}",
            "bundle/checkpoint/weights": b"adapter",
            "request.json": b"remote",
            "sky-state.json": b"forged",
        },
    )
    sky_runner.collect(stage, "fixture")
    assert json.loads((stage / "result.json").read_text())["artifact"]["path"] == str(
        stage / "bundle"
    )
    assert (stage / "bundle" / "checkpoint" / "weights").read_bytes() == b"adapter"
    assert (stage / "request.json").read_text() == "original"
    assert not (stage / "sky-state.json").exists()


@pytest.mark.parametrize(
    "name,data", [("../escape", b"bad"), ("/absolute", b"bad"), ("bundle/link", None)]
)
def test_download_rejects_unsafe_archives(tmp_path, monkeypatch, name, data):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    stage = tmp_path / "stage"
    stage.mkdir()
    _download_bundle(tmp_path, "fixture", {"result.json": b"{}", name: data})
    with pytest.raises(ValueError, match="Unsafe"):
        sky_runner.collect(stage, "fixture")
    assert not (stage / "result.json").exists()


def test_download_checks_checksum_and_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    stage = tmp_path / "stage"
    stage.mkdir()
    path = _download_bundle(tmp_path, "fixture", {"result.json": b"{}"})
    path.write_bytes(b"corrupt" + path.read_bytes()[7:])
    with pytest.raises(ValueError, match="checksum"):
        sky_runner.collect(stage, "fixture")
    monkeypatch.setattr(sky_runner, "MAX_ARTIFACT_BYTES", 1)
    _download_bundle(tmp_path, "oversize", {"result.json": b"{}"})
    with pytest.raises(ValueError, match="transfer limit"):
        sky_runner.collect(stage, "oversize")


def test_remote_archive_exports_regular_outputs_only(tmp_path):
    output, logs = tmp_path / "output", tmp_path / "logs"
    output.mkdir()
    logs.mkdir()
    (output / "result.json").write_text("{}")
    sky_bootstrap.publish_archive(output, logs)
    assert (logs / "final-output" / "firebird-output.part-00000").is_file()
    (output / "escape").symlink_to(logs)
    with pytest.raises(ValueError, match="unsupported"):
        sky_bootstrap.publish_archive(output, logs)


def test_archive_parts_round_trip_with_bounded_members(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(sky_bootstrap, "PART_BYTES", 4096)
    output = tmp_path / "remote-output"
    output.mkdir()
    (output / "result.json").write_text("{}")
    (output / "bundle").mkdir()
    (output / "bundle" / "weights").write_bytes(b"x" * 9000)
    logs = tmp_path / "sky_logs" / "fixture" / "1-firebird-training"
    logs.mkdir(parents=True)
    sky_bootstrap.publish_archive(output, logs)
    descriptor = json.loads((logs / "final-output" / "firebird-output.json").read_text())
    assert len(descriptor["parts"]) > 1
    assert all(part["size"] <= 4096 for part in descriptor["parts"])
    assert not list(logs.glob("*.tar"))
    stage = tmp_path / "stage"
    stage.mkdir()
    sky_runner.collect(stage, "fixture")
    assert (stage / "bundle" / "weights").read_bytes() == b"x" * 9000


def test_remote_bootstrap_runs_protocol_and_makes_downloadable_bundle(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    logs = tmp_path / "sky_logs" / "1-firebird-training"
    logs.mkdir(parents=True)
    (work / "dispatch.json").write_text(
        json.dumps(
            {
                "task_name": "firebird-training",
                "timeout_seconds": 300,
            }
        )
    )
    (work / "request.json").write_text(
        json.dumps(
            {
                "job_id": "example",
                "artifact": None,
                "output_dir": "ignored",
            }
        )
    )
    monkeypatch.chdir(work)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("SKYPILOT_INTERNAL_JOB_ID", "1")

    class Process:
        def __init__(self, argv, **kwargs):
            assert argv[1:3] == ["-m", "firebird_vla.application"]
            request = json.loads(Path(argv[3]).read_text())
            output = Path(request["output_dir"])
            (output / "bundle").mkdir()
            (output / "bundle" / "manifest.json").write_text("{}")
            Path(argv[4]).write_text(
                json.dumps(
                    {
                        "job_id": "example",
                        "artifact": {"path": str(output / "bundle")},
                    }
                )
            )

        def wait(self, timeout):
            assert timeout == 300
            return 0

    monkeypatch.setattr(sky_bootstrap.subprocess, "Popen", Process)
    assert sky_bootstrap.main() == 0
    with tarfile.open(
        fileobj=io.BytesIO((logs / "final-output" / "firebird-output.part-00000").read_bytes())
    ) as archive:
        result = json.load(archive.extractfile("result.json"))
    assert result["artifact"]["path"] == "bundle"


def test_bootstrap_refuses_to_train_without_skypilot_log_identity(tmp_path, monkeypatch):
    (tmp_path / "dispatch.json").write_text(json.dumps({"task_name": "firebird-training"}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SKYPILOT_INTERNAL_JOB_ID", raising=False)
    monkeypatch.setattr(
        sky_bootstrap.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("no training")
    )
    with pytest.raises(RuntimeError, match="training was not started"):
        sky_bootstrap.main()


@pytest.mark.parametrize("failure", [None, "launch", "setup", "download", "cancel", "cleanup"])
def test_dispatch_always_tears_down_owned_cluster(payload, target, tmp_path, monkeypatch, failure):
    calls, events = [], []
    stage = tmp_path / "jobs" / payload["job_id"] / "operation"

    async def event(message):
        events.append(message)

    async def command(argv, directory, **kwargs):
        calls.append(argv)
        assert directory == stage
        assert argv[0] == "/fixture/sky"
        if argv[1:3] == ["api", "status"]:
            return 0, "[]"
        if argv[1] == "launch":
            return (
                (1, "failure")
                if failure == "launch"
                else (0, "Submitted sky.launch request: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee\n")
            )
        if argv[1:3] == ["api", "logs"]:
            if failure == "cancel":
                raise asyncio.CancelledError
            return (1 if failure == "setup" else 0), ""
        if "--sync-down" in argv and failure == "download":
            return 1, ""
        if argv[1] == "down" and failure == "cleanup":
            return 1, ""
        return 0, ""

    async def sync(sky, directory, state, callback, *, final=False):
        calls.append([sky, "checkpoint-download"])
        if failure == "download":
            raise RuntimeError("Checkpoint download pending")

    monkeypatch.setattr(sky_runner, "sync_checkpoints", sync)
    monkeypatch.setattr(sky_runner, "_command", command)
    monkeypatch.setattr(
        sky_runner, "collect", lambda directory, _: (directory / "result.json").write_text("{}")
    )

    async def execute():
        if failure:
            with pytest.raises(asyncio.CancelledError if failure == "cancel" else RuntimeError):
                await sky_runner.run(payload, stage, target, event)
        else:
            assert await sky_runner.run(payload, stage, target, event) == 0

    asyncio.run(execute())
    state = json.loads((stage / sky_runner.STATE_NAME).read_text())
    launch = next(argv for argv in calls if argv[1] == "launch")
    assert "--down" in launch and "--async" in launch
    assert calls[-1][1:4] == ["down", state["cluster"], "-y"]
    assert state["phase"] == (
        "cleanup_failed" if failure in {"cleanup", "launch"} else "cleaned_up"
    )
    if failure == "cleanup":
        assert any("needs attention" in item for item in events)
    if failure is None:
        assert (
            next(i for i, argv in enumerate(calls) if "checkpoint-download" in argv)
            < len(calls) - 1
        )


def test_detached_setup_failure_surfaces_compiler_error_and_still_tears_down(
    payload, target, tmp_path, monkeypatch
):
    stage = tmp_path / "jobs" / payload["job_id"] / "operation"
    calls = []
    secret = "unusual-private-credential"

    async def event(message):
        pass

    async def command(argv, directory, **kwargs):
        calls.append(argv[1])
        if argv[1] == "launch":
            return 0, "Submitted sky.launch request: abcdef01-2345\n"
        if argv[1:3] == ["api", "status"]:
            return 0, "[]"
        if "--follow" in argv:
            (stage / "worker.log").write_text(
                "CMake Error at CMakeLists.txt:189 (find_package):\n"
                "  Could NOT find Protobuf (missing: Protobuf_LIBRARIES Protobuf_INCLUDE_DIR)\n"
                f"Failed URL https://example.invalid/?credential={secret}\n"
                f"Failed token hf_superprivate and credential {secret}\n"
            )
            return 1, "FAILED_SETUP"
        if "--status" in argv:
            return 1, "FAILED_SETUP"
        return 0, ""

    async def sync(*args, **kwargs):
        pass

    monkeypatch.setattr(sky_runner, "_command", command)
    monkeypatch.setattr(sky_runner, "sync_checkpoints", sync)
    with pytest.raises(RuntimeError, match="Could NOT find Protobuf") as error:
        asyncio.run(sky_runner.run(payload, stage, target, event, hf_token=secret))
    assert "down" in calls
    assert secret not in str(error.value)
    assert "hf_superprivate" not in str(error.value)
    assert "example.invalid" not in str(error.value)
    assert len(str(error.value)) < 2000


def test_cleanup_timeout_does_not_skip_down(payload, target, tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    _, state = sky_runner.prepare(payload, stage, target)
    calls = []

    async def command(argv, directory, **kwargs):
        calls.append(argv)
        if argv[1:3] == ["api", "status"]:
            return 0, "[]"
        if argv[1] == "cancel":
            raise TimeoutError
        return 0, ""

    monkeypatch.setattr(sky_runner, "_command", command)
    assert asyncio.run(sky_runner.cleanup(stage, state)) is True
    assert calls[-1][1] == "down"


def test_cleanup_syncs_cloud_checkpoint_metadata_after_cancel_before_down(
    payload, target, tmp_path, monkeypatch
):
    from vla_platform.lifecycle.cloud_storage import encoded, install_descriptor, write_json

    stage = tmp_path / "jobs" / payload["job_id"] / "operation"
    _, state = sky_runner.prepare(payload, stage, target)
    state.update(phase="training", request_id="abcdef01-2345")
    state["target"]["storage_uri"] = "gs://test-bucket/jobs/training-job/operation"
    calls = []

    async def command(argv, directory, **kwargs):
        calls.append(argv[1])
        if argv[1:3] == ["api", "status"]:
            return 0, "[]"
        return 0, ""

    async def sync(sky, directory, actual_state, on_event, *, final=False):
        assert final is True
        assert "cancel" in calls and "down" not in calls
        calls.append("metadata")
        uri = state["target"]["storage_uri"] + "/checkpoints/checkpoint-000005"
        manifest = {
            "metadata": {"remote_uri": uri, "step": 5},
            "files": {"checkpoint/weights": "a" * 64},
        }
        descriptor = {
            "name": "checkpoint-000005",
            "uri": uri,
            "step": 5,
            "label": "Step 5",
            "manifest": manifest,
            "manifest_sha256": hashlib.sha256(encoded(manifest)).hexdigest(),
            "file_bytes": 3 * 1024**3,
        }
        install_descriptor(stage / "cloud-checkpoints/checkpoint-000005", descriptor)
        write_json(stage / "cloud-checkpoints.json", {"checkpoints": [descriptor]})

    monkeypatch.setattr(sky_runner, "_command", command)
    monkeypatch.setattr(sky_runner, "sync_checkpoints", sync)
    assert asyncio.run(sky_runner.cleanup(stage, state)) is True
    assert calls.index("cancel") < calls.index("metadata") < calls.index("down")
    assert (stage / "cloud-checkpoints/checkpoint-000005/remote.json").is_file()
    assert not list(stage.rglob("weights"))
    assert state["final_metadata_synced"] is True


def test_metadata_sync_failure_never_blocks_teardown_and_retries_on_restart(
    payload, target, tmp_path, monkeypatch
):
    stage = tmp_path / "jobs" / payload["job_id"] / "operation"
    _, state = sky_runner.prepare(payload, stage, target)
    state.update(phase="training", request_id="abcdef01-2345")
    state["target"]["storage_uri"] = "gs://test-bucket/jobs/training-job/operation"
    calls = []

    async def command(argv, directory, **kwargs):
        calls.append(argv[1])
        return (0, "[]") if argv[1:3] == ["api", "status"] else (0, "")

    async def unavailable(*args, **kwargs):
        raise RuntimeError("Temporary metadata transport failure")

    monkeypatch.setattr(sky_runner, "_command", command)
    monkeypatch.setattr(sky_runner, "sync_checkpoints", unavailable)
    assert asyncio.run(sky_runner.cleanup(stage, state)) is True
    assert "down" in calls
    assert state["phase"] == "cleaned_up"
    assert state["metadata_sync_error"]
    calls.clear()

    async def recovered(*args, **kwargs):
        calls.append("metadata-only")

    monkeypatch.setattr(sky_runner, "sync_checkpoints", recovered)
    assert asyncio.run(sky_runner.recover(tmp_path)) == []
    assert calls == ["metadata-only"]
    saved = json.loads((stage / sky_runner.STATE_NAME).read_text())
    assert saved["final_metadata_synced"] is True
    assert saved["metadata_sync_error"] is None


@pytest.mark.parametrize(
    "prefix", ["", "The flag is ignored because the server does not support it yet.\n"]
)
def test_cleanup_discovers_lost_launch_id_and_never_cancels_unrelated_requests(
    payload, target, tmp_path, monkeypatch, prefix
):
    stage = tmp_path / "stage"
    _, state = sky_runner.prepare(payload, stage, target)
    state["phase"] = "launching"
    calls, pending = [], True
    owned = "11111111-2222-3333-4444-555555555555"
    unrelated = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

    async def command(argv, directory, **kwargs):
        nonlocal pending
        calls.append(argv)
        if argv[1:3] == ["api", "status"]:
            return 0, prefix + json.dumps(
                [
                    {
                        "cluster_name": state["cluster"],
                        "name": "launch",
                        "request_id": owned,
                        "status": "PENDING" if pending else "CANCELLED",
                    },
                    {
                        "cluster_name": "unrelated",
                        "name": "launch",
                        "request_id": unrelated,
                        "status": "RUNNING",
                    },
                ]
            )
        if argv[1:3] == ["api", "cancel"]:
            assert argv[3] == owned
            pending = False
        return 0, ""

    monkeypatch.setattr(sky_runner, "_command", command)
    assert asyncio.run(sky_runner.cleanup(stage, state)) is True
    assert state["request_id"] == owned
    assert state["phase"] == "cleaned_up"
    assert calls[-1][1] == "down"
    assert not any(unrelated in argv for argv in calls)


def test_unknown_submission_stays_recoverable_when_cluster_is_not_registered(
    payload, target, tmp_path, monkeypatch
):
    stage = tmp_path / "stage"
    _, state = sky_runner.prepare(payload, stage, target)
    state["phase"] = "launching"

    async def command(argv, directory, **kwargs):
        return 0, "[]" if argv[1:3] == ["api", "status"] else "No matching cluster"

    monkeypatch.setattr(sky_runner, "_command", command)
    assert asyncio.run(sky_runner.cleanup(stage, state)) is False
    assert state["phase"] == "cleanup_failed"


def test_cleanup_does_not_discard_unrecognized_diagnostics(payload, target, tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    _, state = sky_runner.prepare(payload, stage, target)
    state["phase"] = "cleanup_failed"
    state["request_id"] = "11111111-2222-3333-4444-555555555555"

    async def command(argv, directory, **kwargs):
        return 0, "Unexpected authentication failure\n[]" if argv[1:3] == ["api", "status"] else ""

    monkeypatch.setattr(sky_runner, "_command", command)
    assert asyncio.run(sky_runner.cleanup(stage, state)) is False
    assert state["phase"] == "cleanup_failed"


def test_queued_workspace_change_is_rejected_before_launch(payload, target, tmp_path):
    async def event(message):
        pass

    with pytest.raises(ValueError, match="queued run"):
        asyncio.run(
            sky_runner.run(payload, tmp_path / "stage", {**target, "workspace": "changed"}, event)
        )
    assert not (tmp_path / "stage").exists()


def test_recovery_ignores_cleaned_runs_and_reports_failed_cleanup(
    payload, target, tmp_path, monkeypatch
):
    data = tmp_path / "data"
    stage = data / "jobs" / payload["job_id"] / "operation"
    _, state = sky_runner.prepare(payload, stage, target)
    calls = []

    async def command(argv, directory, **kwargs):
        calls.append(argv)
        return 1, "failure"

    monkeypatch.setattr(sky_runner, "_command", command)
    failures = asyncio.run(sky_runner.recover(data))
    assert failures[0]["job_id"] == payload["job_id"]
    assert failures[0]["cluster"] == state["cluster"]
    assert "needs attention" in failures[0]["error"]
    state["phase"] = "cleaned_up"
    sky_runner._write_json(stage / sky_runner.STATE_NAME, state)
    calls.clear()
    assert asyncio.run(sky_runner.recover(data)) == []
    assert calls == []


def test_recovery_never_targets_unrelated_cluster(payload, target, tmp_path, monkeypatch):
    stage = tmp_path / "jobs" / payload["job_id"] / "operation"
    _, state = sky_runner.prepare(payload, stage, target)
    state["cluster"] = "my-important-cluster"
    sky_runner._write_json(stage / sky_runner.STATE_NAME, state)

    async def command(*args, **kwargs):
        pytest.fail("must not invoke SkyPilot for an unowned cluster")

    monkeypatch.setattr(sky_runner, "_command", command)
    assert asyncio.run(sky_runner.recover(tmp_path))[0]["error"]


def test_project_verification_failure_prevents_any_launch(payload, target, tmp_path, monkeypatch):
    async def verify(*args, **kwargs):
        raise ValueError("Project mismatch")

    async def event(message):
        pass

    async def command(*args, **kwargs):
        pytest.fail("project mismatch must fail before dispatch")

    monkeypatch.setattr(sky_runner.cloud_compute_catalog, "verify_sky_target", verify)
    monkeypatch.setattr(sky_runner, "_command", command)
    with pytest.raises(ValueError, match="mismatch"):
        asyncio.run(sky_runner.run(payload, tmp_path / "stage", target, event))
    assert not (tmp_path / "stage").exists()


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://user:secret@example.com",
        "https://example.com?token=secret",
        "ftp://example.com",
        "http://example.com/#token",
        "http://[broken",
        "http://bad host",
    ],
)
def test_endpoint_validation_rejects_credentials_and_malformed_urls(endpoint):
    with pytest.raises(ValueError, match="endpoint"):
        sky_runner.validate_endpoint(endpoint)


def test_every_cli_command_uses_the_saved_endpoint(payload, target, tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    sky_runner.prepare(payload, stage, target)
    monkeypatch.setenv("SKYPILOT_API_SERVER_ENDPOINT", "https://different-server.example")

    class Output:
        async def read(self, size):
            return b""

    class Process:
        returncode = 0
        stdout = Output()

        async def wait(self):
            return 0

    async def create(*argv, **kwargs):
        assert kwargs["env"]["SKYPILOT_API_SERVER_ENDPOINT"] == target["sky_api_endpoint"]
        assert "shell" not in kwargs
        return Process()

    async def stop(process):
        pass

    monkeypatch.setattr(sky_runner.asyncio, "create_subprocess_exec", create)
    monkeypatch.setattr(sky_runner, "_stop_process", stop)
    assert asyncio.run(
        sky_runner._command(["/fixture/sky", "logs", "fixture"], stage, timeout=5)
    ) == (0, "")


def checkpoint_transfer(home, cluster, step, *, corrupt=False):
    name = f"checkpoint-{step:06d}"
    weights = f"optimizer+adapter at {step}".encode()
    manifest = json.dumps(
        {
            "schema_version": 1,
            "step": step,
            "files": {"weights": hashlib.sha256(weights).hexdigest()},
        }
    ).encode()
    part = _download_bundle(
        home,
        f"transfer-{step}",
        {
            "manifest.json": manifest,
            "weights": b"corrupt" if corrupt else weights,
        },
    )
    destination = (
        home / "sky_logs" / cluster / "1-firebird-training" / "checkpoint-snapshots" / name
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    part.parent.rename(destination)
    return {
        "name": name,
        "step": step,
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "descriptor": str(destination / "firebird-output.json"),
    }


def test_live_checkpoint_install_is_atomic_verified_and_monotonic(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    stage = tmp_path / "stage"
    stage.mkdir()
    later = checkpoint_transfer(tmp_path, "cluster", 10)
    assert sky_runner.install_checkpoint(stage, "cluster", later) == 10
    earlier = checkpoint_transfer(tmp_path, "cluster", 5)
    assert sky_runner.install_checkpoint(stage, "cluster", earlier) == 5
    assert json.loads((stage / "training" / "latest.json").read_text())["step"] == 10
    assert len(list((stage / "training").glob("checkpoint-*"))) == 2
    assert not Path(earlier["descriptor"]).parent.exists()
    corrupt = checkpoint_transfer(tmp_path, "cluster", 15, corrupt=True)
    with pytest.raises(ValueError, match="checksum"):
        sky_runner.install_checkpoint(stage, "cluster", corrupt)
    assert not (stage / "training" / "checkpoint-000015").exists()
    assert json.loads((stage / "training" / "latest.json").read_text())["step"] == 10


def test_streamed_checkpoint_survives_later_download_timeout(
    payload, target, tmp_path, monkeypatch
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    stage = tmp_path / "stage"
    _, state = sky_runner.prepare(payload, stage, target)
    item = checkpoint_transfer(tmp_path, state["cluster"], 5)
    monkeypatch.setattr(sky_runner.cloud_compute_catalog, "sky_python", lambda _: "/fixture/python")
    events = []

    async def event(message):
        events.append(message)

    async def command(argv, directory, **kwargs):
        await kwargs["on_line"]("FIREBIRD_CHECKPOINT=" + json.dumps(item))
        assert (stage / "training" / "checkpoint-000005").is_dir()
        raise TimeoutError("a later checkpoint download stalled")

    monkeypatch.setattr(sky_runner, "_command", command)
    with pytest.raises(TimeoutError):
        asyncio.run(sky_runner.sync_checkpoints("/fixture/sky", stage, state, event))
    assert events == ["Checkpoint 5 saved locally"]
    assert json.loads((stage / "training" / "latest.json").read_text())["step"] == 5
    assert "checkpoint-000005" in json.loads((stage / "sky-checkpoints.json").read_text())


def test_live_sync_runs_while_training_log_tail_is_still_running(tmp_path, monkeypatch):
    calls, events = [], []

    async def execute():
        finished = asyncio.Event()

        async def event(message):
            events.append(message)

        async def command(argv, directory, **kwargs):
            assert argv[1:4] == ["logs", "cluster", "1"]
            await kwargs["on_event"]("_checkpoint_ready:5")
            await finished.wait()
            return 0, ""

        async def sync(*args, **kwargs):
            assert not finished.is_set()
            calls.append("synced while training")
            await event("Checkpoint 5 saved locally")
            finished.set()

        monkeypatch.setattr(sky_runner, "_command", command)
        monkeypatch.setattr(sky_runner, "sync_checkpoints", sync)
        await sky_runner.follow_training(
            "/fixture/sky",
            tmp_path,
            {
                "cluster": "cluster",
                "config_path": "/fixture/config",
            },
            30,
            event,
        )

    asyncio.run(execute())
    assert calls == ["synced while training"]
    assert events == ["Checkpoint 5 saved locally"]


def test_acknowledgement_is_cpu_only_and_contains_verified_receipts(
    payload, target, tmp_path, monkeypatch
):
    import base64

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    stage = tmp_path / "stage"
    _, state = sky_runner.prepare(payload, stage, target)
    item = checkpoint_transfer(tmp_path, state["cluster"], 5)
    monkeypatch.setattr(sky_runner.cloud_compute_catalog, "sky_python", lambda _: "/fixture/python")
    commands = []

    async def event(message):
        pass

    async def command(argv, directory, **kwargs):
        commands.append(argv)
        if argv[0] == "/fixture/python":
            await kwargs["on_line"]("FIREBIRD_CHECKPOINT=" + json.dumps(item))
        else:
            assert argv[1] == "exec" and argv[2] == state["cluster"]
            assert argv[argv.index("--gpus") + 1] == "none"
            assert argv[argv.index("--cpus") + 1] == "0.1"
            assert json.loads(base64.urlsafe_b64decode(argv[-1])) == {
                "checkpoint-000005": item["manifest_sha256"],
            }
            assert (stage / "training" / "checkpoint-000005").is_dir()
        return 0, ""

    monkeypatch.setattr(sky_runner, "_command", command)
    asyncio.run(sky_runner.sync_checkpoints("/fixture/sky", stage, state, event))
    assert len(commands) == 2


def test_too_many_checkpoints_rejected_before_provisioning(payload, target, tmp_path):
    payload["parameters"]["training"] = {"steps": 20000, "save_every": 1}
    with pytest.raises(ValueError, match="at most 10000"):
        sky_runner.prepare(payload, tmp_path / "stage", target)
    assert not (tmp_path / "stage").exists()


def test_cancellation_after_step_five_keeps_the_canonical_local_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    stage = tmp_path / "stage"
    stage.mkdir()
    item = checkpoint_transfer(tmp_path, "cluster", 5)

    async def execute():
        saved, tail_stopped = asyncio.Event(), asyncio.Event()

        async def event(message):
            if message == "Checkpoint 5 saved locally":
                saved.set()

        async def command(argv, directory, **kwargs):
            assert argv[1:4] == ["logs", "cluster", "1"]
            await kwargs["on_event"]("_checkpoint_ready:5")
            try:
                await asyncio.Future()
            finally:
                tail_stopped.set()

        async def sync(*args, **kwargs):
            step = sky_runner.install_checkpoint(stage, "cluster", item)
            await event(f"Checkpoint {step} saved locally")

        monkeypatch.setattr(sky_runner, "_command", command)
        monkeypatch.setattr(sky_runner, "sync_checkpoints", sync)
        task = asyncio.create_task(
            sky_runner.follow_training(
                "/fixture/sky",
                stage,
                {
                    "cluster": "cluster",
                    "config_path": "/fixture/config",
                },
                30,
                event,
            )
        )
        await asyncio.wait_for(saved.wait(), 1)
        assert not task.done() and not tail_stopped.is_set()
        assert (stage / "training" / "checkpoint-000005" / "manifest.json").is_file()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert tail_stopped.is_set()

    asyncio.run(execute())
    assert json.loads((stage / "training" / "latest.json").read_text()) == {
        "checkpoint": "checkpoint-000005",
        "step": 5,
    }
    assert (stage / "training" / "checkpoint-000005" / "weights").is_file()


@pytest.mark.parametrize("valid", [True, False])
def test_final_only_checkpoints_are_verified_before_resume_pointer(tmp_path, monkeypatch, valid):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    stage = tmp_path / "stage"
    stage.mkdir()
    weights = b"final fallback checkpoint"
    manifest = json.dumps(
        {
            "schema_version": 1,
            "step": 5,
            "files": {"weights": hashlib.sha256(weights).hexdigest()},
        }
    ).encode()
    _download_bundle(
        tmp_path,
        "cluster",
        {
            "result.json": b"{}",
            "training/checkpoint-000005/manifest.json": manifest,
            "training/checkpoint-000005/weights": weights if valid else b"corrupt",
            "training/latest.json": b'{"step":999,"checkpoint":"untrusted"}',
        },
    )
    if valid:
        sky_runner.collect(stage, "cluster")
        assert json.loads((stage / "training" / "latest.json").read_text()) == {
            "checkpoint": "checkpoint-000005",
            "step": 5,
        }
    else:
        with pytest.raises(ValueError, match="checksum"):
            sky_runner.collect(stage, "cluster")
        assert not (stage / "training" / "latest.json").exists()
