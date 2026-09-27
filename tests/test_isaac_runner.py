"""Offline simulation ownership tests. No SDK, credentials or paid cloud calls."""

import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from vla_platform.lifecycle import isaac_runner as runner


@pytest.fixture
def profile(tmp_path):
    root = tmp_path / "runner"
    sky = root / "workers/skypilot"
    sky.mkdir(parents=True)
    isaac = root / "workers/isaac_sim"
    isaac.mkdir()
    for name in (
        "rollout_launch.py",
        "rollout_sdk.py",
        "sky.sh",
        "launch-rollout.sh",
        "config.yaml",
    ):
        (sky / name).write_text("fixture\n")
    (isaac / "scene.usda").write_text("fixture cup scene\n")
    task = sky / "rollout.local.yaml"
    task.write_text("fixture operator task\n")
    key = tmp_path / "fixture-credential.json"
    key.write_text("fake fixture only, never a real credential")
    return runner.SimulationProfile(
        "so101-cup-episode-001",
        "Cup experiment",
        root,
        task,
        "robotics-demo",
        key,
        "gs://fixture-results/experiments",
        runner._sha(task),
        True,
    )


@pytest.fixture
def checkpoint():
    return {
        "policy_type": "act",
        "model_id": "sha256:" + "a" * 64,
        "camera_key": "observation.images.front",
        "width": 640,
        "height": 360,
        "state_dim": 6,
        "action_dim": 6,
        "chunk_size": 100,
        "action_steps": 100,
    }


def config(profile):
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in asdict(profile).items()
    }


def test_profile_loader_public_boundary_and_identity(profile, tmp_path):
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"schema_version": 1, "profiles": [config(profile)]}))
    (loaded,) = runner.load_profiles(path)
    assert loaded == profile
    assert loaded.python == profile.skypilot / ".venv/bin/python"
    assert set(loaded.public()) == {
        "id",
        "label",
        "architectures",
        "experimental",
        "task_object",
        "provider",
        "accelerators",
        "task_success",
    }
    assert str(profile.credential_file) not in json.dumps(loaded.public())
    identity = profile.identity_hash()
    profile.credential_file.write_text("changed fake key")
    assert profile.identity_hash() == identity
    (profile.runner_root / "workers/isaac_sim/scene.usda").write_text("changed geometry")
    assert profile.identity_hash() != identity
    profile.task.write_text("changed task")
    with pytest.raises(ValueError, match="task changed"):
        profile.identity_hash()


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", "../escape"),
        ("task_sha256", "z" * 64),
        ("accept_eula", 1),
        ("runner_root", "relative"),
        ("project_id", "bad;command"),
        ("results_uri", "https://outside"),
        ("results_uri", "gs://fixture-results/a//b"),
        ("sky_api_endpoint", "http://user:key@localhost"),
    ],
)
def test_invalid_profiles_rejected(profile, tmp_path, field, value):
    record = config(profile)
    record[field] = value
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"schema_version": 1, "profiles": [record]}))
    with pytest.raises(ValueError):
        runner.load_profiles(path)


def test_duplicate_profile_extra_fields_and_schema_rejected(profile, tmp_path):
    path = tmp_path / "profiles.json"
    for value in (
        {"schema_version": True, "profiles": []},
        {"schema_version": 1, "profiles": [config(profile)] * 2},
        {"schema_version": 1, "profiles": [config(profile) | {"command": "bad"}]},
    ):
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError):
            runner.load_profiles(path)
    assert runner.load_profiles(None) == ()


def test_environment_does_not_forward_unrelated_credentials(profile, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-secret")
    monkeypatch.setenv("HTTPS_PROXY", "http://bad")
    monkeypatch.setenv("SKYPILOT_API_SERVER_ENDPOINT", "http://unrelated")
    env = profile.environment()
    assert not {"OPENROUTER_API_KEY", "HTTPS_PROXY", "SKYPILOT_API_SERVER_ENDPOINT"} & env.keys()
    assert env["GOOGLE_APPLICATION_CREDENTIALS"] == str(profile.credential_file)
    assert env["SIM_PROJECT_ID"] == profile.project_id
    with pytest.raises(ValueError, match="license"):
        replace(profile, accept_eula=False).environment()


@pytest.mark.parametrize("family", ["act", "smolvla"])
def test_both_families_admitted_without_quality_claim(profile, checkpoint, family):
    record = runner.admit(profile, {"checkpoint": checkpoint | {"policy_type": family}})
    assert record["model_id"] == checkpoint["model_id"]
    assert record["task_success"] is None and record["calibration_verified"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("policy_type", "openvla"),
        ("state_dim", 7),
        ("action_dim", 7),
        ("width", True),
        ("height", 1921),
        ("chunk_size", 0),
        ("action_steps", 101),
        ("camera_key", "camera"),
        ("model_id", "not-a-fingerprint"),
    ],
)
def test_incompatible_checkpoint_fails_before_cloud(profile, checkpoint, field, value):
    with pytest.raises(ValueError):
        runner.admit(profile, checkpoint | {field: value})


def context(profile, checkpoint):
    return {
        "schema_version": 1,
        "group_name": "isaac-act-test-bbbbbbbb",
        "rollout_id": "b" * 32,
        "model_id": checkpoint["model_id"],
        "manifest_sha256": "c" * 64,
        "results_prefix": profile.results_uri + "/" + "b" * 32,
    }


def receipt(ctx):
    return {
        "schema_version": 1,
        "skypilot": "0.13.0",
        "group_name": ctx["group_name"],
        "job_id": 42,
        "request_id": "req-1",
        "tasks": [
            {"task_name": "isaac", "task_id": 0, "is_primary_in_job_group": True},
            {"task_name": "vla", "task_id": 1, "is_primary_in_job_group": False},
        ],
    }


def states(rec, primary="SUCCEEDED", auxiliary="CANCELLED"):
    return [
        {
            "job_id": 42,
            "job_name": rec["group_name"],
            "task_id": idx,
            "task_name": name,
            "is_primary_in_job_group": idx == 0,
            "is_job_group": True,
            "execution": "parallel",
            "status": state,
        }
        for idx, (name, state) in enumerate((("isaac", primary), ("vla", auxiliary)))
    ]


def own(profile, checkpoint, directory):
    target = directory / "receipts"
    target.mkdir(parents=True)
    ctx = context(profile, checkpoint)
    rec = receipt(ctx)
    runner._write(target / "launch-context.json", ctx)
    runner._write(target / "submission.json", rec)
    runner._write(directory / "request.json", runner.admit(profile, checkpoint))
    return ctx, rec


def test_observation_rejects_mismatched_or_incomplete_groups(profile, checkpoint):
    rec = receipt(context(profile, checkpoint))
    assert runner._states(rec, states(rec)) == {"isaac": "SUCCEEDED", "vla": "CANCELLED"}
    for key, value in (
        ("job_id", 43),
        ("job_name", "foreign"),
        ("task_id", 9),
        ("is_primary_in_job_group", False),
        ("execution", "serial"),
        ("status", "MAGIC"),
    ):
        rows = states(rec)
        rows[0][key] = value
        with pytest.raises(ValueError):
            runner._states(rec, rows)
    with pytest.raises(ValueError):
        runner._states(rec, states(rec)[:1])


def test_successful_owned_run(profile, checkpoint, tmp_path, monkeypatch):
    directory = tmp_path / "job"
    calls = []
    events = []

    async def command(p, args, **kwargs):
        calls.append(args)
        if "-c" in args:
            return 0, json.dumps(checkpoint).encode()
        if "--receipt-dir" in args:
            ctx, rec = own(profile, checkpoint, directory)
            assert runner._json(directory / "request.json")["model_id"] == ctx["model_id"]
            return 0, b""
        if "observe" in args:
            runner._write(
                directory / "observation.json", states(receipt(context(profile, checkpoint)))
            )
            return 0, b""
        if "--collect" in args:
            runner._write(
                directory / "artifacts.json",
                {
                    "run_id": "d" * 32,
                    "artifacts": [
                        {
                            "path": "artifacts/outputs/video.mp4",
                            "sha256": "e" * 64,
                            "bytes": 10,
                            "generation": "12",
                        }
                    ],
                },
            )
            return 0, b""
        raise AssertionError(args)

    async def event(*args):
        events.append(args)

    monkeypatch.setattr(runner, "_command", command)

    async def execute():
        result = await runner.run(
            profile,
            tmp_path / "checkpoint",
            directory,
            event,
            30,
            expected_profile_sha256=profile.identity_hash(),
        )
        assert result["execution_status"] == "succeeded" and result["task_success"] is None
        assert result["resource_deletion"] == "unverified"
        with pytest.raises(ValueError, match="never resubmit"):
            await runner.run(profile, tmp_path / "checkpoint", directory, event, 30)

    asyncio.run(execute())
    assert not any("cancel" in row for row in calls)
    assert [row[0] for row in events] == ["preparing", "running", "running", "saving"]


def test_changed_profile_prevents_dispatch(profile, tmp_path, monkeypatch):
    identity = profile.identity_hash()
    (profile.runner_root / "workers/isaac_sim/scene.usda").write_text("different")

    async def unexpected(*a, **k):
        raise AssertionError("Must not spawn")

    monkeypatch.setattr(runner, "_command", unexpected)
    with pytest.raises(ValueError, match="changed after"):
        asyncio.run(
            runner.run(
                profile,
                tmp_path,
                tmp_path / "job",
                unexpected,
                30,
                expected_profile_sha256=identity,
            )
        )


@pytest.mark.parametrize("complete", [True, False])
def test_recovery_uses_only_owned_job_or_group(
    profile, checkpoint, tmp_path, monkeypatch, complete
):
    directory = tmp_path / "job"
    ctx, rec = own(profile, checkpoint, directory)
    if not complete:
        (directory / "receipts/submission.json").unlink()
    calls = []

    async def command(p, args, **kwargs):
        calls.append(args)
        return 0, b""

    monkeypatch.setattr(runner, "_command", command)
    result = asyncio.run(runner.recover(profile, directory))
    assert result["status"] == "cancellation_requested"
    assert result["resource_deletion"] == "unverified"
    assert calls[0][-2:] == (["42", "--yes"] if complete else [ctx["group_name"], "--yes"])
    assert "launch" not in calls[0]


def test_recovery_does_not_use_foreign_results_prefix(profile, checkpoint, tmp_path, monkeypatch):
    directory = tmp_path / "job"
    ctx, _ = own(profile, checkpoint, directory)
    ctx["results_prefix"] = "gs://foreign-bucket/results/" + "b" * 32
    runner._write(directory / "receipts/launch-context.json", ctx)

    async def unexpected(*a, **k):
        raise AssertionError("Must not spawn")

    monkeypatch.setattr(runner, "_command", unexpected)
    result = asyncio.run(runner.recover(profile, directory))
    assert result["status"] == "cleanup_unknown"


def test_failed_status_requests_cancellation(profile, checkpoint, tmp_path, monkeypatch):
    directory = tmp_path / "job"
    commands = []

    async def command(p, args, **kwargs):
        commands.append(args)
        if "-c" in args:
            return 0, json.dumps(checkpoint).encode()
        if "--receipt-dir" in args:
            own(profile, checkpoint, directory)
        elif "observe" in args:
            runner._write(
                directory / "observation.json",
                states(receipt(context(profile, checkpoint)), "FAILED"),
            )
        elif "cancel" not in args:
            raise AssertionError(args)
        return 0, b""

    async def event(*args):
        pass

    monkeypatch.setattr(runner, "_command", command)
    with pytest.raises(ValueError, match="execution failed"):
        asyncio.run(runner.run(profile, tmp_path, directory, event, 30))
    assert sum("cancel" in row for row in commands) == 1
    assert not (directory / "report.json").exists()


def test_repeated_cancellation_waits_for_remote_cancel(profile, checkpoint, tmp_path, monkeypatch):
    directory = tmp_path / "job"
    observed = asyncio.Event()
    cancelling = asyncio.Event()
    release = asyncio.Event()
    done = []

    async def command(p, args, **kwargs):
        if "-c" in args:
            return 0, json.dumps(checkpoint).encode()
        if "--receipt-dir" in args:
            own(profile, checkpoint, directory)
        elif "observe" in args:
            observed.set()
            await asyncio.Event().wait()
        elif "cancel" in args:
            cancelling.set()
            await release.wait()
            done.append(True)
        return 0, b""

    async def event(*args):
        pass

    monkeypatch.setattr(runner, "_command", command)

    async def execute():
        task = asyncio.create_task(runner.run(profile, tmp_path, directory, event, 30))
        await observed.wait()
        task.cancel()
        await cancelling.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert done == [True]

    asyncio.run(execute())
    assert not (directory / "report.json").exists()


def test_real_command_output_limit(profile):
    with pytest.raises(ValueError, match="output limit"):
        asyncio.run(
            runner._command(profile, [sys.executable, "-c", "print('x'*200000)"], capture=True)
        )


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group regression")
def test_real_command_timeout_reaps_descendant(profile, tmp_path):
    child = tmp_path / "child.pid"
    source = (
        "import subprocess,sys,time,pathlib; "
        "p=subprocess.Popen([sys.executable,'-c',"
        "'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)"
    )
    with pytest.raises(TimeoutError):
        asyncio.run(
            runner._command(profile, [sys.executable, "-c", source, str(child)], timeout=0.3)
        )
    pid = int(child.read_text())
    # Kernel death/reparenting can lag SIGKILL; a zombie cannot compute.
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        state = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
        ).stdout.strip()
        if not state or state.startswith("Z"):
            break
        time.sleep(0.01)
    assert not state or state.startswith("Z")


@pytest.mark.skipif(os.name != "posix", reason="POSIX FIFO boundary")
def test_record_fifo_is_rejected_without_blocking(tmp_path):
    path = tmp_path / "fifo"
    os.mkfifo(path)
    with pytest.raises(ValueError):
        runner._json(path)


def test_collector_reads_only_owned_generation_bound_files(
    profile, checkpoint, tmp_path, monkeypatch
):
    directory, records, calls = collection_fixture(profile, checkpoint, tmp_path, monkeypatch)
    runner._collect(directory)
    result = runner._json(directory / "artifacts.json")
    assert len(result["artifacts"]) == 4
    assert result["run_id"] == "d" * 32
    for artifact in result["artifacts"]:
        assert artifact["sha256"] == runner._sha(directory / artifact["path"])
    assert all(call[1]["allow_redirects"] is False for call in calls)
    assert all(
        call[0].startswith("https://storage.googleapis.com/storage/v1/b/fixture-results/o")
        for call in calls
    )
    assert all(call[1]["params"].get("generation") == "123" for call in calls[1:])


@pytest.mark.parametrize(
    "fault",
    [
        "foreign-prefix",
        "two-runs",
        "duplicate-object",
        "oversize",
        "truncated",
        "wrong-model",
        "wrong-manifest",
        "redirect",
        "unfinished",
    ],
)
def test_collector_rejects_unowned_or_incomplete_results(
    profile, checkpoint, tmp_path, monkeypatch, fault
):
    directory, records, calls = collection_fixture(
        profile, checkpoint, tmp_path, monkeypatch, fault
    )
    with pytest.raises(ValueError):
        runner._collect(directory)
    assert not (directory / "artifacts.json").exists()


def collection_fixture(profile, checkpoint, tmp_path, monkeypatch, fault=None):
    from types import ModuleType

    directory = tmp_path / "job"
    ctx, rec = own(profile, checkpoint, directory)
    runner._write(directory / "request.json", {"model_id": checkpoint["model_id"]})
    rollout = {
        "status": "succeeded",
        "model_id": checkpoint["model_id"],
        "manifest_sha256": "c" * 64,
    }
    if fault == "wrong-model":
        rollout["model_id"] = "sha256:" + "f" * 64
    if fault == "wrong-manifest":
        rollout["manifest_sha256"] = "f" * 64
    result = {
        "status": "succeeded",
        "run_id": "d" * 32,
        "mode": "experimental",
        "exit_code": 0,
        "rollout_result": rollout,
    }
    if fault == "unfinished":
        result["status"] = "running"
    content = {
        "job-result.json": json.dumps(result).encode(),
        "outputs/result.json": json.dumps(rollout).encode(),
        "outputs/trajectory.jsonl": b"{}\n",
        "outputs/video.mp4": b"generated-test-video-placeholder",
    }
    prefix = "experiments/" + "b" * 32 + "/" + "d" * 32 + "/"
    if fault == "foreign-prefix":
        prefix = "foreign/" + "d" * 32 + "/"
    records = [
        {"name": prefix + name, "generation": "123", "size": str(len(data))}
        for name, data in content.items()
    ]
    if fault == "two-runs":
        records.append(
            {
                "name": "experiments/" + "b" * 32 + "/" + "e" * 32 + "/job-result.json",
                "generation": "1",
                "size": "2",
            }
        )
    if fault == "duplicate-object":
        records.append(dict(records[0]))
    if fault == "oversize":
        records[0]["size"] = str(runner.JSON_LIMIT + 1)
    if fault == "truncated":
        records[0]["size"] = str(int(records[0]["size"]) + 1)
    calls = []

    class Response:
        status_code = 302 if fault == "redirect" else 200

        def __init__(self, data):
            self.data = data

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, size):
            yield self.data

    class Session:
        def __init__(self, credentials):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url, **kwargs):
            from urllib.parse import unquote

            calls.append((url, kwargs))
            if kwargs["params"].get("alt") == "media":
                name = unquote(url.rsplit("/o/", 1)[1]).removeprefix(prefix)
                return Response(content[name])
            return Response(json.dumps({"items": records}).encode())

    google = ModuleType("google")
    auth = ModuleType("google.auth")
    transport = ModuleType("google.auth.transport")
    requests = ModuleType("google.auth.transport.requests")
    google.auth = auth
    auth.transport = transport
    transport.requests = requests
    auth.default = lambda **kwargs: ("fake-fixture", None)
    requests.AuthorizedSession = Session
    for name, module in [
        ("google", google),
        ("google.auth", auth),
        ("google.auth.transport", transport),
        ("google.auth.transport.requests", requests),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    return directory, records, calls


def test_accepted_model_fingerprint_checked_before_launch(
    profile, checkpoint, tmp_path, monkeypatch
):
    async def command(p, args, **kwargs):
        assert "-c" in args
        return 0, json.dumps(checkpoint).encode()

    async def event(*args):
        raise AssertionError("Must not reach launch")

    monkeypatch.setattr(runner, "_command", command)
    with pytest.raises(ValueError, match="checkpoint changed"):
        asyncio.run(
            runner.run(
                profile,
                tmp_path,
                tmp_path / "job",
                event,
                30,
                expected_model_id="sha256:" + "f" * 64,
            )
        )


def test_gcloud_config_stays_server_only(profile, tmp_path):
    profile = replace(profile, gcloud_config=tmp_path / "private-gcloud")
    assert profile.environment()["CLOUDSDK_CONFIG"] == str(tmp_path / "private-gcloud")
    assert "private-gcloud" not in json.dumps(profile.public())
    assert profile.identity_hash() != replace(profile, gcloud_config=None).identity_hash()


def test_nonfinite_json_and_duplicate_keys_rejected(tmp_path):
    path = tmp_path / "record.json"
    for text in ('{"x":1e309}', '{"x":NaN}', '{"x":1,"x":2}'):
        path.write_text(text)
        with pytest.raises(ValueError):
            runner._json(path)


def test_cancel_during_spawn_waits_for_owned_child(profile, monkeypatch):
    async def scenario():
        created = asyncio.Event()
        release = asyncio.Event()
        processes = []
        original = asyncio.create_subprocess_exec

        async def spawn(*args, **kwargs):
            process = await original(*args, **kwargs)
            processes.append(process)
            created.set()
            await release.wait()
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        task = asyncio.create_task(
            runner._command(
                profile, [sys.executable, "-c", "import time; time.sleep(60)"], timeout=30
            )
        )
        await created.wait()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert processes[0].returncode is not None

    asyncio.run(scenario())


def test_receipt_creation_failure_does_not_launch(profile, tmp_path, checkpoint, monkeypatch):
    async def command(p, args, **kwargs):
        assert "-c" in args
        return 0, json.dumps(checkpoint).encode()

    async def event(*args):
        raise AssertionError("Must not launch")

    monkeypatch.setattr(runner, "_command", command)
    monkeypatch.setattr(runner, "_write", lambda *args: (_ for _ in ()).throw(OSError("full disk")))
    with pytest.raises(OSError):
        asyncio.run(runner.run(profile, tmp_path, tmp_path / "job", event, 30))


def test_full_disk_cannot_prevent_owned_cancellation(profile, checkpoint, tmp_path, monkeypatch):
    directory = tmp_path / "job"
    own(profile, checkpoint, directory)
    calls = []

    async def command(p, args, **kwargs):
        calls.append(args)
        return 0, b""

    monkeypatch.setattr(runner, "_command", command)
    monkeypatch.setattr(runner, "_write", lambda *args: (_ for _ in ()).throw(OSError("full disk")))
    result = asyncio.run(runner.recover(profile, directory))
    assert result["status"] == "cancellation_requested"
    assert result["receipt_saved"] is False
    assert result["resource_deletion"] == "unverified"
    assert len(calls) == 1 and calls[0][-2:] == ["42", "--yes"]


@pytest.mark.parametrize("changed", ["profile", "model"])
def test_recovery_refuses_changed_control_or_model_identity(
    profile, checkpoint, tmp_path, monkeypatch, changed
):
    directory = tmp_path / "job"
    own(profile, checkpoint, directory)
    if changed == "profile":
        (profile.skypilot / "config.yaml").write_text("changed endpoint")
    else:
        accepted = runner._json(directory / "request.json")
        accepted["model_id"] = "sha256:" + "f" * 64
        runner._write(directory / "request.json", accepted)

    async def unexpected(*args, **kwargs):
        raise AssertionError("Must not cancel another target")

    monkeypatch.setattr(runner, "_command", unexpected)
    result = asyncio.run(runner.recover(profile, directory))
    assert result["status"] == "cleanup_unknown"
