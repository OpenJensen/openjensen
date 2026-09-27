"""Recording preparation tests use generated local metadata, never robot evidence."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from vla_platform.contracts import IntakeRequest


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def recording_request():
    return {
        "source": "local",
        "snapshot_for_training": True,
        "recordings": {
            "schema_version": 1,
            "configuration_sha256": "a" * 64,
            "captures": [
                {
                    "session_id": "b" * 32,
                    "session_sha256": "c" * 64,
                    "episodes": [{"episode_id": "d" * 32, "receipt_sha256": "e" * 64}],
                }
            ],
        },
    }


def test_recording_request_keeps_dataset_operation_and_legacy_wire():
    value = IntakeRequest.model_validate(recording_request())
    assert value.recordings.captures[0].session_id == "b" * 32
    assert value.recordings.timeout_seconds == 600
    assert value.path is None and value.snapshot_for_training
    assert "recordings" not in IntakeRequest(repo_id="local/example").model_dump()
    assert "recordings" not in IntakeRequest(source="local", path="data").model_dump()


@pytest.mark.parametrize(
    "key,value",
    [
        ("source", "huggingface"),
        ("path", "/arbitrary"),
        ("repo_id", "local/override"),
        ("snapshot_for_training", False),
        ("revision", "other"),
    ],
)
def test_recordings_cannot_override_intake_boundary(key, value):
    request = recording_request()
    request[key] = value
    with pytest.raises(ValueError):
        IntakeRequest.model_validate(request)


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", True),
        ("timeout_seconds", True),
        ("timeout_seconds", 59),
        ("timeout_seconds", 1801),
        ("timeout_seconds", "600"),
        ("output", "/arbitrary"),
        ("configuration_sha256", "a" * 64 + "\n"),
    ],
)
def test_recording_recipe_strict_fields(key, value):
    request = recording_request()
    request["recordings"][key] = value
    with pytest.raises(ValueError):
        IntakeRequest.model_validate(request)


def test_recording_selection_uniqueness_and_aggregate_budget():
    request = recording_request()
    request["recordings"]["captures"] *= 2
    with pytest.raises(ValueError):
        IntakeRequest.model_validate(request)
    request = recording_request()
    episode = request["recordings"]["captures"][0]["episodes"][0]
    request["recordings"]["captures"][0]["episodes"] = [episode, copy.deepcopy(episode)]
    with pytest.raises(ValueError):
        IntakeRequest.model_validate(request)


def test_catalog_has_no_private_paths_and_binds_project(tmp_path):
    from vla_platform.datasets.recordings import catalog, load_config, resolve_selection

    config, root = configuration(tmp_path)
    selected = capture(root)
    settings = load_config(config)
    result = catalog(settings, "project")
    assert result["captures"][0]["session_id"] == selected["session_id"]
    assert str(tmp_path) not in json.dumps(result)
    request = recording_request()["recordings"]
    request["configuration_sha256"] = settings.identity
    request["captures"] = [selected]
    from vla_platform.datasets.recording_contracts import RecordingPreparation

    assert resolve_selection(settings, "project", RecordingPreparation(**request))[0][
        "path"
    ] == str(root / "session")
    with pytest.raises(ValueError, match="project"):
        catalog(settings, "other")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    return sha(raw)


def configuration(tmp_path):
    root = tmp_path / "captures"
    root.mkdir()
    worker = tmp_path / "worker/firebird_teaching"
    worker.mkdir(parents=True)
    for name in ("__init__.py", "prepare_dataset.py", "dataset.py", "contracts.py", "journal.py"):
        (worker / name).write_text("# fixed worker fixture\n")
    isaac = worker.parent.parent / "isaac_sim/sim_worker"
    for name in ("__init__.py", "rollout/__init__.py", "rollout/contracts.py"):
        target = isaac / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# fixed contract fixture\n")
    executable = tmp_path / "python"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    config = tmp_path / "configuration.json"
    write(
        config,
        {
            "schema_version": 1,
            "dataset_python": str(executable),
            "worker_root": str(worker.parent),
            "ffmpeg": str(executable),
            "ffprobe": str(executable),
            "projects": [{"project_id": "project", "capture_root": str(root)}],
        },
    )
    return config, root


def capture(root):
    folder = root / "session"
    meta = {
        "schema_version": 1,
        "session_id": "b" * 32,
        "origin": "synthetic",
        "lineage_group": "fixture-scene",
        "controller": "joint_position_targets",
        "state_units": "radians",
        "action_units": "radians",
        "timebase": "simulation_seconds",
        "joint_names": ["joint_0"],
        "camera_key": "observation.images.front",
        "camera_prim": "/World/Camera",
        "scene_sha256": "0" * 64,
        "width": 32,
        "height": 32,
        "fps": 30,
        "physics_hz": 120,
    }
    session_hash = write(folder / "session.json", meta)
    episode_hash = write(
        folder / ("d" * 32) / "episode.json",
        {
            "schema_version": 1,
            "episode_id": "d" * 32,
            "frames": 2,
            "finalized": True,
            "outcome": "unknown",
            "termination": "finish",
            "trajectory_sha256": "a" * 64,
            "events_sha256": "b" * 64,
        },
    )
    return {
        "session_id": meta["session_id"],
        "session_sha256": session_hash,
        "episodes": [{"episode_id": "d" * 32, "receipt_sha256": episode_hash}],
    }


def selection_for(config, selected):
    from vla_platform.datasets.recordings import load_config

    request = recording_request()
    request["recordings"]["configuration_sha256"] = load_config(config).identity
    request["recordings"]["captures"] = [selected]
    return request


def test_unfinalized_episode_is_not_catalogued(tmp_path):
    from vla_platform.datasets.recordings import catalog, load_config

    config, root = configuration(tmp_path)
    capture(root)
    (root / "session" / ("e" * 32)).mkdir()
    assert len(catalog(load_config(config), "project")["captures"][0]["episodes"]) == 1


@pytest.mark.parametrize("change", ["session", "episode", "config", "worker", "executable"])
def test_reviewed_identity_change_rejected(tmp_path, change):
    from vla_platform.datasets.recordings import load_config, resolve_selection

    config, root = configuration(tmp_path)
    selected = capture(root)
    request = IntakeRequest.model_validate(selection_for(config, selected))
    target = {
        "session": root / "session/session.json",
        "episode": root / "session" / ("d" * 32) / "episode.json",
        "config": config,
        "worker": tmp_path / "worker/firebird_teaching/dataset.py",
        "executable": tmp_path / "python",
    }[change]
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        resolve_selection(load_config(config), "project", request.recordings)


@pytest.mark.parametrize(
    "which", ["capture_root", "session", "episode", "metadata", "config", "worker"]
)
def test_catalog_rejects_symlinked_data_and_configuration(tmp_path, which):
    from vla_platform.datasets.recordings import catalog, load_config

    config, root = configuration(tmp_path)
    capture(root)
    node = {
        "capture_root": root,
        "session": root / "session",
        "episode": root / "session" / ("d" * 32),
        "metadata": root / "session/session.json",
        "config": config,
        "worker": tmp_path / "worker",
    }[which]
    saved = tmp_path / "saved"
    node.rename(saved)
    node.symlink_to(saved, target_is_directory=saved.is_dir())
    with pytest.raises((ValueError, OSError)):
        catalog(load_config(config), "project")


def test_venv_python_symlink_is_bound_without_rewriting_invocation(tmp_path):
    from vla_platform.datasets.recordings import executable_identity, load_config

    config, _ = configuration(tmp_path)
    original = tmp_path / "python"
    actual = tmp_path / "python-base"
    original.rename(actual)
    original.symlink_to(actual)
    identity = executable_identity(str(original))
    assert identity["invoked"] == str(original) and identity["target"] == str(actual)
    before = load_config(config).identity
    actual.write_text("#!/bin/sh\nexit 1\n")
    assert load_config(config).identity != before


@pytest.mark.parametrize("alter", ["overlap", "duplicate", "traversal", "schema_bool", "extra"])
def test_configuration_fails_closed(tmp_path, alter):
    from vla_platform.datasets.recordings import load_config

    config, root = configuration(tmp_path)
    value = json.loads(config.read_bytes())
    if alter in {"overlap", "duplicate"}:
        value["projects"].append(
            {
                "project_id": "other" if alter == "overlap" else "project",
                "capture_root": str(root / "nested"),
            }
        )
    elif alter == "traversal":
        value["projects"][0]["capture_root"] += "/../captures"
    elif alter == "schema_bool":
        value["schema_version"] = True
    else:
        value["command"] = "arbitrary"
    write(config, value)
    with pytest.raises(ValueError):
        load_config(config)


@pytest.mark.parametrize(
    "field,value", [("frames", True), ("finalized", 1), ("schema_version", True)]
)
def test_catalog_rejects_loose_finalized_receipt(tmp_path, field, value):
    from vla_platform.datasets.recordings import catalog, load_config

    config, root = configuration(tmp_path)
    capture(root)
    receipt = root / "session" / ("d" * 32) / "episode.json"
    data = json.loads(receipt.read_bytes())
    data[field] = value
    write(receipt, data)
    with pytest.raises(ValueError):
        catalog(load_config(config), "project")


def test_catalog_enforces_metadata_and_entry_bounds(tmp_path, monkeypatch):
    from vla_platform.datasets import recordings as rec

    config, root = configuration(tmp_path)
    capture(root)
    loaded = rec.load_config(config)
    monkeypatch.setattr(rec, "CATALOG_BYTES", 5)
    with pytest.raises(ValueError, match="byte limit"):
        rec.catalog(loaded, "project")
    monkeypatch.setattr(rec, "CATALOG_BYTES", 16384)
    monkeypatch.setattr(rec, "CATALOG_ENTRIES", 1)
    with pytest.raises(ValueError, match="entry limit"):
        rec.catalog(loaded, "project")


def test_no_configuration_reports_setup_not_ready(tmp_path):
    from vla_platform.datasets.recordings import options

    response = options(tmp_path / "missing", "project")
    assert response["configured"] is response["runtime_verified"] is False
    assert str(tmp_path) not in json.dumps(response)


def test_catalog_routes_and_intake_admit_project_identity_before_any_worker(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from vla_platform.api import create_app
    from vla_platform.datasets.recordings import Recordings
    from vla_platform.settings import Settings

    config, root = configuration(tmp_path)
    selected = capture(root)
    started = []

    async def spy(self, job):
        started.append(job)

    monkeypatch.setattr(Recordings, "run", spy)
    with TestClient(
        create_app(Settings(data_dir=tmp_path / "app", recording_config=config))
    ) as client:
        project = client.post(
            "/api/v1/projects", json={"name": "Generated capture fixture"}
        ).json()["id"]
        value = json.loads(config.read_bytes())
        value["projects"][0]["project_id"] = project
        write(config, value)
        options = client.get(f"/api/v1/projects/{project}/recordings/options")
        assert options.status_code == 200 and options.json()["runtime_verified"] is False
        catalog = client.get(f"/api/v1/projects/{project}/recordings")
        assert catalog.status_code == 200 and str(tmp_path) not in catalog.text
        assert catalog.headers["cache-control"] == "no-store"
        assert client.get("/api/v1/projects/absent/recordings").status_code == 404
        body = selection_for(config, selected)
        invalid = copy.deepcopy(body)
        invalid["recordings"]["captures"][0]["session_id"] = "f" * 32
        assert client.post(f"/api/v1/projects/{project}/intakes", json=invalid).status_code == 404
        invalid = copy.deepcopy(body)
        invalid["recordings"]["captures"][0]["session_sha256"] = "f" * 64
        assert client.post(f"/api/v1/projects/{project}/intakes", json=invalid).status_code == 409
        assert client.get(f"/api/v1/projects/{project}/jobs").json() == [] and started == []
        response = client.post(f"/api/v1/projects/{project}/intakes", json=body)
        assert response.status_code == 202
        assert response.json()["kind"] == "dataset.inspect"
        assert (
            response.json()["request"]["recordings"]
            == IntakeRequest.model_validate(body).recordings.model_dump()
        )
        assert len(client.get(f"/api/v1/projects/{project}/jobs").json()) == 1
        assert len(started) == 1
        # Ordinary local intake retains its independent configuration requirement.
        assert (
            client.post(
                f"/api/v1/projects/{project}/intakes", json={"source": "local", "path": "x"}
            ).status_code
            == 422
        )


def test_snapshot_verification_binds_all_bytes_and_descriptor(tmp_path):
    import threading
    import time

    from vla_platform.contracts import DatasetSnapshot
    from vla_platform.datasets.recordings import checked_snapshot
    from vla_platform.datasets.snapshots import MANIFEST, canonical, descriptor

    raw = b"generated dataset bytes"
    manifest = {
        "schema_version": 1,
        "files": [{"path": "meta/info.json", "size": len(raw), "sha256": sha(raw)}],
        "total_episodes": 1,
        "total_frames": 2,
        "lineage_validated": True,
        "warnings": [],
    }
    digest = sha(canonical(manifest))
    folder = tmp_path / digest
    (folder / "meta").mkdir(parents=True)
    (folder / "meta/info.json").write_bytes(raw)
    (folder / MANIFEST).write_bytes(canonical(manifest))
    snapshot = DatasetSnapshot.model_validate(descriptor(manifest, digest))

    def check():
        return checked_snapshot(
            tmp_path,
            snapshot,
            {"meta/info.json": sha(raw)},
            deadline=time.monotonic() + 5,
            stop=threading.Event(),
        )

    assert check() == manifest
    (folder / "meta/info.json").write_bytes(b"changed dataset bytes")
    with pytest.raises(ValueError, match="bytes"):
        check()


def test_inventory_budget_and_cancel_apply_even_to_empty_files(tmp_path):
    import threading
    import time

    from vla_platform.datasets.recordings import inventory

    (tmp_path / "empty").touch()
    stop = threading.Event()
    stop.set()
    with pytest.raises(TimeoutError):
        inventory(tmp_path, deadline=time.monotonic() + 1, stop=stop)
    with pytest.raises(ValueError, match="entry limit"):
        inventory(
            tmp_path, deadline=time.monotonic() + 1, stop=threading.Event(), budget=[0, 50000]
        )


@pytest.mark.parametrize(
    "name,value",
    [
        ("origin", []),
        ("origin", {}),
        ("outcome", []),
        ("outcome", {}),
        ("termination", []),
        ("termination", {}),
    ],
)
def test_malformed_catalog_choices_return_422_not_500(tmp_path, name, value):
    from fastapi.testclient import TestClient
    from vla_platform.api import create_app
    from vla_platform.settings import Settings

    config, root = configuration(tmp_path)
    capture(root)
    node = (
        root / "session/session.json"
        if name == "origin"
        else root / "session" / ("d" * 32) / "episode.json"
    )
    record = json.loads(node.read_bytes())
    record[name] = value
    write(node, record)
    with TestClient(
        create_app(Settings(data_dir=tmp_path / "app", recording_config=config))
    ) as client:
        project = client.post("/api/v1/projects", json={"name": "Malformed fixture"}).json()["id"]
        settings = json.loads(config.read_bytes())
        settings["projects"][0]["project_id"] = project
        write(config, settings)
        response = client.get(f"/api/v1/projects/{project}/recordings")
        assert response.status_code == 422 and str(tmp_path) not in response.text


def test_real_supervised_process_failure_keeps_bounded_private_diagnostic(tmp_path):
    import asyncio
    import os
    import sys
    from types import SimpleNamespace

    from vla_platform.datasets.recordings import RecordingError, Recordings
    from vla_platform.lifecycle.service import Lifecycle
    from vla_platform.settings import Settings

    owner = SimpleNamespace(settings=Settings(data_dir=tmp_path / "app"))
    owner.lifecycle = Lifecycle(owner)
    adapter = Recordings(owner)

    async def run():
        with pytest.raises(RecordingError, match="worker failed"):
            await adapter.execute(
                [sys.executable, "-c", "import sys;sys.stderr.write('x'*20000);sys.exit(7)"],
                tmp_path,
                {"PATH": os.defpath},
                tmp_path / "log.json",
            )

    asyncio.run(run())
    assert len(json.loads((tmp_path / "log.json").read_bytes())["stderr_tail"]) == 4096


def test_execute_repeated_cancel_drains_owned_cleanup_before_return(tmp_path):
    import asyncio
    from types import SimpleNamespace

    from vla_platform.datasets.recordings import Recordings

    async def run():
        started, stop_started, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        class Process:
            stderr = None

            async def wait(self):
                started.set()
                await asyncio.Event().wait()

        class Owner:
            async def act_spawn_owned(self, *args, **kwargs):
                return Process()

            async def act_stop_owned(self, process, **kwargs):
                from vla_platform.lifecycle.simulation import finish_owned

                stop_started.set()
                await finish_owned(release.wait())

        adapter = Recordings(SimpleNamespace(lifecycle=Owner()))

        # A no-data pipe still exercises the bounded drain path.
        async def empty_tail(stream):
            return ""

        from unittest.mock import patch

        with patch("vla_platform.lifecycle.native_replay.error_tail", empty_tail):
            task = asyncio.create_task(adapter.execute([], tmp_path, {}, tmp_path / "log.json"))
            await started.wait()
            task.cancel()
            await stop_started.wait()
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert (tmp_path / "log.json").is_file()

    asyncio.run(run())


def converted_fixture(tmp_path):
    import threading
    import time
    from types import SimpleNamespace

    from vla_platform.datasets.recordings import (
        canonical,
        inventory,
        load_config,
        resolve_selection,
    )

    config, root = configuration(tmp_path)
    selected = capture(root)
    request = IntakeRequest.model_validate(selection_for(config, selected))
    sources = resolve_selection(load_config(config), "project", request.recordings)
    source = root / "session"
    meta = json.loads((source / "session.json").read_bytes())
    receipt = json.loads((source / ("d" * 32) / "episode.json").read_bytes())
    output = tmp_path / "converted"

    def inv(folder):
        return inventory(folder, deadline=time.monotonic() + 5, stop=threading.Event())

    write(
        output / "meta/firebird-lineage.json",
        {
            "schema_version": 1,
            "episodes": [
                {"episode_index": 0, "origin": "synthetic", "lineage_group": "fixture-scene"}
            ],
        },
    )
    write(
        output / "meta/firebird-demonstrations.json",
        {
            "schema_version": 1,
            "writer_upstream_revision": "e595b7902714ba51f91e47523f66f89c5181b649",
            "lerobot_version": "0.6.2",
            "task_success_verified": False,
            "controller": meta["controller"],
            "state_units": meta["state_units"],
            "action_units": meta["action_units"],
            "timebase": meta["timebase"],
            "joint_order": meta["joint_names"],
            "camera_prim": meta["camera_prim"],
            "action_column": "action",
            "requested_action_column": "teaching.requested_action",
            "sources": [
                {
                    "capture_path": str(source),
                    "source_session_id": meta["session_id"],
                    "scene_sha256": meta["scene_sha256"],
                    "lineage_group": meta["lineage_group"],
                    "origin": meta["origin"],
                    "source_files": inv(source),
                }
            ],
            "episodes": [
                {
                    "episode_index": 0,
                    "capture_episode_id": "d" * 32,
                    "source_session_id": meta["session_id"],
                    "reset_id": "d" * 32,
                    "outcome": "unknown",
                    "termination": "finish",
                    "events": [],
                    "source_trajectory_sha256": receipt["trajectory_sha256"],
                }
            ],
        },
    )
    job = SimpleNamespace(id="job-fixture", request=request)
    response = {
        "schema_version": 1,
        "job_id": job.id,
        "configuration_sha256": request.recordings.configuration_sha256,
        "selection_sha256": sha(canonical(request.recordings.model_dump())),
        "files": inv(output),
        "conversion": {
            "status": "finalized",
            "path": str(output),
            "episodes": 1,
            "frames": 2,
            "readback_verified": True,
            "source_preserved": True,
            "task_success_verified": False,
            "sources": 1,
            "lineage_groups": 1,
        },
    }
    return job, sources, output, response, inv


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "schema_bool",
        "job",
        "selection",
        "file",
        "source",
        "frames",
        "success",
        "lineage",
        "units",
        "source_hash",
    ],
)
def test_conversion_response_rechecks_owned_bytes_and_provenance(tmp_path, damage):
    import threading
    import time

    from vla_platform.datasets.recordings import checked_conversion

    job, sources, output, response, inv = converted_fixture(tmp_path)
    if damage == "schema_bool":
        response["schema_version"] = True
    elif damage == "job":
        response["job_id"] = "other"
    elif damage == "selection":
        response["selection_sha256"] = "0" * 64
    elif damage == "file":
        (output / "extra").write_text("unmanifested")
    elif damage == "source":
        (Path(sources[0]["path"]) / "extra").write_text("source changed")
    elif damage == "frames":
        response["conversion"]["frames"] = True
    elif damage == "success":
        response["conversion"]["task_success_verified"] = True
    elif damage in {"lineage", "units", "source_hash"}:
        node = (
            output
            / "meta"
            / ("firebird-lineage.json" if damage == "lineage" else "firebird-demonstrations.json")
        )
        data = json.loads(node.read_bytes())
        if damage == "lineage":
            data["episodes"][0]["lineage_group"] = "invented-independent-group"
        elif damage == "units":
            data["action_units"] = "normalized"
        else:
            data["sources"][0]["source_files"]["session.json"] = "0" * 64
        write(node, data)
        response["files"] = inv(output)

    def check():
        return checked_conversion(
            response, job, sources, output, deadline=time.monotonic() + 5, stop=threading.Event()
        )

    if damage is None:
        _, summary = check()
        assert summary.source_count == summary.lineage_group_count == 1
        assert summary.task_success_verified is False
    else:
        with pytest.raises(ValueError):
            check()


def test_cancelled_read_thread_is_drained_and_does_not_publish():
    import asyncio
    import threading

    from vla_platform.datasets.recordings import checked_io

    started, finished = threading.Event(), threading.Event()

    def reading(*, stop):
        started.set()
        assert stop.wait(5), "test cancellation was not delivered"
        finished.set()
        raise TimeoutError("stopped")

    async def run():
        task = asyncio.create_task(checked_io(reading))
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()

    asyncio.run(run())


def test_actual_job_cancel_reaps_writer_and_never_registers_dataset(tmp_path):
    import os
    import sys
    import time

    from fastapi.testclient import TestClient
    from vla_platform.api import create_app
    from vla_platform.settings import Settings

    config, root = configuration(tmp_path)
    selected = capture(root)
    pidfile = tmp_path / "owned-pid"
    executable = tmp_path / "python"
    executable.write_text(
        f"#!{sys.executable}\nimport os,time\nfrom pathlib import Path\n"
        f"Path({str(pidfile)!r}).write_text(str(os.getpid()))\ntime.sleep(60)\n"
    )
    with TestClient(
        create_app(Settings(data_dir=tmp_path / "app", recording_config=config))
    ) as client:
        project = client.post(
            "/api/v1/projects", json={"name": "Owned cancellation fixture"}
        ).json()["id"]
        value = json.loads(config.read_bytes())
        value["projects"][0]["project_id"] = project
        write(config, value)
        response = client.post(
            f"/api/v1/projects/{project}/intakes", json=selection_for(config, selected)
        )
        assert response.status_code == 202
        job_id = response.json()["id"]
        limit = time.monotonic() + 15
        while not pidfile.exists() and time.monotonic() < limit:
            time.sleep(0.01)
        assert pidfile.exists(), client.get(f"/api/v1/jobs/{job_id}").json()
        pid = int(pidfile.read_text())
        cancelled = client.post(f"/api/v1/jobs/{job_id}/cancel")
        assert cancelled.status_code == 200
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        assert job["status"] == "cancelled" and job["result"] is None
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        assert not (tmp_path / "app/dataset-snapshots").exists()
        assert not (
            tmp_path / "app/jobs" / job_id / "recording-preparation/verification.json"
        ).exists()


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("stop_fails", [False, True])
def test_cleanup_failure_overrides_cancel_and_stalled_drain(tmp_path, cancelled, stop_fails):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import patch

    from vla_platform.datasets.recordings import RecordingError, Recordings

    async def run():
        started, stopping, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        drained = asyncio.Event()

        class Process:
            stderr = None

            async def wait(self):
                started.set()
                if cancelled:
                    await asyncio.Event().wait()
                return 0

        class Owner:
            async def act_spawn_owned(self, *args, **kwargs):
                return Process()

            async def act_stop_owned(self, process, **kwargs):
                stopping.set()
                await release.wait()
                if stop_fails:
                    raise PermissionError("private process details")

        async def stalled_tail(stream):
            try:
                await asyncio.Event().wait()
            finally:
                drained.set()

        adapter = Recordings(SimpleNamespace(lifecycle=Owner()))
        with patch("vla_platform.lifecycle.native_replay.error_tail", stalled_tail):
            task = asyncio.create_task(adapter.execute([], tmp_path, {}, tmp_path / "log.json"))
            await started.wait()
            if cancelled:
                task.cancel()
            await stopping.wait()
            if cancelled:
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
            release.set()
            with pytest.raises(RecordingError, match="cleanup is unverified") as error:
                await task
            assert "private process details" not in str(error.value)
            assert drained.is_set()

    asyncio.run(run())


def test_cleanup_failure_is_visible_on_already_cancelled_saved_job(tmp_path):
    import asyncio
    from types import SimpleNamespace

    from vla_platform.datasets.recordings import RecordingError, Recordings

    async def run():
        job = SimpleNamespace(
            id="owned",
            project_id="project",
            status="queued",
            result=None,
            error=None,
            request=IntakeRequest.model_validate(recording_request()),
        )
        saved = []

        async def get(ident):
            assert ident == job.id
            return job

        async def save(current):
            saved.append((current.status, current.error, current.result))

        owner = SimpleNamespace(
            native_slots=asyncio.Semaphore(1),
            lock=asyncio.Lock(),
            get=get,
            save=save,
            settings=SimpleNamespace(data_dir=tmp_path),
        )
        adapter = Recordings(owner)

        async def interrupted_stage(current, message):
            current.status = "cancelled"
            raise RecordingError(
                "Recording preparation cleanup is unverified; inspect private evidence"
            )

        adapter.stage = interrupted_stage
        await adapter.run(job)
        assert saved[-1][0] == "cancelled"
        assert "cleanup is unverified" in saved[-1][1]
        assert saved[-1][2] is None

    asyncio.run(run())
