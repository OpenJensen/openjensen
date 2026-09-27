"""CPU replay adapter protocol fixtures are not model/robot task evidence."""

import asyncio
import copy
import hashlib
import json
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_native_distillation import fixture as _dataset_fixture
from test_native_replay_contract import configured, request
from vla_platform.lifecycle import native_replay as nr
from vla_platform.lifecycle.contracts import LifecycleResult, PolicyArtifact
from vla_platform.lifecycle.native_quantization import inventory, model_identity
from vla_platform.lifecycle.service import Lifecycle

dataset_fixture = _dataset_fixture


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(nr.canonical(value))


def manifest(root, metadata):
    files = {
        p.relative_to(root).as_posix(): digest(p.read_bytes())
        for p in root.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    write(root / "manifest.json", {"schema_version": 1, "metadata": metadata, "files": files})
    return digest((root / "manifest.json").read_bytes())


@pytest.fixture
def fixture(dataset_fixture):
    f = dataset_fixture
    root = f.data_dir / "jobs/packed/artifact"
    policy = root / "policy"
    policy.mkdir(parents=True)
    for name in f.admitted["files"]:
        if name != "model.safetensors":
            (policy / name).write_bytes((f.admitted["path"] / name).read_bytes())
    config = json.loads((policy / "config.json").read_bytes())
    config["use_vae"] = False
    write(policy / "config.json", config)
    (policy / "model.fbq").write_bytes(b"protocol-packed-weights")
    write(
        policy / "encoding.json",
        {
            "schema_version": 1,
            "format": "firebird_quant",
            "format_version": 1,
            "policy_family": "act",
            "weights": "model.fbq",
            "compute_dtype": "float32",
            "runtime": nr.RUNTIME,
            "recipe": {
                "bits": 8,
                "group_size": 64,
                "min_elements": 128,
                "min_ndim": 2,
                "include": [],
                "exclude": [],
            },
        },
    )
    metadata = {
        "architecture": "act",
        "format": "firebird_quant",
        "format_version": 1,
        "model_id": model_identity(inventory(policy)),
        "precision": "int8",
        "policy_subdirectory": "policy",
        "inference_only": True,
        "training_resume_supported": False,
        **nr.FLAGS,
    }
    source = PolicyArtifact(
        id="packed:operation",
        project_id="project",
        job_id="packed",
        label="Protocol packed ACT",
        format="native_quantized",
        path="jobs/packed/artifact",
        manifest_sha256=manifest(root, metadata),
        file_bytes=1,
        metadata=metadata,
    )
    req = request(
        artifact_id=source.id,
        native_replay={
            "adapter": "act-packed-observation-v1",
            "selection": [
                {"episode_index": 0, "frame_index": 0},
                {"episode_index": 2, "frame_index": 1},
            ],
            "coordinate_attestation": "generated_fixture",
            "units": ["generated_fixture"] * 6,
        },
    )
    admitted = nr.source_info(source, f.data_dir)
    data = nr.dataset_info(f.profile, f.data_dir / "dataset-snapshots", admitted, req.native_replay)
    worker = f.root / "workers/isaac_sim"
    for name in (
        "native_replay.py",
        "native_replay_prepare.py",
        "native_replay_runtime.py",
        "native_replay_contracts.py",
        "backend.py",
        "client.py",
        "server.py",
    ):
        path = worker / "sim_worker/rollout" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# protocol fixture")
    path = worker / "sim_worker/policy_http_server.py"
    path.write_text("# protocol fixture")
    for name in ("native_application.py", "native_consumer.py", "native_package.py"):
        path = worker.parent / "firebird_quant/src/firebird_quant" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# protocol fixture")
    runtime = configured(worker)
    data["implementation"] = nr.implementation_identity(runtime)
    lifecycle = SimpleNamespace(
        settings=SimpleNamespace(data_dir=f.data_dir),
        runtime=lambda _: runtime,
        compute=SimpleNamespace(require_enabled=lambda _: None),
        artifact=AsyncMock(return_value=source),
        execution=f.lifecycle.execution,
        event=AsyncMock(),
        publish=AsyncMock(),
    )
    return SimpleNamespace(
        root=f.root,
        data_dir=f.data_dir,
        source=source,
        admitted=admitted,
        data=data,
        profile=f.profile,
        runtime=runtime,
        lifecycle=lifecycle,
        job=SimpleNamespace(id="replay", project_id="project", request=req),
    )


def prepared(f, root):
    root.mkdir(parents=True)
    samples = []
    for index, selected in enumerate(f.data["selection"]):
        raw = bytes([index]) * (32 * 32 * 3)
        name = f"frame-{index:06d}.rgb"
        (root / name).write_bytes(raw)
        ancestry = f.data["lineage"][selected["episode_index"]]
        samples.append(
            {
                **selected,
                "timestamp_seconds": selected["frame_index"] / 30,
                "task": "Generated protocol fixture",
                "state": [0.0] * 6,
                "origin": ancestry["origin"],
                "lineage_group": ancestry["lineage_group"],
                "image": {
                    "file": name,
                    "width": 32,
                    "height": 32,
                    "bytes": len(raw),
                    "sha256": digest(raw),
                },
            }
        )
    doc = {
        "schema_version": 1,
        "format": "native-policy-observations-v1",
        "source": f.data["source"],
        "camera_key": f.admitted["camera"],
        "semantics": f.data["semantics"],
        "samples": samples,
    }
    write(root / "manifest.json", doc)
    files = {
        p.name: {"bytes": p.stat().st_size, "sha256": digest(p.read_bytes())}
        for p in root.iterdir()
    }
    response = {
        "schema_version": 1,
        "path": str(root),
        "manifest_sha256": files["manifest.json"]["sha256"],
        "files": files,
        "observations": len(samples),
        "source": f.data["source"],
        "scope": (
            "Original immutable inputs; policy-coordinate compatibility "
            "is operator-attested for recorded sources"
        ),
        "episode_lengths": [{"episode_index": e, "length": 2} for e in (0, 2)],
    }
    return doc, response


def output(f, root, doc, corpus_files):
    root.mkdir(parents=True)
    records = []
    for index, sample in enumerate(doc["samples"]):
        actions = [[0.0] * 6 for _ in range(100)]
        records.append(
            {
                "sample_index": index,
                "episode_index": sample["episode_index"],
                "frame_index": sample["frame_index"],
                "timestamp_seconds": sample["timestamp_seconds"],
                "input_rgb_sha256": sample["image"]["sha256"],
                "input_state_sha256": digest(nr.canonical(sample["state"])),
                "actions": actions,
                "reset_repeat_exact": True,
                "requests": [
                    {"seconds": 0.1, "actions_sha256": digest(nr.canonical(actions))}
                    for _ in range(2)
                ],
            }
        )
    predictions = {
        "schema_version": 1,
        "model_id": f.admitted["model_id"],
        "versions": nr.RUNTIME,
        "device": "cpu",
        "mode": "independent_observation_replay",
        "records": records,
        "server_closed": True,
        "external_network_disabled": True,
        "floating_master_reads_blocked": True,
        **nr.FLAGS,
    }
    write(root / "predictions.json", predictions)
    write(
        root / "lineage.json",
        {
            "schema_version": 1,
            "source": {
                "files": f.admitted["files"],
                "model_id": f.admitted["model_id"],
                "artifact_id": f.source.id,
                "artifact_manifest_sha256": f.source.manifest_sha256,
            },
            "observation_manifest_sha256": corpus_files["manifest.json"]["sha256"],
            "observation_files": corpus_files,
            "observations": doc,
            "implementation_sha256": f.data["implementation"],
        },
    )
    report = {
        "stage": "native_replay",
        "mode": "independent_observation_replay",
        "model_id": f.admitted["model_id"],
        "device": "cpu",
        "versions": nr.RUNTIME,
        "observation_source": f.data["source"],
        "observations": len(doc["samples"]),
        "action_shape": [100, 6],
        "reset_repeat_exact": True,
        "coordinate_semantics": f.data["semantics"],
        "elapsed_seconds": 0.1,
        "server_closed": True,
        **nr.FLAGS,
        "scope": nr.SCOPE,
    }
    write(root / "report.json", report)
    manifest(
        root,
        {
            "architecture": "act",
            "model_id": f.admitted["model_id"],
            "recipe": "native-observation-replay-v1",
            "device": "cpu",
            **nr.FLAGS,
        },
    )
    return {
        "schema_version": 1,
        "job_id": f.job.id,
        "operation": "policy.run",
        "artifact": {
            "path": str(root),
            "format": "native_run_record",
            "label": "CPU observation replay",
        },
        "report": report,
    }


def check(f, response, doc, corpus_files, root):
    return nr.check_result(response, f.job, f.source, f.admitted, f.data, doc, corpus_files, root)


def test_complete_replay_report_never_promotes_task_success(fixture):
    f = fixture
    doc, res = prepared(f, f.root / "corpus")
    _, files = nr.check_corpus(f.root / "corpus", res, f.data, f.admitted)
    response = output(f, f.root / "result", doc, files)
    metadata, _ = check(f, response, doc, files, f.root / "result")
    assert metadata["metadata"]["task_success"] is None
    assert metadata["metadata"]["quality_verified"] is False


@pytest.mark.parametrize(
    "fault", ["changed", "extra", "master", "link", "registered", "encoding", "boolclaim"]
)
def test_packed_admission_rejects_tamper_or_wrong_representation(fixture, fault):
    f = fixture
    root = f.admitted["path"]
    outer = root.parent
    if fault == "changed":
        (root / "model.fbq").write_bytes(b"changed")
    if fault in {"extra", "master"}:
        (root / ("extra.json" if fault == "extra" else "model.safetensors")).write_bytes(
            b"notallowed"
        )
    if fault == "link":
        (root / "model.fbq").unlink()
        (root / "model.fbq").symlink_to(root / "config.json")
    if fault == "registered":
        f.source.manifest_sha256 = "0" * 64
    if fault == "encoding":
        value = json.loads((root / "encoding.json").read_bytes())
        value["recipe"] = None
        write(root / "encoding.json", value)
    if fault == "boolclaim":
        f.source.metadata["format_version"] = True
    if fault in {"extra", "master", "encoding", "boolclaim"}:
        f.source.manifest_sha256 = manifest(outer, f.source.metadata)
    with pytest.raises((ValueError, OSError)):
        nr.source_info(f.source, f.data_dir)


@pytest.mark.parametrize("fault", ["attestation", "dimension", "names", "outofrange", "budget"])
def test_dataset_admission_is_explicit_and_bounded(fixture, fault):
    f = fixture
    recipe = f.job.request.native_replay.model_copy(deep=True)
    admitted = copy.deepcopy(f.admitted)
    if fault == "attestation":
        recipe.coordinate_attestation = "policy_recorded_coordinates"
    if fault == "dimension":
        admitted["image_shape"] = [3, 64, 32]
    if fault == "names":
        recipe.units = [""] * 6
    if fault == "outofrange":
        recipe.selection[0].frame_index = 100
    if fault == "budget":
        recipe.selection = [
            SimpleNamespace(model_dump=lambda i=i: {"episode_index": 0, "frame_index": i})
            for i in range(32)
        ]
    with pytest.raises(ValueError):
        nr.dataset_info(f.profile, f.data_dir / "dataset-snapshots", admitted, recipe)


@pytest.mark.parametrize(
    "fault",
    [
        "short",
        "badlength",
        "boollength",
        "duplength",
        "missinglength",
        "wrongframe",
        "origin",
        "state",
        "pixel",
        "extra",
    ],
)
def test_prepared_native_frames_and_lengths_checked_even_if_rehashed(fixture, fault):
    f = fixture
    root = f.root / "corpus"
    doc, res = prepared(f, root)
    if fault == "short":
        doc["samples"].pop()
        res["observations"] -= 1
    if fault == "badlength":
        res["episode_lengths"][1]["length"] = 1
    if fault == "boollength":
        res["episode_lengths"][0]["length"] = True
    if fault == "duplength":
        res["episode_lengths"][1]["episode_index"] = 0
    if fault == "missinglength":
        res["episode_lengths"].pop()
    if fault == "wrongframe":
        doc["samples"][0]["frame_index"] = True
    if fault == "origin":
        doc["samples"][0]["origin"] = "recorded"
    if fault == "state":
        doc["samples"][0]["state"][0] = 1e300
    if fault == "pixel":
        (root / "frame-000000.rgb").write_bytes(b"bad")
    if fault == "extra":
        (root / "extra").write_bytes(b"extra")
    write(root / "manifest.json", doc)
    res["manifest_sha256"] = digest((root / "manifest.json").read_bytes())
    res["files"] = {
        p.name: {"sha256": digest(p.read_bytes()), "bytes": p.stat().st_size}
        for p in root.iterdir()
    }
    with pytest.raises(ValueError):
        nr.check_corpus(root, res, f.data, f.admitted)


@pytest.mark.parametrize(
    "fault",
    [
        "quality",
        "task",
        "identity",
        "shape",
        "count",
        "actionhash",
        "finite",
        "worker",
        "source",
        "boolschema",
        "responsepath",
    ],
)
def test_rehashed_result_cannot_promote_or_substitute_evidence(fixture, fault):
    f = fixture
    doc, res = prepared(f, f.root / "corpus")
    _, files = nr.check_corpus(f.root / "corpus", res, f.data, f.admitted)
    root = f.root / "result"
    response = output(f, root, doc, files)
    prediction = json.loads((root / "predictions.json").read_bytes())
    lineage = json.loads((root / "lineage.json").read_bytes())
    if fault == "quality":
        prediction["quality_verified"] = True
    if fault == "task":
        response["report"]["task_success"] = True
    if fault == "identity":
        prediction["model_id"] = "sha256:" + "0" * 64
    if fault == "shape":
        prediction["records"][0]["actions"].pop()
    if fault == "count":
        prediction["records"].pop()
    if fault == "actionhash":
        prediction["records"][0]["requests"][0]["actions_sha256"] = "0" * 64
    if fault == "finite":
        prediction["records"][0]["actions"][0][0] = True
    if fault == "worker":
        lineage["implementation_sha256"] = {}
    if fault == "source":
        lineage["source"]["artifact_id"] = "other"
    if fault == "boolschema":
        response["schema_version"] = True
    if fault == "responsepath":
        response["artifact"]["path"] = str(f.root)
    write(root / "predictions.json", prediction)
    write(root / "lineage.json", lineage)
    write(root / "report.json", response["report"])
    metadata = json.loads((root / "manifest.json").read_bytes())["metadata"]
    manifest(root, metadata)
    with pytest.raises(ValueError):
        check(f, response, doc, files, root)


def test_orchestration_stages_snapshot_and_registers_only_report(fixture, monkeypatch):
    f = fixture

    async def execute(life, runtime, mode, req, res):
        payload = json.loads(req.read_bytes())
        if mode == "prepare":
            assert payload["dataset_snapshot"]["path"] != str(f.data["root"])
            _, response = prepared(f, Path(payload["output_dir"]))
        else:
            inputs = Path(payload["observations"]["path"])
            doc = json.loads((inputs / "manifest.json").read_bytes())
            files = {
                p.name: {"sha256": digest(p.read_bytes()), "bytes": p.stat().st_size}
                for p in inputs.iterdir()
            }
            response = output(f, Path(payload["output_dir"]) / "native-run", doc, files)
        write(res, response)
        return response

    monkeypatch.setattr(nr, "execute", execute)
    result = asyncio.run(nr.run(f.lifecycle, f.job))
    assert result.artifacts[0].format == "native_run_record"
    assert result.artifacts[0].parent_ids == [f.source.id]
    assert result.reports[0]["dataset_job_id"] == f.job.request.dataset_job_id
    assert result.reports[0]["task_success"] is None
    f.lifecycle.publish.assert_awaited_once()


def test_source_mutation_during_cancel_is_not_suppressed(fixture, monkeypatch):
    f = fixture

    async def execute(*args):
        (f.admitted["path"] / "model.fbq").write_bytes(b"changed")
        raise asyncio.CancelledError

    monkeypatch.setattr(nr, "execute", execute)
    with pytest.raises(ValueError):
        asyncio.run(nr.run(f.lifecycle, f.job))
    f.lifecycle.publish.assert_not_awaited()


def test_command_fixed_paths_no_credentials(fixture, monkeypatch):
    f = fixture
    monkeypatch.setenv("OPENROUTER_API_KEY", "private")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "private")
    for mode in ("prepare", "run"):
        argv, cwd, env = nr.command(
            f.runtime, mode, f.root / "request.json", f.root / "result.json"
        )
        assert argv[2].startswith("sim_worker.rollout.native_replay")
        assert env["CUDA_VISIBLE_DEVICES"] == ""
        assert "OPENROUTER_API_KEY" not in env and "GOOGLE_APPLICATION_CREDENTIALS" not in env
    with pytest.raises(ValueError):
        nr.command(f.runtime, "arbitrary", f.root / "r", f.root / "o")


@pytest.mark.parametrize("mode", ["prepare", "run"])
@pytest.mark.parametrize("cancel_window", ["running", "spawn", "cleanup"])
def test_real_owned_child_cancellation_is_reaped_even_when_repeated(
    fixture, monkeypatch, mode, cancel_window
):
    f = fixture
    real_spawn = asyncio.create_subprocess_exec
    seen = []
    entered = None
    release = None

    class Owner:
        act_spawn_owned = Lifecycle.act_spawn_owned
        act_stop_owned = Lifecycle.act_stop_owned

        async def stop(self, process, unused, grace_seconds=5):
            if cancel_window == "cleanup":
                entered.set()
                await release.wait()
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()

    async def launch(*args, **kwargs):
        p = await real_spawn(*args, **kwargs)
        seen.append(p)
        if cancel_window == "spawn":
            entered.set()
            await release.wait()
        return p

    code = "import time;time.sleep(60)" if cancel_window != "cleanup" else "pass"
    monkeypatch.setattr(
        nr, "command", lambda *_: ([sys.executable, "-c", code], str(f.root), dict(os.environ))
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)

    async def scenario():
        nonlocal entered, release
        entered, release = asyncio.Event(), asyncio.Event()
        task = asyncio.create_task(nr.execute(Owner(), f.runtime, mode, f.root / "r", f.root / "o"))
        async with asyncio.timeout(5):
            while not seen:
                await asyncio.sleep(0.01)
            if cancel_window != "running":
                await entered.wait()
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0.01)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert len(seen) == 1 and seen[0].returncode is not None
    with pytest.raises(ProcessLookupError):
        os.kill(seen[0].pid, 0)


def test_one_deadline_during_preparation_never_starts_model(fixture, monkeypatch):
    f = fixture
    f.job.request.timeout_seconds = 0.05
    called = []

    async def admitted(*args):
        return f.runtime, f.source, f.admitted, f.data

    async def execute(*args):
        called.append(args[2])
        await asyncio.sleep(2)

    monkeypatch.setattr(nr, "admission", admitted)
    monkeypatch.setattr(nr, "execute", execute)
    with pytest.raises(TimeoutError):
        asyncio.run(nr.run(f.lifecycle, f.job))
    assert called in ([], ["prepare"])
    f.lifecycle.publish.assert_not_awaited()


def completed(f):
    doc, prepared_response = prepared(f, f.root / "corpus")
    _, corpus_files = nr.check_corpus(f.root / "corpus", prepared_response, f.data, f.admitted)
    root = f.data_dir / "jobs" / f.job.id / "operation/worker-output/native-run"
    response = output(f, root, doc, corpus_files)
    manifest_doc, files = check(f, response, doc, corpus_files, root)
    artifact = PolicyArtifact(
        id=f"{f.job.id}:operation",
        project_id="project",
        job_id=f.job.id,
        label="CPU replay",
        format="native_run_record",
        path=root.relative_to(f.data_dir).as_posix(),
        manifest_sha256=files["manifest.json"]["sha256"],
        file_bytes=sum(item["bytes"] for name, item in files.items() if name != "manifest.json"),
        parent_ids=[f.source.id],
        metadata=manifest_doc["metadata"],
    )
    report = {
        **response["report"],
        "operation": "policy.run",
        "source_artifact_id": f.source.id,
        "source_artifact_manifest_sha256": f.source.manifest_sha256,
        "dataset_job_id": f.job.request.dataset_job_id,
        "dataset_snapshot_id": f.data["descriptor"]["id"],
    }
    f.job.kind = "policy.run"
    f.job.status = "succeeded"
    f.job.result = LifecycleResult(artifacts=[artifact], reports=[report])
    f.lifecycle.artifact = AsyncMock(return_value=artifact)
    f.lifecycle.execution.get = AsyncMock(return_value=f.job)
    return artifact, root


def test_historical_record_full_outputs_survive_original_model_removal(fixture, monkeypatch):
    f = fixture
    artifact, _ = completed(f)
    (f.admitted["path"] / "model.fbq").unlink()
    monkeypatch.setattr(nr, "source_info", lambda *_: pytest.fail("Historical read reopened model"))
    result = asyncio.run(nr.read_record(f.lifecycle, "project", artifact.id))
    assert result["source_kind"] == "generated_fixture"
    assert result["coordinate_names"] == f.data["semantics"]["action_names"]
    assert len(result["records"]) == 2 and len(result["records"][0]["actions"]) == 100
    assert result["records"][1]["frame_index"] == 1


@pytest.mark.parametrize(
    "fault",
    [
        "cross-project",
        "running",
        "failed",
        "format",
        "registration",
        "tamper",
        "missing",
        "saved-report",
        "source",
        "extra",
        "path",
    ],
)
def test_historical_read_fails_closed_on_missing_changed_or_wrong_job(fixture, fault):
    f = fixture
    artifact, root = completed(f)
    if fault == "cross-project":
        f.job.project_id = "other"
    if fault in {"running", "failed"}:
        f.job.status = fault
    if fault == "format":
        artifact.format = "simulation_record"
    if fault == "registration":
        f.job.result.artifacts = []
    if fault == "tamper":
        (root / "predictions.json").write_bytes(b"{}")
    if fault == "missing":
        (root / "predictions.json").unlink()
    if fault == "saved-report":
        f.job.result.reports[0]["coordinate_semantics"]["units"][0] = "wrong"
    if fault == "source":
        f.job.result.reports[0]["source_artifact_id"] = "other"
    if fault == "extra":
        (root / "extra.json").write_bytes(b"{}")
    if fault == "path":
        artifact.path = "../elsewhere"
    with pytest.raises(ValueError):
        asyncio.run(nr.read_record(f.lifecycle, "project", artifact.id))


@pytest.mark.parametrize("fault", ["project", "failed", "metadata-only"])
def test_admission_requires_this_projects_succeeded_snapshot(fixture, fault):
    f = fixture
    dataset = f.lifecycle.execution.get.return_value
    if fault == "project":
        dataset.project_id = "other"
    if fault == "failed":
        dataset.status = "failed"
    if fault == "metadata-only":
        dataset.result.inspection_scope = "metadata"
    with pytest.raises(ValueError):
        asyncio.run(nr.validate(f.lifecycle, "project", f.job.request))
    f.lifecycle.publish.assert_not_awaited()


def test_aggregate_budget_rejected_before_preparer_or_rgb_io(fixture, monkeypatch):
    f = fixture
    manifest = copy.deepcopy(f.data["manifest"])
    manifest["total_frames"] = 64
    manifest["features"][f.admitted["camera"]]["shape"] = [1920, 1920, 3]
    admitted = copy.deepcopy(f.admitted)
    admitted["image_shape"] = [3, 1920, 1920]
    recipe = f.job.request.native_replay.model_copy(deep=True)
    recipe.selection = [
        SimpleNamespace(model_dump=lambda i=i: {"episode_index": 0, "frame_index": i})
        for i in range(32)
    ]
    monkeypatch.setattr(nr, "verify_snapshot", lambda *_: manifest)
    with pytest.raises(ValueError, match="128MiB"):
        nr.dataset_info(f.profile, f.data_dir / "dataset-snapshots", admitted, recipe)


def test_unbounded_integer_measurement_rejected_without_overflow():
    with pytest.raises(ValueError):
        nr.number(10**1000)
