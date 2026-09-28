"""Distillation orchestration protocol fixtures contain no ML/robot-quality evidence."""

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
from test_distillation_contract import configured, recipe, request
from vla_platform.contracts import DatasetProfile, DatasetSnapshot
from vla_platform.datasets import snapshots
from vla_platform.lifecycle import native_distillation as nd
from vla_platform.lifecycle.contracts import PolicyArtifact
from vla_platform.lifecycle.native_quantization import inventory, source_info
from vla_platform.lifecycle.service import Lifecycle


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(nd.canonical(value))


def manifest(root, metadata):
    files = {
        p.relative_to(root).as_posix(): digest(p.read_bytes())
        for p in root.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    write(root / "manifest.json", {"schema_version": 1, "metadata": metadata, "files": files})
    return digest((root / "manifest.json").read_bytes())


@pytest.fixture
def fixture(tmp_path):
    data_dir = tmp_path / "data"
    teacher_root = data_dir / "jobs/teacher/artifact"
    teacher = teacher_root / "policy"
    config = {
        "type": "act",
        "use_vae": True,
        "chunk_size": 100,
        "n_action_steps": 100,
        "dim_model": 512,
        "dim_feedforward": 3200,
        "n_encoder_layers": 4,
        "n_decoder_layers": 1,
        "n_heads": 8,
        "dropout": 0.1,
        "replace_final_stride_with_dilation": False,
        "vision_backbone": "resnet18",
        "normalization_mapping": {"ACTION": "MEAN_STD"},
        "input_features": {
            "observation.state": {"shape": [6]},
            "observation.images.front": {"shape": [3, 32, 32]},
        },
        "output_features": {"action": {"shape": [6]}},
    }
    write(teacher / "config.json", config)
    (teacher / "model.safetensors").write_bytes(b"protocol-teacher" * 100)
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        write(teacher / name, {"steps": [{"state_file": "stats.safetensors"}]})
    (teacher / "stats.safetensors").write_bytes(b"protocol-statistics")
    metadata = {"architecture": "act"}
    source = PolicyArtifact(
        id="teacher:operation",
        project_id="project",
        job_id="teacher",
        label="Protocol ACT",
        format="native_checkpoint",
        path="jobs/teacher/artifact",
        manifest_sha256=manifest(teacher_root, metadata),
        file_bytes=1,
        metadata=metadata,
    )
    snap_manifest = {
        "schema_version": 1,
        "format": "lerobot_v3",
        "fps": 30,
        "total_episodes": 3,
        "total_frames": 6,
        "warnings": [],
        "lineage_validated": True,
        "lineage": [
            {"episode_index": i, "lineage_group": f"group{i}", "origin": "synthetic"}
            for i in range(3)
        ],
        "features": {
            "observation.state": {"shape": [6], "names": [f"j{i}" for i in range(6)]},
            "action": {"shape": [6], "names": [f"j{i}" for i in range(6)]},
            "observation.images.front": {"shape": [32, 32, 3]},
        },
        "files": [{"path": "meta/info.json", "size": 3, "sha256": digest(b"{}\n")}],
    }
    raw = snapshots.canonical(snap_manifest)
    snap_sha = digest(raw)
    snaproot = data_dir / "dataset-snapshots" / snap_sha
    (snaproot / "meta").mkdir(parents=True)
    (snaproot / "meta/info.json").write_bytes(b"{}\n")
    (snaproot / snapshots.MANIFEST).write_bytes(raw)
    profile = DatasetProfile.model_construct(
        snapshot=DatasetSnapshot(**snapshots.descriptor(snap_manifest, snap_sha)),
        inspection_scope="complete_snapshot",
    )
    selection = recipe()
    selection.update(coordinate_attestation="generated_fixture", frame_stride=1, steps=2)
    req = request(native_distillation=selection, artifact_id=source.id)
    admitted = source_info(source, data_dir, require_inference=False)
    data = nd.dataset_info(
        profile, data_dir / "dataset-snapshots", admitted, req.native_distillation
    )
    data["episode_lengths"] = {0: 2, 1: 2, 2: 2}
    worker = tmp_path / "workers/policy_distillation"
    for p in [
        worker / "src/firebird_distill/prepare.py",
        worker / "src/firebird_distill/application.py",
        worker.parent / "act_optimizer/src/firebird_act/application.py",
        worker.parent / "smolvla_qlora/src/firebird_vla/local_dataset.py",
    ]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# protocol fixture")
    for name in ("__init__.py", "contracts.py", "runtime.py", "provenance.py"):
        (worker / "src/firebird_distill" / name).write_text("# protocol fixture")
    for name in ("bundle.py", "probe.py", "control_schema.py"):
        (worker.parent / "act_optimizer/src/firebird_act" / name).write_text("# protocol fixture")
    for name in ("control_contract.py", "control_schema.py"):
        (worker.parent / "smolvla_qlora/src/firebird_vla" / name).write_text("# protocol fixture")
    runtime = configured(worker)
    data["implementation"] = nd.implementation_identity(runtime)
    lifecycle = SimpleNamespace(
        settings=SimpleNamespace(data_dir=data_dir),
        runtime=lambda _: runtime,
        compute=SimpleNamespace(require_enabled=lambda _: None),
        artifact=AsyncMock(return_value=source),
        execution=SimpleNamespace(
            get=AsyncMock(
                return_value=SimpleNamespace(
                    project_id="project", status="succeeded", result=profile
                )
            )
        ),
        event=AsyncMock(),
        publish=AsyncMock(),
    )
    job = SimpleNamespace(id="student", project_id="project", request=req)
    return SimpleNamespace(
        root=tmp_path,
        data_dir=data_dir,
        source=source,
        admitted=admitted,
        data=data,
        profile=profile,
        runtime=runtime,
        lifecycle=lifecycle,
        job=job,
    )


def prepared(f, root):
    root.mkdir(parents=True)
    samples = []
    for episode, (split, group) in f.data["selected"].items():
        for frame in range(2):
            name = f"sample-{len(samples):06d}.safetensors"
            raw = f"protocol{episode}-{frame}".encode()
            (root / name).write_bytes(raw)
            samples.append(
                {
                    "file": name,
                    "sha256": digest(raw),
                    "bytes": len(raw),
                    "episode_id": episode,
                    "lineage_group": group,
                    "frame_index": frame,
                    "episode_length": 2,
                    "split": split,
                }
            )
    doc = {
        "schema_version": 1,
        "format": "act-observation-corpus-v1",
        "source": f.data["source"],
        "semantics": f.data["semantics"],
        "camera": f.data["camera"],
        "image_shape": [3, 32, 32],
        "chunk_size": 100,
        "samples": samples,
    }
    write(root / "manifest.json", doc)
    response = {
        "path": str(root),
        "manifest_sha256": digest((root / "manifest.json").read_bytes()),
        "samples": len(samples),
        "source": f.data["source"],
    }
    return doc, response


def output(f, directory, doc, corpus_sha):
    policy = directory / "policy"
    policy.mkdir(parents=True)
    for name in f.admitted["files"]:
        (policy / name).write_bytes((f.admitted["path"] / name).read_bytes())
    (policy / "model.safetensors").write_bytes(b"protocol-student")
    config = json.loads((policy / "config.json").read_text())
    config.update(
        use_vae=False,
        dim_model=256,
        dim_feedforward=1024,
        n_encoder_layers=2,
        n_decoder_layers=1,
        n_heads=4,
        dropout=0.0,
    )
    write(policy / "config.json", config)
    files = inventory(policy)
    metadata = {
        "architecture": "act",
        "recipe": "act-action-distillation-v1",
        "model_id": nd.model_id(policy, files),
        "policy_inventory_sha256": digest(nd.canonical(files)),
        "precision": "fp32",
        "policy_subdirectory": "policy",
        "inference_only": True,
        "training_resume_supported": False,
        "fresh_reload_verified": True,
        "cpu_reload_verified": True,
        "quality_verified": False,
        "calibration_verified": False,
        "speedup_verified": False,
        "isaac_runtime_verified": False,
        "task_success": None,
        "dataset_kind": f.data["source"]["kind"],
        "teacher_artifact_id": f.source.id,
        "teacher_artifact_manifest_sha256": f.source.manifest_sha256,
    }
    predictions = [
        {
            "sample_sha256": s["sha256"],
            "split": s["split"],
            "raw_sha256": "a" * 64,
            "postprocessed_sha256": "b" * 64,
            "queue_and_reset_exact": True,
        }
        for s in doc["samples"]
    ]
    measures = {
        "valid_action_coordinates": 18,
        "teacher_normalized_l1": 0.4,
        "teacher_normalized_rmse": 0.5,
        "demonstration_per_coordinate_l1": [0.3] * 6,
    }
    train = {
        "schema_version": 1,
        "adapter": "act-act-v1",
        "versions": nd.RUNTIME,
        "device": "cpu",
        "dtype": "float32",
        "dataset_kind": f.data["source"]["kind"],
        "steps": 2,
        "training_losses": [0.5, 0.3],
        "gradient_norms": [0.7, 0.4],
        "untuned_student": {s: measures for s in ["train", "validation"]},
        "trained_student": {s: measures for s in nd.SPLITS},
        "selection": "fixed last step; final split assessed only after saving immutable student",
        "teacher_weights_bytes": f.admitted["files"]["model.safetensors"]["bytes"],
        "teacher_inference_tensor_bytes": 1000,
        "student_weights_bytes": files["model.safetensors"]["bytes"],
        "student_parameters": 3,
        "trainable_parameters": 2,
        "teacher_gradients_absent": True,
        "copied_backbone_unchanged": True,
        "action_head_changed": True,
        "policy_files": files,
        "teacher_target_sha256": {s["file"]: "c" * 64 for s in doc["samples"]},
        "predictions": predictions,
        "quality_verified": False,
        "calibration_verified": False,
        "speedup_verified": False,
        "task_success": None,
        "teacher_training_overlap": "unknown; student-only held-out split",
        "training_resume_supported": False,
    }
    write(directory / "training.json", train)
    write(
        directory / "verification.json",
        {
            "schema_version": 1,
            "versions": nd.RUNTIME,
            "policy_files": files,
            "predictions": predictions,
        },
    )
    write(
        directory / "lineage.json",
        {
            "schema_version": 1,
            "teacher": {
                "files": f.admitted["files"],
                "artifact_id": f.source.id,
                "artifact_manifest_sha256": f.source.manifest_sha256,
            },
            "corpus_manifest_sha256": corpus_sha,
            "corpus": doc,
            "recipe": f.job.request.native_distillation.model_dump(include=nd.RECIPE_KEYS),
            "teacher_training_overlap": "unknown",
            "student_splits_disjoint": True,
            "implementation_sha256": f.data["implementation"],
        },
    )
    manifest(directory, metadata)
    return {
        "schema_version": 1,
        "job_id": f.job.id,
        "operation": "policy.distill",
        "artifact": {"path": str(directory), "format": "native_checkpoint"},
        "report": {
            **train,
            "fresh_reload_verified": True,
            "elapsed_seconds": 1.2,
            "scope": "protocol fixture",
        },
    }


def check(f, response, doc, corpus_sha, directory):
    return nd.check_result(
        response, f.job, f.source, f.admitted, f.data, doc, corpus_sha, directory
    )


def test_native_vae_teacher_and_generated_snapshot_admitted(fixture):
    f = fixture
    assert asyncio.run(nd.validate(f.lifecycle, "project", f.job.request)) is None
    with pytest.raises(ValueError, match="inference-only"):
        source_info(f.source, f.data_dir)
    nd.unchanged(f.source, f.admitted, f.data, f.data_dir)


@pytest.mark.parametrize(
    "changes", [{"project_id": "other"}, {"status": "running"}, {"result": None}]
)
def test_dataset_cannot_cross_project_or_use_unfinished_profile(fixture, changes):
    value = fixture.lifecycle.execution.get.return_value
    for k, v in changes.items():
        setattr(value, k, v)
    with pytest.raises(ValueError, match="completed immutable"):
        asyncio.run(nd.validate(fixture.lifecycle, "project", fixture.job.request))


@pytest.mark.parametrize("change", ["group", "attestation", "shape", "names", "small-teacher"])
def test_dataset_rejects_leakage_and_coordinate_guessing(fixture, monkeypatch, change):
    f = fixture
    m = copy.deepcopy(f.data["manifest"])
    config = json.loads((f.admitted["path"] / "config.json").read_text())
    if change == "group":
        m["lineage"][1]["lineage_group"] = m["lineage"][0]["lineage_group"]
    if change == "attestation":
        f.job.request.native_distillation.coordinate_attestation = "teacher_recorded_coordinates"
    if change == "shape":
        m["features"]["observation.images.front"]["shape"] = [64, 64, 3]
    if change == "names":
        m["features"]["action"]["names"] = list(reversed(m["features"]["action"]["names"]))
    if change == "small-teacher":
        config["dim_model"] = 256
        write(f.admitted["path"] / "config.json", config)
    monkeypatch.setattr(nd, "resolve_snapshot", lambda *_: f.data["root"])
    monkeypatch.setattr(nd, "verify_snapshot", lambda *_: m)
    with pytest.raises(ValueError):
        nd.dataset_info(
            f.profile,
            f.data_dir / "dataset-snapshots",
            f.admitted,
            f.job.request.native_distillation,
        )


def test_commands_are_fixed_offline_and_exclude_credentials(fixture, monkeypatch):
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "private")
    monkeypatch.setenv("HF_TOKEN", "private")
    for mode in ["prepare", "application"]:
        argv, _, env = nd.command(fixture.runtime, mode, Path("/request"), Path("/response"))
        assert argv[1:3] == ["-m", "firebird_distill." + mode]
        assert "HF_TOKEN" not in env and "GOOGLE_APPLICATION_CREDENTIALS" not in env
        assert env["CUDA_VISIBLE_DEVICES"] == "" and env["OMP_NUM_THREADS"] == "1"
    with pytest.raises(ValueError):
        nd.command(fixture.runtime, "other", Path("/r"), Path("/o"))


@pytest.mark.parametrize(
    "change", ["bytes", "split", "float", "omit-frame", "extra", "source", "shape"]
)
def test_preparation_receipt_must_bind_all_selected_bytes_and_rows(fixture, change):
    f = fixture
    root = f.root / "corpus"
    doc, res = prepared(f, root)
    if change == "bytes":
        (root / doc["samples"][0]["file"]).write_bytes(b"changed")
    elif change == "extra":
        (root / "extra").write_bytes(b"extra")
    else:
        if change == "split":
            doc["samples"][0]["split"] = "final"
        if change == "float":
            doc["samples"][0]["frame_index"] = 0.0
        if change == "omit-frame":
            row = doc["samples"].pop()
            (root / row["file"]).unlink()
            res["samples"] -= 1
        if change == "source":
            doc["source"] = {**doc["source"], "identity": "other"}
        if change == "shape":
            doc["image_shape"] = [True, 32, 32]
        write(root / "manifest.json", doc)
        res["manifest_sha256"] = digest((root / "manifest.json").read_bytes())
    with pytest.raises(ValueError):
        nd.check_corpus(root, res, f.data, f.job.request.native_distillation)


def test_complete_package_is_accepted_with_fixed_native_identity(fixture):
    f = fixture
    doc, res = prepared(f, f.root / "corpus")
    nd.check_corpus(f.root / "corpus", res, f.data, f.job.request.native_distillation)
    directory = f.root / "student"
    response = output(f, directory, doc, res["manifest_sha256"])
    result, _ = check(f, response, doc, res["manifest_sha256"], directory)
    assert result["metadata"]["quality_verified"] is False
    assert result["metadata"]["model_id"].startswith("sha256:")


@pytest.mark.parametrize(
    "change",
    [
        "nan",
        "quality",
        "missing-final",
        "wrong-count",
        "incomplete-reload",
        "bool-version",
        "bool-lineage",
        "processor",
        "wrong-model",
        "wrong-source",
        "not-smaller",
        "coerced-shape",
    ],
)
def test_rehashed_but_invalid_worker_output_cannot_be_registered(fixture, change):
    f = fixture
    doc, res = prepared(f, f.root / "corpus")
    directory = f.root / "student"
    response = output(f, directory, doc, res["manifest_sha256"])
    report = response["report"]
    m = json.loads((directory / "manifest.json").read_text())
    metadata = m["metadata"]
    if change == "nan":
        report["training_losses"][0] = float("nan")
    if change == "quality":
        report["quality_verified"] = True
    if change == "missing-final":
        report["trained_student"].pop("final")
    if change == "wrong-count":
        report["trained_student"]["final"]["valid_action_coordinates"] = 6
    if change == "incomplete-reload":
        report["predictions"].pop()
    if change == "bool-version":
        response["schema_version"] = True
    if change == "bool-lineage":
        p = directory / "lineage.json"
        lineage = json.loads(p.read_text())
        lineage["schema_version"] = True
        write(p, lineage)
    if change == "processor":
        (directory / "policy/stats.safetensors").write_bytes(b"replaced")
    if change == "wrong-model":
        metadata["model_id"] = "sha256:" + "0" * 64
    if change == "wrong-source":
        metadata["teacher_artifact_id"] = "other"
    if change == "not-smaller":
        report["teacher_inference_tensor_bytes"] = 1
    if change == "coerced-shape":
        p = directory / "policy/config.json"
        c = json.loads(p.read_text())
        c["n_heads"] = 4.0
        write(p, c)
    if change != "nan":
        write(
            directory / "training.json",
            {
                k: v
                for k, v in report.items()
                if k not in {"fresh_reload_verified", "elapsed_seconds", "scope"}
            },
        )
    manifest(directory, metadata)
    with pytest.raises(ValueError):
        check(f, response, doc, res["manifest_sha256"], directory)


def test_orchestration_stages_snapshot_and_publishes_only_validated_artifact(fixture, monkeypatch):
    f = fixture

    async def execute(_life, _runtime, mode, req, res):
        value = json.loads(req.read_text())
        if mode == "metadata":
            response = {
                "schema_version": 1,
                "snapshot_id": f.data["descriptor"]["id"],
                "snapshot_manifest_sha256": f.data["descriptor"]["manifest_sha256"],
                "episode_lengths": [{"episode_id": e, "length": 2} for e in [0, 1, 2]],
            }
        elif mode == "prepare":
            assert value["dataset_snapshot"]["path"] != str(f.data["root"])
            _, response = prepared(f, Path(value["output_dir"]))
        else:
            assert 1 <= value["timeout_seconds"] <= 600
            doc = json.loads((Path(value["dataset"]["path"]) / "manifest.json").read_text())
            response = output(
                f,
                Path(value["output_dir"]) / "distilled-policy",
                doc,
                value["dataset"]["manifest_sha256"],
            )
        write(res, response)
        return response

    monkeypatch.setattr(nd, "execute", execute)
    result = asyncio.run(nd.run(f.lifecycle, f.job))
    assert result.artifacts[0].parent_ids == [f.source.id]
    assert result.reports[0]["dataset_snapshot_id"] == f.data["descriptor"]["id"]
    assert result.reports[0]["stage"] == "distilling"
    f.lifecycle.publish.assert_awaited_once()


@pytest.mark.parametrize("mode", ["prepare", "application"])
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
        nd, "command", lambda *_: ([sys.executable, "-c", code], str(f.root), dict(os.environ))
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)

    async def scenario():
        nonlocal entered, release
        entered, release = asyncio.Event(), asyncio.Event()
        task = asyncio.create_task(nd.execute(Owner(), f.runtime, mode, f.root / "r", f.root / "o"))
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


def test_deadline_during_preparation_never_starts_training(fixture, monkeypatch):
    f = fixture
    f.job.request.timeout_seconds = 0.05
    called = []

    async def execute(*args):
        called.append(args[2])
        await asyncio.sleep(2)

    monkeypatch.setattr(nd, "execute", execute)
    with pytest.raises(TimeoutError):
        asyncio.run(nd.run(f.lifecycle, f.job))
    assert called in ([], ["metadata"])  # The one deadline also covers admission/staging.
    f.lifecycle.publish.assert_not_awaited()


@pytest.mark.parametrize("version", ["2.11.0+cpu", "2.11.0+cu128"])
def test_runtime_wheel_variant_retained_and_exact_in_fresh_reload(fixture, version):
    f = fixture
    doc, prepared_response = prepared(f, f.root / "corpus")
    directory = f.root / "student"
    response = output(f, directory, doc, prepared_response["manifest_sha256"])
    response["report"]["versions"] = {**nd.RUNTIME, "torch": version}
    write(
        directory / "training.json",
        {
            k: v
            for k, v in response["report"].items()
            if k not in {"fresh_reload_verified", "elapsed_seconds", "scope"}
        },
    )
    verification = json.loads((directory / "verification.json").read_text())
    verification["versions"] = response["report"]["versions"]
    write(directory / "verification.json", verification)
    metadata = json.loads((directory / "manifest.json").read_text())["metadata"]
    manifest(directory, metadata)
    check(f, response, doc, prepared_response["manifest_sha256"], directory)
    verification["versions"] = nd.RUNTIME
    write(directory / "verification.json", verification)
    manifest(directory, metadata)
    with pytest.raises(ValueError, match="Fresh-process"):
        check(f, response, doc, prepared_response["manifest_sha256"], directory)


def test_consistently_truncated_corpus_rejected_against_native_source_lengths(fixture):
    f = fixture
    root = f.root / "corpus"
    doc, response = prepared(f, root)
    retained = []
    for row in doc["samples"]:
        if row["frame_index"] == 0:
            row["episode_length"] = 1
            retained.append(row)
        else:
            (root / row["file"]).unlink()
    for i, row in enumerate(retained):
        target = f"sample-{i:06d}.safetensors"
        (root / row["file"]).rename(root / target)
        row["file"] = target
    doc["samples"] = retained
    write(root / "manifest.json", doc)
    response.update(samples=3, manifest_sha256=digest((root / "manifest.json").read_bytes()))
    with pytest.raises(ValueError, match="episode/split"):
        nd.check_corpus(root, response, f.data, f.job.request.native_distillation)


@pytest.mark.parametrize("fault", ["bool", "duplicate", "missing", "other-source", "too-long"])
def test_native_metadata_receipt_selection_and_bounds(fixture, fault):
    f = fixture
    response = {
        "schema_version": 1,
        "snapshot_id": f.data["descriptor"]["id"],
        "snapshot_manifest_sha256": f.data["descriptor"]["manifest_sha256"],
        "episode_lengths": [{"episode_id": e, "length": 2} for e in [0, 1, 2]],
    }
    assert nd.check_lengths(response, f.data) == {0: 2, 1: 2, 2: 2}
    if fault == "bool":
        response["episode_lengths"][0]["length"] = True
    if fault == "duplicate":
        response["episode_lengths"][1]["episode_id"] = 0
    if fault == "missing":
        response["episode_lengths"].pop()
    if fault == "other-source":
        response["snapshot_id"] = "sha256:" + "f" * 64
    if fault == "too-long":
        response["episode_lengths"][0]["length"] = 6
    with pytest.raises(ValueError):
        nd.check_lengths(response, f.data)
