"""Catalog visibility cannot grant execution to a missing model adapter."""

import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import insert
from vla_platform.api import create_app
from vla_platform.compute_settings import ComputeSettings, ComputeSettingsUpdate
from vla_platform.contracts import DatasetProfile, IntakeRequest, Job, now
from vla_platform.lifecycle.contracts import PolicyArtifact, PolicyRequest
from vla_platform.lifecycle.runtime import Runtime, RuntimeCatalog
from vla_platform.lifecycle.service import Lifecycle
from vla_platform.lifecycle.training_catalog import TRAINING_MODEL_BY_ID, public_training_models
from vla_platform.local_worker_registry import LocalWorkerRegistry
from vla_platform.settings import Settings
from vla_platform.storage import jobs


def runtime(**overrides):
    return Runtime.model_validate(
        {
            "id": "trainer",
            "label": "Protocol fixture",
            "device": "cuda",
            "worker_root": str(Path(__file__).parent / "fixtures/native_worker"),
            "training_root": str(Path(__file__).parent / "fixtures/native_worker"),
            "training_python": sys.executable,
            "vendor": "fixture",
            "build": "fixture",
            **overrides,
        }
    )


def test_default_worker_cannot_advertise_diagnostic_models_as_dataset_trainers():
    models = public_training_models(RuntimeCatalog(runtimes=[runtime()]))
    assert [model["id"] for model in models if model["available"]] == ["smolvla"]
    assert models[0]["runtime_ids"] == ["trainer"]
    with pytest.raises(ValidationError, match="only SmolVLA"):
        runtime(training_model_ids=["smolvla", "openvla"])


@pytest.mark.parametrize(
    "ids,match",
    [
        (["unknown"], "not registered"),
        (["openvla", "openvla"], "unique"),
    ],
)
def test_operator_adapter_still_requires_registered_pinned_models(ids, match):
    with pytest.raises(ValidationError, match=match):
        runtime(training_module="custom_training.application", training_model_ids=ids)


def test_cpu_training_environment_does_not_make_gpu_models_available():
    models = public_training_models(RuntimeCatalog(runtimes=[runtime(device="cpu")]))
    assert not any(model["available"] for model in models)


def test_custom_operator_module_executes_the_selected_pinned_model(tmp_path):
    trainer = runtime(training_module="custom_training.application", training_model_ids=["openvla"])
    config = tmp_path / "runtimes.json"
    config.write_text(RuntimeCatalog(runtimes=[trainer]).model_dump_json())
    app = create_app(Settings(data_dir=tmp_path / "data", runtime_config=config))
    with TestClient(app) as client:
        client.put("/api/v1/compute-settings", json={"local": {"enabled": True}})
        pid = client.post("/api/v1/projects", json={"name": "Native adapter"}).json()["id"]
        record = Job(
            id="dataset",
            project_id=pid,
            status="succeeded",
            created_at=now(),
            updated_at=now(),
            request=IntakeRequest(repo_id="fixture/dataset"),
            result=DatasetProfile(
                source="huggingface",
                repo_id="fixture/dataset",
                revision="a" * 40,
                format="lerobot_v2",
                total_episodes=2,
                total_frames=20,
                fps=10,
                features={"action": {"shape": [7]}},
                metadata_sha256="b" * 64,
                inspected_at=now(),
                warnings=[],
            ),
        )

        async def seed():
            async with app.state.execution.storage.engine.begin() as connection:
                await connection.execute(
                    insert(jobs).values(
                        id=record.id,
                        project_id=pid,
                        status=record.status,
                        record=record.model_dump(),
                    )
                )

        client.portal.call(seed)
        models = client.get("/api/v1/policy-options").json()["training_models"]
        assert [model["id"] for model in models if model["available"]] == ["openvla"]
        base = {
            "operation": "policy.finetune",
            "runtime_id": "trainer",
            "dataset_job_id": "dataset",
        }
        url = f"/api/v1/projects/{pid}/policy-jobs"
        # A registered module is not an API-controlled command, and one model's
        # adapter does not grant permission to execute every catalog entry.
        assert (
            client.post(url, json={**base, "training_module": "arbitrary.module"}).status_code
            == 422
        )
        missing = client.post(url, json=base)
        assert missing.status_code == 422
        assert "no training adapter for SmolVLA" in missing.json()["detail"]
        model = TRAINING_MODEL_BY_ID["openvla"]
        recipe = {"model_id": model.model_id, "model_revision": model.model_revision}
        assert (
            client.post(
                url,
                json={
                    **base,
                    "training": {**recipe, "model_revision": "main"},
                },
            ).status_code
            == 422
        )
        response = client.post(url, json={**base, "training": recipe})
        assert response.status_code == 202, response.text
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            job = client.get(f"/api/v1/jobs/{response.json()['id']}").json()
            if job["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.025)
        assert job["status"] == "succeeded", job
        assert job["request"]["training"] == recipe
        artifact = job["result"]["artifacts"][0]
        assert artifact["metadata"]["fixture_only"] is True
        assert artifact["metadata"]["base_model"] == {
            "repository": model.model_id,
            "revision": model.model_revision,
        }


def test_catalog_checkpoint_metadata_matches_existing_research_catalog():
    path = (
        Path(__file__).parents[1] / "workers/smolvla_qlora/configs/benchmarks/open_weight_vlas.json"
    )
    reference = json.loads(path.read_text())
    for entry in reference["models"]:
        if entry["id"] in {"smolvla", "pi05"}:
            continue  # Dataset fine-tuning uses the base, not the LIBERO benchmark checkpoint.
        model = TRAINING_MODEL_BY_ID[entry["id"]]
        assert model.model_id == entry["checkpoint"]["source"]
        assert model.model_revision == entry["checkpoint"]["revision"]
        assert model.checkpoint_subdirectory == entry["checkpoint"]["subdirectory"]


@pytest.mark.parametrize("model_id", ["openvla", "smolvla"])
@pytest.mark.parametrize("resume_kind", ["job", "artifact"])
@pytest.mark.parametrize("stored_recipe", [True, False])
def test_resume_preserves_checkpoint_contract_and_null_recipe(
    tmp_path, model_id, resume_kind, stored_recipe
):
    model = TRAINING_MODEL_BY_ID[model_id]
    recipe = {
        "model_id": model.model_id,
        "model_revision": "c" * 40,  # A historical, pinned revision must stay resumable.
        "camera_keys": ["observation.images.top", "observation.images.wrist"],
        "steps": 500,
    }
    original = Job(
        id="prior",
        project_id="project",
        kind="policy.finetune",
        status="interrupted",
        request=PolicyRequest(
            operation="policy.finetune",
            runtime_id="trainer",
            dataset_job_id="dataset",
            training_method="qlora",
            training=recipe,
        ),
        created_at=now(),
        updated_at=now(),
    )
    dataset = SimpleNamespace(
        project_id="project",
        status="succeeded",
        result=DatasetProfile(
            source="huggingface",
            repo_id="fixture/dataset",
            revision="a" * 40,
            format="lerobot_v2",
            total_episodes=2,
            total_frames=20,
            fps=10,
            features={"action": {"shape": [7]}},
            metadata_sha256="b" * 64,
            inspected_at=now(),
            warnings=[],
        ),
    )
    lifecycle = Lifecycle.__new__(Lifecycle)
    lifecycle.settings = SimpleNamespace(data_dir=tmp_path)
    lifecycle.compute = ComputeSettings(tmp_path)
    lifecycle.compute.update(ComputeSettingsUpdate.model_validate({"local": {"enabled": True}}))
    lifecycle.execution = SimpleNamespace(
        get=AsyncMock(
            side_effect=lambda jid: {
                "prior": original,
                "dataset": dataset,
            }.get(jid)
        )
    )
    lifecycle.local_workers = LocalWorkerRegistry(
        tmp_path,
        RuntimeCatalog(
            runtimes=[
                runtime(
                    training_module="custom_training.application",
                    training_model_ids=[model_id],
                )
            ]
        ),
    )
    directory = tmp_path / "jobs/prior/operation/training"
    checkpoint = directory / "checkpoint-000100"
    artifact = PolicyArtifact(
        id="prior:operation",
        project_id="project",
        job_id="prior",
        label="Saved checkpoint",
        format="training_checkpoint",
        path="jobs/prior/operation/bundle",
        manifest_sha256="d" * 64,
        file_bytes=1,
        metadata={
            "method": "qlora",
            "base_model": {"repository": model.model_id, "revision": recipe["model_revision"]},
        },
    )
    if resume_kind == "artifact":
        checkpoint = tmp_path / artifact.path / "checkpoint"
        lifecycle.artifact = AsyncMock(return_value=artifact)
    checkpoint.mkdir(parents=True)
    files = {}
    for name in (
        "stats.json",
        "splits.json",
        "training.pt",
        "probe.safetensors",
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
        "policy/config.json",
    ):
        path = checkpoint / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"Native checkpoint inventory fixture, not model weights")
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if stored_recipe:
        (checkpoint / "recipe.json").write_text(json.dumps({**recipe, "method": "qlora"}))
    if stored_recipe:
        files["recipe.json"] = hashlib.sha256((checkpoint / "recipe.json").read_bytes()).hexdigest()
    (checkpoint / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "step": 100, "files": files})
    )
    if resume_kind == "job":
        (directory / "latest.json").write_text(json.dumps({"checkpoint": checkpoint.name}))
    request = PolicyRequest(
        operation="policy.finetune",
        runtime_id="trainer",
        dataset_job_id="wrong-dataset",
        training=None,
        resume_job_id="prior" if resume_kind == "job" else None,
        artifact_id=artifact.id if resume_kind == "artifact" else None,
    )
    if resume_kind == "job" and not stored_recipe:
        # Discovery now rejects missing recipe evidence rather than selecting an
        # incomplete checkpoint because latest.json happens to point to it.
        with pytest.raises(ValueError, match="completed checkpoint"):
            asyncio.run(lifecycle.validate("project", request))
        return
    asyncio.run(lifecycle.validate("project", request))
    assert request.training is None
    assert request.training_method == "qlora"
    assert request.dataset_job_id == "dataset"
    assert original.request.training == recipe
    if model_id == "openvla":
        lifecycle.local_workers = LocalWorkerRegistry(
            tmp_path, RuntimeCatalog(runtimes=[runtime()])
        )
        with pytest.raises(ValueError, match="no training adapter for OpenVLA"):
            asyncio.run(lifecycle.validate("project", request))


def test_native_cloud_adapter_exposes_executable_profiles_and_full_method():
    from vla_platform.lifecycle.native_profiles import NATIVE_PROFILES

    models = public_training_models(
        RuntimeCatalog(
            runtimes=[
                runtime(execution="skypilot", training_model_ids=["smolvla", *NATIVE_PROFILES])
            ]
        )
    )
    ready = {model["id"]: model for model in models if model["available"]}
    assert set(ready) == {"smolvla", *NATIVE_PROFILES}
    assert ready["act"]["initialization"] == "scratch"
    assert ready["act"]["methods"] == ["full"]
    assert ready["pi05"]["model_revision"]
    assert ready["pi05"]["model_id"].startswith("lerobot/")


@pytest.mark.parametrize("model_id", ["act", "vqbet", "psi0", "vla_jepa"])
@pytest.mark.parametrize("artifact_kind", ["step", "final", "legacy"])
@pytest.mark.parametrize("supplied_recipe", [None, {"steps": 200}])
def test_chained_cloud_resume_uses_checkpoint_recipe_and_camera_lineage(
    tmp_path, model_id, artifact_kind, supplied_recipe
):
    from vla_platform.lifecycle.cloud_storage import encoded, install_descriptor, write_json
    from vla_platform.lifecycle.sky_runner import effective_training_recipe

    model = TRAINING_MODEL_BY_ID[model_id]
    cameras = ["observation.images.front"]
    if model.required_cameras == 2:
        cameras.append("observation.images.wrist")
    recipe = {
        "model_id": model.model_id,
        "model_revision": model.model_revision,
        "camera_keys": cameras,
        "steps": 200,
        "warmup_steps": 10,
        "learning_rate": 0.0001,
        "method": "full",
    }
    first_recipe = {key: value for key, value in recipe.items() if key != "method"}
    if artifact_kind == "legacy":
        first_recipe.pop("camera_keys")  # Old request stored cameras only in artifact metadata.
    first = Job(
        id="first",
        project_id="project",
        kind="policy.finetune",
        status="interrupted",
        request=PolicyRequest(
            operation="policy.finetune",
            runtime_id="trainer",
            dataset_job_id="dataset",
            training_method="full",
            training=first_recipe,
        ),
        created_at=now(),
        updated_at=now(),
    )
    resumed = Job(
        id="resumed",
        project_id="project",
        kind="policy.finetune",
        status="interrupted",
        request=PolicyRequest(
            operation="policy.finetune",
            runtime_id="trainer",
            dataset_job_id="dataset",
            training_method="full",
            training=None,
            artifact_id="first:operation",
        ),
        created_at=now(),
        updated_at=now(),
    )
    features = {
        "action": {"dtype": "float32", "shape": [6]},
        "observation.state": {"dtype": "float32", "shape": [6]},
        **{key: {"dtype": "video", "shape": [32, 32, 3]} for key in cameras},
    }
    dataset = SimpleNamespace(
        project_id="project",
        status="succeeded",
        result=DatasetProfile(
            source="huggingface",
            repo_id="fixture/dataset",
            revision="a" * 40,
            format="lerobot_v2" if model_id == "psi0" else "lerobot_v3",
            total_episodes=2,
            total_frames=20,
            fps=10,
            features=features,
            metadata_sha256="b" * 64,
            inspected_at=now(),
            warnings=[],
        ),
    )
    lifecycle = Lifecycle.__new__(Lifecycle)
    lifecycle.settings = SimpleNamespace(data_dir=tmp_path)
    lifecycle.compute = ComputeSettings(tmp_path)
    lifecycle.compute.update(ComputeSettingsUpdate.model_validate({"local": {"enabled": True}}))
    lifecycle.local_workers = LocalWorkerRegistry(
        tmp_path,
        RuntimeCatalog(
            runtimes=[
                runtime(
                    training_module="custom_training.application",
                    training_model_ids=[model_id],
                    gpu_memory_mib=81920,
                )
            ]
        ),
    )
    lifecycle.execution = SimpleNamespace(
        get=AsyncMock(
            side_effect=lambda key: {"first": first, "resumed": resumed, "dataset": dataset}.get(
                key
            )
        )
    )
    prior = PolicyArtifact(
        id="first:operation",
        project_id="project",
        job_id="first",
        label="Earlier checkpoint",
        format="training_checkpoint",
        path="jobs/first/operation/bundle",
        manifest_sha256="a" * 64,
        file_bytes=1,
        metadata={},
    )
    if artifact_kind != "legacy":
        write_json(tmp_path / prior.path / "checkpoint/recipe.json", recipe)
    name = "checkpoint-000060" if artifact_kind == "step" else "operation"
    path = (
        "jobs/resumed/operation/cloud-checkpoints/checkpoint-000060"
        if artifact_kind == "step"
        else "jobs/resumed/operation/bundle"
    )
    uri = "gs://fixture-bucket/jobs/resumed/operation/" + name
    metadata = {
        "storage": "gcs",
        "remote_uri": uri,
        "step": 60,
        "method": "full",
        "base_model": {"repository": model.model_id, "revision": model.model_revision},
        "camera_keys": cameras,
    }
    files = {"checkpoint/weights.bin": "b" * 64}
    if artifact_kind != "legacy":
        files["checkpoint/recipe.json"] = hashlib.sha256(encoded(recipe)).hexdigest()
    manifest = {"schema_version": 1, "files": files, "metadata": metadata}
    descriptor = {
        "uri": uri,
        "manifest": manifest,
        "file_bytes": 1,
        "manifest_sha256": hashlib.sha256(encoded(manifest)).hexdigest(),
    }
    install_descriptor(tmp_path / path, descriptor)
    selected = PolicyArtifact(
        id="resumed:" + name,
        project_id="project",
        job_id="resumed",
        label="Resumed checkpoint",
        format="training_checkpoint",
        path=path,
        manifest_sha256=descriptor["manifest_sha256"],
        file_bytes=1,
        metadata=metadata,
    )
    if artifact_kind == "step":
        write_json(
            tmp_path / "jobs/resumed/operation/training/checkpoint-000060/recipe.json", recipe
        )
    lifecycle.artifact = AsyncMock(
        side_effect=lambda _, key: {
            prior.id: prior,
            selected.id: selected,
        }[key]
    )
    followup = PolicyRequest(
        operation="policy.finetune",
        runtime_id="trainer",
        dataset_job_id="wrong-dataset",
        artifact_id=selected.id,
        training=supplied_recipe,
    )
    saved, method, dataset_id = asyncio.run(
        lifecycle.resumed_training_contract("project", followup)
    )
    assert saved["camera_keys"] == cameras and saved["steps"] == 200
    assert method == "full" and dataset_id == "dataset"
    asyncio.run(lifecycle.validate("project", followup))
    assert followup.training is None
    assert followup.dataset_job_id == "dataset" and followup.training_method == "full"
    dispatched = effective_training_recipe(
        {"parameters": followup.model_dump(), "artifact": selected.model_dump()}
    )
    assert dispatched["model_id"] == model.model_id
    for overrides in [{"steps": 300}, {"model_id": "lerobot/smolvla_base"}]:
        invalid = followup.model_copy(update={"training": overrides})
        with pytest.raises(ValueError, match="must preserve the saved recipe"):
            asyncio.run(lifecycle.validate("project", invalid))
    if artifact_kind == "step":
        sidecar = tmp_path / "jobs/resumed/operation/training/checkpoint-000060/recipe.json"
        sidecar.write_text("{}")
        with pytest.raises(ValueError, match="differs from its registered manifest"):
            asyncio.run(lifecycle.resumed_training_contract("project", followup))
