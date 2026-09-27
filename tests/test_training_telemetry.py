"""Training observations remain inspectable without a GPU or external cloud calls."""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import insert
from test_sky_lifecycle import PROJECT, finished, request, write_result
from test_sky_lifecycle import cloud as cloud
from vla_platform.api import create_app
from vla_platform.contracts import Job, now
from vla_platform.lifecycle import telemetry
from vla_platform.lifecycle.contracts import PolicyRequest
from vla_platform.lifecycle.service import Lifecycle
from vla_platform.settings import Settings
from vla_platform.storage import jobs


def test_cloud_metrics_and_resolved_recipe_survive_runtime_change_and_reload(cloud):
    state, runner, workspace = cloud
    snapshots = []

    async def exercise():
        async with workspace() as execution:

            async def run(payload, stage_dir, target, on_event):
                async def emit(value):
                    await on_event("_telemetry:" + json.dumps(value))

                await emit(
                    {
                        "phase": "preparing",
                        "message": "Training recipe resolved",
                        "recipe": {
                            "steps": 20,
                            "seed": 17,
                            "model_id": "model/pinned",
                            "model_revision": "c" * 40,
                            "output_dir": "/private",
                        },
                        "environment": {"torch": "fixture", "api_key": "never-return"},
                    }
                )
                await emit(
                    {
                        "phase": "training",
                        "step": 4,
                        "total_steps": 20,
                        "train_loss": 0.8,
                        "learning_rate": 0.001,
                        "elapsed_seconds": 10,
                    }
                )
                await emit(
                    {
                        "phase": "training",
                        "step": 8,
                        "total_steps": 20,
                        "train_loss": 0.4,
                        "validation_loss": 0.5,
                        "learning_rate": 0.0005,
                        "elapsed_seconds": 18,
                    }
                )
                await emit(
                    {
                        "phase": "validation",
                        "step": 8,
                        "message": "Evaluating held-out episodes",
                        "elapsed_seconds": 18,
                    }
                )
                current = await execution.get(payload["job_id"])
                snapshots.append(await telemetry.snapshot(execution.lifecycle, current))
                await emit({"checkpoint_saved": "checkpoint-000008", "step": 8})
                (stage_dir / "worker.log").write_text(
                    "Stockout in first zone\nBearer private-token\nhf_abcdef\n"
                )
                write_result(payload, stage_dir)
                return 0

            runner.run = run
            job = await execution.submit(PROJECT, request(training={"steps": 20}))
            done = await finished(execution, job)
            assert done.status == "succeeded"
            state["runtime"].label = "Changed after acceptance"
            execution.lifecycle = Lifecycle(execution)
            restored = await telemetry.snapshot(execution.lifecycle, done)
            assert restored.completed_steps == 8  # Never pretend success observed all 20 steps.
            assert restored.percent == 40
            assert restored.eta_seconds is None
            assert restored.phase == "completed"
            assert restored.latest.validation_loss == 0.5
            assert restored.checkpoints[0].step == 8
            config = restored.reproducibility
            assert config["recipe_source"] == "worker_resolved"
            assert config["recipe"]["seed"] == 17
            assert config["runtime"]["label"] != "Changed after acceptance"
            assert config["dataset"]["revision"] == "a" * 40
            assert config["model"]["revision"] == "c" * 40
            assert "never-return" not in json.dumps(config)
            assert "/private" not in json.dumps(config)
            assert "private-token" not in str(restored.logs)
            assert "hf_abcdef" not in str(restored.logs)

    asyncio.run(exercise())
    live = snapshots[0]
    assert live.phase == "validation"
    assert live.current_action == "Evaluating held-out episodes"
    assert live.percent == 40
    assert live.elapsed_seconds == 18
    assert live.eta_seconds == 24
    assert live.latest.train_loss == 0.4
    assert live.latest.timestamp


def test_failed_setup_keeps_unknown_optimizer_progress_and_original_recipe(cloud):
    _, runner, workspace = cloud

    async def run(payload, stage_dir, target, on_event):
        raise RuntimeError("Dataset download unavailable")

    runner.run = run

    async def exercise():
        async with workspace() as execution:
            job = await execution.submit(PROJECT, request(training={"steps": 9}))
            done = await finished(execution, job)
            result = await telemetry.snapshot(execution.lifecycle, done)
            assert result.phase == "failed"
            assert result.total_steps == 9
            assert result.completed_steps is None
            assert result.percent is None
            assert result.eta_seconds is None
            assert result.latest is None
            assert result.reproducibility["recipe_source"] == "accepted_request"

    asyncio.run(exercise())


def test_api_reads_per_step_metrics_partial_writes_and_downloads_reproducibility(tmp_path):
    settings = Settings(data_dir=tmp_path)
    with TestClient(create_app(settings)) as client:
        execution = client.app.state.execution
        job = Job(
            id="training-fixture",
            project_id="project",
            kind="policy.finetune",
            status="running",
            created_at=now(),
            updated_at=now(),
            request=PolicyRequest(
                operation="policy.finetune",
                runtime_id="missing-old-runtime",
                dataset_job_id="old-dataset",
                training={"steps": 10},
            ),
        )

        async def insert_job():
            async with execution.storage.engine.begin() as connection:
                await connection.execute(
                    insert(jobs).values(
                        id=job.id,
                        project_id=job.project_id,
                        status=job.status,
                        record=job.model_dump(),
                    )
                )

        client.portal.call(insert_job)
        training = tmp_path / "jobs" / job.id / "operation/training"
        training.mkdir(parents=True)
        (training / "metrics.jsonl").write_text(
            '{"step": 2, "train_loss": 0.7, "elapsed_seconds": 5}\n'
            '{"step": 3, "train_loss": 0.6, "elapsed_seconds": 7}\n{"step":'
        )
        result = client.get(f"/api/v1/jobs/{job.id}/training")
        assert result.status_code == 200, result.text
        result = result.json()
        assert result["percent"] == 30
        assert result["latest"]["timestamp"] is None
        assert result["latest"]["train_loss"] == 0.6
        assert result["eta_seconds"] == 14
        assert result["reproducibility"]["runtime"] is None
        download = client.get(f"/api/v1/jobs/{job.id}/training/reproducibility")
        assert download.status_code == 200
        assert "attachment" in download.headers["content-disposition"]
        assert download.json()["request"]["training"] == {"steps": 10}
        assert client.get("/api/v1/jobs/unknown/training").status_code == 404
        (training / "metrics.jsonl").unlink()
        (training.parent / "worker.log").write_text(
            'SkyPilot: {"step": 5, "checkpoint_saved": "checkpoint-000005"}\n'
            'SkyPilot: {"step": 10, "train_loss": 0.138, "elapsed_seconds": 143.99}\n'
        )
        legacy = client.get(f"/api/v1/jobs/{job.id}/training").json()
        assert legacy["completed_steps"] == 10
        assert legacy["latest"]["train_loss"] == 0.138
        assert legacy["latest"]["elapsed_seconds"] == 143.99
        assert legacy["checkpoints"][0]["step"] == 5
        assert not (training.parent.parent / "training-metrics.jsonl").exists()
        # Old saved events/errors must get the same protection as newly observed text.
        sensitive = "Authorization: Bearer example-secret hf_exampletoken"
        job.status = "failed"
        job.error = sensitive
        client.portal.call(execution.save, job)
        (training.parent.parent / "events.jsonl").write_text(
            json.dumps(
                {
                    "sequence": 1,
                    "stage": "failed",
                    "timestamp": now(),
                    "message": sensitive,
                    "data": {
                        "phase": "failed",
                        "message": sensitive,
                        "Authorization": "private-auth",
                    },
                }
            )
            + "\n"
        )
        original = download.json()
        original.pop("accepted_runtime", None)
        original["runtime"] = {
            "training_image": "worker:v1",
            "conversion_python": "/private/python",
            "evaluation_python": "/private/evaluation-python",
        }
        (training.parent.parent / "reproducibility.json").write_text(json.dumps(original))
        (training.parent / "request.json").write_text(
            json.dumps(
                {
                    "runtime": {
                        "training_image": "worker:v2",
                        "conversion_vendor": "/private/vendor",
                        "evaluation_python": "/private/evaluation-python",
                        "Authorization": "private-auth",
                        "notes": [sensitive],
                    }
                }
            )
        )
        (training.parent / "worker.log").write_text(sensitive)
        secured = client.get(f"/api/v1/jobs/{job.id}/training")
        downloaded = client.get(f"/api/v1/jobs/{job.id}/training/reproducibility")
        events = client.get(f"/api/v1/jobs/{job.id}/events")
        for response in (secured, downloaded, events):
            assert response.status_code == 200
            for private in ("example-secret", "hf_exampletoken", "private-auth", "/private/"):
                assert private not in response.text
        provenance = downloaded.json()
        assert provenance["runtime_source"] == "worker_request"
        assert provenance["runtime"]["training_image"] == "worker:v2"
        assert provenance["accepted_runtime"]["training_image"] == "worker:v1"
        parsed = telemetry.parse_observation(json.dumps({"phase": "failed", "message": sensitive}))
        assert "example-secret" not in str(parsed)
        assert "hf_exampletoken" not in str(parsed)


@pytest.mark.parametrize("execution_kind", ["native", "skypilot"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", -1),
        ("seed", True),
        ("seed", 2147483648),
        ("eval_every", 0),
        ("validation_fraction", 1),
        ("validation_fraction", 0),
        ("validation_fraction", float("inf")),
        ("gradient_accumulation_steps", 1.5),
        ("learning_rate", -0.01),
        ("learning_rate", float("nan")),
    ],
)
def test_invalid_training_form_rejected_before_worker_or_allocation(
    cloud, execution_kind, field, value
):
    state, _, workspace = cloud
    state["runtime"].execution = execution_kind

    async def exercise():
        async with workspace() as execution:
            with pytest.raises(ValueError):
                await execution.submit(
                    PROJECT, request(training={"steps": 100, "warmup_steps": 0, field: value})
                )
            assert not any(job.kind == "policy.finetune" for job in await execution.list(PROJECT))
            assert state["native_calls"] == 0

    asyncio.run(exercise())


def test_invalid_or_nonfinite_diagnostics_do_not_become_optimizer_metrics():
    assert telemetry.parse_observation("GPU ready") is None
    assert telemetry.parse_observation('{"phase": {"details": "download"}}') is None
    assert telemetry.parse_observation('{"step": true, "train_loss": 1}') is None
    value = telemetry.parse_observation(
        'SkyPilot: {"step": 2, "train_loss": NaN, "learning_rate": 0.001}'
    )
    assert value == {"step": 2, "learning_rate": 0.001}
