"""Scene orchestration failures must never turn into simulator acceptance."""

import asyncio
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from vla_platform.api import create_app
from vla_platform.contracts import IntakeRequest, Job, now
from vla_platform.datasets.inspect import profile
from vla_platform.scenes import service
from vla_platform.scenes.agent import SceneError, command
from vla_platform.scenes.contracts import Proposal, Review, SceneRequest, SceneRun
from vla_platform.scenes.models import catalog
from vla_platform.settings import Settings


def source(project="project"):
    intake = IntakeRequest(repo_id=service.SOURCE)
    raw = json.dumps(
        {
            "codebase_version": "v3.0",
            "total_episodes": 30,
            "total_frames": 4500,
            "fps": 30,
            "features": {
                "observation.images.front": {"dtype": "video", "shape": [1080, 1920, 3]},
                "observation.state": {"dtype": "float32", "shape": [6]},
                "action": {"dtype": "float32", "shape": [6]},
            },
        }
    ).encode()
    return Job(
        id="source",
        project_id=project,
        status="succeeded",
        request=intake,
        result=profile(raw, intake, service.REVISION),
        created_at=now(),
        updated_at=now(),
    )


def body(**changes):
    return SceneRequest(request_id="a" * 32, source_job_id="source", **changes)


def proposal():
    template = (
        Path(__file__).resolve().parents[1]
        / "workers/isaac_sim/scenes/so101-pickup/scene_config.json"
    )
    config = json.loads(template.read_text())
    values = {
        key: {field: config[key][field] for field in model.model_fields}
        for key, model in (
            ("camera", __import__("vla_platform.scenes.contracts", fromlist=["Camera"]).Camera),
            ("robot", __import__("vla_platform.scenes.contracts", fromlist=["Robot"]).Robot),
            ("cup", __import__("vla_platform.scenes.contracts", fromlist=["Cup"]).Cup),
            ("box", __import__("vla_platform.scenes.contracts", fromlist=["Box"]).Box),
        )
    }
    return Proposal(
        matches_template=True,
        **values,
        observations=["Thin cup and shelf visible."],
        uncertainties=["Metric dimensions unmeasured."],
    )


def setup(tmp_path, monkeypatch):
    scenes = service.Scenes(tmp_path)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.delenv("JENSEN_SCENE_MODEL", raising=False)
    monkeypatch.setattr(scenes, "options", lambda: {"configured": True, **catalog()})
    monkeypatch.setenv("JENSEN_SCENE_PYTHON", "fixture-python")
    calls = []

    async def evidence(run, *args):
        folder = scenes.folder(run.id)
        frame = folder / "frame-0.jpg"
        frame.write_bytes(b"fixture")
        service.write(folder / "evidence.json", {"revision": service.REVISION})
        run.artifacts = ["evidence.json", "frame-0.jpg"]
        run.gates["source_evidence"] = True
        return [frame], {}

    async def role(work, name, schema, *args, **kwargs):
        calls.append(name)
        result = (
            proposal()
            if schema is Proposal
            else Review(
                supported=True,
                consistent_with_images=True,
                issues=[],
                uncertainties=["Runtime unchecked."],
            )
        )
        service.write(work / f"{name}.json", result.model_dump())
        return result, {"input_tokens": 10, "output_tokens": 5}

    async def build(args, **kwargs):
        bundle = Path(args[-2])
        bundle.mkdir()
        (bundle / "scene.usda").write_text("#usda 1.0")
        service.write(bundle / "structure.json", {"passed": True, "errors": []})
        return b""

    monkeypatch.setattr(scenes, "evidence", evidence)
    monkeypatch.setattr(service, "run_role", role)
    monkeypatch.setattr(service, "command", build)
    return scenes, calls


def test_bundle_never_claims_runtime_or_visual_acceptance(tmp_path, monkeypatch):
    scenes, calls = setup(tmp_path, monkeypatch)

    async def run():
        result = await scenes.create("project", body(), source(), None)
        await scenes.tasks[result.id]
        terminal = scenes.get(result.id)
        assert terminal.status == "needs_simulation"
        assert terminal.gates["usd_structure"] and terminal.gates["agent_review"]
        assert not any(
            terminal.gates[k] for k in ("isaac_runtime", "visual_match", "calibrated_replay")
        )
        assert terminal.usage == {"input_tokens": 20, "output_tokens": 10}
        assert scenes.artifact(result.id, "scene-bundle.zip").is_file()
        import zipfile

        with zipfile.ZipFile(scenes.artifact(result.id, "scene-bundle.zip")) as archive:
            assert not any("auth" in name for name in archive.namelist())
            acceptance = json.loads(archive.read("acceptance.json"))
            assert acceptance["status"] == "needs_simulation"
            assert acceptance["agent"] == terminal.agent.model_dump()
        duplicate = await scenes.create("project", body(), source(), None)
        assert duplicate.id == result.id and calls == ["proposal", "review"]
        with pytest.raises(SceneError):
            await scenes.create("project", body(episode_index=2), source(), None)
        with pytest.raises(SceneError):
            scenes.artifact(result.id, "../run.json")

    asyncio.run(run())


def test_restart_preserves_interruption_without_rerunning(tmp_path):
    scenes = service.Scenes(tmp_path)
    scenes.folder("a" * 32).mkdir(parents=True)
    run = SceneRun(
        id="a" * 32,
        project_id="project",
        request=body(),
        status="running",
        stage="planning",
        created_at=now(),
        updated_at=now(),
        message="started",
    )
    scenes.save(run)
    auth = scenes.folder(run.id) / "attempt-1/.codex-runtime/auth.json"
    auth.parent.mkdir(parents=True)
    auth.write_text("private-fixture")
    restarted = service.Scenes(tmp_path)
    asyncio.run(restarted.reconcile())
    assert restarted.get(run.id).status == "interrupted"
    assert not auth.exists() and not restarted.tasks


def test_cancel_before_worker_starts_reaches_terminal_state(tmp_path, monkeypatch):
    scenes, calls = setup(tmp_path, monkeypatch)

    async def run():
        result = await scenes.create("project", body(), source(), None)
        terminal = await scenes.cancel(result.id)
        assert terminal.status == "cancelled" and not calls
        assert scenes.tasks[result.id].done()

    asyncio.run(run())


def test_mismatched_dataset_and_cross_project_rejected(tmp_path, monkeypatch):
    scenes, _ = setup(tmp_path, monkeypatch)

    async def run():
        with pytest.raises(SceneError):
            await scenes.create("another", body(), source(), None)
        changed = source()
        changed.result = changed.result.model_copy(update={"revision": "b" * 40})
        with pytest.raises(SceneError):
            await scenes.create("project", body(), changed, None)
        assert not scenes.directory.exists()

    asyncio.run(run())


def test_agent_review_rejection_is_bounded_and_retained(tmp_path, monkeypatch):
    scenes, calls = setup(tmp_path, monkeypatch)

    async def role(work, name, schema, *args, **kwargs):
        calls.append(name)
        result = (
            proposal()
            if schema is Proposal
            else Review(
                supported=True,
                consistent_with_images=False,
                issues=["Camera target is wrong."],
                uncertainties=["Pose not measured."],
            )
        )
        service.write(work / f"{name}.json", result.model_dump())
        return result, {}

    monkeypatch.setattr(service, "run_role", role)

    async def run():
        result = await scenes.create("project", body(max_attempts=2), source(), None)
        await scenes.tasks[result.id]
        terminal = scenes.get(result.id)
        assert terminal.status == "needs_review" and terminal.attempt == 2
        assert calls == ["proposal", "review", "proposal", "review"]
        assert not terminal.gates["usd_structure"]
        assert "attempt-2/review.json" in terminal.artifacts

    asyncio.run(run())


def test_failed_worker_error_is_not_returned_as_success_or_secret(tmp_path, monkeypatch):
    scenes, _ = setup(tmp_path, monkeypatch)

    async def failed(*args, **kwargs):
        raise RuntimeError("secret-key-fixture-must-never-appear")

    monkeypatch.setattr(service, "run_role", failed)

    async def run():
        result = await scenes.create("project", body(), source(), None)
        await scenes.tasks[result.id]
        terminal = scenes.get(result.id)
        assert terminal.status == "failed"
        assert "secret-key" not in terminal.model_dump_json()
        assert not terminal.gates["usd_structure"]

    asyncio.run(run())


def test_agent_output_rejects_nonfinite_or_impossible_geometry():
    value = proposal().model_dump()
    value["cup"]["height"] = float("nan")
    with pytest.raises(ValidationError):
        Proposal.model_validate(value)
    value = proposal().model_dump()
    value["camera"]["target"] = value["camera"]["position"]
    with pytest.raises(ValidationError):
        Proposal.model_validate(value)
    value = proposal().model_dump()
    value["box"]["size"] = [0.001, 0.001, 0.001]
    with pytest.raises(ValidationError):
        Proposal.model_validate(value)


def test_timeout_stops_owned_process_group(tmp_path):
    import sys

    async def run():
        with pytest.raises(SceneError, match="time budget"):
            await command(
                [
                    sys.executable,
                    "-c",
                    "import os,time; open('pid','w').write(str(os.getpid())); time.sleep(30)",
                ],
                cwd=tmp_path,
                timeout=0.2,
            )
        pid = int((tmp_path / "pid").read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)

    asyncio.run(run())


def test_api_options_and_empty_project_runs(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        result = client.get("/api/v1/scene-generation/options")
        assert result.status_code == 200 and result.json()["isaac_verified"] is False
        project = client.post("/api/v1/projects", json={"name": "Scene fixture"}).json()["id"]
        assert client.get(f"/api/v1/projects/{project}/scenes").json() == []
        assert client.get("/api/v1/scenes/not-a-run").status_code == 404
        missing = client.post(f"/api/v1/projects/{project}/scenes", json=body().model_dump())
        assert missing.status_code == 404


def test_model_catalog_filters_non_image_hidden_and_unsupported_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))

    def entry(slug, **updates):
        return dict(
            slug=slug,
            display_name=slug,
            visibility="list",
            supported_in_api=True,
            input_modalities=["text", "image"],
            supported_reasoning_levels=[{"effort": "high"}],
            **updates,
        )

    values = [
        entry("fixture-vision"),
        entry("fixture-text"),
        entry("fixture-hidden"),
        entry("fixture-no-api"),
    ]
    values[1]["input_modalities"] = ["text"]
    values[2]["visibility"] = "hide"
    values[3]["supported_in_api"] = False
    (tmp_path / "models_cache.json").write_text(json.dumps({"models": values}))
    options = catalog()
    assert options["model_catalog_source"] == "codex_catalog"
    assert options["models"] == [
        {
            "id": "fixture-vision",
            "label": "fixture-vision",
            "reasoning_efforts": ["high"],
            "default_reasoning_effort": "high",
        }
    ]
    (tmp_path / "models_cache.json").write_text("invalid catalog")
    assert catalog()["model_catalog_source"] == "bundled"


def test_model_and_power_pin_both_roles_and_survive_default_changes(tmp_path, monkeypatch):
    scenes, _ = setup(tmp_path, monkeypatch)
    seen = []
    original = service.run_role

    async def role(*args, **kwargs):
        seen.append(kwargs.copy())
        return await original(*args, **kwargs)

    monkeypatch.setattr(service, "run_role", role)

    async def run():
        request = body(model="gpt-6-luna", reasoning_effort="high")
        result = await scenes.create("project", request, source(), None)
        monkeypatch.setenv("JENSEN_SCENE_MODEL", "gpt-6-astra")
        await scenes.tasks[result.id]
        terminal = scenes.get(result.id)
        assert terminal.status == "needs_simulation"
        assert terminal.agent.model == "gpt-6-luna"
        assert terminal.agent.reasoning_effort == "high"
        assert seen == [{"model": "gpt-6-luna", "reasoning_effort": "high"}] * 2
        assert (await scenes.create("project", request, source(), None)).agent == terminal.agent
        with pytest.raises(SceneError, match="different scene request"):
            await scenes.create(
                "project", body(model="gpt-6-astra", reasoning_effort="high"), source(), None
            )

    asyncio.run(run())


@pytest.mark.parametrize(
    "changes", [{"model": "unknown-model"}, {"model": "gpt-6-luna", "reasoning_effort": "ultra"}]
)
def test_unsupported_model_or_power_rejected_before_work(tmp_path, monkeypatch, changes):
    scenes, calls = setup(tmp_path, monkeypatch)

    async def run():
        with pytest.raises(ValueError):
            await scenes.create("project", body(**changes), source(), None)
        assert not scenes.directory.exists() and not calls

    asyncio.run(run())


@pytest.mark.parametrize(
    "changes", [{"model": "--model"}, {"model": "model\n--flag"}, {"reasoning_effort": "infinite"}]
)
def test_agent_choice_contract_rejects_invalid_values(changes):
    with pytest.raises(ValidationError):
        body(**changes)


def test_agent_command_uses_explicit_model_and_power_and_removes_auth(tmp_path, monkeypatch):
    from vla_platform.scenes import agent

    observed = []
    monkeypatch.setattr(agent.ApiConnections, "key", lambda *args: "private-test-credential")

    async def fake(args, **kwargs):
        observed.extend(map(str, args))
        assert kwargs["env"]["CODEX_API_KEY"] == "private-test-credential"
        (tmp_path / "review.json").write_text(
            json.dumps(
                {
                    "supported": True,
                    "consistent_with_images": True,
                    "issues": [],
                    "uncertainties": ["No runtime proof"],
                }
            )
        )
        return b'{"type":"turn.completed","usage":{"output_tokens":3}}'

    monkeypatch.setattr(agent, "command", fake)

    async def run():
        result, usage = await agent.run_role(
            tmp_path,
            "review",
            Review,
            "fixture",
            [],
            tmp_path,
            model="gpt-6-astra",
            reasoning_effort="xhigh",
        )
        assert result.supported and usage["output_tokens"] == 3

    asyncio.run(run())
    assert observed[observed.index("--model") + 1] == "gpt-6-astra"
    assert 'model_reasoning_effort="xhigh"' in observed
    assert "private-test-credential" not in " ".join(observed)
    assert not (tmp_path / ".codex-runtime/auth.json").exists()
