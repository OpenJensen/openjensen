"""Terminal admission mirrors real lifecycle boundaries; these are not model benchmarks."""

import asyncio
import copy

import pytest

pytest.importorskip("textual")
from test_tui import job
from test_tui_lifecycle import LifecycleServer, context, recipe
from vla_platform.contracts import Job
from vla_platform.tui_client import ApiError
from vla_platform.tui_lifecycle import canonical, template

MODES = ("resume", "export", "evaluate", "engine_run", "gguf_quantize")


class ExtendedServer(LifecycleServer):
    def __init__(self):
        super().__init__()
        self.models.append(
            {"id": "smolvla", "model_id": "fixture/smolvla", "methods": ["lora", "qlora"]}
        )
        runtime = copy.deepcopy(self.runtimes[0])
        runtime.update(
            id="engine",
            act_export=True,
            engine_evaluation=True,
            run=True,
            simulation=True,
            training_model_ids=["act", "smolvla"],
        )
        self.runtimes.append(runtime)
        for ident, model in (("act-training", "act"), ("smol-training", "smolvla")):
            request = recipe("train")
            request["runtime_id"] = "engine"
            request["training"]["model_id"] = f"fixture/{model}"
            request["training_method"] = "full" if model == "act" else "lora"
            self.jobs.append(
                Job.model_validate(
                    {
                        **job(ident, status="failed", policy=True),
                        "request": request,
                    }
                ).model_dump(mode="json")
            )
            artifact = copy.deepcopy(self.artifacts[0])
            artifact.update(id=ident, job_id=ident, format="training_checkpoint")
            artifact["metadata"] = {
                "architecture": model,
                "method": request["training_method"],
                "training_backend": "lerobot" if model == "act" else "smolvla",
                "base_model": {"repository": f"fixture/{model}"},
                "dataset": {"source": "huggingface", "revision": "b" * 40},
            }
            self.artifacts.append(artifact)
        gguf = copy.deepcopy(self.artifacts[-1])
        gguf.update(id="gguf", format="gguf")
        gguf["metadata"].update(precision="float", task="libero_object")
        self.artifacts.append(gguf)


def request(mode):
    value = {
        "operation": {
            "resume": "policy.finetune",
            "export": "policy.export",
            "evaluate": "policy.evaluate",
            "engine_run": "policy.run",
            "gguf_quantize": "policy.quantize",
        }[mode],
        "runtime_id": "engine",
        "timeout_seconds": 600,
        "artifact_id": "act-training" if mode in {"resume", "export"} else "gguf",
    }
    if mode == "resume":
        value.update(dataset_job_id="data", training_method="full")
    elif mode == "export":
        value["training_method"] = "full"
    elif mode == "gguf_quantize":
        value["precision"] = {"language": "Q8_0", "vision": None}
    else:
        value["evaluation"] = {
            "mode": "engine",
            "suite": "libero_object",
            "warmups": 1,
            "repetitions": 2,
        }
    return value


def current(server=None):
    return asyncio.run(context(server or ExtendedServer()))


@pytest.mark.parametrize("suite", ["libero_object", "libero_spatial"])
def test_protocol_draft_never_chooses_robot_tasks_states_or_tolerances(suite):
    evaluation = template("evaluate", "engine", "gguf", protocol=suite)["evaluation"]
    assert evaluation["mode"] == "libero" and evaluation["suite"] == suite
    assert evaluation["initial_states"] == evaluation["final_states"] == []
    assert evaluation["seed"] is None and evaluation["warmups"] is None
    if suite == "libero_spatial":
        assert evaluation["steps"] == 280  # Required by the existing benchmark contract.
        assert evaluation["task_ids"] == []
        assert all(item is None for item in evaluation["parity_limits"].values())
    else:
        assert evaluation["steps"] is None and evaluation["task_id"] is None


@pytest.mark.parametrize("mode", MODES)
def test_extended_recipe_review_preserves_explicit_request(mode):
    reviewed = current().review(mode, canonical(request(mode)))
    for key, value in request(mode).items():
        if key == "evaluation":
            assert all(reviewed.request[key][field] == item for field, item in value.items())
        else:
            assert reviewed.request[key] == value
    with pytest.raises(ApiError):
        current().review(mode, canonical(template(mode, "engine", "gguf", "data")))


def test_resume_saved_lineage_and_recipe_are_not_editable():
    ctx = current()
    original = ctx.review("resume", canonical(request("resume")))
    for changes in (
        {"training": {"steps": 999}},
        {"training_method": "lora"},
        {"dataset_job_id": "wrong"},
        {"resume_job_id": "act-training"},
    ):
        with pytest.raises(ApiError):
            ctx.review("resume", canonical({**request("resume"), **changes}))
    ctx.jobs[1]["request"]["training"]["steps"] += 1
    assert (
        ctx.review("resume", canonical(request("resume"))).context_sha256 != original.context_sha256
    )


def test_interrupted_job_resume_requires_owned_terminal_training_lineage():
    value = request("resume")
    value.pop("artifact_id")
    value["resume_job_id"] = "act-training"
    ctx = current()
    assert ctx.review("resume", canonical(value)).request["resume_job_id"] == "act-training"
    for status in ("running", "queued", "succeeded"):
        ctx.jobs[1]["status"] = status
        with pytest.raises(ApiError):
            ctx.review("resume", canonical(value))


def test_legacy_checkpoint_resume_uses_metadata_instead_of_guessing_smolvla():
    ctx = current()
    ctx.jobs[1]["request"].update(training=None, training_method="lora")
    assert ctx.review("resume", canonical(request("resume"))).request["training_method"] == "full"
    ctx.artifacts[2]["metadata"]["base_model"] = "invalid"
    with pytest.raises(ApiError, match="saved model"):
        ctx.review("resume", canonical(request("resume")))


def test_resume_review_binds_intermediate_checkpoint_manifest():
    ctx = current()
    intermediate = copy.deepcopy(ctx.jobs[1])
    intermediate.update(id="resumed-training")
    intermediate["request"].update(artifact_id="act-training", training=None)
    ctx.jobs.append(intermediate)
    source = copy.deepcopy(ctx.artifacts[2])
    source.update(id="resumed-checkpoint", job_id="resumed-training")
    ctx.artifacts.append(source)
    value = {**request("resume"), "artifact_id": "resumed-checkpoint"}
    reviewed = ctx.review("resume", canonical(value))
    ctx.artifacts[2]["manifest_sha256"] = "c" * 64
    assert ctx.review("resume", canonical(value)).context_sha256 != reviewed.context_sha256


def test_export_adapter_and_cloud_download_boundaries():
    ctx = current()
    ctx.artifacts[2]["metadata"]["dataset"]["source"] = "local"
    with pytest.raises(ApiError, match="Hugging Face"):
        ctx.review("export", canonical(request("export")))
    server = ExtendedServer()
    server.artifacts[2]["metadata"]["storage"] = "gcs"
    with pytest.raises(ApiError, match="completed"):
        current(server).review("export", canonical(request("export")))
    server.jobs[1]["status"] = "succeeded"
    server.artifacts[2]["metadata"]["reload_verified"] = True
    assert (
        current(server).review("export", canonical(request("export"))).artifact["id"]
        == "act-training"
    )
    server.runtimes[-1]["provider"] = "gcp"
    server.runtimes[-1]["execution"] = "skypilot"
    with pytest.raises(ApiError, match="runtime"):
        current(server).review("export", canonical(request("export")))


@pytest.mark.parametrize("mode", ["evaluate", "engine_run"])
def test_engine_has_no_task_quality_claim_and_libero_requires_explicit_protocol(mode):
    ctx = current()
    value = request(mode)
    value["evaluation"] = {"mode": "libero", "suite": "libero_object"}
    with pytest.raises(ApiError):
        ctx.review(mode, canonical(value))
    value["evaluation"].update(
        task_id=0,
        initial_states=[0],
        final_states=[1],
        seed=0,
        steps=200,
        warmups=1,
        repetitions=2,
    )
    assert ctx.review(mode, canonical(value)).request["evaluation"]["mode"] == "libero"
    ctx.runtimes[-1]["simulation"] = False
    with pytest.raises(ApiError, match="LIBERO"):
        ctx.review(mode, canonical(value))


@pytest.mark.parametrize("mode", ["evaluate", "engine_run", "gguf_quantize"])
def test_engine_rejects_act_and_native_recipe_smuggling(mode):
    value = request(mode)
    value["artifact_id"] = "policy"
    with pytest.raises(ApiError):
        current().review(mode, canonical(value))
    value = request(mode)
    value["training"] = {"steps": 2}
    with pytest.raises(ApiError):
        current().review(mode, canonical(value))


def test_cloud_engine_refuses_simulation_and_resume_refuses_local_materialization_assumption():
    ctx = current()
    ctx.runtimes[-1].update(provider="gcp", execution="skypilot")
    assert ctx.review("evaluate", canonical(request("evaluate"))).runtime["provider"] == "gcp"
    value = request("evaluate")
    value["evaluation"].update(
        mode="libero", task_id=0, initial_states=[0], final_states=[1], seed=1, steps=200
    )
    with pytest.raises(ApiError):
        ctx.review("evaluate", canonical(value))
    ctx.runtimes[-1].update(provider="local", execution="native")
    ctx.artifacts[2]["metadata"]["storage"] = "gcs"
    with pytest.raises(ApiError, match="Cloud checkpoint"):
        ctx.review("resume", canonical(request("resume")))


def test_resume_cycle_missing_lineage_and_cloud_interrupted_job_are_rejected():
    ctx = current()
    ctx.jobs[1]["request"].update(training=None, artifact_id="act-training")
    with pytest.raises(ApiError, match="cycle"):
        ctx.review("resume", canonical(request("resume")))
    ctx = current()
    ctx.jobs[1]["compute_target"] = {"provider": "cloud fixture"}
    value = request("resume")
    value.update(artifact_id=None, resume_job_id="act-training")
    with pytest.raises(ApiError, match="Cloud checkpoint"):
        ctx.review("resume", canonical(value))
    ctx.jobs.pop(1)
    with pytest.raises(ApiError):
        ctx.review("resume", canonical(value))


@pytest.mark.parametrize("bad", [True, "1", 1.0])
def test_evaluation_does_not_coerce_a_budget(bad):
    value = request("evaluate")
    value["evaluation"]["warmups"] = bad
    with pytest.raises(ApiError):
        current().review("evaluate", canonical(value))


@pytest.mark.parametrize("metadata", ["untrusted", [], 7])
def test_malformed_nested_export_metadata_is_a_public_validation_error(metadata):
    ctx = current()
    ctx.artifacts[2]["metadata"]["dataset"] = metadata
    with pytest.raises(ApiError, match="Hugging Face"):
        ctx.review("export", canonical(request("export")))


@pytest.mark.parametrize("mode", MODES)
def test_pilot_extended_workflows_require_consent_then_send_one_exact_request(tmp_path, mode):
    from test_tui import until
    from test_tui_lifecycle import open_form, reviewed
    from textual.widgets import Button, Select, TextArea
    from vla_platform.tui import FirebirdApp

    async def scenario():
        server = ExtendedServer()
        app = FirebirdApp(client=server.client(), poll_seconds=100, journal_dir=tmp_path / "state")
        async with app.run_test(size=(100, 40)) as pilot:
            form = await open_form(server, tmp_path, "train", app, pilot)
            form.query_one("#lifecycle-mode", Select).value = mode
            await pilot.pause()
            form.query_one("#recipe-editor", TextArea).load_text(canonical(request(mode)))
            await pilot.pause()
            assert not server.posts
            await reviewed(form, pilot)
            form.query_one("#recipe-submit", Button).press()
            await until(lambda: app.screen is app.default_screen and app.job_id == "accepted")
            assert len(server.posts) == 1
            assert server.posts[0] == form.accepted["request"]
            assert form.journal.read() is None

    asyncio.run(scenario())


def test_pilot_resume_draft_preserves_lineage_and_stale_lineage_cannot_submit(tmp_path):
    from test_tui import until
    from test_tui_lifecycle import open_form, reviewed
    from textual.widgets import Button, Select, TextArea
    from vla_platform.tui import FirebirdApp

    async def scenario():
        server = ExtendedServer()
        app = FirebirdApp(client=server.client(), poll_seconds=100, journal_dir=tmp_path / "state")
        async with app.run_test(size=(48, 18)) as pilot:
            form = await open_form(server, tmp_path, "train", app, pilot)
            form.query_one("#lifecycle-mode", Select).value = "resume"
            await pilot.pause()
            form.query_one("#recipe-runtime", Select).value = "engine"
            form.query_one("#recipe-resume", Select).value = "act-training"
            form.query_one("#recipe-build", Button).press()
            await pilot.pause()
            from vla_platform.tui_lifecycle import parse_recipe

            draft = parse_recipe(form.query_one("#recipe-editor", TextArea).text)
            from textual.widgets import Static

            assert "artifact_id" in draft, {
                "mode": form.query_one("#lifecycle-mode", Select).value,
                "artifact": form.selected("artifact"),
                "resume": form.selected("resume"),
                "error": str(form.query_one("#lifecycle-error", Static).content),
                "build_disabled": form.query_one("#recipe-build", Button).disabled,
            }
            assert draft["dataset_job_id"] == "data" and draft["training_method"] == "full"
            assert draft["artifact_id"] is None and draft["timeout_seconds"] is None
            assert draft["resume_job_id"] == "act-training" and not server.posts
            draft["timeout_seconds"] = 600
            form.query_one("#recipe-editor", TextArea).load_text(canonical(draft))
            await pilot.pause()
            await reviewed(form, pilot)
            server.jobs[1]["request"]["training"]["steps"] += 1
            form.query_one("#recipe-submit", Button).press()
            await until(lambda: form.reviewed is None and not form.busy)
            assert not server.posts and form.journal.read() is None

    asyncio.run(scenario())
