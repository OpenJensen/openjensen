"""Recipe/persistence/Pilot boundaries; generated protocol records are not ML evidence."""

import asyncio
import copy
import json
import os
from pathlib import Path

import httpx
import pytest
from filelock import Timeout

pytest.importorskip("textual")
from test_tui import Server, choose_project, job, until
from textual.widgets import Button, Checkbox, Select, Static, TextArea
from vla_platform.contracts import DatasetProfile, Job
from vla_platform.lifecycle.runtime import PublicRuntime
from vla_platform.tui import FirebirdApp
from vla_platform.tui_client import ApiError
from vla_platform.tui_lifecycle import (
    OPERATIONS,
    AttemptJournal,
    Context,
    artifact_allowed,
    canonical,
    dataset_allowed,
    template,
)
from vla_platform.tui_lifecycle_forms import LifecycleForm

SHA = "a" * 64
BASE_MODES = ("train", "distill", "quantize", "replay")


def dataset():
    return DatasetProfile(
        source="local",
        revision=f"metadata-sha256:{SHA}",
        format="lerobot_v3",
        total_episodes=3,
        total_frames=12,
        fps=30,
        features={
            "action": {"shape": [6]},
            "observation.state": {"shape": [6]},
            "observation.images.front": {"dtype": "video", "shape": [3, 32, 32]},
        },
        metadata_sha256=SHA,
        inspected_at="2026-09-27T00:00:00Z",
        warnings=[],
        inspection_scope="complete_snapshot",
        snapshot={
            "id": f"sha256:{SHA}",
            "manifest_sha256": SHA,
            "total_bytes": 100,
            "file_count": 3,
            "total_episodes": 3,
            "total_frames": 12,
            "lineage_validated": True,
            "warnings": [],
        },
    ).model_dump(mode="json")


def recipe(mode):
    value = template(mode, mode, "packed" if mode == "replay" else "policy", "data")
    value["timeout_seconds"] = 600
    if mode == "train":
        value.update(
            training_method="full",
            training={
                "model_id": "fixture/act",
                "steps": 12,
                "batch_size": 1,
                "learning_rate": 0.0001,
                "camera_key": "observation.images.front",
            },
        )
    elif mode == "distill":
        value["native_distillation"].update(
            steps=1,
            learning_rate=0.0001,
            seed=1,
            frame_stride=1,
            splits={"train": [0], "validation": [1], "final": [2]},
            coordinate_attestation="generated_fixture",
            units=["generated"] * 6,
        )
    elif mode == "quantize":
        value["native_quantization"]["bits"] = 8
    else:
        value["native_replay"].update(
            selection=[{"episode_index": 0, "frame_index": 0}],
            coordinate_attestation="generated_fixture",
            units=["generated"] * 6,
        )
    return value


class LifecycleServer(Server):
    def __init__(self):
        super().__init__()
        self.jobs = [{**job("data", status="succeeded"), "result": dataset()}]
        self.artifacts = [
            dict(
                id=ident,
                project_id="p",
                job_id="previous",
                label="Generated fixture",
                format=fmt,
                path=f"jobs/previous/{ident}",
                manifest_sha256=SHA,
                file_bytes=100,
                parent_ids=[],
                metadata={"architecture": "act", "inference_only": True},
            )
            for ident, fmt in [("policy", "inference_export"), ("packed", "native_quantized")]
        ]
        self.runtimes = [
            PublicRuntime(
                id=mode,
                label=mode,
                provider="local",
                provider_label="Local fixture",
                region=None,
                enabled=True,
                device="cuda" if mode == "train" else "cpu",
                training=mode == "train",
                training_model_ids=["act"],
                simulation=False,
                gpu_name=None,
                gpu_memory_mib=None,
                training_gpu_count=None,
                **({OPERATIONS[mode][2]: True} if mode != "train" else {}),
            ).model_dump(mode="json")
            for mode in BASE_MODES
        ]
        self.models = [{"id": "act", "model_id": "fixture/act", "methods": ["full"]}]
        self.invalid_ack = False
        self.post_gate = None
        self.context_gate = None

    async def handle(self, request):
        path = request.url.path.removeprefix("/api/v1")
        if path == "/policy-options" or path.endswith("/artifacts"):
            if path == "/policy-options" and self.context_gate:
                await self.context_gate.wait()
            self.calls.append((request.method, request.url.path, request.content))
            return httpx.Response(
                200,
                json=(
                    {"runtimes": self.runtimes, "training_models": self.models}
                    if path == "/policy-options"
                    else self.artifacts
                ),
            )
        if request.method == "POST" and path.endswith("/policy-jobs"):
            self.calls.append((request.method, request.url.path, request.content))
            if self.post_timeout:
                raise httpx.ReadTimeout("lost acknowledgment")
            if self.invalid_ack:
                return httpx.Response(200, json={})
            value = Job.model_validate(
                {
                    **job("accepted", status="queued", policy=True),
                    "kind": json.loads(request.content)["operation"],
                    "request": json.loads(request.content),
                }
            ).model_dump(mode="json")
            self.jobs.append(value)
            if self.post_gate:
                await self.post_gate.wait()
            return httpx.Response(202, json=value)
        return await super().handle(request)

    @property
    def posts(self):
        return [json.loads(raw) for method, _, raw in self.calls if method == "POST"]


async def context(server):
    client = server.client()
    try:
        return await Context.fetch(client, "p")
    finally:
        await client.close()


@pytest.mark.parametrize("mode", BASE_MODES)
def test_complete_recipes_preserved_and_templates_require_choices(mode):
    current = asyncio.run(context(LifecycleServer()))
    reviewed = current.review(mode, canonical(recipe(mode)))
    for key, value in recipe(mode).items():
        assert reviewed.request[key] == value
    with pytest.raises(ApiError):
        current.review(mode, canonical(template(mode, mode, "policy", "data")))
    assert (
        reviewed.context_sha256
        != current.review(mode, canonical({**recipe(mode), "timeout_seconds": 300})).context_sha256
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s.artifacts[0].update(project_id="q"),
        lambda s: s.artifacts.append(copy.deepcopy(s.artifacts[0])),
        lambda s: s.runtimes.append(copy.deepcopy(s.runtimes[0])),
        lambda s: s.models[0].update(methods=[{}]),
        lambda s: s.models.append(copy.deepcopy(s.models[0])),
        lambda s: s.projects.clear(),
    ],
)
def test_invalid_or_foreign_context_is_rejected(mutation):
    server = LifecycleServer()
    mutation(server)
    with pytest.raises(ApiError):
        asyncio.run(context(server))


@pytest.mark.parametrize(
    "raw", ['{"a":1,"a":2}', '{"x":NaN}', '{"x":1e309}', "[]", "{", " " * 65537]
)
def test_recipe_json_bounds_and_ambiguity(raw):
    from vla_platform.tui_lifecycle import parse_recipe

    with pytest.raises(ApiError):
        parse_recipe(raw)


def test_recipe_capability_dataset_and_identity_gates():
    current = asyncio.run(context(LifecycleServer()))
    for item in current.runtimes:
        item["enabled"] = False
    for mode in BASE_MODES:
        with pytest.raises(ApiError, match="runtime"):
            current.review(mode, canonical(recipe(mode)))
    server = LifecycleServer()
    server.artifacts[0]["metadata"]["storage"] = "gcs"
    assert not artifact_allowed(server.artifacts[0], "quantize")
    server.jobs[0]["result"]["snapshot"]["lineage_validated"] = False
    assert not dataset_allowed(server.jobs[0], "distill")
    assert dataset_allowed(server.jobs[0], "replay")
    request = recipe("quantize")
    request["artifact_id"] = "foreign"
    with pytest.raises(ApiError, match="project-owned"):
        asyncio.run(context(server)).review("quantize", canonical(request))


def journal(tmp_path):
    return AttemptJournal(tmp_path / "private", "http://127.0.0.1:9999", "p")


def review():
    return asyncio.run(context(LifecycleServer())).review("quantize", canonical(recipe("quantize")))


def test_journal_private_restart_exact_ack_and_cross_endpoint_isolation(tmp_path):
    saved = journal(tmp_path)
    record = saved.begin(review())
    assert saved.read() == record == journal(tmp_path).read()
    assert "training" not in saved.path.read_text()
    assert "http" not in saved.path.read_text()
    if os.name == "posix":
        assert saved.root.stat().st_mode & 0o777 == 0o700
        assert saved.path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ApiError, match="earlier"):
        saved.begin(review())
    with pytest.raises(ApiError, match="changed"):
        saved.finish({**record, "recipe_sha256": "b" * 64})
    saved.finish(record, uncertain=True)
    with pytest.raises(ApiError, match="changed"):
        saved.finish(record)
    saved.finish(saved.read())
    assert saved.read() is None
    assert AttemptJournal(saved.root, "http://127.0.0.1:9998", "p").path != saved.path


def test_second_client_cannot_clear_inflight_claim(tmp_path):
    saved, other = journal(tmp_path), journal(tmp_path)
    other.lock.timeout = 0.01
    with saved.claim(review()):
        with pytest.raises(Timeout):
            other.read()
    assert other.read()["state"] == "pending"


@pytest.mark.parametrize("target", ["record", "lock"])
@pytest.mark.skipif(os.name != "posix", reason="POSIX named-pipe and symlink test")
def test_journal_special_files_rejected_without_blocking(tmp_path, target):
    saved = journal(tmp_path)
    saved.prepare()
    path = saved.path if target == "record" else Path(saved.lock.lock_file)
    os.mkfifo(path, 0o600)
    with pytest.raises(ApiError):
        saved.read()
    path.unlink()
    path.symlink_to(tmp_path / "missing")
    with pytest.raises((ApiError, OSError)):
        saved.read()


@pytest.mark.parametrize("content", ["{}", "x" * 4097, '{"x":1,"x":2}', "\ufffd"])
def test_bad_journal_cannot_be_reset_implicitly(tmp_path, content):
    saved = journal(tmp_path)
    saved.prepare()
    saved.path.write_text(content)
    saved.path.chmod(0o600)
    with pytest.raises(ApiError):
        saved.begin(review())
    assert saved.path.read_text() == content


async def open_form(server, tmp_path, mode, app, pilot):
    await choose_project(app, pilot)
    app.action_view("detail-tab")
    app.query_one("#detail", TextArea).focus()
    # Keep the screen stable while Pilot waits for key dispatch; its old-screen
    # callbacks otherwise race the async API read/modal push. No product delay.
    server.context_gate = asyncio.Event()
    await pilot.press("f5")
    server.context_gate.set()
    await until(lambda: isinstance(app.screen, LifecycleForm) and app.screen.is_mounted)
    await pilot.pause()
    form = app.screen
    form.query_one("#lifecycle-mode", Select).value = mode
    await pilot.pause()
    form.query_one("#recipe-editor", TextArea).load_text(json.dumps(recipe(mode)))
    await pilot.pause()
    return form


async def reviewed(form, pilot):
    form.query_one("#recipe-review", Button).press()
    await until(lambda: form.reviewed is not None and not form.busy)
    await pilot.pause()
    form.query_one("#recipe-consent", Checkbox).value = True
    await pilot.pause()


@pytest.mark.parametrize("mode", BASE_MODES)
def test_pilot_explicit_complete_submission_one_post_and_job_monitor(tmp_path, mode):
    async def scenario():
        server = LifecycleServer()
        app = FirebirdApp(client=server.client(), poll_seconds=100, journal_dir=tmp_path / "state")
        async with app.run_test(size=(100, 40)) as pilot:
            form = await open_form(server, tmp_path, mode, app, pilot)
            assert not server.posts
            await reviewed(form, pilot)
            assert not form.query_one("#recipe-submit", Button).disabled
            form.query_one("#recipe-submit", Button).focus()
            await pilot.press("enter")
            await until(lambda: app.screen is app.default_screen and app.job_id == "accepted")
            assert len(server.posts) == 1
            assert server.posts[0] == form.accepted["request"]
            assert form.journal.read() is None
            await until(lambda: '"queued"' in app.query_one("#detail", TextArea).text)

    asyncio.run(scenario())


def test_pilot_cloud_separate_consent_review_edit_and_resize(tmp_path):
    async def scenario():
        server = LifecycleServer()
        server.runtimes[0].update(
            provider="gcp", provider_label="Google Cloud", execution="skypilot"
        )
        app = FirebirdApp(client=server.client(), poll_seconds=100, journal_dir=tmp_path / "state")
        async with app.run_test(size=(48, 18)) as pilot:
            form = await open_form(server, tmp_path, "train", app, pilot)
            await reviewed(form, pilot)
            assert form.query_one("#recipe-submit", Button).disabled
            form.query_one("#recipe-cloud-consent", Checkbox).value = True
            await pilot.pause()
            assert not form.query_one("#recipe-submit", Button).disabled
            form.query_one("#recipe-editor", TextArea).load_text(
                json.dumps(recipe("train"), indent=2)
            )
            await pilot.pause()
            assert form.reviewed is None
            assert form.query_one("#recipe-submit", Button).disabled
            assert not form.query_one("#recipe-consent", Checkbox).value
            await pilot.press("escape")
            assert app.screen is app.default_screen
            assert not server.posts

    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["source", "runtime", "project"])
def test_changed_context_between_review_and_submit_sends_nothing(tmp_path, change):
    async def scenario():
        server = LifecycleServer()
        app = FirebirdApp(client=server.client(), poll_seconds=100, journal_dir=tmp_path / "state")
        async with app.run_test(size=(100, 40)) as pilot:
            form = await open_form(server, tmp_path, "quantize", app, pilot)
            await reviewed(form, pilot)
            if change == "source":
                server.artifacts[0]["manifest_sha256"] = "b" * 64
            elif change == "runtime":
                server.runtimes[2]["enabled"] = False
            else:
                app.project_id = "q"
            form.query_one("#recipe-submit", Button).press()
            await until(lambda: form.reviewed is None and not form.busy)
            assert not server.posts
            assert form.journal.read() is None

    asyncio.run(scenario())


@pytest.mark.parametrize("invalid_ack", [False, True])
def test_uncertain_post_survives_restart_requires_later_history_and_exact_ack(
    tmp_path, invalid_ack
):
    async def scenario():
        server = LifecycleServer()
        server.post_timeout, server.invalid_ack = not invalid_ack, invalid_ack
        for restart in (False, True):
            app = FirebirdApp(
                client=server.client(), poll_seconds=100, journal_dir=tmp_path / "state"
            )
            async with app.run_test(size=(100, 40)) as pilot:
                form = await open_form(server, tmp_path, "quantize", app, pilot)
                if not restart:
                    await reviewed(form, pilot)
                    form.query_one("#recipe-submit", Button).press()
                    await until(lambda: form.attempt is not None and not form.busy)
                    assert form.attempt["state"] == "uncertain"
                else:
                    assert form.blocked and form.query_one("#recipe-review", Button).disabled
                    assert form.query_one("#attempt-clear", Button).disabled
                    form.query_one("#attempt-history", Button).press()
                    await until(lambda: form.history_reviewed and not form.busy)
                    assert form.query_one("#attempt-clear", Button).disabled
                    form.query_one("#attempt-consent", Checkbox).value = True
                    await pilot.pause()
                    form.query_one("#attempt-clear", Button).press()
                    await until(lambda: not form.blocked)
                    assert form.journal.read() is None
                    assert form.reviewed is None  # Never resubmit on acknowledgment.
                assert len(server.posts) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["begin", "cleanup"])
def test_storage_errors_block_writes_but_keep_accepted_receipt(tmp_path, monkeypatch, phase):
    async def scenario():
        server = LifecycleServer()
        app = FirebirdApp(client=server.client(), poll_seconds=100, journal_dir=tmp_path / "state")
        async with app.run_test(size=(100, 40)) as pilot:
            form = await open_form(server, tmp_path, "quantize", app, pilot)
            await reviewed(form, pilot)

            def fail(*args, **kwargs):
                raise OSError("private filesystem error")

            monkeypatch.setattr(form.journal, "begin" if phase == "begin" else "finish", fail)
            form.query_one("#recipe-submit", Button).press()
            await until(lambda: form.storage_failed and not form.busy)
            assert len(server.posts) == (1 if phase == "cleanup" else 0)
            assert form.query_one("#recipe-review", Button).disabled
            assert "private filesystem error" not in str(
                form.query_one("#lifecycle-error", Static).content
            )
            if phase == "cleanup":
                assert form.accepted["id"] == "accepted"
                assert "accepted" in str(form.query_one("#accepted-receipt", Static).content)
                assert form.journal.read()["state"] == "pending"
                form.action_back()
                await until(lambda: app.job_id == "accepted")

    asyncio.run(scenario())


def test_cancelled_send_preserves_pending_receipt_and_never_reposts(tmp_path):
    async def scenario():
        server = LifecycleServer()
        server.post_gate = asyncio.Event()
        app = FirebirdApp(client=server.client(), poll_seconds=100, journal_dir=tmp_path / "state")
        async with app.run_test(size=(100, 40)) as pilot:
            form = await open_form(server, tmp_path, "quantize", app, pilot)
            await reviewed(form, pilot)
            form.query_one("#recipe-submit", Button).press()
            await until(lambda: len(server.posts) == 1)
            assert any(item["id"] == "accepted" for item in server.jobs)
            pending = form.journal.path.read_text()
            form.workers.cancel_all()
            await until(lambda: not form.busy)
            assert json.loads(pending)["state"] == "pending"
            assert form.journal.read()["state"] == "pending"
            assert len(server.posts) == 1

    asyncio.run(scenario())


def test_accepted_cleanup_directory_flush_failure_keeps_visible_receipt(tmp_path, monkeypatch):
    async def scenario():
        server = LifecycleServer()
        app = FirebirdApp(client=server.client(), poll_seconds=100, journal_dir=tmp_path / "state")
        async with app.run_test(size=(100, 40)) as pilot:
            form = await open_form(server, tmp_path, "quantize", app, pilot)
            await reviewed(form, pilot)
            original_sync = form.journal._sync

            def fail_cleanup_only():
                if not form.journal.path.exists():
                    raise OSError("directory flush failed")
                original_sync()

            monkeypatch.setattr(form.journal, "_sync", fail_cleanup_only)
            form.query_one("#recipe-submit", Button).press()
            await until(lambda: form.storage_failed and not form.busy)
            assert len(server.posts) == 1 and form.accepted["id"] == "accepted"
            assert "accepted" in str(form.query_one("#accepted-receipt", Static).content)
            assert form.query_one("#recipe-submit", Button).disabled
            assert form.query_one("#recipe-review", Button).disabled

    asyncio.run(scenario())


def test_draft_build_and_saved_recipe_copy_never_submit(tmp_path):
    async def scenario():
        server = LifecycleServer()
        server.jobs.append(
            Job.model_validate(
                {
                    **job("old", status="failed", policy=True),
                    "request": recipe("train"),
                }
            ).model_dump(mode="json")
        )
        app = FirebirdApp(client=server.client(), poll_seconds=100, journal_dir=tmp_path / "state")
        async with app.run_test(size=(100, 40)) as pilot:
            await choose_project(app, pilot)
            from textual.widgets import OptionList

            app.query_one("#jobs", OptionList).highlighted = 1
            app.action_lifecycle()
            await until(lambda: isinstance(app.screen, LifecycleForm))
            await pilot.pause()  # Screen identity precedes completion of child mounting.
            form = app.screen
            form.query_one("#recipe-copy", Button).press()
            await pilot.pause()
            assert (
                json.loads(form.query_one("#recipe-editor", TextArea).text)
                == server.jobs[1]["request"]
            )
            form.query_one("#recipe-runtime", Select).value = "train"
            form.query_one("#recipe-dataset", Select).value = "data"
            form.query_one("#recipe-build", Button).press()
            await pilot.pause()
            draft = json.loads(form.query_one("#recipe-editor", TextArea).text)
            assert draft["training"]["steps"] is None and draft["timeout_seconds"] is None
            assert draft["dataset_job_id"] == "data"
            assert not server.posts

    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["method", "model", "camera", "foreign-camera"])
def test_training_recipe_requires_explicit_registered_model_method_and_camera(change):
    current = asyncio.run(context(LifecycleServer()))
    value = recipe("train")
    if change == "method":
        value.pop("training_method")
    elif change == "model":
        value["training"]["model_id"] = "unregistered/model"
    elif change == "camera":
        value["training"].pop("camera_key")
    else:
        value["training"]["camera_key"] = "missing.camera"
    with pytest.raises(ApiError):
        current.review("train", canonical(value))


def test_invalid_recipe_reports_user_field_without_response_blame():
    current = asyncio.run(context(LifecycleServer()))
    value = recipe("quantize")
    value["native_quantization"]["bits"] = 3
    with pytest.raises(ApiError, match="Recipe needs correction: native_quantization.bits"):
        current.review("quantize", canonical(value))
