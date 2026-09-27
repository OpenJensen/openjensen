"""Labelled terminal recipes, using generated protocol fixtures, not ML-quality evidence."""

import asyncio
import copy
import json

import pytest

pytest.importorskip("textual")
from test_tui import until
from test_tui_lifecycle import BASE_MODES, open_form, recipe, reviewed
from test_tui_policy_modes import MODES, ExtendedServer, context, request
from textual.widgets import Button, Checkbox, Input, Select, SelectionList, Static, TextArea
from vla_platform.tui import FirebirdApp
from vla_platform.tui_client import ApiError
from vla_platform.tui_lifecycle import canonical, template
from vla_platform.tui_recipe_fields import Field, RecipeFields, display_value, parse_value


def complete(mode):
    return recipe(mode) if mode in BASE_MODES else request(mode)


@pytest.mark.parametrize(
    "kind,value",
    [
        ("integer", True),
        ("integer", 1.0),
        ("integer", "1"),
        ("number", float("inf")),
        ("number", 10**400),
        ("number", False),
        ("integers", [True]),
        ("integers", [1.2]),
        ("integers", "1, 2"),
        ("observations", [{"episode_index": 0, "frame_index": 1, "extra": 2}]),
    ],
)
def test_existing_malformed_draft_values_are_not_silently_repaired(kind, value):
    with pytest.raises(ApiError):
        display_value(Field("test", "Budget", kind), value)


@pytest.mark.parametrize(
    "kind,text",
    [
        ("integer", "1.5"),
        ("integer", "true"),
        ("integer", "1e2"),
        ("number", "NaN"),
        ("number", "1e9999"),
        ("number", ""),
        ("integers", "1,,2"),
        ("integers", "1-10"),
        ("integers", "1,"),
        ("observations", "0:1:2"),
        ("observations", "0:1,"),
    ],
)
def test_invalid_user_field_text_never_becomes_a_request(kind, text):
    with pytest.raises(ApiError):
        parse_value(Field("test", "Selection", kind), text)


@pytest.mark.parametrize("mode", (*BASE_MODES, *MODES))
def test_pilot_fields_apply_preserves_recipe_then_requires_review_and_one_submit(tmp_path, mode):
    async def scenario():
        server = ExtendedServer()
        app = FirebirdApp(client=server.client(), poll_seconds=100, journal_dir=tmp_path / "state")
        async with app.run_test(size=(48, 18)) as pilot:
            form = await open_form(server, tmp_path, "train", app, pilot)
            form.query_one("#lifecycle-mode", Select).value = mode
            await pilot.pause()
            value = complete(mode)
            if mode == "train":
                value["training"]["adapter_option"] = {"preserve": [1, 2]}
            form.query_one("#recipe-editor", TextArea).load_text(canonical(value))
            await pilot.pause()
            await reviewed(form, pilot)
            before = len(server.calls)
            form.query_one("#recipe-fields", Button).focus()
            await pilot.press("enter")
            await until(lambda: isinstance(app.screen, RecipeFields) and app.screen.is_mounted)
            editor = app.screen
            assert editor.query_one("#field-timeout_seconds", Input).has_focus
            await pilot.press("ctrl+shift+a", "3", "0", "0")
            assert editor.query_one("#field-timeout_seconds", Input).value == "300"
            editor.query_one("#fields-apply", Button).focus()
            await pilot.press("enter")
            await until(lambda: app.screen is form)
            await pilot.pause()
            await pilot.pause()
            assert len(server.calls) == before and not server.posts
            assert json.loads(form.query_one("#recipe-editor", TextArea).text) == {
                **value,
                "timeout_seconds": 300,
            }
            assert form.reviewed is None
            assert not form.query_one("#recipe-consent", Checkbox).value
            assert form.query_one("#recipe-submit", Button).disabled
            await reviewed(form, pilot)
            form.query_one("#recipe-submit", Button).press()
            await until(lambda: app.screen is app.default_screen and app.job_id == "accepted")
            assert server.posts == [form.accepted["request"]]
            assert server.posts[0]["timeout_seconds"] == 300

    asyncio.run(scenario())


def test_pilot_build_training_without_json_edits_and_cancel_preserves_exact_original(tmp_path):
    async def scenario():
        server = ExtendedServer()
        app = FirebirdApp(client=server.client(), poll_seconds=100, journal_dir=tmp_path / "state")
        async with app.run_test(size=(64, 22)) as pilot:
            form = await open_form(server, tmp_path, "train", app, pilot)
            form.query_one("#recipe-runtime", Select).value = "train"
            form.query_one("#recipe-dataset", Select).value = "data"
            form.query_one("#recipe-build", Button).press()
            await pilot.pause()
            original = form.query_one("#recipe-editor", TextArea).text
            form.query_one("#recipe-fields", Button).press()
            await until(lambda: isinstance(app.screen, RecipeFields) and app.screen.is_mounted)
            editor = app.screen
            editor.query_one("#field-timeout_seconds", Input).value = "600"
            editor.query_one("#field-training-model_id", Select).value = "fixture/act"
            editor.query_one("#field-training_method", Select).value = "full"
            editor.query_one("#field-training-steps", Input).value = "12"
            editor.query_one("#field-training-batch_size", Input).value = "1"
            editor.query_one("#field-training-learning_rate", Input).value = "0.0001"
            editor.query_one("#field-training-camera_keys", SelectionList).select(
                "observation.images.front"
            )
            editor.query_one("#fields-apply", Button).press()
            await until(lambda: app.screen is form)
            await pilot.pause()
            draft = json.loads(form.query_one("#recipe-editor", TextArea).text)
            assert draft["training"]["camera_keys"] == ["observation.images.front"]
            assert draft["training"]["steps"] == 12
            assert original != form.query_one("#recipe-editor", TextArea).text
            await reviewed(form, pilot)
            prior_review = form.reviewed
            exact = form.query_one("#recipe-editor", TextArea).text
            form.query_one("#recipe-fields", Button).press()
            await until(lambda: isinstance(app.screen, RecipeFields) and app.screen.is_mounted)
            app.screen.query_one("#field-training-steps", Input).value = "999"
            await pilot.press("escape")
            assert app.screen is form
            assert form.query_one("#recipe-editor", TextArea).text == exact
            assert form.reviewed is prior_review
            assert form.query_one("#recipe-consent", Checkbox).value
            assert not server.posts

    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["project", "draft", "invalid", "stale-runtime"])
def test_pilot_changed_context_and_invalid_values_never_submit(tmp_path, change):
    async def scenario():
        server = ExtendedServer()
        app = FirebirdApp(client=server.client(), poll_seconds=100, journal_dir=tmp_path / "state")
        async with app.run_test(size=(48, 18)) as pilot:
            form = await open_form(server, tmp_path, "quantize", app, pilot)
            original = form.query_one("#recipe-editor", TextArea).text
            form.query_one("#recipe-fields", Button).press()
            await until(lambda: isinstance(app.screen, RecipeFields) and app.screen.is_mounted)
            editor = app.screen
            if change == "project":
                app.project_id = "changed"
            elif change == "draft":
                form.query_one("#recipe-editor", TextArea).load_text("{}")
            elif change == "invalid":
                editor.query_one("#field-timeout_seconds", Input).value = "NaN"
            else:
                server.runtimes[2]["enabled"] = False
            editor.query_one("#fields-apply", Button).press()
            if change == "stale-runtime":
                await until(lambda: app.screen is form)
                await pilot.pause()
                form.query_one("#recipe-review", Button).press()
                await until(
                    lambda: "runtime" in str(form.query_one("#lifecycle-error", Static).content)
                )
                assert form.reviewed is None
            else:
                await pilot.pause()
                assert app.screen is editor
                assert str(editor.query_one("#fields-error", Static).content)
                assert form.query_one("#recipe-editor", TextArea).text == (
                    "{}" if change == "draft" else original
                )
            assert not server.posts and form.journal.read() is None

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ("distill", "replay", "evaluate", "engine_run", "gguf_quantize"))
def test_structured_fields_cover_new_drafts_without_implicit_budgets(mode):
    ctx = asyncio.run(context(ExtendedServer()))
    draft = template(
        mode,
        mode if mode in BASE_MODES else "engine",
        "packed" if mode == "replay" else "policy",
        "data",
        protocol="libero_spatial" if mode in {"evaluate", "engine_run"} else None,
    )
    editor = RecipeFields(mode, draft, ctx)
    assert editor.initial["timeout_seconds"] == ""
    paths = {field.path for field in editor.fields}
    if mode in {"evaluate", "engine_run"}:
        assert {
            "evaluation.task_ids",
            "evaluation.parity_limits.max_rmse",
            "evaluation.final_states",
        } <= paths
        assert editor.initial["evaluation.steps"] == "280"
    if mode in {"distill", "replay"}:
        key = "native_distillation" if mode == "distill" else "native_replay"
        assert editor.initial[f"{key}.units"] == [""] * 6


@pytest.mark.parametrize("mode", ("resume", "export"))
def test_saved_training_identity_and_recipe_have_no_editable_override(mode):
    ctx = asyncio.run(context(ExtendedServer()))
    draft = complete(mode)
    before = copy.deepcopy(draft)
    editor = RecipeFields(mode, draft, ctx)
    assert [field.path for field in editor.fields] == ["timeout_seconds"]
    assert draft == before


def test_pilot_advanced_adapter_change_is_rejected_and_noop_apply_revokes_consent(tmp_path):
    async def scenario():
        server = ExtendedServer()
        server.jobs[0]["result"]["features"]["observation.images.wrist"] = {
            "dtype": "video",
            "shape": [3, 32, 32],
        }
        app = FirebirdApp(client=server.client(), poll_seconds=100, journal_dir=tmp_path / "state")
        async with app.run_test(size=(48, 18)) as pilot:
            form = await open_form(server, tmp_path, "train", app, pilot)
            draft = recipe("train")
            draft["runtime_id"] = "engine"
            draft["training"].pop("camera_key")
            draft["training"]["camera_keys"] = [
                "observation.images.wrist",
                "observation.images.front",
            ]
            draft["training"]["adapter_option"] = {"keep": True}
            form.query_one("#recipe-editor", TextArea).load_text(canonical(draft))
            await pilot.pause()
            await reviewed(form, pilot)
            exact = form.query_one("#recipe-editor", TextArea).text
            form.query_one("#recipe-fields", Button).press()
            await until(lambda: isinstance(app.screen, RecipeFields) and app.screen.is_mounted)
            editor = app.screen
            assert (
                editor.query_one("#field-training-camera_keys", SelectionList).selected
                == draft["training"]["camera_keys"]
            )
            editor.query_one("#field-training-model_id", Select).value = "fixture/smolvla"
            editor.query_one("#field-training_method", Select).value = "lora"
            editor.query_one("#fields-apply", Button).press()
            await pilot.pause()
            assert app.screen is editor
            assert "advanced training" in str(editor.query_one("#fields-error", Static).content)
            assert form.query_one("#recipe-editor", TextArea).text == exact
            editor.query_one("#field-training-model_id", Select).value = "fixture/act"
            editor.query_one("#field-training_method", Select).value = "full"
            editor.query_one("#fields-apply", Button).press()
            await until(lambda: app.screen is form)
            await pilot.pause()
            assert json.loads(form.query_one("#recipe-editor", TextArea).text) == draft
            assert form.reviewed is None and not form.query_one("#recipe-consent", Checkbox).value
            assert not server.posts

    asyncio.run(scenario())


@pytest.mark.parametrize("protocol", ("engine", "libero_object", "libero_spatial"))
def test_pilot_evaluation_from_blank_fields_requires_disjoint_explicit_selections(
    tmp_path, protocol
):
    async def scenario():
        server = ExtendedServer()
        if protocol == "libero_spatial":
            server.artifacts[-1]["metadata"]["task"] = "libero_spatial"
        app = FirebirdApp(client=server.client(), poll_seconds=100, journal_dir=tmp_path / "state")
        async with app.run_test(size=(48, 18)) as pilot:
            form = await open_form(server, tmp_path, "train", app, pilot)
            form.query_one("#lifecycle-mode", Select).value = "evaluate"
            await pilot.pause()
            for name, value in (
                ("runtime", "engine"),
                ("artifact", "gguf"),
                ("protocol", protocol),
            ):
                form.query_one(f"#recipe-{name}", Select).value = value
            form.query_one("#recipe-build", Button).press()
            await pilot.pause()
            form.query_one("#recipe-fields", Button).press()
            await until(lambda: isinstance(app.screen, RecipeFields) and app.screen.is_mounted)
            editor = app.screen
            values = {
                "timeout_seconds": "600",
                "evaluation-warmups": "1",
                "evaluation-repetitions": "2",
            }
            if protocol != "engine":
                values.update(
                    {
                        "evaluation-initial_states": "0,1",
                        "evaluation-final_states": "1,2",
                        "evaluation-seed": "0",
                        "evaluation-steps": "280",
                    }
                )
                if protocol == "libero_spatial":
                    values.update(
                        {
                            "evaluation-task_ids": "0,1",
                            "evaluation-parity_limits-profile": "generated-software-test",
                            "evaluation-parity_limits-max_rmse": "0.1",
                            "evaluation-parity_limits-max_abs_error": "0.2",
                        }
                    )
                else:
                    values["evaluation-task_id"] = "0"
            for name, value in values.items():
                editor.query_one(f"#field-{name}", Input).value = value
            if protocol != "engine":
                editor.query_one("#fields-apply", Button).press()
                await pilot.pause()
                assert app.screen is editor
                assert "disjoint" in str(editor.query_one("#fields-error", Static).content)
                editor.query_one("#field-evaluation-final_states", Input).value = "2,3"
            editor.query_one("#fields-apply", Button).press()
            await until(lambda: app.screen is form)
            await pilot.pause()
            draft = json.loads(form.query_one("#recipe-editor", TextArea).text)
            assert draft["evaluation"]["mode"] == ("engine" if protocol == "engine" else "libero")
            if protocol != "engine":
                assert draft["evaluation"]["final_states"] == [2, 3]
                assert draft["evaluation"]["initial_states"] == [0, 1]
            assert not server.posts
            await reviewed(form, pilot)
            assert not form.query_one("#recipe-submit", Button).disabled

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ("distill", "replay"))
def test_pilot_native_recorded_data_fields_do_not_invent_units_or_episode_splits(tmp_path, mode):
    async def scenario():
        server = ExtendedServer()
        app = FirebirdApp(client=server.client(), poll_seconds=100, journal_dir=tmp_path / "state")
        async with app.run_test(size=(48, 18)) as pilot:
            form = await open_form(server, tmp_path, mode, app, pilot)
            for name, value in (
                ("runtime", mode),
                ("dataset", "data"),
                ("artifact", "packed" if mode == "replay" else "policy"),
            ):
                form.query_one(f"#recipe-{name}", Select).value = value
            form.query_one("#recipe-build", Button).press()
            await pilot.pause()
            form.query_one("#recipe-fields", Button).press()
            await until(lambda: isinstance(app.screen, RecipeFields) and app.screen.is_mounted)
            editor = app.screen
            editor.query_one("#fields-apply", Button).press()
            await pilot.pause()
            assert app.screen is editor and not server.posts
            editor.query_one("#field-timeout_seconds", Input).value = "600"
            prefix = "native_distillation" if mode == "distill" else "native_replay"
            editor.query_one(
                f"#field-{prefix}-coordinate_attestation", Select
            ).value = "generated_fixture"
            for index in range(6):
                editor.query_one(f"#field-{prefix}-units-{index}", Input).value = "generated"
            if mode == "distill":
                values = {
                    "steps": "1",
                    "learning_rate": "0.0001",
                    "seed": "1",
                    "frame_stride": "1",
                    "splits-train": "0",
                    "splits-validation": "1",
                    "splits-final": "2",
                }
            else:
                values = {"selection": "0:0,0:0"}
            for name, value in values.items():
                editor.query_one(f"#field-{prefix}-{name}", Input).value = value
            if mode == "replay":
                editor.query_one("#fields-apply", Button).press()
                await pilot.pause()
                assert app.screen is editor
                assert "distinct" in str(editor.query_one("#fields-error", Static).content)
                editor.query_one(f"#field-{prefix}-selection", Input).value = "0:0"
            editor.query_one("#fields-apply", Button).press()
            await until(lambda: app.screen is form)
            await pilot.pause()
            assert json.loads(form.query_one("#recipe-editor", TextArea).text) == complete(mode)
            assert not server.posts

    asyncio.run(scenario())
