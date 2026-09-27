"""Editable recipes with fresh, identity-bound review and one explicit API submission."""

import asyncio
import json

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import HorizontalScroll, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Collapsible, Label, Select, Static, TextArea

from vla_platform.cli_client import validate_acknowledgment
from vla_platform.tui_client import ApiError, plain, segment
from vla_platform.tui_lifecycle import (
    GUIDANCE,
    OPERATIONS,
    AttemptJournal,
    Context,
    artifact_allowed,
    canonical,
    dataset_allowed,
    parse_recipe,
    runtime_allowed,
    template,
)
from vla_platform.tui_policy_modes import (
    interrupted_training,
    resume_context,
    visible_resume_recipe,
)
from vla_platform.tui_recipe_fields import RecipeFields


class LifecycleForm(ModalScreen):
    BINDINGS = [("escape", "back", "Back")]
    CSS = """
    LifecycleForm { align: center middle; background: $background 70%; }
    LifecycleForm > VerticalScroll { width: 94; max-width: 100%; height: 95%;
        border: round $accent; padding: 1 2; background: $surface; }
    LifecycleForm Label, LifecycleForm Static { height: auto; margin-bottom: 1; }
    LifecycleForm Select, LifecycleForm Checkbox { margin-bottom: 1; }
    LifecycleForm HorizontalScroll { height: 3; }
    #recipe-editor { height: 16; min-height: 8; margin-bottom: 1; }
    #review-recipe { height: 12; }
    #lifecycle-error { color: $warning; }
    .hidden { display: none; }
    """

    def __init__(
        self, client, context: Context, journal: AttemptJournal, still_current, saved=None
    ):
        super().__init__()
        self.client, self.context, self.journal = client, context, journal
        self.still_current, self.saved = still_current, saved
        self.reviewed = None
        self.busy = False
        self.blocked = False
        self.storage_failed = False
        self.attempt = None
        self.history_reviewed = False
        self.accepted = None

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Label(Text(f"New lifecycle job · {plain(self.context.project['name'])}"))
            yield Static(
                "The application owns execution. Choose explicit inputs, edit the recipe, "
                "then review before submitting. No job starts while navigating.",
                markup=False,
            )
            yield Select(
                [(x[0], key) for key, x in OPERATIONS.items()],
                value="train",
                allow_blank=False,
                id="lifecycle-mode",
            )
            yield Static("", id="recipe-guidance", markup=False)
            yield Label(
                "Evaluation protocol (choose before building the draft)", id="protocol-label"
            )
            yield Select(
                [
                    ("Engine inference checks", "engine"),
                    ("LIBERO Object", "libero_object"),
                    ("LIBERO Spatial", "libero_spatial"),
                ],
                prompt="Choose a protocol",
                id="recipe-protocol",
            )
            yield Label("Registered compute (configured capability, not model validation)")
            yield Select([], prompt="Choose compute", id="recipe-runtime")
            yield Label("Completed dataset intake")
            yield Select([], prompt="Choose a dataset when required", id="recipe-dataset")
            yield Label("Registered policy artifact")
            yield Select([], prompt="Choose a policy when required", id="recipe-artifact")
            yield Label("Interrupted training job (alternative to a checkpoint)", id="resume-label")
            yield Select(
                [],
                prompt="Choose one interrupted job, or use the policy selector",
                id="recipe-resume",
            )
            yield Static("", id="model-guidance", markup=False)
            with HorizontalScroll():
                yield Button("Build / replace draft", id="recipe-build")
                yield Button(
                    "Copy selected job recipe", id="recipe-copy", disabled=self.saved is None
                )
            yield Static(
                "Null/empty required fields must be chosen. Fixed adapter identifiers "
                "come from the API contract; no episode, budget or coordinate units "
                "are inferred. Changing selectors does not overwrite your draft.",
                markup=False,
            )
            yield Button("Edit with labelled fields", id="recipe-fields")
            with Collapsible(title="Advanced: complete JSON recipe", collapsed=True):
                yield TextArea("{}", id="recipe-editor", show_line_numbers=True)
            yield Button("Review exact recipe", id="recipe-review", variant="primary")
            yield Static("", id="accepted-receipt", markup=False)
            yield Static("", id="lifecycle-error", markup=False)
            with VerticalScroll(id="recipe-reviewed", classes="hidden"):
                yield Static("", id="review-summary", markup=False)
                yield TextArea("", id="review-recipe", read_only=True)
                yield Checkbox(
                    "I reviewed this exact source, recipe and compute target", id="recipe-consent"
                )
                yield Checkbox(
                    "I authorize this paid Google Cloud job; timeout is not a spending cap",
                    id="recipe-cloud-consent",
                )
                yield Button(
                    "Submit this job once", id="recipe-submit", variant="primary", disabled=True
                )
            with VerticalScroll(id="attempt-recovery", classes="hidden"):
                yield Static("", id="attempt-summary", markup=False)
                yield Button("Load saved job history", id="attempt-history")
                yield Static("", id="attempt-jobs", markup=False)
                yield Checkbox(
                    "I inspected history for this exact attempt before allowing a new request",
                    id="attempt-consent",
                    disabled=True,
                )
                yield Button("Clear this attempt block", id="attempt-clear", disabled=True)
            yield Button("Back to jobs", id="lifecycle-back")

    def message(self, value):
        self.query_one("#lifecycle-error", Static).update(Text(plain(value)))

    def on_mount(self):
        self.mode_changed()
        self.load_journal()
        self.query_one("#lifecycle-mode", Select).focus()

    def action_back(self):
        if self.busy:
            self.message(
                "A request is in progress. Closing the TUI preserves pending evidence; "
                "it does not cancel application jobs."
            )
        else:
            self.dismiss(self.accepted)

    @on(Button.Pressed, "#lifecycle-back")
    def back(self):
        self.action_back()

    def load_journal(self):
        try:
            self.attempt = self.journal.read()
            self.blocked = self.storage_failed or self.attempt is not None
        except Exception:
            self.blocked = True
            self.message(
                "Attempt storage is unavailable or invalid. Restore it and reopen; "
                "inspect saved jobs before any new submission."
            )
        self.history_reviewed = False
        self.query_one("#attempt-recovery").set_class(self.attempt is None, "hidden")
        if self.attempt:
            self.query_one("#attempt-summary", Static).update(
                Text(
                    f"Earlier {self.attempt['state']} attempt {self.attempt['attempt_id']}\n"
                    f"{self.attempt['operation']} · recipe SHA256 {self.attempt['recipe_sha256']}\n"
                    "Outcome may already be recorded. A refresh alone does not authorize retry."
                )
            )
        self.controls()

    def controls(self):
        self.query_one("#recipe-review", Button).disabled = self.busy or self.blocked
        consent = self.query_one("#recipe-consent", Checkbox).value
        cloud = self.reviewed is not None and self.reviewed.runtime["provider"] == "gcp"
        cloud_ok = self.query_one("#recipe-cloud-consent", Checkbox).value
        self.query_one("#recipe-submit", Button).disabled = (
            self.busy
            or self.blocked
            or self.reviewed is None
            or not consent
            or (cloud and not cloud_ok)
        )
        self.query_one("#attempt-history", Button).disabled = self.busy
        self.query_one("#attempt-consent", Checkbox).disabled = (
            not self.history_reviewed or self.busy
        )
        self.query_one("#attempt-clear", Button).disabled = (
            self.busy
            or not self.history_reviewed
            or not self.query_one("#attempt-consent", Checkbox).value
        )
        mode = str(self.query_one("#lifecycle-mode", Select).value)
        for name in (
            "recipe-build",
            "recipe-copy",
            "lifecycle-mode",
            "recipe-runtime",
            "recipe-dataset",
            "recipe-artifact",
            "recipe-resume",
            "recipe-protocol",
            "recipe-editor",
            "recipe-fields",
        ):
            self.query_one("#" + name).disabled = (
                self.busy
                or (name == "recipe-copy" and self.saved is None)
                or (name == "recipe-dataset" and mode not in {"train", "distill", "replay"})
                or (name == "recipe-artifact" and mode == "train")
                or (name == "recipe-resume" and mode != "resume")
                or (name == "recipe-protocol" and mode not in {"evaluate", "engine_run"})
            )

    @on(Checkbox.Changed)
    def consent_changed(self):
        self.controls()

    @on(Select.Changed, "#lifecycle-mode")
    def mode_changed(self):
        mode = str(self.query_one("#lifecycle-mode", Select).value)
        self.reviewed = None
        self.query_one("#recipe-reviewed").add_class("hidden")
        self.query_one("#recipe-guidance", Static).update(Text(GUIDANCE[mode]))
        for selector in ("#protocol-label", "#recipe-protocol"):
            self.query_one(selector).set_class(mode not in {"evaluate", "engine_run"}, "hidden")
        self.query_one("#recipe-runtime", Select).set_options(
            [
                (Text(plain(f"{r['label']} · {r['provider_label']} · {r['device']}")), r["id"])
                for r in self.context.runtimes
                if runtime_allowed(r, mode)
            ]
        )
        self.query_one("#recipe-artifact", Select).set_options(
            [
                (Text(plain(f"{a['label']} · {a['id']}")), a["id"])
                for a in self.context.artifacts
                if artifact_allowed(a, mode)
            ]
        )
        self.query_one("#recipe-dataset", Select).set_options(
            [
                (Text(plain(f"{j['id']} · {(j.get('result') or {}).get('source', '')}")), j["id"])
                for j in self.context.jobs
                if dataset_allowed(j, mode)
            ]
        )
        self.query_one("#recipe-artifact", Select).disabled = mode == "train"
        self.query_one("#recipe-dataset", Select).disabled = mode not in {
            "train",
            "distill",
            "replay",
        }
        self.query_one("#recipe-resume", Select).set_options(
            [
                (Text(plain(f"{j['id']} · {j['status']}")), j["id"])
                for j in self.context.jobs
                if interrupted_training(j)
            ]
        )
        for selector in ("#resume-label", "#recipe-resume"):
            self.query_one(selector).set_class(mode != "resume", "hidden")
        models = "\n".join(
            f"{m['model_id']}: {', '.join(m['methods'])} · {m.get('status', 'unknown')}"
            for m in self.context.models
        )
        cameras = "\n".join(
            f"Dataset {j['id']} camera features: "
            + ", ".join(
                key
                for key in (j.get("result") or {}).get("features", {})
                if key.startswith("observation.images.")
            )
            for j in self.context.jobs
            if dataset_allowed(j, "train")
        )
        self.query_one("#model-guidance", Static).update(
            Text(plain(models + "\n" + cameras) if mode == "train" else "")
        )

    def selected(self, name):
        selector = self.query_one("#recipe-" + name, Select)
        return None if selector.is_blank() else str(selector.value)

    @on(Button.Pressed, "#recipe-build")
    def build(self):
        mode = str(self.query_one("#lifecycle-mode", Select).value)
        value = template(
            mode,
            self.selected("runtime"),
            self.selected("artifact"),
            self.selected("dataset"),
            self.selected("resume") if mode == "resume" else None,
            self.selected("protocol") if mode in {"evaluate", "engine_run"} else None,
        )
        if mode == "resume" and (value["artifact_id"] or value["resume_job_id"]):
            try:
                saved, metadata, _ = resume_context(
                    self.context, value["artifact_id"], value["resume_job_id"]
                )
            except ApiError as exc:
                self.message(str(exc))
                return
            value.update(
                dataset_job_id=saved.get("dataset_job_id"),
                training_method=visible_resume_recipe(saved, metadata).get(
                    "method", saved["training_method"]
                ),
            )
        if mode == "export":
            source = next(
                (a for a in self.context.artifacts if a["id"] == value["artifact_id"]), None
            )
            if source:
                value["training_method"] = source["metadata"].get("method")
        self.query_one("#recipe-editor", TextArea).load_text(json.dumps(value, indent=2))
        self.message("Draft replaced. Open labelled fields to choose settings, then review.")
        self.query_one("#recipe-fields", Button).focus()

    @on(Button.Pressed, "#recipe-fields")
    def edit_fields(self):
        if self.busy:
            return
        mode = str(self.query_one("#lifecycle-mode", Select).value)
        raw = self.query_one("#recipe-editor", TextArea).text

        def unchanged():
            return (
                self.still_current()
                and mode == str(self.query_one("#lifecycle-mode", Select).value)
                and raw == self.query_one("#recipe-editor", TextArea).text
            )

        def applied(value):
            if value is None:
                return
            if not unchanged():
                self.message(
                    "Project or draft changed. Reopen the composer; no fields were applied."
                )
                return
            self.edited()
            self.query_one("#recipe-editor", TextArea).load_text(json.dumps(value, indent=2))
            self.message("Fields applied to the draft. Review the exact recipe before submitting.")
            self.query_one("#recipe-review", Button).focus()

        try:
            editor = RecipeFields(mode, parse_recipe(raw), self.context, unchanged)
        except ApiError as exc:
            self.message(str(exc))
            return
        self.app.push_screen(editor, applied)

    @on(Button.Pressed, "#recipe-copy")
    def copy(self):
        if self.saved:
            self.query_one("#recipe-editor", TextArea).load_text(json.dumps(self.saved, indent=2))
            self.message(
                "Copied for editing only. Check current operation/inputs; "
                "this is not a retry or resume."
            )

    @on(TextArea.Changed, "#recipe-editor")
    def edited(self):
        self.reviewed = None
        self.query_one("#recipe-reviewed").add_class("hidden")
        self.query_one("#recipe-consent", Checkbox).value = False
        self.query_one("#recipe-cloud-consent", Checkbox).value = False
        self.controls()

    def start(self, method):
        if self.busy:
            return
        self.busy = True
        self.controls()
        self.run_worker(method, group="lifecycle", exit_on_error=False)

    @on(Button.Pressed, "#recipe-review")
    def review(self):
        self.start(self.prepare_review)

    async def prepare_review(self):
        raw = self.query_one("#recipe-editor", TextArea).text
        mode = str(self.query_one("#lifecycle-mode", Select).value)
        try:
            context = await Context.fetch(self.client, self.context.project["id"])
            if not self.still_current():
                raise ApiError("Project selection changed. Reopen the composer.")
            reviewed = context.review(mode, raw)
            self.context, self.reviewed = context, reviewed
            target = reviewed.runtime
            summary = (
                f"Project {context.project['id']} · {target['label']} · "
                f"{target['provider_label']}\n"
                f"Runtime {target['id']} · timeout {reviewed.request['timeout_seconds']}s\n"
            )
            if reviewed.artifact:
                summary += (
                    f"Policy {reviewed.artifact['id']}\n"
                    f"Manifest {reviewed.artifact['manifest_sha256']}\n"
                )
            if reviewed.dataset:
                summary += (
                    f"Dataset {reviewed.dataset['id']} · immutable identity retained in review\n"
                )
            if reviewed.source_jobs:
                summary += (
                    "Saved lineage: " + " → ".join(j["id"] for j in reviewed.source_jobs) + "\n"
                )
            self.query_one("#review-summary", Static).update(Text(plain(summary + GUIDANCE[mode])))
            self.query_one("#review-recipe", TextArea).load_text(
                json.dumps(reviewed.request, indent=2)
            )
            self.query_one("#recipe-consent", Checkbox).value = False
            self.query_one("#recipe-cloud-consent", Checkbox).value = False
            self.query_one("#recipe-cloud-consent").set_class(target["provider"] != "gcp", "hidden")
            self.query_one("#recipe-reviewed").remove_class("hidden")
            self.message(
                "Review is current. The server still validates all source files "
                "and runtime admission."
            )
        except ApiError as exc:
            self.reviewed = None
            self.message(str(exc))
        finally:
            self.busy = False
            self.controls()

    @on(Button.Pressed, "#recipe-submit")
    def submit(self):
        if self.query_one("#recipe-submit", Button).disabled:
            return
        self.start(self.send_once)

    async def send_once(self):
        previous = self.reviewed
        record = None
        try:
            context = await Context.fetch(self.client, self.context.project["id"])
            current = context.review(previous.mode, canonical(previous.request))
            if not self.still_current() or current.context_sha256 != previous.context_sha256:
                raise ApiError(
                    "Source, dataset, compute or project changed. "
                    "Review again; no submission was sent."
                )
            with self.journal.claim(previous) as record:
                value = await self.client.request(
                    "POST",
                    f"/projects/{segment(context.project['id'])}/policy-jobs",
                    previous.request,
                )
                validate_acknowledgment(
                    self.client,
                    f"/projects/{segment(context.project['id'])}/policy-jobs",
                    previous.request,
                    value,
                )
                # Preserve the accepted receipt before any fallible disk cleanup.
                self.accepted = value
                self.query_one("#accepted-receipt", Static).update(
                    Text(
                        f"Job accepted: {plain(value['id'])} · {plain(value['status'])}. "
                        "Follow it in Jobs; do not submit it again."
                    )
                )
                self.journal.finish(record)
            self.dismiss(self.accepted)
        except asyncio.CancelledError:
            self.reviewed = None
            if self.is_mounted:
                self.load_journal()
            raise
        except Exception as exc:
            if record is not None and self.accepted is None:
                try:
                    self.journal.finish(record, uncertain=True)
                except Exception:
                    pass  # Preserve whichever durable record remains; never retry the POST.
            if isinstance(exc, ApiError):
                self.message(str(exc) + " No automatic retry was made.")
            else:
                self.storage_failed = True
                self.message(
                    "Attempt storage is unavailable. Inspect saved jobs and restore storage before "
                    "another request. No automatic retry was made."
                )
            self.reviewed = None
            self.load_journal()
        finally:
            self.busy = False
            if self.is_mounted:
                self.controls()

    @on(Button.Pressed, "#attempt-history")
    def history(self):
        self.start(self.load_history)

    async def load_history(self):
        try:
            context = await Context.fetch(self.client, self.context.project["id"])
            if not self.still_current():
                raise ApiError("Project selection changed; reopen this project.")
            current = self.journal.read()
            if (
                current is None
                or self.attempt is None
                or canonical(current) != canonical(self.attempt)
            ):
                raise ApiError("Attempt changed. Reopen and inspect the current attempt.")
            self.context = context
            lines = [
                f"{j['id']} · {j['kind']} · {j['status']} · {j['created_at']}" for j in context.jobs
            ]
            self.query_one("#attempt-jobs", Static).update(
                Text(
                    plain(
                        "Fresh saved history (last50):\n" + "\n".join(lines[-50:])
                        if lines
                        else "No recorded jobs."
                    )
                )
            )
            self.history_reviewed = True
            self.query_one("#attempt-consent", Checkbox).value = False
        except ApiError, OSError:
            self.history_reviewed = False
            self.message(
                "Could not confirm current attempt and history. Keep the block and refresh."
            )
        finally:
            self.busy = False
            self.controls()

    @on(Button.Pressed, "#attempt-clear")
    def clear(self):
        if self.query_one("#attempt-clear", Button).disabled:
            return
        try:
            self.journal.finish(self.attempt)
            self.load_journal()
            self.message(
                "This exact attempt was acknowledged after history review. "
                "Review any new recipe explicitly."
            )
        except Exception:
            self.blocked = True
            self.message(
                "Attempt changed or storage failed. "
                "Nothing was resubmitted; reopen and inspect history."
            )
            self.controls()
