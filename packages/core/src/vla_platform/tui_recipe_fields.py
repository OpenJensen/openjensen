"""Local, labelled editors for existing recipes; execution remains in LifecycleForm."""

import copy
import math
import re
from dataclasses import dataclass

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import HorizontalScroll, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, SelectionList, Static

from vla_platform.tui_client import ApiError, plain
from vla_platform.tui_lifecycle import GUIDANCE, OPERATIONS, canonical


@dataclass(frozen=True)
class Field:
    path: str
    label: str
    kind: str = "integer"
    choices: tuple = ()

    @property
    def ident(self):
        return "field-" + self.path.replace(".", "-")


def at(value, path):
    for part in path.split("."):
        if not isinstance(value, dict):
            raise ApiError(f"{path}: expected an object. Correct the advanced recipe first.")
        value = value.get(part)
    return value


def put(value, path, item):
    parts = path.split(".")
    for part in parts[:-1]:
        value = value[part]
    value[parts[-1]] = item


def fields_for(mode, draft, context):
    fields = [Field("timeout_seconds", "Execution timeout in seconds (30–86400)")]
    if mode == "train":
        runtime = next((r for r in context.runtimes if r["id"] == draft.get("runtime_id")), {})
        models = tuple(
            m["model_id"]
            for m in context.models
            if m["id"] in runtime.get("training_model_ids", [])
        )
        methods = tuple(
            dict.fromkeys(
                method for m in context.models if m["model_id"] in models for method in m["methods"]
            )
        )
        fields += [
            Field("training.model_id", "Registered model", "choice", models),
            Field(
                "training_method",
                "Training method (must match the chosen model)",
                "choice",
                methods,
            ),
            Field("training.steps", "Training steps"),
            Field("training.batch_size", "Batch size"),
            Field("training.learning_rate", "Learning rate", "number"),
            Field(
                "training.camera_keys",
                "Dataset cameras — Space selects; order is preserved",
                "cameras",
            ),
        ]
    elif mode == "quantize":
        fields += [Field("native_quantization.bits", "Packed ACT precision", "choice", (8, 4))]
    elif mode in {"distill", "replay"}:
        prefix = "native_distillation" if mode == "distill" else "native_replay"
        if mode == "distill":
            fields += [
                Field(f"{prefix}.steps", "Student training steps (1–10000)"),
                Field(f"{prefix}.learning_rate", "Learning rate (0.0000001–0.001)", "number"),
                Field(f"{prefix}.seed", "Random seed"),
                Field(f"{prefix}.frame_stride", "Frame stride"),
                Field(
                    f"{prefix}.splits.train", "Training episode IDs, comma separated", "integers"
                ),
                Field(
                    f"{prefix}.splits.validation",
                    "Validation episode IDs, comma separated",
                    "integers",
                ),
                Field(
                    f"{prefix}.splits.final",
                    "Unused final episode IDs, comma separated",
                    "integers",
                ),
            ]
        else:
            fields += [
                Field(
                    f"{prefix}.selection",
                    "Observations: episode:frame pairs, comma separated",
                    "observations",
                )
            ]
        attestation = (
            "teacher_recorded_coordinates" if mode == "distill" else "policy_recorded_coordinates"
        )
        fields += [
            Field(
                f"{prefix}.coordinate_attestation",
                "Coordinate evidence (generated_fixture is only for generated data)",
                "choice",
                (attestation, "generated_fixture"),
            ),
            Field(
                f"{prefix}.units",
                "Six coordinate units in the recorded action-vector order",
                "units",
            ),
        ]
    elif mode == "gguf_quantize":
        fields += [
            Field("precision.language", "Language precision", "choice", ("Q8_0", "Q4_0")),
            Field(
                "precision.vision",
                "Vision precision (blank keeps original)",
                "optional_choice",
                ("Q8_0",),
            ),
        ]
    elif mode in {"evaluate", "engine_run"}:
        evaluation = draft.get("evaluation")
        if not isinstance(evaluation, dict) or evaluation.get("mode") not in {"engine", "libero"}:
            raise ApiError(
                "Choose an evaluation protocol, then build the draft before editing fields."
            )
        fields += [
            Field("evaluation.warmups", "Warmup iterations (1–100)"),
            Field("evaluation.repetitions", "Measured repetitions (2–1000)"),
        ]
        if evaluation["mode"] == "libero":
            spatial = evaluation.get("suite") == "libero_spatial"
            fields += [
                Field(
                    "evaluation.task_ids" if spatial else "evaluation.task_id",
                    "Task IDs (0–9), comma separated" if spatial else "Task ID (0–9)",
                    "integers" if spatial else "integer",
                ),
                Field("evaluation.initial_states", "Search state IDs, comma separated", "integers"),
                Field(
                    "evaluation.final_states", "Unused final state IDs, comma separated", "integers"
                ),
                Field("evaluation.seed", "Random seed"),
                Field("evaluation.steps", "Horizon (Spatial requires 280; Object 1–500)"),
            ]
            if spatial:
                fields += [
                    Field(
                        "evaluation.parity_limits.profile", "Parity threshold profile name", "text"
                    ),
                    Field("evaluation.parity_limits.max_rmse", "Maximum action RMSE", "number"),
                    Field(
                        "evaluation.parity_limits.max_abs_error",
                        "Maximum absolute action error",
                        "number",
                    ),
                ]
    return fields


def display_value(field, value):
    """Reject lossy conversions of an existing draft instead of silently repairing it."""
    if value is None:
        return ""
    kind = field.kind
    if kind == "integer" and type(value) is int:
        return str(value)
    if kind == "number" and type(value) in (int, float):
        try:
            if math.isfinite(value):
                return str(value)
        except OverflowError:
            pass
    if kind == "text" and isinstance(value, str):
        return value
    if kind == "integers" and isinstance(value, list) and all(type(x) is int for x in value):
        return ", ".join(map(str, value))
    if (
        kind == "observations"
        and isinstance(value, list)
        and all(
            isinstance(x, dict)
            and set(x) == {"episode_index", "frame_index"}
            and all(type(n) is int for n in x.values())
            for x in value
        )
    ):
        return ", ".join(f"{x['episode_index']}:{x['frame_index']}" for x in value)
    raise ApiError(
        f"{field.label}: the draft has an unsupported value; correct the advanced recipe first."
    )


def parse_value(field, raw):
    raw = raw.strip()
    if not raw:
        raise ApiError(f"{field.label}: choose a value.")
    if field.kind == "text":
        return raw
    if field.kind == "integer":
        if re.fullmatch(r"-?[0-9]+", raw):
            return int(raw)
    elif field.kind == "number":
        try:
            value = float(raw)
            if math.isfinite(value):
                return value
        except ValueError:
            pass
    elif field.kind == "integers":
        if re.fullmatch(r"[0-9]+(?:\s*,\s*[0-9]+)*", raw):
            return [int(x.strip()) for x in raw.split(",")]
    elif field.kind == "observations":
        if re.fullmatch(r"[0-9]+\s*:\s*[0-9]+(?:\s*,\s*[0-9]+\s*:\s*[0-9]+)*", raw):
            return [
                dict(zip(("episode_index", "frame_index"), map(int, x.split(":")), strict=True))
                for x in raw.split(",")
            ]
    raise ApiError(f"{field.label}: enter a finite number or the displayed list format.")


class RecipeFields(ModalScreen):
    """Editing has no HTTP client, job submission or durable-attempt side effects."""

    BINDINGS = [("escape", "cancel", "Keep original draft")]
    CSS = """
    RecipeFields { align: center middle; background: $background 70%; }
    RecipeFields > VerticalScroll { width: 88; max-width: 100%; height: 95%;
        border: round $accent; padding: 1 2; background: $surface; }
    RecipeFields Label, RecipeFields Static { height: auto; margin-bottom: 1; }
    RecipeFields Input, RecipeFields Select { margin-bottom: 1; }
    RecipeFields SelectionList { height: auto; max-height: 8; margin-bottom: 1; }
    RecipeFields HorizontalScroll { height: 3; }
    #fields-error { color: $warning; }
    """

    def __init__(self, mode, draft, context, still_current=lambda: True):
        super().__init__()
        self.mode, self.context = mode, context
        self.still_current = still_current
        self.draft = copy.deepcopy(draft)
        if self.draft.get("operation") != OPERATIONS[mode][1]:
            raise ApiError("Build a draft for the selected workflow first.")
        self.fields = fields_for(mode, draft, context)
        self.initial = {}
        self.cameras = []
        for field in self.fields:
            value = at(self.draft, field.path)
            if field.kind in {"choice", "optional_choice"}:
                if value is not None and not any(
                    type(value) is type(x) and value == x for x in field.choices
                ):
                    raise ApiError(
                        f"{field.label}: the saved choice is unavailable. "
                        "Correct the advanced recipe or rebuild explicitly."
                    )
                self.initial[field.path] = value
            elif field.kind == "units":
                if value is None:
                    value = []
                if (
                    not isinstance(value, list)
                    or len(value) not in (0, 6)
                    or any(not isinstance(x, str) for x in value)
                ):
                    raise ApiError(
                        "Coordinate units must be empty or six recorded units. "
                        "Correct the advanced recipe first."
                    )
                self.initial[field.path] = value or [""] * 6
            elif field.kind == "cameras":
                legacy = at(self.draft, "training.camera_key")
                if value is None and legacy is not None:
                    value = [legacy]
                value = [] if value is None else value
                dataset = next(
                    (j for j in context.jobs if j["id"] == draft.get("dataset_job_id")), {}
                )
                features = (dataset.get("result") or {}).get("features", {})
                self.cameras = [key for key in features if key.startswith("observation.images.")]
                if (
                    not isinstance(value, list)
                    or any(not isinstance(x, str) or x not in self.cameras for x in value)
                    or len(value) != len(set(value))
                ):
                    raise ApiError(
                        "Saved cameras do not match this dataset. "
                        "Correct the advanced recipe or rebuild explicitly."
                    )
                if legacy is not None and value != [legacy]:
                    raise ApiError(
                        "Both camera_key and camera_keys are present with different values. "
                        "Resolve this in the advanced recipe first."
                    )
                self.initial[field.path] = value
            else:
                self.initial[field.path] = display_value(field, value)
                if len(self.initial[field.path]) > 8192:
                    raise ApiError(f"{field.label}: use the advanced recipe for this long value.")

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Label(Text(f"{OPERATIONS[self.mode][0]} · recipe settings"))
            yield Static(GUIDANCE[self.mode], markup=False)
            yield Static(
                Text(
                    plain(
                        f"Compute: {self.draft.get('runtime_id')}\n"
                        f"Dataset: {self.draft.get('dataset_job_id') or '—'}\n"
                        f"Policy: {self.draft.get('artifact_id') or '—'}\n"
                        f"Interrupted job: {self.draft.get('resume_job_id') or '—'}"
                    )
                )
            )
            if self.mode in {"resume", "export"}:
                yield Static(
                    "Saved model, dataset and training method are retained. "
                    "Only the execution timeout is editable here.",
                    markup=False,
                )
            yield Static(
                "Applying changes only updates the draft. Review and consent are still required "
                "before any job starts. Advanced settings are preserved.",
                markup=False,
            )
            for field in self.fields:
                yield Label(field.label)
                initial = self.initial[field.path]
                if field.kind in {"choice", "optional_choice"}:
                    yield Select(
                        [(Text(plain(x)), x) for x in field.choices],
                        value=Select.NULL if initial is None else initial,
                        prompt="Choose explicitly",
                        id=field.ident,
                    )
                elif field.kind == "units":
                    for index, unit in enumerate(initial):
                        yield Label(f"Coordinate {index + 1} unit")
                        yield Input(unit, id=f"{field.ident}-{index}", max_length=80)
                elif field.kind == "cameras":
                    # SelectionList retains supplied selected order, then newly selected order.
                    keys = [*initial, *(key for key in self.cameras if key not in initial)]
                    yield SelectionList(
                        *[(Text(plain(key)), key, key in initial) for key in keys], id=field.ident
                    )
                else:
                    yield Input(initial, id=field.ident, max_length=8192)
            yield Static("", id="fields-error", markup=False)
            with HorizontalScroll():
                yield Button("Apply fields to draft", id="fields-apply", variant="primary")
                yield Button("Keep original draft", id="fields-cancel")

    def on_mount(self):
        self.query_one("#field-timeout_seconds", Input).focus()

    def action_cancel(self):
        self.dismiss(None)

    @on(Button.Pressed, "#fields-cancel")
    def cancel(self):
        self.action_cancel()

    def value(self):
        result = copy.deepcopy(self.draft)
        for field in self.fields:
            if field.kind in {"choice", "optional_choice"}:
                widget = self.query_one("#" + field.ident, Select)
                if widget.is_blank() and field.kind != "optional_choice":
                    raise ApiError(f"{field.label}: choose a value.")
                value = None if widget.is_blank() else widget.value
            elif field.kind == "units":
                value = [
                    self.query_one(f"#{field.ident}-{i}", Input).value.strip() for i in range(6)
                ]
            elif field.kind == "cameras":
                value = list(self.query_one("#" + field.ident, SelectionList).selected)
                # Preserve a legacy single-camera recipe when it remains equivalent.
                if "camera_key" in result["training"]:
                    if "camera_keys" not in result["training"] and value == [
                        result["training"]["camera_key"]
                    ]:
                        continue
                    result["training"].pop("camera_key")
            else:
                value = parse_value(field, self.query_one("#" + field.ident, Input).value)
            put(result, field.path, value)
        if self.mode == "train":
            changed_adapter = result["training"]["model_id"] != self.draft["training"].get(
                "model_id"
            ) or result["training_method"] != self.draft.get("training_method")
            extra = set(self.draft["training"]) - {
                "model_id",
                "steps",
                "batch_size",
                "learning_rate",
                "camera_key",
                "camera_keys",
            }
            if changed_adapter and extra:
                raise ApiError(
                    "This draft contains advanced training settings. Build a fresh draft "
                    "before changing its model or method; no settings were discarded."
                )
        self.context.review(self.mode, canonical(result))
        return result

    @on(Button.Pressed, "#fields-apply")
    def apply(self):
        try:
            if not self.still_current():
                raise ApiError(
                    "Project or draft changed. Keep the original draft and reopen the composer."
                )
            self.dismiss(self.value())
        except (ApiError, ValueError) as exc:
            error = self.query_one("#fields-error", Static)
            error.update(Text(plain(str(exc))))
            error.scroll_visible()
