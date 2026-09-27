"""Explicit, bounded artifact downloads on the terminal's existing connection."""

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path

from rich.text import Text
from textual import on
from textual.containers import HorizontalScroll, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static, TextArea

from vla_platform.cli_client import DEFAULT_DOWNLOAD_LIMIT, MAX_DOWNLOAD_LIMIT, download_artifact
from vla_platform.lifecycle.contracts import PolicyArtifact
from vla_platform.tui_client import ApiError, plain, segment


async def artifacts(api, project_id):
    records = api.validate(
        list[PolicyArtifact],
        await api.request("GET", f"/projects/{segment(project_id)}/artifacts"),
    )
    api.unique(records)
    if any(item["project_id"] != project_id for item in records):
        raise ApiError("Application returned artifacts for another project.")
    if any(not re.fullmatch(r"[a-f0-9]{64}", item["manifest_sha256"]) for item in records):
        raise ApiError("Artifact registry returned an invalid manifest identity.")
    return records


@dataclass(frozen=True)
class DownloadDraft:
    artifact_id: str
    output: Path
    max_bytes: int
    timeout: int
    expected_sha256: str | None

    @classmethod
    def parse(cls, artifact_id, output, max_bytes, timeout, expected):
        if not artifact_id or not output.strip():
            raise ApiError("Choose a policy artifact and an explicit local output file.")
        if not re.fullmatch(r"[0-9]+", max_bytes) or not re.fullmatch(r"[0-9]+", timeout):
            raise ApiError("Enter whole numbers for the byte limit and deadline.")
        limit, seconds = int(max_bytes), int(timeout)
        if not 1 <= limit <= MAX_DOWNLOAD_LIMIT or not 1 <= seconds <= 3600:
            raise ApiError("Use a 1-byte–100 GiB limit and a 1–3600 second deadline.")
        expected = expected.strip().lower() or None
        if expected is not None and not re.fullmatch(r"[a-f0-9]{64}", expected):
            raise ApiError("Expected archive SHA256 must contain 64 hexadecimal characters.")
        path = Path(output.strip()).expanduser().absolute()
        if path.exists() or path.is_symlink():
            raise ApiError("Output already exists; choose a new file. Nothing was overwritten.")
        if not path.parent.is_dir():
            raise ApiError("Output parent directory must already exist.")
        return cls(artifact_id, path, limit, seconds, expected)


class ArtifactDownloadForm(ModalScreen):
    BINDINGS = [("escape", "back", "Back / stop transfer")]
    CSS = """
    ArtifactDownloadForm { align: center middle; background: $background 70%; }
    ArtifactDownloadForm > VerticalScroll {
        width: 86; max-width: 100%; height: auto; max-height: 95%;
        border: round $accent; padding: 1 2; background: $surface;
    }
    ArtifactDownloadForm Label, ArtifactDownloadForm Static { height: auto; margin-bottom: 1; }
    ArtifactDownloadForm Input, ArtifactDownloadForm Select { margin-bottom: 1; }
    ArtifactDownloadForm HorizontalScroll { height: 3; }
    #download-message { color: $warning; }
    #download-receipt { height: 9; }
    """

    def __init__(self, api, project_id, records, current):
        super().__init__()
        self.api, self.project_id, self.records, self.current = api, project_id, records, current
        self.transfer_task = None
        self.stop_requested = False
        self.busy = False
        self.reviewed = None
        self.receipt = None
        self.closing = False

    def compose(self):
        with VerticalScroll():
            yield Label("Save a policy archive")
            yield Static(
                "Saves a TAR file on this computer. It does not extract files or run a model. "
                "The byte limit covers the archive, including its packaging overhead.",
                markup=False,
            )
            yield Select(
                [
                    (Text(plain(f"{a['label']} · {a['format']} · {a['id']}")), a["id"])
                    for a in self.records
                ],
                prompt="Choose a registered artifact",
                id="download-artifact",
            )
            yield Label("New output file on this computer")
            yield Input(placeholder="/existing/folder/policy.tar", id="download-output")
            yield Label("Maximum archive bytes")
            yield Input(str(DEFAULT_DOWNLOAD_LIMIT), id="download-limit")
            yield Label("Deadline in seconds")
            yield Input("600", id="download-timeout")
            yield Label("Expected archive SHA256 (optional; not the manifest hash)")
            yield Input(id="download-sha", max_length=64)
            yield Static(
                "Choose a policy and output, then review.", id="download-message", markup=False
            )
            yield Static("", id="download-summary", markup=False)
            with HorizontalScroll():
                yield Button("Review download", id="download-review", variant="primary")
                yield Button("Save archive", id="download-submit", disabled=True)
                yield Button("Stop transfer", id="download-stop", disabled=True)
                yield Button("Back", id="download-back")
            yield TextArea("No archive saved yet.", read_only=True, id="download-receipt")

    def message(self, text):
        if not self.closing:
            for widget in self.query("#download-message").results(Static):
                widget.update(Text(plain(text)))

    def controls(self):
        if self.closing:
            return
        for name in ("artifact", "output", "limit", "timeout", "sha", "review"):
            for widget in self.query("#download-" + name):
                widget.disabled = self.busy
        for name, disabled in {
            "submit": self.busy or self.reviewed is None,
            "stop": not self.busy,
            "back": self.busy,
        }.items():
            for widget in self.query("#download-" + name).results(Button):
                widget.disabled = disabled

    def draft(self):
        selected = self.query_one("#download-artifact", Select)
        return DownloadDraft.parse(
            None if selected.is_blank() else str(selected.value),
            *[
                self.query_one("#download-" + name, Input).value
                for name in ("output", "limit", "timeout", "sha")
            ],
        )

    def start(self, coroutine):
        self.stop_requested = False
        self.transfer_task = asyncio.create_task(coroutine)
        self.transfer_task.add_done_callback(self.finished)

    def finished(self, task):
        # Cancellation before a coroutine starts does not enter its finally block.
        if self.transfer_task is task:
            self.busy = False
            if task.cancelled():
                self.message(
                    "Operation stopped. No completed output was published; "
                    "application jobs were unchanged."
                )
            self.controls()

    @on(Input.Changed)
    @on(Select.Changed)
    def changed(self):
        self.reviewed = None
        for widget in self.query("#download-summary").results(Static):
            widget.update("")
        self.controls()

    @on(Button.Pressed, "#download-review")
    def review(self):
        if self.busy or self.closing:
            return
        try:
            draft = self.draft()
        except (ApiError, OSError, ValueError) as exc:
            self.message(str(exc))
            return
        self.busy = True
        self.reviewed = None
        self.controls()
        self.start(self.read_review(draft))

    async def read_review(self, draft):
        try:
            records = await artifacts(self.api, self.project_id)
            if not self.current():
                raise ApiError("Project changed. Close this form and select the current project.")
            item = next((a for a in records if a["id"] == draft.artifact_id), None)
            if item is None:
                raise ApiError("Artifact is no longer registered. Reopen the artifact list.")
            self.reviewed = (draft, item)
            for widget in self.query("#download-summary").results(Static):
                widget.update(
                    Text(
                        plain(
                            f"{item['label']} · {item['id']}\nManifest: {item['manifest_sha256']}\n"
                            f"Output: {draft.output}\n"
                            f"At most {draft.max_bytes:,} bytes in {draft.timeout}s.\n"
                            "No overwrite, extraction, provider launch or automatic retry."
                        )
                    )
                )
            self.message("Ready. Save archive to start the transfer.")
        except asyncio.CancelledError:
            self.message("Review stopped. No archive transfer started.")
            raise
        except ApiError as exc:
            self.message(str(exc))
        finally:
            self.busy = False
            self.controls()

    @on(Button.Pressed, "#download-submit")
    def submit(self):
        if self.busy or self.closing or self.reviewed is None:
            return
        try:
            draft, item = self.reviewed
            if not self.current() or self.draft() != draft:
                raise ApiError("Project or form changed. Review again before downloading.")
        except (ApiError, OSError, ValueError) as exc:
            self.reviewed = None
            self.controls()
            self.message(str(exc))
            return
        self.busy = True
        self.reviewed = None
        self.controls()
        self.message("Downloading… Stop transfer cancels only this local download.")
        self.start(self.transfer(draft, item))

    async def transfer(self, draft, item):
        try:
            self.receipt = await download_artifact(
                self.project_id,
                draft.artifact_id,
                draft.output,
                draft.max_bytes,
                draft.timeout,
                draft.expected_sha256,
                api=self.api,
                expected_artifact=item,
            )
            for widget in self.query("#download-receipt").results(TextArea):
                widget.load_text(plain(json.dumps(self.receipt, indent=2)))
            self.message(
                "Archive saved. The receipt verifies bytes and registry stability, "
                "not model quality."
            )
        except asyncio.CancelledError:
            self.message(
                "Transfer stopped. No completed output was published; jobs were not cancelled."
            )
            raise
        except (ApiError, OSError, ValueError) as exc:
            self.message(f"{exc} No automatic retry was made.")
        finally:
            self.busy = False
            self.controls()

    @on(Button.Pressed, "#download-stop")
    def stop(self):
        if self.transfer_task and not self.transfer_task.done() and not self.stop_requested:
            self.stop_requested = True
            self.message("Stopping transfer and removing partial output…")
            self.transfer_task.cancel()

    async def stop_transfer(self):
        task = self.transfer_task
        if task is not None:
            self.stop()
            # Drain exactly this owned task. Repeated UI/parent cancellation must
            # not interrupt an HTTP stream already closing after the first stop.
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
            if not task.cancelled():
                task.result()

    async def on_unmount(self):
        self.closing = True
        await self.stop_transfer()

    @on(Button.Pressed, "#download-back")
    def action_back(self):
        if self.busy:
            self.stop()
        else:
            self.dismiss(self.receipt)
