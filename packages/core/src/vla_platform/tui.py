"""Optional Textual workbench. The application API remains the only job owner."""

import json
from datetime import datetime
from functools import partial

from pydantic import ValidationError
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.containers import HorizontalScroll, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    OptionList,
    Select,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)
from textual.widgets.option_list import Option

from vla_platform.contracts import IntakeRequest, Job, Project, ProjectCreate
from vla_platform.tui_client import ApiClient, ApiError, plain, segment

ACTIVE = {"queued", "running"}


class Form(ModalScreen):
    BINDINGS = [("escape", "dismiss(None)", "Back")]
    CSS = """
    Form { align: center middle; background: $background 70%; }
    Form > VerticalScroll {
        width: 70; max-width: 100%; height: auto; max-height: 95%;
        border: round $accent; padding: 1 2; background: $surface;
    }
    Form Label, Form Static { height: auto; margin-bottom: 1; }
    Form Input, Form Select, Form Checkbox { margin-bottom: 1; }
    Form HorizontalScroll { height: 3; }
    #form-error { color: $error; }
    """

    def error(self, message):
        self.query_one("#form-error", Static).update(Text(plain(message)))

    @on(Button.Pressed, "#back")
    def back(self):
        self.dismiss(None)


class ProjectForm(Form):
    AUTO_FOCUS = "#name"

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Label("Create a project")
            yield Input(placeholder="Project name", id="name", max_length=100)
            yield Static("", id="form-error")
            with HorizontalScroll():
                yield Button("Create project", id="submit", variant="primary")
                yield Button("Back", id="back")

    @on(Button.Pressed, "#submit")
    @on(Input.Submitted)
    def submit(self):
        try:
            value = ProjectCreate(name=self.query_one("#name", Input).value)
        except ValidationError:
            self.error("Enter a project name (1–100 characters).")
            return
        self.dismiss(value.model_dump())


class IntakeForm(Form):
    AUTO_FOCUS = "#source"

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Label("Inspect a robotics dataset")
            yield Select(
                [("Hugging Face repository", "huggingface"), ("Local application folder", "local")],
                value="huggingface",
                allow_blank=False,
                id="source",
            )
            yield Label("Repository (owner/dataset)", id="target-label")
            yield Input(placeholder="owner/dataset", id="target", max_length=4096)
            yield Label("Revision (branch, tag or commit; main by default)", id="revision-label")
            yield Input(value="main", id="revision", max_length=200)
            yield Checkbox("Prepare immutable local training copy", id="snapshot", disabled=True)
            yield Static(
                "Local paths refer to the application host and its allowed dataset folder. "
                "A training copy validates and copies data/media; it does not start training.",
                markup=False,
            )
            yield Static("", id="form-error")
            with HorizontalScroll():
                yield Button("Submit intake", id="submit", variant="primary")
                yield Button("Back", id="back")

    @on(Select.Changed, "#source")
    def source_changed(self):
        local = self.query_one("#source", Select).value == "local"
        self.query_one("#target-label", Label).update(
            "Dataset folder on the application host" if local else "Repository (owner/dataset)"
        )
        self.query_one("#revision", Input).disabled = local
        self.query_one("#snapshot", Checkbox).disabled = not local

    @on(Button.Pressed, "#submit")
    @on(Input.Submitted)
    def submit(self):
        source = self.query_one("#source", Select).value
        target = self.query_one("#target", Input).value.strip()
        payload = (
            {
                "source": source,
                "path": target,
                "snapshot_for_training": self.query_one("#snapshot", Checkbox).value,
            }
            if source == "local"
            else {
                "source": source,
                "repo_id": target,
                "revision": self.query_one("#revision", Input).value,
            }
        )
        try:
            value = IntakeRequest.model_validate(payload)
        except ValidationError:
            self.error("Enter a valid owner/dataset repository or a nonempty allowed local path.")
            return
        self.dismiss(value.model_dump(exclude_none=True))


class CancelForm(Form):
    def __init__(self, job):
        super().__init__()
        self.job = job

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Label(Text(f"Cancel job {plain(self.job['id'])}?"))
            yield Static(
                Text(f"{plain(self.job['kind'])} · last observed {plain(self.job['status'])}")
            )
            yield Static(
                "This requests cancellation through the application. "
                "A cancellation receipt is not proof of cloud resource deletion.",
                markup=False,
            )
            with HorizontalScroll():
                yield Button("Keep job", id="back")
                yield Button("Cancel this job", id="confirm", variant="error")

    def on_mount(self):
        self.query_one("#back", Button).focus()

    @on(Button.Pressed, "#confirm")
    def confirm(self):
        self.dismiss(True)


class FirebirdApp(App):
    ENABLE_COMMAND_PALETTE = False
    TITLE = "OPEN JENSEN"
    SUB_TITLE = "Dataset → Fine-tune → Distill → Quantize → Evaluate → Run"
    BINDINGS = [
        ("f1", "view('projects-tab')", "Projects"),
        ("f2", "view('jobs-tab')", "Jobs"),
        ("f3", "view('detail-tab')", "Details"),
        ("ctrl+n", "new_project", "New project"),
        ("f4", "intake", "Intake"),
        ("f8", "cancel_job", "Cancel job"),
        ("ctrl+r", "refresh", "Refresh"),
        ("ctrl+q", "quit", "Quit"),
    ]
    CSS = """
    #connection, #notice, #project-label { height: auto; max-height: 5; padding: 0 1; }
    #notice { color: $warning; }
    #actions { height: 3; }
    #actions Button { margin-right: 1; }
    TabbedContent { height: 1fr; }
    OptionList { height: 1fr; }
    #detail { height: 1fr; }
    """

    def __init__(self, base="http://127.0.0.1:8000", *, client=None, poll_seconds=3):
        super().__init__()
        self.client = client or ApiClient(base)
        self.poll_seconds = poll_seconds
        self.projects = []
        self.jobs = []
        self.project_id = None
        self.job_id = None
        self.epoch = 0
        self.read_worker = None
        self.writing = False
        self.connected = False
        self.last_read = "Not yet connected"

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(Text(f"Connecting to {self.client.base}"), id="connection")
        yield Static("", id="notice", markup=False)
        with HorizontalScroll(id="actions"):
            yield Button("New project", id="new-project")
            yield Button("Intake", id="intake", disabled=True)
            yield Button("Refresh", id="refresh")
            yield Button("Cancel selected job", id="cancel-job", disabled=True)
        with TabbedContent(initial="projects-tab", id="views"):
            with TabPane("Projects", id="projects-tab"):
                yield OptionList(id="projects")
            with TabPane("Jobs", id="jobs-tab"):
                yield Static("Select a project and press Enter.", id="project-label", markup=False)
                yield OptionList(id="jobs")
            with TabPane("Job details", id="detail-tab"):
                yield TextArea("Select a job and press Enter.", read_only=True, id="detail")
        yield Footer()

    def on_mount(self):
        self.set_interval(self.poll_seconds, self.poll)
        self.action_refresh()
        self.query_one("#projects", OptionList).focus()

    async def on_unmount(self):
        await self.client.close()

    def notice(self, value):
        if not self.is_running:
            return
        self.query_one("#notice", Static).update(Text(plain(value)))

    def controls(self):
        # Queued highlight messages may arrive while the screen is being dismantled.
        if not self.is_running:
            return
        self.query_one("#new-project", Button).disabled = self.writing
        self.query_one("#intake", Button).disabled = (
            self.writing
            or not self.connected
            or not any(p["id"] == self.project_id for p in self.projects)
        )
        job = self.highlighted_job()
        self.query_one("#cancel-job", Button).disabled = (
            self.writing or not self.connected or job is None or job["status"] not in ACTIVE
        )

    def highlighted_job(self):
        options = self.query_one("#jobs", OptionList)
        if options.highlighted is None:
            return None
        ident = options.get_option_at_index(options.highlighted).id
        return next((job for job in self.jobs if job["id"] == ident), None)

    def poll(self):
        # Slow requests finish within their own deadline; timer never restarts them forever.
        if self.read_worker is None or self.read_worker.is_finished:
            self.action_refresh()

    def action_refresh(self):
        if not self.is_running:
            return
        self.read_worker = self.run_worker(self.fetch_records, group="read", exclusive=True)

    @staticmethod
    def options(widget, records, render):
        highlighted = (
            widget.get_option_at_index(widget.highlighted).id
            if widget.highlighted is not None
            else None
        )
        widget.clear_options()
        widget.add_options(
            [Option(Text(plain(render(record))), id=record["id"]) for record in records]
        )
        widget.highlighted = next(
            (i for i, record in enumerate(records) if record["id"] == highlighted),
            0 if records else None,
        )

    async def fetch_records(self):
        epoch, project_id, job_id = self.epoch, self.project_id, self.job_id
        try:
            projects = await self.client.projects()
            if not self.is_running or epoch != self.epoch:
                return
            self.projects = projects
            self.options(
                self.query_one("#projects", OptionList),
                projects,
                lambda p: f"{p['name']} · {p['id']}",
            )
            if project_id and not any(p["id"] == project_id for p in projects):
                self.project_id = self.job_id = None
                self.jobs = []
                self.query_one("#jobs", OptionList).clear_options()
                self.query_one("#detail", TextArea).load_text("Project no longer available.")
                self.query_one("#project-label", Static).update("Select a project.")
                self.notice("The selected project is no longer available. Select another project.")
                project_id = None
            if project_id:
                jobs = await self.client.jobs(project_id)
                if not self.is_running or epoch != self.epoch or project_id != self.project_id:
                    return
                self.jobs = jobs
                self.options(
                    self.query_one("#jobs", OptionList),
                    jobs,
                    lambda j: f"{j['status']} · {j['kind']} · {j['id']}",
                )
                if job_id and any(j["id"] == job_id for j in jobs):
                    job = await self.client.job(job_id)
                    events = await self.client.request("GET", f"/jobs/{segment(job_id)}/events")
                    if not isinstance(events, list):
                        raise ApiError("Application returned invalid job events.")
                    if not self.is_running or epoch != self.epoch or job_id != self.job_id:
                        return
                    if job["project_id"] != project_id:
                        raise ApiError("Application returned another project's job.")
                    raw = plain(
                        json.dumps({"job": job, "events": events}, indent=2, ensure_ascii=False)
                    )
                    if len(raw) > 120000:
                        raw = raw[:120000] + "\n[Display truncated; use firebird jobs show/events.]"
                    self.query_one("#detail", TextArea).load_text(raw)
                elif job_id:
                    self.job_id = None
                    self.query_one("#detail", TextArea).load_text("Job no longer available.")
            self.connected = True
            self.last_read = datetime.now().astimezone().strftime("%H:%M:%S %Z")
            self.query_one("#connection", Static).update(
                Text(f"{self.client.base} · Last refreshed {self.last_read}")
            )
        except ApiError as error:
            if self.is_running and epoch == self.epoch:
                self.connected = False
                self.query_one("#connection", Static).update(
                    Text(
                        f"Offline or unavailable · Last refreshed {self.last_read}\n{plain(error)}"
                    )
                )
        finally:
            self.controls()

    def action_view(self, tab):
        if self.screen is not self.default_screen:
            return
        self.query_one("#views", TabbedContent).active = tab
        target = {"projects-tab": "projects", "jobs-tab": "jobs", "detail-tab": "detail"}[tab]
        self.call_after_refresh(self.query_one("#" + target).focus)

    @on(OptionList.OptionSelected, "#projects")
    def project_selected(self, event):
        selected = next((p for p in self.projects if p["id"] == event.option.id), None)
        if selected is None:
            self.notice("Project selection changed during refresh. Select an available project.")
            return
        self.epoch += 1
        self.project_id = event.option.id
        self.job_id = None
        self.jobs = []
        self.query_one("#jobs", OptionList).clear_options()
        self.query_one("#detail", TextArea).load_text("Select a job and press Enter.")
        self.query_one("#project-label", Static).update(Text(f"Project: {plain(selected['name'])}"))
        self.controls()
        self.action_view("jobs-tab")
        self.action_refresh()

    @on(OptionList.OptionHighlighted, "#jobs")
    def job_highlighted(self):
        self.controls()

    @on(OptionList.OptionSelected, "#jobs")
    def job_selected(self, event):
        if not any(job["id"] == event.option.id for job in self.jobs):
            self.notice("Job selection changed during refresh. Select an available job.")
            return
        self.epoch += 1
        self.job_id = event.option.id
        self.query_one("#detail", TextArea).load_text(f"Loading job {plain(self.job_id)}…")
        self.action_view("detail-tab")
        self.action_refresh()

    def action_new_project(self):
        if not self.writing and self.screen is self.default_screen:
            self.push_screen(ProjectForm(), lambda payload: self.mutate("project", payload))

    def action_intake(self):
        if self.screen is not self.default_screen:
            return
        if self.query_one("#intake", Button).disabled:
            return
        project_id = self.project_id
        self.push_screen(IntakeForm(), lambda payload: self.mutate("intake", payload, project_id))

    def action_cancel_job(self):
        if self.screen is not self.default_screen:
            return
        if self.query_one("#cancel-job", Button).disabled:
            return
        job = self.highlighted_job()
        project_id = self.project_id
        self.push_screen(
            CancelForm(job), lambda ok: self.mutate("cancel", job if ok else None, project_id)
        )

    def mutate(self, kind, payload, project_id=None):
        if payload is None or self.writing:
            return
        if kind != "project" and (
            project_id != self.project_id
            or not self.connected
            or not any(p["id"] == project_id for p in self.projects)
        ):
            self.notice(
                "Selection or connection changed. Review the current project before acting."
            )
            return
        if kind == "cancel" and (self.highlighted_job() or {}).get("id") != payload["id"]:
            self.notice("Job selection changed. Cancellation was not sent.")
            return
        self.writing = True
        self.controls()
        self.run_worker(partial(self.write, kind, payload, project_id), group="mutation")

    async def write(self, kind, payload, project_id):
        try:
            if kind == "project":
                result = await self.client.request("POST", "/projects", payload)
                result = self.client.validate(Project, result)
                self.notice(f"Project created: {result['name']}. Select it in Projects.")
            elif kind == "intake":
                result = await self.client.request(
                    "POST", f"/projects/{segment(project_id)}/intakes", payload
                )
                result = self.client.validate(Job, result)
                if result["project_id"] != project_id or result["kind"] != "dataset.inspect":
                    raise ApiError(
                        "Intake response identity differs; inspect jobs before retrying."
                    )
                self.notice(f"Intake accepted: {result['id']}. Follow its actual status in Jobs.")
            else:
                current = await self.client.job(payload["id"])
                if (
                    current["project_id"] != project_id
                    or current["status"] not in ACTIVE
                    or self.project_id != project_id
                    or (self.highlighted_job() or {}).get("id") != payload["id"]
                ):
                    self.notice("Job status or selection changed. Cancellation was not sent.")
                    return
                result = await self.client.request("POST", f"/jobs/{segment(payload['id'])}/cancel")
                result = self.client.validate(Job, result)
                if result["id"] != payload["id"] or result["project_id"] != project_id:
                    raise ApiError(
                        "Cancellation response identity differs; refresh to inspect the outcome."
                    )
                self.notice(f"Cancellation response: {result['id']} · {result['status']}.")
        except ApiError as error:
            self.notice(f"{error} No automatic retry was made; refresh to inspect the outcome.")
        finally:
            self.writing = False
            self.controls()
            self.action_refresh()

    @on(Button.Pressed)
    def button(self, event):
        action = {
            "new-project": self.action_new_project,
            "intake": self.action_intake,
            "refresh": self.action_refresh,
            "cancel-job": self.action_cancel_job,
        }.get(event.button.id)
        if action:
            action()
