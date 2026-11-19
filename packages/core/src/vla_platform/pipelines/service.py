"""Dataset pipelines: ordered steps over an episode manifest.

Modelled on speech data processors: a run is a list of steps, each step reads the
manifest the previous one wrote and writes its own, so every stage can be inspected.
A step whose inputs and settings are unchanged is not repeated, which makes a stopped
run cheap to continue and lets review decisions re-run only what follows them.
"""

import asyncio
import hashlib
import json
import os
import re
import signal
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from vla_platform.datasets.local_preview import reader_python
from vla_platform.pipelines import generative, vision

WORKER = Path(__file__).with_name("worker.py")
ACTIVE = {"queued", "running"}
# Steps that run in the dataset reader's interpreter, and the worker operation for each.
WORKER_STEPS = {
    "read_dataset": "read",
    "gripper_events": "gripper",
    "sample_frames": "frames",
    "jitter": "jitter",
    "trim_idle": "trim",
}
STEPS = {
    *WORKER_STEPS,
    "vision_review",
    "generative_edit",
    "keep_where",
    "human_review",
    "export_dataset",
}
CAMERA_NOTES = {
    "camera_4": "mounted on the gripper, looking past the jaws",
    "wrist": "mounted on the wrist, looking past the jaws",
    "front": "a fixed overview of the table, the arm and the goal",
}
# One-click recipes. "check" finds bad demonstrations; the operator then reviews and saves.
RECIPES = {
    "check": [
        {"processor": "read_dataset"},
        {"processor": "gripper_events"},
        {"processor": "sample_frames", "params": {"frames": 10}},
        {"processor": "vision_review", "params": {"camera_notes": CAMERA_NOTES}},
        {
            "processor": "keep_where",
            # Vision decides whether the task finished; the gripper trace, which sees every
            # close, decides whether it took more than one grasp.
            "params": {
                "conditions": {
                    "vision.outcome": ["success", "success_after_retry"],
                    "gripper.verdict": ["clean"],
                }
            },
        },
    ],
    "augment": [
        {"processor": "read_dataset"},
        {"processor": "jitter", "params": {"copies": 2}},
    ],
    "check_free": [
        {"processor": "read_dataset"},
        {"processor": "gripper_events"},
        {"processor": "sample_frames", "params": {"frames": 10}},
        {"processor": "keep_where", "params": {"conditions": {"gripper.verdict": ["clean"]}}},
    ],
}


class PipelineError(ValueError):
    def __init__(self, message, status=422):
        super().__init__(message)
        self.status = status


def now():
    return datetime.now(UTC).isoformat()


def fingerprint(previous, step, extra=None):
    value = json.dumps([previous, step, extra], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        for entry in entries:
            stream.write(json.dumps(entry, allow_nan=False, sort_keys=True) + "\n")
    temporary.replace(path)


def label(entry, path):
    """``vision.outcome`` -> entry["labels"]["vision"]["outcome"], or None."""
    value = entry["labels"]
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


class Pipelines:
    def __init__(self, data_dir, *, library=None, api_keys=None):
        self.data_dir = Path(data_dir).resolve()
        self.root = self.data_dir / "dataset-pipelines"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.library = library
        self.api_keys = api_keys
        self.tasks = {}
        self.limit = asyncio.Semaphore(1)
        self.lock = asyncio.Lock()
        for run in self.records():
            if run["status"] in ACTIVE:
                run.update(status="interrupted", message="Stopped by an app restart. Continue it.")
                self.save(run)

    # Records -------------------------------------------------------------------------

    def directory(self, ident):
        if not re.fullmatch(r"[a-f0-9]{32}", ident or ""):
            raise PipelineError("Pipeline not found", 404)
        return self.root / ident

    def get(self, ident, project_id=None):
        path = self.directory(ident) / "run.json"
        try:
            run = json.loads(path.read_text())
        except OSError, ValueError:
            raise PipelineError("Pipeline not found", 404) from None
        if project_id is not None and run["project_id"] != project_id:
            raise PipelineError("Pipeline not found", 404)
        return run

    def records(self, project_id=None):
        runs = []
        for path in self.root.glob("*/run.json"):
            try:
                run = json.loads(path.read_text())
            except OSError, ValueError:
                continue
            if project_id is None or run["project_id"] == project_id:
                runs.append(run)
        return sorted(runs, key=lambda run: run["created_at"], reverse=True)

    def save(self, run):
        run["updated_at"] = now()
        directory = self.directory(run["id"])
        directory.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".run-", dir=directory)
        with os.fdopen(fd, "w") as stream:
            json.dump(run, stream, indent=2, allow_nan=False)
        os.replace(temporary, directory / "run.json")
        return run

    def commit(self, run):
        """Save what execution owns without overwriting review decisions made meanwhile."""
        current = self.get(run["id"])
        for key in ("steps", "results", "status", "message"):
            current[key] = run[key]
        run.update({key: current[key] for key in ("review", "saved_datasets")})
        return self.save(current)

    def step_folder(self, run, index):
        return (
            self.directory(run["id"]) / "steps" / f"{index:02d}-{run['steps'][index]['processor']}"
        )

    def manifest(self, ident, index=None, project_id=None):
        run = self.get(ident, project_id)
        done = [i for i, result in enumerate(run["results"]) if result.get("status") == "succeeded"]
        if not done:
            return run, []
        index = done[-1] if index is None else index
        if index not in done:
            raise PipelineError("That step has not finished", 409)
        return run, read_jsonl(self.step_folder(run, index) / "manifest.jsonl")

    # Lifecycle -----------------------------------------------------------------------

    def create(self, project_id, *, steps, name, dataset_id=None, root=None):
        steps = [self._step(step) for step in steps]
        if not steps or steps[0]["processor"] != "read_dataset":
            raise PipelineError("A pipeline starts by reading a dataset")
        if dataset_id is not None:
            if self.library is None:
                raise PipelineError("The dataset library is unavailable", 503)
            source = self.library.get(dataset_id, project_id)
            root = self.library.resolve(dataset_id, project_id)
            label_name = source["name"]
        elif root is not None:
            label_name = Path(root).name
        else:
            raise PipelineError("Choose a dataset")
        run = {
            "id": uuid4().hex,
            "project_id": project_id,
            "name": name or f"Check {label_name}",
            "dataset": {"id": dataset_id, "name": label_name, "root": str(root)},
            "steps": steps,
            "results": [{} for _ in steps],
            "review": {},
            "status": "queued",
            "message": "Waiting to start",
            "saved_datasets": [],
            "created_at": now(),
        }
        self.save(run)
        self.start(run["id"])
        return run

    def _step(self, step):
        processor = step.get("processor")
        if processor not in STEPS:
            raise PipelineError(f"Unknown step {processor!r}")
        params = step.get("params") or {}
        if not isinstance(params, dict):
            raise PipelineError("Step settings must be an object")
        return {"processor": processor, "params": params}

    def start(self, ident):
        if ident in self.tasks and not self.tasks[ident].done():
            return
        task = asyncio.create_task(self.execute(ident))
        self.tasks[ident] = task
        task.add_done_callback(lambda done: self.tasks.pop(ident, None))

    async def extend(self, ident, project_id, steps):
        """Append steps (for example review and save) and run what has not run yet."""
        async with self.lock:
            run = self.get(ident, project_id)
            if run["status"] in ACTIVE:
                raise PipelineError("Wait for this pipeline to finish first", 409)
            for step in steps:
                run["steps"].append(self._step(step))
                run["results"].append({})
            run.update(status="queued", message="Waiting to start")
            self.save(run)
        self.start(ident)
        return run

    async def set_review(self, ident, project_id, decisions):
        async with self.lock:
            run = self.get(ident, project_id)
            run["review"].update({key: bool(value) for key, value in decisions.items()})
            return self.save(run)

    async def resume(self, ident, project_id):
        run = self.get(ident, project_id)
        if run["status"] not in ACTIVE:
            run.update(status="queued", message="Waiting to start")
            self.save(run)
        self.start(ident)
        return run

    async def close(self):
        for task in list(self.tasks.values()):
            task.cancel()
        for task in list(self.tasks.values()):
            try:
                await task
            except BaseException:
                pass

    # Execution -----------------------------------------------------------------------

    async def execute(self, ident):
        async with self.limit:
            run = self.get(ident)
            run.update(status="running", message="Starting")
            self.commit(run)
            previous = run["dataset"]["root"]
            try:
                for index, step in enumerate(run["steps"]):
                    extra = run["review"] if step["processor"] == "human_review" else None
                    key = fingerprint(previous, step, extra)
                    result = run["results"][index]
                    folder = self.step_folder(run, index)
                    if (
                        result.get("status") == "succeeded"
                        and result.get("fingerprint") == key
                        and (folder / "manifest.jsonl").is_file()
                    ):
                        previous = key
                        continue
                    # Everything after a changed step is stale.
                    for later in range(index, len(run["steps"])):
                        run["results"][later] = {}
                    run["results"][index] = {"status": "running", "started_at": now()}
                    run["message"] = f"Step {index + 1} of {len(run['steps'])}: {step['processor']}"
                    self.commit(run)
                    source = (
                        None if index == 0 else self.step_folder(run, index - 1) / "manifest.jsonl"
                    )
                    summary = await self._run_step(run, index, step, source, folder)
                    entries = read_jsonl(folder / "manifest.jsonl")
                    run = self.get(ident)
                    run["results"][index] = {
                        "status": "succeeded",
                        "fingerprint": key,
                        "started_at": run["results"][index].get("started_at"),
                        "finished_at": now(),
                        "entries": len(entries),
                        "kept": sum(1 for entry in entries if entry["keep"]),
                        "summary": summary,
                    }
                    self.commit(run)
                    previous = key
                run.update(status="succeeded", message="Finished")
            except asyncio.CancelledError:
                run = self.get(ident)
                self._fail_running(run, "Stopped before it finished. Continue it.")
                run.update(status="interrupted", message="Stopped before it finished. Continue it.")
                self.commit(run)
                raise
            except (PipelineError, vision.VisionError, OSError, ValueError) as error:
                run = self.get(ident)
                self._fail_running(run, str(error))
                run.update(status="failed", message=str(error))
            self.commit(run)

    def _fail_running(self, run, message):
        for result in run["results"]:
            if result.get("status") == "running":
                result.update(status="failed", error=message, finished_at=now())

    async def _run_step(self, run, index, step, source, folder):
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / "manifest.jsonl"
        params = step["params"]
        processor = step["processor"]
        if processor in WORKER_STEPS:
            payload = {**params, "operation": WORKER_STEPS[processor], "output": str(output)}
            if processor == "read_dataset":
                payload["root"] = run["dataset"]["root"]
            else:
                payload["input"] = str(source)
            if processor == "gripper_events":
                payload.setdefault("action_names", self._action_names(run))
            if processor in {"sample_frames", "jitter"}:
                name = "frames" if processor == "sample_frames" else "augmented"
                payload["folder"] = str(self.directory(run["id"]) / name)
            return await self.worker(payload)
        entries = read_jsonl(source)
        if processor == "vision_review":
            return await self._vision(run, index, entries, params, output)
        if processor == "generative_edit":
            return await self._generative(run, entries, params, output)
        if processor == "keep_where":
            conditions = params.get("conditions") or {}
            for entry in entries:
                checks = [
                    label(entry, path) in allowed
                    for path, allowed in conditions.items()
                    if label(entry, path) is not None
                ]
                entry["keep"] = all(checks)
                entry["suggested"] = entry["keep"]
            write_jsonl(output, entries)
            return {"rule": conditions}
        if processor == "human_review":
            # Review is about recorded episodes; augmented copies are made again after it.
            entries = [entry for entry in entries if entry["origin"] == "recorded"]
            changed = 0
            for entry in entries:
                decision = run["review"].get(entry["id"])
                if decision is not None and decision != entry["keep"]:
                    entry["keep"] = decision
                    changed += 1
                if decision is not None:
                    entry["labels"]["review"] = {"keep": decision}
            write_jsonl(output, entries)
            return {"changed_by_review": changed}
        if processor == "export_dataset":
            return await self._export(run, entries, params, source, folder)
        raise PipelineError(f"Unknown step {processor!r}")

    def _action_names(self, run):
        info = json.loads((Path(run["dataset"]["root"]) / "meta" / "info.json").read_text())
        return info["features"]["action"].get("names") or []

    async def _vision(self, run, index, entries, params, output):
        key = self.api_keys.key("openai") if self.api_keys is not None else None
        key = key or os.environ.get("OPENAI_API_KEY")
        if not key:
            write_jsonl(output, entries)
            return {"skipped": "Save an OpenAI API key in Cloud & APIs to add the vision review."}

        async def progress(done, total):
            current = self.get(run["id"])
            current["message"] = f"Vision review: {done} of {total} episodes"
            self.save(current)
            run["message"] = current["message"]

        # The Decisions API (fixed answers with probabilities) costs about 1/70 of a
        # free-text vision review and found the same failed demonstrations.
        judge = vision.review if params.get("api") == "responses" else vision.decide
        summary = await judge(entries, key, params, progress)
        write_jsonl(output, entries)
        return summary

    async def _generative(self, run, entries, params, output):
        key = self.api_keys.key("openai") if self.api_keys is not None else None
        key = key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise PipelineError("Save an OpenAI API key in Cloud & APIs; it checks every edit.")
        provider = generative.provider_from(params, self.api_keys)

        async def progress(done, total):
            current = self.get(run["id"])
            current["message"] = f"Generative edits: {done} of {total} episodes checked"
            self.save(current)
            run["message"] = current["message"]

        try:
            summary = await generative.edit_entries(
                entries,
                params,
                self.directory(run["id"]) / "generative",
                provider=provider,
                decisions_key=key,
                progress=progress,
            )
        finally:
            await provider.close()
        write_jsonl(output, entries)
        return summary

    async def _export(self, run, entries, params, source, folder):
        if not any(entry["keep"] for entry in entries):
            raise PipelineError("No episodes are kept, so there is nothing to save")
        name = params.get("name") or f"{run['dataset']['name']}-cleaned"
        payload = {"operation": "export", "input": str(source), "task": params.get("task")}
        if self.library is None or run["dataset"]["id"] is None:
            target = Path(params.get("folder") or self.directory(run["id"]) / "export")
            result = await self.worker({**payload, "output": str(target)})
            saved = {"folder": str(target), "name": name, **result}
        else:

            async def produce(target):
                return await self.worker({**payload, "output": str(target)}, timeout=3600)

            dataset, result = await self.library.build(
                run["project_id"],
                name,
                produce,
                {"derived_from": {"pipeline_id": run["id"], "dataset_id": run["dataset"]["id"]}},
                stage="saving",
            )
            saved = {"dataset_id": dataset["id"], "name": dataset["name"], **result}
        write_jsonl(folder / "manifest.jsonl", entries)
        current = self.get(run["id"])
        current["saved_datasets"].append(saved)
        self.save(current)
        return saved

    async def worker(self, payload, timeout=1800):
        executable = str(reader_python(self.data_dir))
        process = await asyncio.create_subprocess_exec(
            executable,
            "-I",
            "-B",
            str(WORKER),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=os.name == "posix",
        )
        try:
            async with asyncio.timeout(timeout):
                raw, _ = await process.communicate(json.dumps(payload).encode())
        except BaseException:
            if process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL) if os.name == "posix" else process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
            raise
        try:
            result = json.loads(raw)
        except ValueError:
            raise PipelineError("The dataset reader returned no result") from None
        if process.returncode or "error" in result:
            raise PipelineError(result.get("error", "The dataset reader failed"))
        return result

    # Views ---------------------------------------------------------------------------

    def media_path(self, ident, relative):
        directory = self.directory(ident)
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or path.suffix != ".jpg":
            raise PipelineError("Image not found", 404)
        if not path.is_file():
            raise PipelineError("Image not found", 404)
        return path

    def episodes(self, ident, project_id):
        """The latest manifest, trimmed to what the review page shows."""
        run, entries = self.manifest(ident, project_id=project_id)
        directory = self.directory(ident).resolve()
        rows = []
        for entry in entries:
            strip = label(entry, "frames.strip")
            vision_label = entry["labels"].get("vision") or {}
            gripper = entry["labels"].get("gripper") or {}
            rows.append(
                {
                    "id": entry["id"],
                    "episode_index": entry["episode_index"],
                    "origin": entry["origin"],
                    "keep": run["review"].get(entry["id"], entry["keep"]),
                    "suggested": entry.get("suggested", entry["keep"]),
                    "reviewed": entry["id"] in run["review"],
                    "strip": (
                        Path(strip).resolve().relative_to(directory).as_posix()
                        if strip and Path(strip).resolve().is_relative_to(directory)
                        else None
                    ),
                    "gripper": {
                        key: gripper[key] for key in ("verdict", "missed_grasps") if key in gripper
                    }
                    or None,
                    "vision": {
                        key: vision_label[key]
                        for key in ("outcome", "grasp_attempts", "confidence", "notes")
                        if key in vision_label
                    }
                    or None,
                    "waypoints": {
                        name: result["reached"]
                        for name, result in (vision_label.get("waypoints") or {}).items()
                    },
                }
            )
        return {"pipeline": run, "episodes": rows}
