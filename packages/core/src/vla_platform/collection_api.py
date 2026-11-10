"""Project-scoped Pi collection runs. Physical execution stays in Robot Lab."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import os
import re
import secrets
import sys
import time
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from vla_platform.agent_control.robot_target import RobotTarget
from vla_platform.agent_control.settings import Profiles
from vla_platform.robot_lab_bridge import RobotLabBridge

router = APIRouter(prefix="/api/v1", tags=["Agent collection"])
ROOT = Path(__file__).resolve().parents[4]
TERMINAL = {"saved", "stopped", "failed", "interrupted"}


def save(path, value):
    pending = path.with_suffix(".tmp")
    with pending.open("w") as stream:
        os.chmod(pending, 0o600)
        json.dump(value, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


class CollectionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["mock", "physical"] = "mock"
    model: Literal["gpt-6-astra"] = "gpt-6-astra"
    task: str = Field(default="Place one object into the tray", min_length=1, max_length=1000)
    camera_indices: list[StrictInt] = Field(default=[0, 1], min_length=2, max_length=2)
    fps: int = Field(default=10, ge=1, le=20, strict=True)
    max_steps: int = Field(default=16, ge=1, le=40, strict=True)
    max_seconds: int = Field(default=120, ge=10, le=300, strict=True)
    max_tokens: int = Field(default=16000, ge=1024, le=64000, strict=True)
    expected_generation: int = Field(default=0, ge=0, strict=True)
    confirm: str = ""
    operator_present: bool = Field(default=False, strict=True)
    workspace_clear: bool = Field(default=False, strict=True)


class CollectionRuns:
    def __init__(self, app, data_dir):
        self.app = app
        self.root = Path(data_dir) / "agent-collection"
        self.root.mkdir(mode=0o700, exist_ok=True)
        self.tasks = {}
        self.live = {}
        for path in self.root.glob("*/run.json"):
            value = json.loads(path.read_text())
            if value["status"] not in TERMINAL:
                value.update(
                    status="interrupted",
                    error=(
                        "Application restarted; motion is never resumed. "
                        "Raw companion recording is retained."
                    ),
                )
                save(path, value)

    def directory(self, ident):
        if not re.fullmatch(r"[a-f0-9]{32}", ident):
            raise ValueError("Invalid run identity")
        return self.root / ident

    def get(self, ident):
        return json.loads((self.directory(ident) / "run.json").read_text())

    def update(self, ident, **changes):
        value = self.get(ident)
        value.update(changes, updated_at=time.time())
        save(self.directory(ident) / "run.json", value)
        return value

    def list(self, project_id):
        return sorted(
            [
                json.loads(p.read_text())
                for p in self.root.glob("*/run.json")
                if json.loads(p.read_text())["project_id"] == project_id
            ],
            key=lambda x: x["created_at"],
            reverse=True,
        )

    async def start(self, project_id, config):
        if any(not task.done() for task in self.tasks.values()):
            raise ValueError("Finish the current collection run before starting another")
        if len(set(config.camera_indices)) != 2 or any(
            not 0 <= i <= 63 for i in config.camera_indices
        ):
            raise ValueError("Choose two distinct camera indices between 0 and 63")
        if config.mode == "physical":
            if (
                config.confirm != "ARM AI CONTROL"
                or not config.operator_present
                or not config.workspace_clear
            ):
                raise ValueError("Confirm ARM AI CONTROL, operator presence and a clear workspace")
            if not Profiles(self.root.parent).shared_key():
                raise ValueError("Configure an OpenAI API key in Settings first")
        ident = uuid4().hex
        directory = self.directory(ident)
        directory.mkdir(mode=0o700)
        value = {
            "id": ident,
            "project_id": project_id,
            "episode_id": ident + "-0",
            "created_at": time.time(),
            "status": "starting",
            "config": config.model_dump(),
            "frames": 0,
            "error": None,
            "dataset": None,
            "proposal": None,
            "task_success_verified": False,
        }
        save(directory / "run.json", value)
        self.live[ident] = {
            "operator": time.monotonic(),
            "approved": asyncio.Event(),
            "bridge": None,
            "recording": False,
            "epoch": None,
            "stopping": False,
        }
        self.tasks[ident] = asyncio.create_task(self._run(ident, config))
        return value

    async def event(self, ident, event, *, mirror=True):
        value = {
            "run_id": ident,
            "episode_id": ident + "-0",
            "core_monotonic_ns": time.monotonic_ns(),
            "unix_ns": time.time_ns(),
            **event,
        }
        with (self.directory(ident) / "trace.jsonl").open("a") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        live = self.live[ident]
        if mirror and live["recording"]:
            await live["bridge"].request(
                "POST", "collection/event", payload={"run_id": ident, "event": value}
            )

    async def pulse(self, ident, target, epoch, config):
        while True:
            live = self.live[ident]
            if live.get("model_stopped"):
                await asyncio.sleep(0.4)
                continue
            if config.mode == "physical" and time.monotonic() - live["operator"] > 1.5:
                raise ValueError("Operator page heartbeat expired")
            await target.heartbeat(ident, epoch)
            if live["recording"]:
                state = await live["bridge"].request("GET", "collection/status")
                self.update(ident, frames=state.get("frames", 0), recording=state)
                if state["status"] not in {"recording", "saved"}:
                    raise ValueError(state.get("error") or "Recording stopped")
            await asyncio.sleep(0.4)

    async def _run(self, ident, config):
        live = self.live[ident]
        live["entered"] = True
        worker = mock = pulse = None
        bridge = self.app.state.robot_lab
        target = None
        outcome, error = "saved", None
        try:
            if config.mode == "mock":
                token = secrets.token_hex(32)
                python = (
                    os.environ.get("JENSEN_COLLECTION_PYTHON")
                    or os.environ.get("FIREBIRD_CPU_READER_PYTHON")
                    or sys.executable
                )
                mock = await asyncio.create_subprocess_exec(
                    python,
                    str(ROOT / "workers/robot_lab/collection_mock.py"),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                mock.stdin.write(
                    (
                        json.dumps(
                            {"root": str(self.directory(ident) / "mock-companion"), "token": token}
                        )
                        + "\n"
                    ).encode()
                )
                await mock.stdin.drain()
                ready = json.loads(await asyncio.wait_for(mock.stdout.readline(), 15))
                bridge = RobotLabBridge(self.directory(ident))
                bridge.config = lambda: {
                    "url": "http://127.0.0.1:" + str(ready["port"]),
                    "token": token,
                    "label": "Synthetic companion",
                }
                generation = ready["generation"]
            else:
                generation = config.expected_generation
            live["bridge"] = bridge
            # Detect an outdated USB companion before transferring controller authority.
            await bridge.request("GET", "collection/status")
            target = RobotTarget(bridge)
            acquired = await target.acquire(
                ident,
                {
                    "confirm": "ARM AI CONTROL",
                    "expected_generation": generation,
                    "camera_index": config.camera_indices[0],
                    "operator_present": True,
                    "workspace_clear": True,
                },
            )
            epoch = live["epoch"] = acquired["authority_epoch"]
            pulse = asyncio.create_task(self.pulse(ident, target, epoch, config))
            await bridge.request(
                "POST",
                "collection/start",
                payload={
                    "run_id": ident,
                    "camera_indices": config.camera_indices,
                    "fps": config.fps,
                    "task": config.task,
                    "model": "deterministic-pi" if config.mode == "mock" else config.model,
                    "environment_kind": "synthetic" if config.mode == "mock" else "physical",
                },
            )
            live["recording"] = True
            await self.event(
                ident,
                {
                    "type": "run_started",
                    "configuration": config.model_dump(),
                    "runtime": "pi-agent-core@1.5.1",
                    "authority_epoch": epoch,
                },
            )
            self.update(ident, status="running")
            node = os.environ.get("JENSEN_PI_NODE", "node")
            worker = await asyncio.create_subprocess_exec(
                node,
                str(ROOT / "workers/pi_collection/worker.mjs"),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                limit=2 * 1024**2,
                env={
                    k: v
                    for k, v in os.environ.items()
                    if not any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET"))
                },
            )
            start = {
                "type": "start",
                **config.model_dump(),
                "api_key": Profiles(self.root.parent).shared_key()
                if config.mode == "physical"
                else None,
            }
            worker.stdin.write((json.dumps(start) + "\n").encode())
            await worker.stdin.drain()
            observations, calls = {}, {}
            deadline = time.monotonic() + config.max_seconds
            while True:
                if pulse.done():
                    await pulse
                if time.monotonic() >= deadline:
                    raise ValueError("Episode time budget reached")
                read = asyncio.create_task(worker.stdout.readline())
                done, _ = await asyncio.wait(
                    {read, pulse},
                    timeout=min(1, deadline - time.monotonic()),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if pulse in done:
                    read.cancel()
                    await pulse
                if read not in done:
                    read.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await read
                    continue
                line = read.result()
                if not line:
                    raise ValueError("Pi worker exited before completion")
                message = json.loads(line)
                kind = message["type"]
                if kind == "tool":
                    call_id = message["id"]
                    fingerprint = hashlib.sha256(
                        json.dumps(message, sort_keys=True).encode()
                    ).hexdigest()
                    if call_id in calls:
                        previous, response = calls[call_id]
                        if previous != fingerprint:
                            raise ValueError("Tool identity reused with different arguments")
                    else:
                        if len(calls) >= config.max_steps:
                            raise ValueError("Tool budget reached")
                        await self.event(ident, {**message, "type": "tool_request"})
                        operation = asyncio.create_task(
                            self.tool(ident, target, epoch, config, message, observations)
                        )
                        try:
                            done, _ = await asyncio.wait(
                                {operation, pulse},
                                timeout=max(0.01, deadline - time.monotonic()),
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            if pulse in done:
                                await pulse
                            if operation not in done:
                                raise ValueError("Episode time budget reached")
                            response = await operation
                        finally:
                            if not operation.done():
                                operation.cancel()
                                with contextlib.suppress(asyncio.CancelledError):
                                    await operation
                        calls[call_id] = (fingerprint, response)
                    worker.stdin.write(
                        (json.dumps({"type": "result", "id": call_id, **response}) + "\n").encode()
                    )
                    await worker.stdin.drain()
                elif kind == "done":
                    if message.get("error"):
                        raise ValueError("Model returned an error; inspect trace")
                    await self.event(ident, message)
                    if not live.get("model_stopped"):
                        raise ValueError("Model ended or budget reached before episode stop")
                    break
                elif kind == "fatal":
                    raise ValueError(message["error"])
                else:
                    await self.event(ident, message)
        except asyncio.CancelledError:
            outcome, error = "stopped", "Stopped by operator"
        except Exception as exc:
            outcome, error = "failed", str(exc)
        finally:
            live["finalizing"] = True
            self.update(ident, status="finalizing")
            # Stop remains independent of worker/model/recording finalization.
            if worker and worker.returncode is None:
                worker.terminate()
            if outcome != "saved" and target and live["epoch"] is not None:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(target.release(ident, live["epoch"]), 3)
            if live["recording"]:
                with contextlib.suppress(Exception):
                    await self.event(
                        ident, {"type": "run_ended", "outcome": outcome, "error": error}
                    )
                try:
                    state = await bridge.request(
                        "POST", "collection/finish", payload={"run_id": ident}
                    )
                    self.update(ident, recording=state, frames=state.get("frames", 0))
                    if state["status"] != "saved" and outcome == "saved":
                        outcome, error = "failed", state.get("error") or "Recording incomplete"
                    with contextlib.suppress(Exception):
                        await target.release(ident, epoch)
                    if state.get("frames", 0) >= 2 and state["status"] != "finalizing":
                        dataset = await bridge.import_dataset(
                            self.get(ident)["project_id"],
                            state["dataset_id"],
                            self.app.state.dataset_library,
                        )
                        self.update(ident, dataset=dataset)
                except Exception as exc:
                    outcome, error = (
                        "failed",
                        "Recording retained; import/finalization failed: " + str(exc),
                    )
            if target and live["epoch"] is not None:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(target.release(ident, live["epoch"]), 3)
            if pulse:
                pulse.cancel()
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await pulse
            for process in (worker, mock):
                if process and process.returncode is None:
                    with contextlib.suppress(ProcessLookupError):
                        process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), 3)
                    except TimeoutError:
                        process.kill()
                        await process.wait()
            if config.mode == "mock" and bridge is not self.app.state.robot_lab:
                await bridge.close()
            live["recording"] = False
            self.update(ident, status=outcome, error=error, proposal=None)
            self.live.pop(ident, None)

    async def tool(self, ident, target, epoch, config, message, observations):
        name, args, call_id = message["name"], message["args"], message["id"]
        command_id = hashlib.sha256((ident + "\0" + call_id).encode()).hexdigest()
        if name == "stop":
            # Model stop concludes the episode; finalizer saves then releases torque.
            self.live[ident]["model_stopped"] = True
            await self.live[ident]["bridge"].request(
                "POST", "collection/stop", payload={"run_id": ident}
            )
            return {"result": {"stopped": True, "task_success_verified": False}}
        if self.live[ident].get("model_stopped"):
            raise ValueError("Tool called after stop")
        if name == "observe":
            raw = await target.observe(ident, epoch)
            side = await self.live[ident]["bridge"].request(
                "GET",
                "camera.jpg",
                query={
                    "index": config.camera_indices[1],
                    "cameras": ",".join(map(str, config.camera_indices)),
                },
                image=True,
            )
            side = side[0] if isinstance(side, tuple) else side
            oid = raw["observation_id"]
            directory = self.directory(ident) / "observations"
            directory.mkdir(exist_ok=True)
            wrist = base64.b64decode(raw["image_data_url"].split(",", 1)[1])
            for role, data in (("wrist", wrist), ("side", side)):
                (directory / (oid + "-" + role + ".jpg")).write_bytes(data)
            clean = {k: v for k, v in raw.items() if k != "image_data_url"}
            clean.update(
                side_image_sha256=hashlib.sha256(side).hexdigest(),
                side_acquired_core_monotonic_ns=time.monotonic_ns(),
            )
            save(directory / (oid + ".json"), clean)
            await self.live[ident]["bridge"].request(
                "POST",
                "collection/observation",
                payload={
                    "run_id": ident,
                    "observation_id": oid,
                    "metadata": clean,
                    "images": {
                        "wrist": base64.b64encode(wrist).decode(),
                        "side": base64.b64encode(side).decode(),
                    },
                },
            )
            observations[oid] = (raw, time.monotonic())
            self.update(ident, observation_id=oid)
            await self.event(
                ident, {"type": "observation", "tool_call_id": call_id, "observation": clean}
            )
            return {
                "result": clean,
                "content": [
                    {"type": "text", "text": json.dumps(clean)},
                    {
                        "type": "image",
                        "data": base64.b64encode(wrist).decode(),
                        "mimeType": "image/jpeg",
                    },
                    {
                        "type": "image",
                        "data": base64.b64encode(side).decode(),
                        "mimeType": "image/jpeg",
                    },
                ],
            }
        if name not in {"move", "set_gripper"}:
            raise ValueError("Unregistered tool")
        oid = args.get("observation_id")
        if oid not in observations:
            raise ValueError("Unknown model observation")
        original, seen = observations[oid]
        if config.mode == "physical":
            proposal = {
                "tool_call_id": call_id,
                "command_id": command_id,
                "tool": name,
                "arguments": args,
                "observation_id": oid,
            }
            self.live[ident]["approved"].clear()
            self.update(ident, status="awaiting_approval", proposal=proposal)
            await asyncio.wait_for(self.live[ident]["approved"].wait(), 60)
            await self.event(
                ident,
                {"type": "operator_approved", "tool_call_id": call_id, "command_id": command_id},
            )
            self.update(ident, status="running", proposal=None)
        if time.monotonic() - seen > 60:
            raise ValueError("Model observation expired; observe again")
        fresh = await target.observe(ident, epoch)
        if any(
            abs(fresh["joints"][j] - original["joints"][j]) > (2 if j == "gripper" else 1)
            for j in fresh["joints"]
        ):
            raise ValueError("Robot drifted from model observation; proposal rejected")
        await self.live[ident]["bridge"].request(
            "POST",
            "collection/observation",
            payload={
                "run_id": ident,
                "observation_id": fresh["observation_id"],
                "metadata": {k: v for k, v in fresh.items() if k != "image_data_url"},
                "images": {"wrist": fresh["image_data_url"].split(",", 1)[1]},
            },
        )
        targets = {"gripper": args["target"]} if name == "set_gripper" else args["targets"]
        requested = {"targets": targets, "duration_seconds": args["duration_seconds"]}
        await self.event(
            ident,
            {
                "type": "command_requested",
                "tool_call_id": call_id,
                "command_id": command_id,
                "requested": requested,
                "model_observation_id": oid,
                "validation_observation": {k: v for k, v in fresh.items() if k != "image_data_url"},
            },
        )
        receipt = await target.execute(
            ident,
            epoch,
            fresh,
            command_id,
            requested,
            config.model,
            command_id,
            model_observation=original,
        )
        await self.event(
            ident,
            {
                "type": "command_result",
                "tool_call_id": call_id,
                "command_id": command_id,
                "receipt": receipt,
            },
        )
        if receipt["status"] != "completed":
            raise ValueError("Movement did not complete; no retry")
        return {"result": receipt}

    async def stop(self, ident):
        live = self.live.get(ident)
        if live and not live.get("finalizing") and not live.get("stopping"):
            live["stopping"] = True
            self.tasks[ident].cancel()
            if not live.get("entered"):
                self.update(ident, status="stopped", error="Stopped before startup")
                self.live.pop(ident, None)
            # Cancellation enters finalization, whose first hardware operation is release.
        return self.get(ident)

    async def close(self):
        for ident in list(self.live):
            await self.stop(ident)
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)


async def project(request, project_id):
    if await request.app.state.projects.get(project_id) is None:
        raise HTTPException(404, "Project not found")
    return request.app.state.collection_runs


def owned(service, project_id, ident):
    try:
        value = service.get(ident)
    except ValueError, FileNotFoundError:
        raise HTTPException(404, "Run not found") from None
    if value["project_id"] != project_id:
        raise HTTPException(404, "Run not found")
    return value


@router.get("/projects/{project_id}/robot-lab/collection-runs")
async def runs(project_id: str, request: Request):
    return (await project(request, project_id)).list(project_id)


@router.post("/projects/{project_id}/robot-lab/collection-runs", status_code=201)
async def start(project_id: str, payload: CollectionInput, request: Request):
    service = await project(request, project_id)
    try:
        return await service.start(project_id, payload)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None


@router.post("/projects/{project_id}/robot-lab/collection-runs/{ident}/{action}")
async def action(project_id: str, ident: str, action: str, request: Request):
    service = await project(request, project_id)
    value = owned(service, project_id, ident)
    live = service.live.get(ident)
    if action == "stop":
        return await service.stop(ident)
    if action == "heartbeat" and live:
        live["operator"] = time.monotonic()
        return {"ok": True}
    if action == "approve" and live and value["proposal"]:
        body = await request.json()
        if body.get("command_id") != value["proposal"]["command_id"]:
            raise HTTPException(409, "Proposal changed")
        live["approved"].set()
        return {"ok": True}
    raise HTTPException(409, "Run action unavailable")


@router.get("/projects/{project_id}/robot-lab/collection-runs/{ident}/trace")
async def trace(project_id: str, ident: str, request: Request):
    service = await project(request, project_id)
    owned(service, project_id, ident)
    path = service.directory(ident) / "trace.jsonl"
    if not path.exists():
        raise HTTPException(404, "No trace yet")
    return FileResponse(path, media_type="application/x-ndjson", filename=ident + "-trace.jsonl")


@router.get("/projects/{project_id}/robot-lab/collection-runs/{ident}/image/{role}")
async def image(project_id: str, ident: str, role: str, request: Request):
    service = await project(request, project_id)
    value = owned(service, project_id, ident)
    if role not in {"wrist", "side"} or not value.get("observation_id"):
        raise HTTPException(404, "No observation yet")
    return FileResponse(
        service.directory(ident) / "observations" / (value["observation_id"] + "-" + role + ".jpg"),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )
