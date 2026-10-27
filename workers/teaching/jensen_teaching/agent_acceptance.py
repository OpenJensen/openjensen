"""Finite native agent experiments with independent geometric task measurements.

Run in the configured Isaac runtime through the deployment's compute scheduler.
The module never accepts a vendor license, provisions a GPU, or enables hardware.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import random
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from sim_worker.rollout.evaluation import EvaluationSpec

from .contracts import Command, Settings, canonical, decode, finite, integer, text
from .control import Mailbox
from .journal import Journal
from .session import Session


@dataclasses.dataclass(frozen=True)
class Scenario:
    task: str
    clock_policy: str
    seeds: tuple[int, ...]
    max_steps: int
    max_turns: int
    max_seconds: int
    max_output_tokens: int
    max_total_tokens: int
    object_xy_jitter_m: float
    evaluation: EvaluationSpec
    scripted_actions: tuple[dict, ...]

    @classmethod
    def parse(cls, value):
        expected = {
            "schema_version",
            "task",
            "clock_policy",
            "seeds",
            "max_steps",
            "max_turns",
            "max_seconds",
            "max_output_tokens",
            "max_total_tokens",
            "object_xy_jitter_m",
            "evaluation",
            "scripted_actions",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("Native scenario fields do not match the versioned contract")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("Unsupported native scenario schema")
        if value["clock_policy"] not in {"stepped", "continuous"}:
            raise ValueError("Choose an explicit simulation clock")
        seeds = value["seeds"]
        if not isinstance(seeds, list) or not 1 <= len(seeds) <= 20:
            raise ValueError("Choose one to twenty explicit paired seeds")
        seeds = tuple(integer(v, "seed", 0, 2**32 - 1) for v in seeds)
        if len(set(seeds)) != len(seeds):
            raise ValueError("Seeds must be distinct")
        jitter = finite(value["object_xy_jitter_m"], "object jitter")
        if not 0 <= jitter <= 0.03:
            raise ValueError("Object jitter must be within 0.03 meters")
        actions = value["scripted_actions"]
        if not isinstance(actions, list) or len(actions) > 100:
            raise ValueError("Scripted baseline supports at most one hundred actions")
        output_tokens = integer(value["max_output_tokens"], "output token budget", 64, 32768)
        total_tokens = integer(
            value["max_total_tokens"], "total token budget", output_tokens, 1000000
        )
        parsed = []
        for item in actions:
            if not isinstance(item, dict) or set(item) != {"joint", "delta_rad"}:
                raise ValueError("Scripted actions require named joint and delta_rad")
            delta = finite(item["delta_rad"], "delta_rad")
            if not 0 < abs(delta) <= 0.15:
                raise ValueError("Scripted corrections must be nonzero and within 0.15 radians")
            parsed.append({"joint": text(item["joint"], "joint", 96), "delta_rad": delta})
        return cls(
            text(value["task"], "task", 256),
            value["clock_policy"],
            seeds,
            integer(value["max_steps"], "maximum steps", 1, 600),
            integer(value["max_turns"], "maximum turns", 1, 60),
            integer(value["max_seconds"], "maximum seconds", 1, 600),
            output_tokens,
            total_tokens,
            jitter,
            EvaluationSpec.parse(value["evaluation"]),
            tuple(parsed),
        )


def hardware_preflight():
    """Bounded read-only checks of the vendor's memory floor, before importing Kit."""
    reasons, gpus = [], []
    try:
        memory_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        memory_bytes = None
    if memory_bytes is None or memory_bytes < 32_000_000_000:
        reasons.append("Native acceptance requires at least the documented 32 GB system RAM")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            timeout=5,
            check=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            name, memory = line.rsplit(",", 1)
            gpus.append({"name": name.strip(), "memory_mib": int(memory.strip())})
    except (OSError, subprocess.SubprocessError, ValueError):
        reasons.append("A readable NVIDIA GPU inventory is required")
    if not any(gpu["memory_mib"] >= 16000 for gpu in gpus):
        reasons.append("Native acceptance requires at least the documented 16 GB GPU memory")
    return {"system_memory_bytes": memory_bytes, "gpus": gpus, "reasons": reasons}


def runtime_preflight():
    """Read presence and explicit operator gates without importing/starting Kit."""
    hardware = hardware_preflight()
    reasons = list(hardware["reasons"])
    version = None
    if sys.version_info[:2] != (3, 12):
        reasons.append("The pinned native runtime requires Python 3.12")
    try:
        version = importlib.metadata.version("isaacsim")
    except importlib.metadata.PackageNotFoundError:
        reasons.append("The pinned Isaac 6.1 runtime is not installed")
    if version is not None and version != "6.1.0.0":
        reasons.append("Installed Isaac identity is not the pinned 6.1.0.0 runtime")
    if importlib.util.find_spec("isaacsim") is None:
        reasons.append("The native isaacsim package is not importable")
    if os.environ.get("ACCEPT_EULA") != "Y" or os.environ.get(
        "OMNI_KIT_ACCEPT_EULA", ""
    ).upper() not in {"YES", "Y", "1"}:
        reasons.append(
            "Explicit operator license acceptance is missing; no acceptance flag was set"
        )
    return {
        "status": "blocked" if reasons else "ready_for_native_start",
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "isaacsim": version,
        "hardware": hardware,
        "reasons": reasons,
        "native_execution_verified": False,
    }


class ScriptedPlanner:
    def __init__(self, actions):
        self.actions = actions
        self.turn = 0

    def __call__(self, _request, _cancel):
        if self.turn >= len(self.actions):
            return {
                "status": "assessed",
                "model": "scripted-baseline-v1",
                "assessment": {
                    "outcome": "cannot_proceed",
                    "summary": "The declared scripted sequence is exhausted.",
                },
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            }
        value = self.actions[self.turn]
        self.turn += 1
        return {
            "status": "proposed",
            "model": "scripted-baseline-v1",
            "arguments": dict(value),
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }


class ProcessPlanner:
    """Use the same isolated provider protocol as the application, never the motor transport."""

    def __init__(self, python, provider, model, reasoning_effort, worker_root):
        # Preserve the venv executable path; resolving its symlink loses the venv.
        selected = Path(python).absolute()
        selected.resolve(strict=True)
        if not selected.is_file() or not os.access(selected, os.X_OK):
            raise ValueError("Planner Python must be an existing executable")
        root = Path(worker_root).resolve(strict=True)
        if not (root / "jensen_agent_control/__main__.py").is_file():
            raise ValueError("Planner worker source is unavailable")
        self.python, self.worker_root = str(selected), str(root)
        self.provider, self.model, self.reasoning_effort = provider, model, reasoning_effort

    def __call__(self, request, cancel):
        environment = {
            k: v
            for k, v in os.environ.items()
            if k in {"PATH", "HOME", "TMPDIR", "LANG", "SYSTEMROOT", "OPENJENSEN_AGENT_API_KEY"}
        }
        environment.update(
            PYTHONPATH=self.worker_root, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1"
        )
        request = dict(
            request,
            provider=self.provider,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
        )
        process = subprocess.Popen(
            [self.python, "-m", "jensen_agent_control"],
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        pending = canonical(request)
        deadline = time.monotonic() + request["timeout_seconds"] + 2
        try:
            while True:
                if cancel.is_set() or time.monotonic() >= deadline:
                    raise TimeoutError("Native planner request cancelled or timed out")
                try:
                    output, _ = process.communicate(pending, timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    pending = None
            if len(output) > 16 * 1024**2 or process.returncode:
                raise ValueError("Isolated planner failed; no action was admitted")
            result = decode(output, limit=16 * 1024**2)
            if result.get("status") == "error":
                raise ValueError("Isolated planner returned an error; no action was admitted")
            return result
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.communicate()


def _manual(session, mailbox, operation, arguments):
    command = Command(
        uuid.uuid4().hex,
        session.journal.session_id,
        session.episode_id,
        session.revision,
        operation,
        arguments,
    )
    mailbox.submit(dataclasses.asdict(command))
    mailbox.drain(session)
    result = mailbox.get(command.command_id)
    if result["status"] != "acknowledged":
        raise ValueError("Native supervisory command was rejected")
    return result


def _agent(session, mailbox, owner, operation, arguments, *, epoch=None):
    payload = {
        "command_id": uuid.uuid4().hex,
        "session_id": session.journal.session_id,
        "agent_session_id": owner,
        "authority_epoch": session.agent.epoch if epoch is None else epoch,
        "arguments": arguments,
    }
    mailbox.submit(payload, agent_operation=operation)
    mailbox.drain(session)
    return mailbox.get(payload["command_id"])


def run_episode(settings, simulation, output, scenario, planner, *, backend_kind="fixture"):
    """Only the calling owner thread steps the simulator, including while reasoning waits."""
    journal = Journal(output, settings)
    session = Session(settings, simulation, journal)
    mailbox = Mailbox()
    owner = uuid.uuid4().hex
    report = {
        "status": "running",
        "native_execution_verified": False,
        "backend_kind": backend_kind,
        "clock_policy": scenario.clock_policy,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scenario_sha256": hashlib.sha256(canonical(dataclasses.asdict(scenario))).hexdigest(),
        "scene_sha256": settings.scene_sha256,
        "criteria_sha256": scenario.evaluation.digest,
        "proposals": [],
        "receipts": [],
        "planning_wall_seconds": 0.0,
        "simulated_seconds": 0.0,
        "token_budget_charged": 0,
        "token_usage_reported": 0,
        "task_success_verified": False,
        "geometric_success": False,
        "grasp_and_release_verified": False,
        "physical_transfer_verified": False,
    }
    setup_started = time.monotonic()
    started = setup_started
    cancel = threading.Event()
    continuation, previous_receipt = None, None
    heartbeat_at = started
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        _manual(session, mailbox, "task", {"instruction": scenario.task})
        _manual(session, mailbox, "start", {})
        acquired = _agent(
            session,
            mailbox,
            owner,
            "acquire",
            {
                "clock_policy": scenario.clock_policy,
                "expected_revision": session.revision,
                "episode_id": session.episode_id,
                "max_session_seconds": scenario.max_seconds,
                **(
                    {"planning_snapshot_seconds": min(120, scenario.max_seconds)}
                    if scenario.clock_policy == "stepped"
                    else {}
                ),
            },
            epoch=0,
        )
        if acquired["status"] != "acknowledged":
            raise ValueError("Native agent acquisition failed")
        report["setup_wall_seconds"] = time.monotonic() - setup_started
        started = time.monotonic()
        heartbeat_at = started
        action_space = {
            "kind": "simulation",
            "joint_units": {name: "rad" for name in settings.sim.joints},
            "joint_limits": {
                name: list(limits)
                for name, limits in zip(settings.sim.joints, simulation.joint_limits, strict=True)
            },
            "max_delta": {name: 0.15 for name in settings.sim.joints},
            "min_duration_seconds": 0.1,
            "max_duration_seconds": min(2.0, 60 / settings.sim.fps),
        }

        def renew_if_due():
            nonlocal heartbeat_at
            now = time.monotonic()
            if now >= heartbeat_at:
                beat = _agent(session, mailbox, owner, "heartbeat", {})
                if beat["status"] != "acknowledged":
                    raise ValueError("Native owner lease expired")
                heartbeat_at = now + 0.5

        for turn in range(scenario.max_turns):
            if (
                time.monotonic() - started >= scenario.max_seconds
                or session.observation.step >= scenario.max_steps
            ):
                report["termination"] = "budget_exhausted"
                break
            remaining_tokens = scenario.max_total_tokens - report["token_budget_charged"]
            if remaining_tokens < scenario.max_output_tokens:
                report["termination"] = "token_budget_exhausted"
                break
            renew_if_due()
            observed = _agent(session, mailbox, owner, "observe", {})
            if observed["status"] != "acknowledged":
                raise ValueError("Native owner observation was rejected")
            observation = observed["observation"]
            request = {
                "operation": "continue" if continuation else "plan",
                "planning_mode": "task",
                "task": scenario.task,
                "observation": observation,
                "joint_names": list(settings.sim.joints),
                "timeout_seconds": min(90, scenario.max_seconds - (time.monotonic() - started)),
                "max_output_tokens": scenario.max_output_tokens,
                "max_total_tokens": remaining_tokens,
                "action_space": action_space,
                "completed_steps": len(report["receipts"]),
            }
            if continuation:
                request.update(continuation=continuation, receipt=previous_receipt)
            planned_at = time.monotonic()
            future = executor.submit(planner, request, cancel)
            while not future.done():
                now = time.monotonic()
                if (
                    now - started >= scenario.max_seconds
                    or session.observation.step >= scenario.max_steps
                ):
                    cancel.set()
                    raise TimeoutError("Native episode budget expired while reasoning")
                renew_if_due()
                if scenario.clock_policy == "continuous":
                    session.tick()
                    mailbox.publish(session)
                time.sleep(max(0, 1 / settings.sim.fps - (time.monotonic() - now)))
            response = future.result()
            report["planning_wall_seconds"] += time.monotonic() - planned_at
            usage = response.get("usage", {})
            tokens = usage.get("total_tokens", 0) if isinstance(usage, dict) else 0
            if type(tokens) is not int or not 0 <= tokens <= 1000000:
                raise ValueError("Planner token usage is not bounded accounting evidence")
            report["token_usage_reported"] += tokens
            # Reserve each output cap even when provider accounting is absent or lower.
            report["token_budget_charged"] += max(tokens, scenario.max_output_tokens)
            if report["token_budget_charged"] > scenario.max_total_tokens:
                raise ValueError("Planner usage exceeded the episode token budget")
            continuation = response.get("continuation")
            if response.get("status") in {"assessed", "complete"}:
                report["model_assessment"] = response.get(
                    "assessment",
                    {"outcome": "needs_help", "summary": "Provider returned no executable action."},
                )
                report["termination"] = "model_finished"
                break
            arguments = response.get("arguments")
            skill = response.get("skill", "joint_correction")
            expected = (
                {"joint", "delta_rad"}
                if skill == "joint_correction"
                else {"targets", "duration_seconds"}
            )
            if (
                response.get("status") != "proposed"
                or not isinstance(arguments, dict)
                or skill not in {"joint_correction", "joint_segment"}
                or set(arguments) != expected
            ):
                raise ValueError("Native planner did not return a supported bounded skill")
            action_ticks = (
                math.ceil(arguments["duration_seconds"] * settings.sim.fps)
                if skill == "joint_segment"
                else 1
            )
            if session.observation.step + action_ticks > scenario.max_steps:
                report["termination"] = "step_budget_exhausted_before_admission"
                break
            proposal_id = uuid.uuid4().hex
            report["proposals"].append(
                {
                    "id": proposal_id,
                    "observation_id": observation["observation_id"],
                    "observation_sha256": observation["observation_sha256"],
                    "model": response.get("model"),
                    "skill": skill,
                    "arguments": arguments,
                    "usage": usage,
                }
            )
            receipt = _agent(
                session,
                mailbox,
                owner,
                "segment" if skill == "joint_segment" else "execute",
                {
                    **arguments,
                    "observation_id": observation["observation_id"],
                    "observation_sha256": observation["observation_sha256"],
                    "proposal_id": proposal_id,
                    "provider_model": response["model"],
                },
            )
            while receipt["status"] == "executing":
                if time.monotonic() - started >= scenario.max_seconds:
                    report["termination"] = "wall_budget_during_execution"
                    _agent(session, mailbox, owner, "release", {})
                else:
                    renew_if_due()
                    session.tick()
                mailbox.publish(session)
                receipt = mailbox.get(receipt["command_id"])
            report["receipts"].append(receipt)
            if (
                receipt["status"] != "acknowledged"
                or receipt.get("execution", {}).get("status") != "completed"
            ):
                report["termination"] = "action_rejected"
                break
            previous_receipt = {
                "status": "completed",
                "execution": receipt["execution"],
                "task_success_verified": False,
            }
        else:
            report["termination"] = "turn_budget_exhausted"
        report["simulated_seconds"] = session.observation.sim_time
        evidence = getattr(simulation, "task_evidence", lambda: None)()
        report["task_evidence"] = evidence
        report["geometric_success"] = bool(evidence and evidence.get("geometric_success"))
        report["native_execution_verified"] = (
            backend_kind == "isaac"
            and bool(report["receipts"])
            and all(r["status"] == "acknowledged" for r in report["receipts"])
        )
        report["status"] = (
            "executed"
            if report["native_execution_verified"]
            else "fixture_executed"
            if backend_kind == "fixture"
            else "no_verified_execution"
        )
    except Exception as error:
        report.update(status="failed", termination=type(error).__name__, error=str(error))
    finally:
        cancel.set()
        executor.shutdown(wait=True, cancel_futures=True)
        try:
            if session.agent.owner == owner:
                _agent(session, mailbox, owner, "release", {})
            if session.episode_id is not None and session.mode not in {"faulted", "closed"}:
                _manual(session, mailbox, "finish", {})
        finally:
            session.close()
        report["wall_seconds"] = time.monotonic() - started
        (output.parent / "result.json").write_bytes(canonical(report) + b"\n")
    return report


def make_seed_scene(scene, evaluation, output, seed, jitter):
    """Author one small USD override; original assets and their transforms stay intact."""
    from pxr import Sdf, Usd, UsdGeom

    layer = Sdf.Layer.CreateNew(str(output))
    layer.subLayerPaths = [str(Path(scene).resolve(strict=True))]
    stage = Usd.Stage.Open(layer)
    prim = stage.GetPrimAtPath(evaluation.object_prim)
    if not prim:
        raise ValueError("The declared native target object is absent")
    translates = [
        op
        for op in UsdGeom.Xformable(prim).GetOrderedXformOps()
        if op.GetOpName() == "xformOp:translate"
    ]
    if len(translates) != 1:
        raise ValueError("Native paired seeds require one explicit object translation")
    before = translates[0].Get()
    randomizer = random.Random(seed)
    offset = (randomizer.uniform(-jitter, jitter), randomizer.uniform(-jitter, jitter), 0.0)
    translates[0].Set(type(before)(*(float(before[i]) + offset[i] for i in range(3))))
    stage.GetRootLayer().Save()
    return {
        "seed": seed,
        "object_offset_m": offset,
        "independent_initial_condition": jitter > 0,
        "scene_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--provider", choices=["scripted", "mock", "openai_responses"], default="scripted"
    )
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument(
        "--reasoning-effort", choices=["low", "medium", "high", "xhigh", "max"], default="low"
    )
    parser.add_argument("--planner-python", type=Path)
    parser.add_argument("--planner-source", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    settings = Settings.load(args.settings)
    scenario = Scenario.parse(decode(args.scenario.read_bytes(), limit=65536))
    if scenario.max_steps >= settings.max_steps:
        parser.error("Scenario horizon must leave room within the configured capture bound")
    if settings.task_evaluation is not None and settings.task_evaluation != scenario.evaluation:
        parser.error("Scenario criteria differ from the operator-configured native criteria")
    if args.output.exists():
        parser.error("Choose a new operator-owned output directory")
    args.output.mkdir(parents=True)
    preflight = runtime_preflight()
    (args.output / "preflight.json").write_bytes(canonical(preflight) + b"\n")
    if preflight["status"] == "blocked" or args.preflight_only:
        print(json.dumps(preflight))
        return 2 if preflight["status"] == "blocked" else 0
    if args.provider != "scripted" and (args.planner_python is None or args.planner_source is None):
        parser.error("Provider experiments need an explicit isolated planner Python and source")
    if args.provider == "openai_responses" and not os.environ.get("OPENJENSEN_AGENT_API_KEY"):
        parser.error("Save a selected provider credential securely before a paid experiment")
    from sim_worker.rollout.isaac import IsaacSim

    results = []
    with IsaacSim(settings.sim, scenario.evaluation) as simulation:
        for seed in scenario.seeds:
            directory = args.output / str(seed)
            directory.mkdir()
            scene = directory / "scene.usda"
            identity = make_seed_scene(
                settings.sim.scene, scenario.evaluation, scene, seed, scenario.object_xy_jitter_m
            )
            simulation.select_scene(str(scene))
            selected = dataclasses.replace(
                settings,
                sim=dataclasses.replace(settings.sim, scene=str(scene)),
                scene_sha256=identity["scene_sha256"],
                task_evaluation=scenario.evaluation,
            )
            planner = (
                ScriptedPlanner(scenario.scripted_actions)
                if args.provider == "scripted"
                else ProcessPlanner(
                    args.planner_python,
                    args.provider,
                    args.model,
                    args.reasoning_effort,
                    args.planner_source,
                )
            )
            result = run_episode(
                selected, simulation, directory / "capture", scenario, planner, backend_kind="isaac"
            )
            results.append({"initial_condition": identity, "report": result})
    summary = {
        "schema_version": 1,
        "status": "executed"
        if all(r["report"]["native_execution_verified"] for r in results)
        else "incomplete",
        "provider": args.provider,
        "requested_model": args.model,
        "clock_policy": scenario.clock_policy,
        "task": scenario.task,
        "criteria_sha256": scenario.evaluation.digest,
        "episodes": results,
        "native_execution_verified": all(r["report"]["native_execution_verified"] for r in results),
        "geometric_successes": sum(r["report"]["geometric_success"] for r in results),
        "task_success_verified": False,
        "grasp_and_release_verified": False,
        "physical_transfer_verified": False,
    }
    (args.output / "summary.json").write_bytes(canonical(summary) + b"\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}))
    return 0 if summary["native_execution_verified"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
