"""Pinned SkyPilot submission and bounded, read-only final-state reconciliation.

Invoked only through sky.sh so its isolated runtime/configuration still applies.
A terminal auxiliary cancellation is not proof of infrastructure deletion.
"""

import argparse
import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

SDK_VERSION = "0.13.0"
NOT_FINISHED = 101
RECONCILE_SECONDS = 120
COLLECTION_SECONDS = 40
POLL_SECONDS = 2
JSON_LIMIT = 16 * 1024
QUEUE_FIELDS = (
    "job_id",
    "job_name",
    "task_id",
    "task_name",
    "execution",
    "is_job_group",
    "is_primary_in_job_group",
    "status",
)
ACTIVE = {"PENDING", "SUBMITTED", "STARTING", "RUNNING", "WINDING_DOWN", "RECOVERING", "CANCELLING"}
FAILED = {"FAILED", "FAILED_SETUP", "FAILED_PRECHECKS", "FAILED_NO_RESOURCE", "FAILED_CONTROLLER"}
STATES = ACTIVE | FAILED | {"SUCCEEDED", "CANCELLED"}


def read_json(path):
    """Bound the trusted local subprocess interchange and reject partial files."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > JSON_LIMIT:
            raise ValueError("Invalid or oversized status receipt")
        content = stream.read(JSON_LIMIT + 1)
    if len(content) > JSON_LIMIT:
        raise ValueError("Oversized status receipt")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate key in status receipt")
            result[key] = value
        return result

    return json.loads(content, object_pairs_hook=unique_keys)


def write_json(path, value):
    content = (json.dumps(value, allow_nan=False, sort_keys=True) + "\n").encode()
    if len(content) > JSON_LIMIT or path.is_symlink():
        raise ValueError("Invalid status receipt destination or size")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".status-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def request_id(value):
    # sky.get(None) can select the latest request: never pass an absent ID.
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
        raise ValueError("SkyPilot did not return a valid request ID; submission is uncertain")
    return value


def _runtime():
    if version("skypilot") != SDK_VERSION:
        raise ValueError(f"This adapter requires SkyPilot {SDK_VERSION}")
    import sky

    return sky


def load_dag(path):
    # These internal helpers are the loader/default chain used by the pinned CLI.
    from sky.utils import dag_utils

    dag = dag_utils.load_dag_from_yaml(str(path))
    dag_utils.maybe_infer_and_fill_dag_and_task_names(dag)
    dag_utils.fill_default_config_in_dag_for_job_launch(dag)
    return dag


def identity(dag):
    names = [task.name for task in dag.tasks]
    if (
        not dag.is_job_group()
        or dag.primary_tasks != ["isaac"]
        or len(names) != 2
        or set(names) != {"isaac", "vla"}
        or not isinstance(dag.name, str)
        or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", dag.name)
    ):
        raise ValueError("Expected a named parallel isaac-primary/vla-auxiliary job group")
    return {
        "group_name": dag.name,
        "tasks": [
            {"task_id": index, "task_name": name, "is_primary_in_job_group": name == "isaac"}
            for index, name in enumerate(names)
        ],
    }


def validate_receipt(receipt):
    if (
        not isinstance(receipt, dict)
        or type(receipt.get("schema_version")) is not int
        or receipt["schema_version"] != 1
        or receipt.get("skypilot") != SDK_VERSION
        or type(receipt.get("job_id")) is not int
        or receipt["job_id"] <= 0
        or not isinstance(receipt.get("group_name"), str)
        or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", receipt["group_name"])
    ):
        raise ValueError("Missing or invalid submitted job identity")
    request_id(receipt.get("request_id"))
    tasks = receipt.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 2:
        raise ValueError("Missing submitted task identities")
    names = []
    for index, task in enumerate(tasks):
        if (
            not isinstance(task, dict)
            or type(task.get("task_id")) is not int
            or task["task_id"] != index
            or task.get("task_name") not in {"isaac", "vla"}
            or task.get("is_primary_in_job_group") is not (task["task_name"] == "isaac")
        ):
            raise ValueError("Invalid submitted task identity or role")
        names.append(task["task_name"])
    if set(names) != {"isaac", "vla"}:
        raise ValueError("Duplicate submitted task identities")
    return receipt


def task_states(rows, receipt):
    validate_receipt(receipt)
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("Expected both submitted job-group task records")
    expected = {task["task_name"]: task for task in receipt["tasks"]}
    states = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid task record")
        name = row.get("task_name")
        if not isinstance(name, str) or name not in expected or name in states:
            raise ValueError("Missing, duplicate or unexpected task name")
        if (
            type(row.get("job_id")) is not int
            or row["job_id"] != receipt["job_id"]
            or row.get("job_name") != receipt["group_name"]
            or type(row.get("task_id")) is not int
            or row["task_id"] != expected[name]["task_id"]
            or row.get("execution") != "parallel"
            or row.get("is_job_group") is not True
            or row.get("is_primary_in_job_group") is not expected[name]["is_primary_in_job_group"]
        ):
            raise ValueError("Queue job/task identity or primary role differs from submission")
        state = row.get("status")
        if not isinstance(state, str) or state not in STATES:
            raise ValueError("Missing or unrecognized task status")
        states[name] = state
    return states


def observe(sky, receipt):
    validate_receipt(receipt)
    result = sky.get(
        request_id(
            sky.jobs.queue_v2(
                refresh=False,
                skip_finished=False,
                all_users=False,
                job_ids=[receipt["job_id"]],
                limit=1,
                fields=list(QUEUE_FIELDS),
            )
        )
    )
    if (
        not isinstance(result, tuple)
        or len(result) != 4
        or not isinstance(result[0], list)
        or len(result[0]) != 2
    ):
        raise ValueError("Unexpected SkyPilot queue response")
    # SDK 0.13 returns mapping-compatible ManagedJobRecord objects. Retain only
    # fields needed for identity/status, never user YAML, environment or logs.
    rows = []
    for record in result[0]:
        row = {field: record.get(field) for field in QUEUE_FIELDS}
        row["status"] = getattr(row["status"], "value", row["status"])
        rows.append(row)
    task_states(rows, receipt)
    return rows


@contextmanager
def deferred_signals():
    pending = []
    handlers = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        for number in handlers:
            signal.signal(number, lambda signum, frame: pending.append((signum, frame)))
        yield pending, handlers
    finally:
        for number, handler in handlers.items():
            signal.signal(number, handler)


def run_owned(command, *, timeout, terminate_grace=0, **kwargs):
    """Bound/reap our child without signalling unrelated Sky API daemon groups.

    Defer SIGINT/SIGTERM while Popen creates the child: otherwise an interrupt
    between process creation and returned-handle assignment can orphan it.
    Exec resets the child's temporary Python handlers; no blocked mask leaks.
    """
    process = None
    try:
        with deferred_signals() as (pending, handlers):
            process = subprocess.Popen(command, **kwargs)
        for number, frame in pending:
            handler = handlers[number]
            if callable(handler):
                handler(number, frame)
            elif handler != signal.SIG_IGN:
                raise SystemExit(128 + number)
        code = process.wait(timeout=timeout)
        if code != 0:
            raise subprocess.CalledProcessError(code, command)
        return subprocess.CompletedProcess(command, code)
    except BaseException:
        # Repeated Ctrl-C/termination must not interrupt this bounded cleanup.
        # Retain the original exception and restore the caller's handlers.
        with deferred_signals():
            if process is not None and process.poll() is None:
                if terminate_grace:
                    process.terminate()
                    try:
                        process.wait(timeout=terminate_grace)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                else:
                    process.kill()
                    process.wait(timeout=5)
        raise


def collect(receipt_path, timeout):
    with tempfile.TemporaryDirectory(prefix="firebird-status-") as directory:
        output = Path(directory) / "observation.json"
        run_owned(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "observe",
                "--receipt",
                str(receipt_path),
                "--output",
                str(output),
            ],
            timeout=timeout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return read_json(output)


def reconcile(receipt_path, receipt):
    deadline = time.monotonic() + RECONCILE_SECONDS
    receipt["phase"] = "reconciling"
    write_json(receipt_path, receipt)
    try:
        while (remaining := deadline - time.monotonic()) > 0:
            rows = collect(receipt_path, min(COLLECTION_SECONDS, remaining))
            states = task_states(rows, receipt)
            receipt["last_observation"] = {
                "tasks": states,
                # Pinned SkyPilot's group aggregate considers primary tasks only.
                # This schema has exactly one primary, so the derivation is exact.
                "derived_group_status": states["isaac"],
            }
            receipt.setdefault("first_observation", receipt["last_observation"])
            if time.monotonic() >= deadline:
                break
            primary, auxiliary = states["isaac"], states["vla"]
            if (
                primary in FAILED
                or primary == "CANCELLED"
                or auxiliary in FAILED
                or (auxiliary == "CANCELLED" and primary != "SUCCEEDED")
            ):
                receipt["phase"] = "reconciliation_refused"
                receipt["reason"] = (
                    "Primary cancellation/failure or unexpected auxiliary termination"
                )
                break
            if primary == "SUCCEEDED" and auxiliary in {"SUCCEEDED", "CANCELLED"}:
                receipt.update(phase="reconciled", exit_code=0)
                write_json(receipt_path, receipt)
                print(
                    f"Execution reconciled: primary isaac=SUCCEEDED, auxiliary vla={auxiliary}. "
                    "Group SUCCEEDED is derived from its primary; resource deletion is unverified."
                )
                return 0
            write_json(receipt_path, receipt)
            time.sleep(min(POLL_SECONDS, max(0, deadline - time.monotonic())))
        else:
            receipt["reason"] = "Final-state reconciliation deadline expired"
        if receipt["phase"] == "reconciling":
            receipt["reason"] = "Final-state reconciliation deadline expired"
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
        # Keep a bounded, non-secret category: SDK exceptions may contain URLs or envs.
        receipt["reason"] = f"Final status unavailable ({type(error).__name__})"
    receipt.update(phase="unresolved", exit_code=NOT_FINISHED)
    write_json(receipt_path, receipt)
    print(f"{receipt['reason']}; retaining attached exit {NOT_FINISHED}.", file=sys.stderr)
    return NOT_FINISHED


def launch(sky, path, receipt_path, *, yes=False, detached=False):
    dag = load_dag(path)
    receipt = {
        "schema_version": 1,
        "skypilot": SDK_VERSION,
        **identity(dag),
        "yaml_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "phase": "prepared",
        "request_id": None,
        "job_id": None,
    }
    # Preflight durable output before starting a potentially paid submission.
    if receipt_path.exists() or receipt_path.is_symlink():
        raise ValueError("Submission receipt must not already exist")
    write_json(receipt_path, receipt)
    print(f"Submission receipt: {receipt_path}", flush=True)
    receipt["request_id"] = request_id(
        sky.jobs.launch(
            dag,
            name=dag.name,
            _need_confirmation=not yes,
        )
    )
    print(f"Launch request ID: {receipt['request_id']}", flush=True)
    receipt["phase"] = "submitted_request"
    write_json(receipt_path, receipt)
    result = sky.stream_and_get(receipt["request_id"])
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("Missing submission result; inspect the request receipt, do not resubmit")
    job_ids = [result[0]] if type(result[0]) is int else result[0]
    if (
        not isinstance(job_ids, list)
        or len(job_ids) != 1
        or type(job_ids[0]) is not int
        or job_ids[0] <= 0
    ):
        raise ValueError("Expected one submitted job ID; inspect receipt, do not resubmit")
    receipt.update(job_id=job_ids[0], phase="submitted")
    print(f"Submitted job {receipt['job_id']} ({dag.name}).", flush=True)
    write_json(receipt_path, receipt)
    if detached:
        return 0
    code = sky.jobs.tail_logs(
        name=None,
        job_id=receipt["job_id"],
        follow=True,
        controller=False,
        refresh=False,
    )
    if type(code) is not int or not 0 <= code <= 255:
        raise ValueError("Attached logs did not return a valid completion code")
    receipt.update(attached_exit_code=code, exit_code=code, phase="attached_finished")
    write_json(receipt_path, receipt)
    if code == NOT_FINISHED:
        return reconcile(receipt_path, receipt)
    return code


def _terminated(signum, frame):
    # The outer launcher uses SIGTERM on its wall deadline. Raising lets an
    # in-flight owned-process guard reap its query child before we exit.
    raise SystemExit(128 + signum)


def main():
    signal.signal(signal.SIGTERM, _terminated)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    submit = sub.add_parser("launch")
    submit.add_argument("task", type=Path)
    submit.add_argument("--receipt", type=Path, required=True)
    submit.add_argument("--yes", action="store_true")
    submit.add_argument("--detach-run", action="store_true")
    query = sub.add_parser("observe")
    query.add_argument("--receipt", type=Path, required=True)
    query.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        sky = _runtime()
        if args.operation == "launch":
            return launch(sky, args.task, args.receipt, yes=args.yes, detached=args.detach_run)
        write_json(args.output, observe(sky, read_json(args.receipt)))
        return 0
    except Exception as error:
        print(
            f"Rollout SDK operation failed ({type(error).__name__}); "
            f"inspect receipt {args.receipt}. No automatic resubmission.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
