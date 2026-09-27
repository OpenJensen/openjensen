"""Publish bounded, read-only SkyPilot 0.13 snapshots for the Firebird app.

The app reads JSON only. This process retains the isolated runner credentials;
it never launches jobs, restarts controllers, cancels jobs or loads model weights.
"""

import argparse
import copy
import io
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from urllib.parse import unquote_plus

LOG_BYTES = 48 * 1024
TASKS = ("isaac", "vla")
RUN_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")
ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
SECRET = re.compile(
    r"(\b(?:access_token|refresh_token|client_secret|api[-_]key|password|token|secret|"
    r"hf_token|huggingface_token|hugging_face_hub_token|cloudsdk_auth_access_token|"
    r"authorization|proxy[-_]authorization|cookie|set[-_]cookie|x[-_]api[-_]key)"
    r"\b[\"']?\s*[:=]\s*)"
    r"(?:\"(?:\\.|[^\"\\\r\n])*(?:\"|$)|'(?:\\.|[^'\\\r\n])*(?:'|$)|[^\s\"',}&#]+)",
    re.I | re.M,
)
HEADERS = re.compile(
    r"((?<!\S)(?:authorization|proxy[-_]authorization|cookie|set[-_]cookie|"
    r"x[-_]api[-_]key)\s*[:=]\s*)[^\r\n]*",
    re.I,
)
QUERY_KEYS = {
    "access_token",
    "refresh_token",
    "client_secret",
    "api_key",
    "apikey",
    "password",
    "token",
    "secret",
    "hf_token",
    "signature",
    "sig",
    "x_goog_signature",
    "x_goog_credential",
    "x_goog_security_token",
    "x_amz_signature",
    "x_amz_credential",
    "x_amz_security_token",
}


def _redact_query(match):
    key = unquote_plus(match[2]).casefold().replace("-", "_")
    return match[1] + match[2] + "=[redacted]" if key in QUERY_KEYS else match[0]


def clean_log(text):
    text = ANSI.sub("", text)
    text = re.sub(
        r"-----BEGIN [^-]*PRIVATE KEY-----.*?(?:-----END [^-]*PRIVATE KEY-----|\Z)",
        "[private key redacted]",
        text,
        flags=re.S,
    )
    text = HEADERS.sub(r"\1[redacted]", text)
    text = SECRET.sub(r"\1[redacted]", text)
    text = re.sub(r"(?<![A-Za-z0-9_])hf_[A-Za-z0-9_-]{8,509}(?![A-Za-z0-9_-])", "[redacted]", text)
    text = re.sub(r"\bya29\.[A-Za-z0-9._~-]+", "[redacted]", text)
    text = re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*", "Bearer [redacted]", text)
    text = re.sub(r"(https?://)[^/\s:@]*(?::[^/\s@]*)?@", r"\1[redacted]@", text)
    text = re.sub(r"([?&#])([^=&?#\s\"'<>]+)=([^&#\s\"'<>]*)", _redact_query, text)
    text = "".join(c for c in text if c in "\n\t" or ord(c) >= 32)
    return text.encode("utf-8")[-LOG_BYTES:].decode("utf-8", errors="ignore")


class LogTail(io.TextIOBase):
    """Keep bounded output while receiving SDK log chunks, including huge lines."""

    def __init__(self):
        self._value = ""
        self._pending = ""
        self._private_key = False
        self._discard_line = False

    def _line(self, text):
        if "-----BEGIN " in text and "PRIVATE KEY-----" in text:
            self._private_key = True
            self._value = clean_log(self._value + "[private key redacted]\n")
        if self._private_key:
            if "-----END " in text and "PRIVATE KEY-----" in text:
                self._private_key = False
            return
        self._value = clean_log(self._value + text)

    @property
    def value(self):
        trailing = "" if self._private_key or self._discard_line else self._pending
        return clean_log(self._value + trailing)

    def write(self, text):
        # Buffer complete lines so secrets split across SDK chunks stay redacted.
        for chunk in text.splitlines(keepends=True):
            newline = chunk.endswith(("\n", "\r"))
            if not self._discard_line:
                self._pending += chunk
                if len(self._pending.encode("utf-8")) > LOG_BYTES:
                    self._pending = ""
                    self._discard_line = True
                    self._value = clean_log(self._value + "[oversized log line omitted]\n")
            if newline:
                if not self._discard_line:
                    self._line(self._pending)
                self._pending = ""
                self._discard_line = False
        return len(text)

    def flush(self):
        pass


def select_tasks(rows, job_id, run_id):
    selected = {}
    for row in rows:
        if row.get("job_id") != job_id or row.get("job_name") != run_id:
            raise ValueError("Queue identity did not match the requested job group")
        name = row.get("task_name")
        if name not in TASKS or name in selected:
            raise ValueError("Expected exactly one isaac and one vla task")
        state = row.get("status")
        state = getattr(state, "value", state)
        if not isinstance(state, str) or not re.fullmatch(r"[A-Z_]{1,40}", state):
            raise ValueError("Invalid task state")
        selected[name] = state
    if set(selected) != set(TASKS):
        raise ValueError("Both job-group tasks are not available yet")
    return selected


def read_logs(job_id, task, stream, common, sdk, payloads):
    # Pinned 0.13 API: tail_logs(follow=False) drops the final request result.
    # Verify it explicitly so a 200 stream with a server failure cannot look fresh.
    body = payloads.JobsLogsBody(job_id=job_id, task=task, follow=False, refresh=False, tail=200)
    response = common.make_authenticated_request(
        "POST",
        "/jobs/logs",
        json=json.loads(body.model_dump_json()),
        stream=True,
        timeout=(5, None),
    )
    try:
        sdk.stream_response(
            request_id=common.get_request_id(response),
            response=response,
            output_stream=stream,
            resumable=False,
            get_result=True,
        )
    finally:
        response.close()


def probe(args):
    import sky  # Isolated worker dependency; never imported by the app.

    if version("skypilot") != "0.13.0":
        raise ValueError("This observer requires SkyPilot 0.13.0")
    if args.probe == "queue":
        records, _, _, _ = sky.get(
            sky.jobs.queue_v2(
                refresh=False,
                all_users=False,
                job_ids=[args.job_id],
                limit=1,
                fields=["job_id", "job_name", "task_name", "status"],
            )
        )
        rows = [r.model_dump() if hasattr(r, "model_dump") else r for r in records]
        result = select_tasks(rows, args.job_id, args.run_id)
    else:
        from sky.client import sdk
        from sky.server import common
        from sky.server.requests import payloads

        stream = LogTail()
        read_logs(args.job_id, args.probe, stream, common, sdk, payloads)
        result = stream.value
    Path(args.result).write_text(json.dumps(result), encoding="utf-8")


def fetch(args, kind):
    # A subprocess deadline also bounds SDK calls whose own reads have no timeout.
    with tempfile.TemporaryDirectory(prefix="firebird-cloud-probe-") as directory:
        result = Path(directory) / "result.json"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--probe",
                kind,
                "--job-id",
                str(args.job_id),
                "--run-id",
                args.run_id,
                "--result",
                str(result),
            ],
            check=True,
            timeout=40,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.stat().st_size > 384 * 1024:
            raise ValueError("Observer response exceeded the size limit")
        return json.loads(result.read_text(encoding="utf-8"))


def initial(args):
    return {
        "schema_version": 1,
        "run_id": args.run_id,
        "label": "ACT motion experiment",
        "cluster": args.run_id,
        "job_id": str(args.job_id),
        "collected_at": None,
        "status": "UNKNOWN",
        "collection_error": "Waiting for the first remote collection.",
        "logs": {name: "" for name in TASKS},
        "outcomes": {
            "rollout_completed": None,
            "pickup_success": None,
            "calibration": "unverified" if args.experimental else "unknown",
        },
    }


def resume(args):
    """Recover matching last evidence before an outage or process restart."""
    target = args.output_dir / (args.run_id + ".json")
    if not target.exists() and not target.is_symlink():
        return initial(args)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    if target.is_symlink():
        raise ValueError("Existing snapshot must not be a symlink")
    with os.fdopen(os.open(target, flags), "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("Existing snapshot must be a regular file")
        raw = stream.read(768 * 1024 + 1)
    if len(raw) > 768 * 1024:
        raise ValueError("Existing snapshot is oversized")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Existing snapshot has duplicate fields")
            result[key] = value
        return result

    snapshot = json.loads(raw, object_pairs_hook=unique)
    expected = initial(args)
    valid = (
        isinstance(snapshot, dict)
        and set(snapshot) == set(expected)
        and type(snapshot["schema_version"]) is int
        and snapshot["schema_version"] == 1
        and snapshot["run_id"] == args.run_id
        and snapshot["job_id"] == str(args.job_id)
        and isinstance(snapshot["logs"], dict)
        and set(snapshot["logs"]) == set(TASKS)
        and isinstance(snapshot["outcomes"], dict)
        and set(snapshot["outcomes"]) == set(expected["outcomes"])
    )
    if not valid:
        raise ValueError("Existing snapshot identity or schema is invalid")
    for field, limit in (("label", 120), ("cluster", 100), ("status", 40)):
        if not isinstance(snapshot[field], str) or not 1 <= len(snapshot[field]) <= limit:
            raise ValueError("Existing snapshot text is invalid")
    for value in snapshot["logs"].values():
        if not isinstance(value, str) or len(value.encode("utf-8")) > LOG_BYTES:
            raise ValueError("Existing snapshot log is invalid")
    outcomes = snapshot["outcomes"]
    if outcomes["calibration"] not in ("unknown", "unverified", "verified") or any(
        value is not None and type(value) is not bool
        for value in (outcomes["rollout_completed"], outcomes["pickup_success"])
    ):
        raise ValueError("Existing snapshot outcomes are invalid")
    error = snapshot["collection_error"]
    if error is not None and (not isinstance(error, str) or len(error) > 2000):
        raise ValueError("Existing snapshot error is invalid")
    collected = snapshot["collected_at"]
    if collected is not None:
        if not isinstance(collected, str):
            raise ValueError("Existing snapshot timestamp is invalid")
        timestamp = datetime.fromisoformat(collected)
        if timestamp.tzinfo is None or (timestamp - datetime.now(UTC)).total_seconds() > 5:
            raise ValueError("Existing snapshot timestamp is invalid")
    snapshot["collection_error"] = "Monitor restarted; waiting for fresh remote collection."
    return snapshot


def collect(args, previous, fetcher=fetch):
    snapshot = copy.deepcopy(previous)
    try:
        states = fetcher(args, "queue")
    except (OSError, ValueError, subprocess.SubprocessError):
        snapshot["collection_error"] = (
            "SkyPilot status collection failed or timed out; last evidence retained."
        )
        return snapshot
    failures = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {task: pool.submit(fetcher, args, task) for task in TASKS}
        for task, future in pending.items():
            try:
                snapshot["logs"][task] = clean_log(
                    f"SkyPilot {task} state: {states[task]}\n" + future.result()
                )
            except (OSError, ValueError, subprocess.SubprocessError):
                failures.append(task)
    if failures:
        # Do not relabel partially refreshed logs as a fully fresh observation.
        snapshot["collection_error"] = (
            "SkyPilot log collection failed or timed out for "
            + ", ".join(failures)
            + "; collection time and primary state retain the last complete evidence."
        )
    else:
        snapshot["status"] = states["isaac"]
        snapshot["collected_at"] = datetime.now(UTC).isoformat()
        snapshot["collection_error"] = None
    return snapshot


def publish(directory, snapshot):
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if directory.is_symlink():
        raise ValueError("Feed directory must not be a symlink")
    target = directory / (snapshot["run_id"] + ".json")
    if target.is_symlink():
        raise ValueError("Feed snapshot must not be a symlink")
    fd, name = tempfile.mkstemp(prefix=".snapshot-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(snapshot, stream, ensure_ascii=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--experimental", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--probe", choices=("queue", *TASKS), help=argparse.SUPPRESS)
    parser.add_argument("--result", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.job_id < 1 or not RUN_ID.fullmatch(args.run_id):
        parser.error("A positive job ID and a valid immutable job-group name are required")
    if args.probe:
        probe(args)
        return
    if not args.output_dir:
        parser.error("--output-dir is required")
    snapshot = resume(args)
    publish(args.output_dir, snapshot)
    try:
        while True:
            snapshot = collect(args, snapshot)
            publish(args.output_dir, snapshot)
            print(
                f"Cloud observation: {snapshot['status']}; "
                f"{'collection incomplete' if snapshot['collection_error'] else 'updated'}",
                flush=True,
            )
            if args.once:
                break
            time.sleep(10)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
