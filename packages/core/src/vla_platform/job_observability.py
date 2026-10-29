"""Durable, bounded job diagnostics shared by operation and execution adapters.

The job ledger owns execution state. Diagnostic records explain that state and
are never interpreted as commands or authoritative completion signals.
"""

import io
import json
import logging
import os
import re
import sqlite3
import threading
import time
import zipfile
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

LOGGER = logging.getLogger(__name__)
ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
Severity = Literal["debug", "info", "warning", "error"]


class DiagnosticRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ObservabilityPolicy(DiagnosticRecord):
    max_log_bytes: int = Field(default=16 * 1024**2, ge=65536, le=1024**3, strict=True)
    max_event_records: int = Field(default=10000, ge=100, le=1000000, strict=True)
    max_record_bytes: int = Field(default=16384, ge=1024, le=65536, strict=True)
    retention_days: int = Field(default=30, ge=1, le=3650, strict=True)


class JobLogRecord(DiagnosticRecord):
    sequence: int
    job_id: str
    attempt_id: str
    timestamp: str
    kind: Literal["event", "log"]
    component: str
    stage: str | None = None
    severity: Severity
    stream: Literal["stdout", "stderr"] | None = None
    message: str
    data: dict[str, Any] = Field(default_factory=dict)


class JobLogPage(DiagnosticRecord):
    records: list[JobLogRecord]
    next_cursor: int
    has_more: bool
    retained_from: int | None
    truncated: bool
    available: bool
    discarded_records: int = 0


class JobFailure(DiagnosticRecord):
    code: str
    stage: str
    message: str
    cause: str | None = None
    exit_code: int | None = None
    retryable: bool = False
    cleanup: Literal["unknown", "pending", "succeeded", "failed"] = "unknown"
    attempt_id: str = "1"
    timestamp: str


class ObservabilityError(RuntimeError):
    pass


def timestamp():
    return datetime.now(UTC).isoformat()


def job_key(value: str) -> str:
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise ValueError("Invalid diagnostic job identity")
    return value if value.startswith(("core:", "scene:", "agent:", "dataset:")) else "core:" + value


def redact(text: str) -> str:
    """Sanitize untrusted output before storage or display, including old exports."""
    value = str(text).encode("utf-8", errors="replace").decode("utf-8")
    value = re.sub(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))", "", value)
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", value)
    value = re.sub(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)",
        "[redacted private key]",
        value,
        flags=re.S,
    )
    value = re.sub(
        r"\b(?:hf_[A-Za-z0-9]+|sk-[A-Za-z0-9_-]+|ya29\.[A-Za-z0-9._-]+|AKIA[A-Z0-9]{16})\b",
        "[redacted]",
        value,
    )
    value = re.sub(r"(?i)\b((?:bearer|basic)\s+)\S+", r"\1[redacted]", value)
    value = re.sub(
        r"(?i)((?:[A-Za-z0-9_]*(?:token|secret|password|api_key|authorization|cookie))[\"']?\s*[:=]\s*)(?:[\"'][^\"']*[\"']|[^\s,;]+)",
        r"\1[redacted]",
        value,
    )
    # Endpoints and signed download URLs can expose private hosts and credentials.
    value = re.sub(r"(?:https?|ssh|s3|gs)://[^\s<>\"']+", "[endpoint]", value)
    value = re.sub(r"\b[A-Za-z0-9._-]+@[A-Za-z0-9._-]+(?=[:\s]|$)", "[connection]", value)
    value = re.sub(
        r"(?<![A-Za-z0-9])(?:/(?:home|Users|root|tmp|private|mnt|var|opt)/[^\s\"'<>:,;]*|[A-Za-z]:[\\/][^\s\"'<>]*)",
        "[path]",
        value,
    )
    return value


def public_value(value: Any, depth=0) -> Any:
    if depth > 12:
        return "[depth limit]"
    if isinstance(value, dict):
        result = {}
        for key, item in list(value.items())[:200]:
            name = str(key)
            if re.search(
                r"(?i)(token|secret|password|credential|api.?key|authorization|cookie|^env$|environment|hostname|^host$|endpoint|private.?key|^path$|_path$|_dir$|_root$)",
                name,
            ):
                result[redact(name)[:200]] = "[redacted]"
            else:
                result[redact(name)[:200]] = public_value(item, depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [public_value(item, depth + 1) for item in value[:200]]
    if isinstance(value, str):
        return redact(value)[:16384]
    if value is None or type(value) in (bool, int, float):
        # JSON serialization rejects nonfinite values before SQLite sees them.
        return value
    return redact(str(value))[:16384]


class JobObservability:
    def __init__(self, data_dir: Path, policy: ObservabilityPolicy | None = None):
        self.data_dir = Path(data_dir)
        self.path = self.data_dir / "job-observability.sqlite3"
        self._policy_error = None
        self._configured_policy = None
        try:
            self._configured_policy = policy or self._load_policy()
        except ObservabilityError as exc:
            # A diagnostic configuration error must not disable workspace access.
            # Strict operations remain unavailable until the operator repairs it.
            self._policy_error = str(exc)
        self._lock = threading.RLock()
        self._last_warning = False
        self._next_expiry_check = 0.0
        self._private_key_streams = set()

    @property
    def policy(self):
        if self._policy_error is not None:
            raise ObservabilityError(self._policy_error)
        return self._configured_policy

    def _load_policy(self):
        path = self.data_dir / "job-observability-settings.json"
        try:
            if path.is_symlink():
                raise ObservabilityError(
                    "Diagnostic retention settings need administrator attention"
                )
            if not path.exists():
                return ObservabilityPolicy()
            if not path.is_file() or path.stat().st_size > 16384:
                raise ObservabilityError(
                    "Diagnostic retention settings need administrator attention"
                )
            return ObservabilityPolicy.model_validate_json(path.read_bytes())
        except (OSError, ValueError) as exc:
            raise ObservabilityError("Diagnostic retention settings are invalid") from exc

    @contextmanager
    def _connect(self):
        if self._policy_error is not None:
            raise ObservabilityError(self._policy_error)
        if self.data_dir.is_symlink() or self.path.is_symlink():
            raise ObservabilityError("Diagnostic storage must not be a symbolic link")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=3)
        try:
            os.chmod(self.path, 0o600)
            connection.row_factory = sqlite3.Row
            connection.executescript("""
            CREATE TABLE IF NOT EXISTS diagnostic_records (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL, attempt_id TEXT NOT NULL, timestamp TEXT NOT NULL,
                kind TEXT NOT NULL, component TEXT NOT NULL, stage TEXT,
                severity TEXT NOT NULL, stream TEXT, message TEXT NOT NULL,
                data TEXT NOT NULL, bytes INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS diagnostic_job_cursor
                ON diagnostic_records(job_id, sequence);
            CREATE INDEX IF NOT EXISTS diagnostic_timestamp ON diagnostic_records(timestamp);
            CREATE TABLE IF NOT EXISTS diagnostic_failures (
                job_id TEXT PRIMARY KEY, record TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS diagnostic_retention (
                job_id TEXT PRIMARY KEY, discarded INTEGER NOT NULL DEFAULT 0
            );
        """)
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _bounded(self, message, data):
        encoded = redact(message).encode("utf-8")
        bound = self.policy.max_record_bytes
        clipped = len(encoded) > bound
        if clipped:
            message = (
                encoded[: bound - 100].decode("utf-8", errors="ignore") + "\n[record truncated]"
            )
        else:
            message = encoded.decode("utf-8")
        clean_data = public_value(data or {})
        try:
            serialized = json.dumps(clean_data, allow_nan=False, separators=(",", ":"))
        except TypeError, ValueError:
            serialized = '{"invalid_data_omitted":true}'
        if len(serialized.encode()) > bound:
            serialized = '{"record_data_truncated":true}'
        return message, serialized, len(message.encode()) + len(serialized.encode())

    @staticmethod
    def _record(row):
        value = dict(row)
        value.pop("bytes", None)
        value["data"] = json.loads(value["data"])
        return JobLogRecord.model_validate(value)

    def _prune(self, connection, key):
        # Retention bounds ordinary output independently from critical lifecycle evidence.
        deleted = 0
        for kind, limit, column in (
            ("log", self.policy.max_log_bytes, "bytes"),
            ("event", self.policy.max_event_records, None),
        ):
            aggregate = "SUM(bytes)" if column else "COUNT(*)"
            total = connection.execute(
                f"SELECT COALESCE({aggregate},0) FROM diagnostic_records "
                "WHERE job_id=? AND kind=? AND component != 'diagnostics'",
                (key, kind),
            ).fetchone()[0]
            if total <= limit:
                continue
            rows = connection.execute(
                "SELECT sequence,bytes FROM diagnostic_records "
                "WHERE job_id=? AND kind=? AND component != 'diagnostics' ORDER BY sequence",
                (key, kind),
            )
            remove = []
            for row in rows:
                if total <= limit:
                    break
                total -= row["bytes"] if column else 1
                remove.append((row["sequence"],))
            connection.executemany("DELETE FROM diagnostic_records WHERE sequence=?", remove)
            deleted += len(remove)
        if deleted:
            prior = connection.execute(
                "SELECT discarded FROM diagnostic_retention WHERE job_id=?", (key,)
            ).fetchone()
            connection.execute(
                "INSERT INTO diagnostic_retention(job_id,discarded) VALUES(?,?) "
                "ON CONFLICT(job_id) DO UPDATE SET discarded=discarded+excluded.discarded",
                (key, deleted),
            )
            if prior is None:
                connection.execute(
                    "INSERT INTO diagnostic_records(job_id,attempt_id,timestamp,kind,component,"
                    "stage,severity,stream,message,data,bytes) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        key,
                        "1",
                        timestamp(),
                        "event",
                        "diagnostics",
                        "retention",
                        "warning",
                        None,
                        "Older diagnostic records were removed by the configured retention limit. "
                        "The final failure summary is retained.",
                        "{}",
                        150,
                    ),
                )

    def _expire(self, connection):
        cutoff = (datetime.now(UTC) - timedelta(days=self.policy.retention_days)).isoformat()
        expired = connection.execute(
            "SELECT job_id,COUNT(*) FROM diagnostic_records "
            "WHERE timestamp < ? AND component != 'diagnostics' GROUP BY job_id",
            (cutoff,),
        ).fetchall()
        for key, count in expired:
            connection.execute(
                "DELETE FROM diagnostic_records "
                "WHERE job_id=? AND timestamp < ? AND component != 'diagnostics'",
                (key, cutoff),
            )
            connection.execute(
                "INSERT INTO diagnostic_retention(job_id,discarded) VALUES(?,?) "
                "ON CONFLICT(job_id) DO UPDATE SET discarded=discarded+excluded.discarded",
                (key, count),
            )
        return sum(row[1] for row in expired)

    def expire(self):
        """Apply diagnostic age retention; never touch artifacts or final failure summaries."""
        try:
            with self._lock, self._connect() as connection:
                count = self._expire(connection)
                self._next_expiry_check = time.monotonic() + 3600
                return count
        except (OSError, sqlite3.Error) as exc:
            raise ObservabilityError("Diagnostic retention could not be applied") from exc

    def _stream_message(self, key, attempt_id, stream, message):
        identity = (key, attempt_id, stream)
        with self._lock:
            began = re.search(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", message)
            ended = re.search(r"-----END [A-Z ]*PRIVATE KEY-----", message)
            if identity in self._private_key_streams:
                if ended:
                    self._private_key_streams.discard(identity)
                return "[redacted private key]"
            if began and not ended:
                self._private_key_streams.add(identity)
        return message

    def record(
        self,
        job_id: str,
        message: str,
        *,
        attempt_id="1",
        kind="event",
        component="supervisor",
        stage=None,
        severity="info",
        stream=None,
        data=None,
    ):
        key = job_key(job_id)
        message = self._stream_message(key, str(attempt_id), stream, str(message))
        message, serialized, size = self._bounded(message, data)
        value = JobLogRecord(
            sequence=0,
            job_id=key,
            attempt_id=redact(attempt_id)[:100],
            timestamp=timestamp(),
            kind=kind,
            component=redact(component)[:100],
            stage=redact(stage)[:200] if stage else None,
            severity=severity,
            stream=stream,
            message=message,
            data=json.loads(serialized),
        )
        try:
            with self._lock, self._connect() as connection:
                if time.monotonic() >= self._next_expiry_check:
                    self._expire(connection)
                    self._next_expiry_check = time.monotonic() + 3600
                cursor = connection.execute(
                    "INSERT INTO diagnostic_records(job_id,attempt_id,timestamp,kind,component,"
                    "stage,severity,stream,message,data,bytes) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        key,
                        value.attempt_id,
                        value.timestamp,
                        kind,
                        value.component,
                        value.stage,
                        severity,
                        stream,
                        message,
                        serialized,
                        size,
                    ),
                )
                value.sequence = cursor.lastrowid
                self._prune(connection, key)
            return value
        except (OSError, sqlite3.Error) as exc:
            raise ObservabilityError("Job diagnostics could not be persisted") from exc

    def try_record(self, *args, **kwargs):
        """Output capture must never strand an accepted job when diagnostic storage fails."""
        try:
            return self.record(*args, **kwargs)
        except ObservabilityError, ValueError:
            if not self._last_warning:
                LOGGER.warning(
                    "Job diagnostic storage is unavailable; execution state remains authoritative"
                )
                self._last_warning = True
            return None

    def failure(
        self,
        job_id: str,
        *,
        stage: str,
        message: str,
        code="WORKER_FAILED",
        cause=None,
        exit_code=None,
        retryable=False,
        cleanup="unknown",
        attempt_id="1",
    ):
        key = job_key(job_id)
        summary = JobFailure(
            code=redact(code)[:100],
            stage=redact(stage)[:200],
            message=redact(message)[:4000],
            cause=redact(cause)[:8000] if cause else None,
            exit_code=exit_code,
            retryable=retryable,
            cleanup=cleanup,
            attempt_id=redact(attempt_id)[:100],
            timestamp=timestamp(),
        )
        try:
            with self._lock, self._connect() as connection:
                previous = connection.execute(
                    "SELECT record FROM diagnostic_failures WHERE job_id=?", (key,)
                ).fetchone()
                if previous:
                    try:
                        prior = JobFailure.model_validate_json(previous[0])
                    except ValueError:
                        prior = None
                    if prior is not None and prior.attempt_id == summary.attempt_id:
                        updates = {}
                        if summary.exit_code is None:
                            updates["exit_code"] = prior.exit_code
                        if summary.cause is None:
                            updates["cause"] = prior.cause
                        if summary.cleanup == "unknown":
                            updates["cleanup"] = prior.cleanup
                        if summary.code == "WORKER_FAILED" and prior.code != "WORKER_FAILED":
                            updates.update(
                                code=prior.code,
                                stage=prior.stage,
                                message=prior.message,
                                retryable=prior.retryable,
                            )
                        summary = summary.model_copy(update=updates)
                connection.execute(
                    "INSERT INTO diagnostic_failures(job_id,record) VALUES(?,?) "
                    "ON CONFLICT(job_id) DO UPDATE SET record=excluded.record",
                    (key, summary.model_dump_json()),
                )
            self.record(
                key,
                summary.message,
                stage=stage,
                severity="error",
                attempt_id=attempt_id,
                data={
                    "code": summary.code,
                    "exit_code": exit_code,
                    "retryable": retryable,
                    "cleanup": cleanup,
                },
            )
            return summary
        except (OSError, sqlite3.Error) as exc:
            raise ObservabilityError("Job failure diagnostics could not be persisted") from exc

    def try_failure(self, *args, **kwargs):
        try:
            return self.failure(*args, **kwargs)
        except ObservabilityError, ValueError:
            LOGGER.warning("Job failure diagnostics are unavailable; inspect the saved job error")
            return None

    def get_failure(self, job_id: str):
        key = job_key(job_id)
        try:
            with self._lock, self._connect() as connection:
                row = connection.execute(
                    "SELECT record FROM diagnostic_failures WHERE job_id=?", (key,)
                ).fetchone()
            return JobFailure.model_validate_json(row[0]) if row else None
        except (OSError, sqlite3.Error, ValueError) as exc:
            raise ObservabilityError("Saved failure diagnostics are unavailable") from exc

    def page(self, job_id: str, after=0, limit=200, severity=None, stream=None, query=None):
        key = job_key(job_id)
        if type(after) is not int or after < 0 or type(limit) is not int or not 1 <= limit <= 500:
            raise ValueError("Invalid diagnostic cursor or page size")
        if severity not in {None, "debug", "info", "warning", "error"} or stream not in {
            None,
            "stdout",
            "stderr",
        }:
            raise ValueError("Invalid diagnostic filter")
        terms, parameters = ["job_id=?", "sequence> ?"], [key, after]
        if severity:
            terms.append("severity=?")
            parameters.append(severity)
        if stream:
            terms.append("stream=?")
            parameters.append(stream)
        if query:
            if len(query) > 200:
                raise ValueError("Diagnostic search supports up to 200 characters")
            terms.append("instr(lower(message),lower(?)) > 0")
            parameters.append(query)
        try:
            with self._lock, self._connect() as connection:
                bounds = connection.execute(
                    "SELECT MIN(sequence),MAX(sequence) FROM diagnostic_records WHERE job_id=?",
                    (key,),
                ).fetchone()
                rows = connection.execute(
                    "SELECT * FROM diagnostic_records WHERE "
                    + " AND ".join(terms)
                    + " ORDER BY sequence LIMIT ?",
                    (*parameters, limit + 1),
                ).fetchall()
                discarded = connection.execute(
                    "SELECT discarded FROM diagnostic_retention WHERE job_id=?", (key,)
                ).fetchone()
                has_failure = connection.execute(
                    "SELECT 1 FROM diagnostic_failures WHERE job_id=?", (key,)
                ).fetchone()
            more = len(rows) > limit
            visible = rows[:limit]
            return JobLogPage(
                records=[self._record(row) for row in visible],
                next_cursor=visible[-1]["sequence"] if more else max(after, bounds[1] or 0),
                has_more=more,
                retained_from=bounds[0],
                truncated=bool(discarded),
                available=bool(bounds[0] is not None or has_failure or discarded),
                discarded_records=discarded[0] if discarded else 0,
            )
        except (OSError, sqlite3.Error, ValueError) as exc:
            raise ObservabilityError("Job diagnostic history is unavailable") from exc

    def export(self, job_id: str, specification: dict):
        key = job_key(job_id)
        buffer = io.BytesIO()
        failure = self.get_failure(key)
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            specification_json = json.dumps(public_value(specification), indent=2, allow_nan=False)
            if len(specification_json.encode()) > 4 * 1024**2:
                specification_json = json.dumps({"job_id": key, "specification_truncated": True})
            archive.writestr("job.json", specification_json)
            archive.writestr(
                "failure.json", failure.model_dump_json(indent=2) if failure else "null\n"
            )
            with archive.open("diagnostics.jsonl", "w") as output:
                cursor, written = 0, 0
                while True:
                    page = self.page(key, after=cursor, limit=200)
                    for record in page.records:
                        encoded = (record.model_dump_json() + "\n").encode()
                        written += len(encoded)
                        if written > self.policy.max_log_bytes + 4 * 1024**2:
                            output.write(
                                b'{"message":"Diagnostic export truncated '
                                b'at the export size limit."}\n'
                            )
                            break
                        output.write(encoded)
                    if written > self.policy.max_log_bytes + 4 * 1024**2 or not page.has_more:
                        break
                    cursor = page.next_cursor
            archive.writestr(
                "retention.json",
                json.dumps(
                    {
                        "policy": self.policy.model_dump(),
                        "truncated": page.truncated,
                        "discarded_records": page.discarded_records,
                        "export_limit_reached": written > self.policy.max_log_bytes + 4 * 1024**2,
                    }
                ),
            )
        return buffer.getvalue()
