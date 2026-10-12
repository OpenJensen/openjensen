"""Loopback-only authenticated control mailbox and client, separate from simulator calls."""

from __future__ import annotations

import hashlib
import hmac
import queue
import threading
import time
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .contracts import MAX_JSON, Command, canonical, decode


class Mailbox:
    def __init__(self, *, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.Lock()
        self.pending = queue.PriorityQueue(maxsize=32)
        self.sequence = 0
        self.receipts = OrderedDict()
        self.state = {"mode": "starting", "episode_id": None, "revision": 0}
        self.frame = None
        self.frame_published_monotonic_ns = None

    def submit(self, payload: dict) -> dict:
        command = Command.parse(payload)
        digest = hashlib.sha256(canonical(payload)).hexdigest()
        with self.lock:
            if command.command_id in self.receipts:
                receipt = self.receipts[command.command_id]
                if receipt["request_sha256"] != digest:
                    raise ValueError("Command ID was reused with different contents")
                return dict(receipt)
            if self.pending.full():
                raise ValueError("Teaching command queue is full")
            receipt = {
                "command_id": command.command_id,
                "status": "queued",
                "request_sha256": digest,
                "received_monotonic_ns": time.monotonic_ns(),
            }
            self.receipts[command.command_id] = receipt
            priority = 0 if command.operation in {"pause", "reset", "finish"} else 1
            self.sequence += 1
            self.pending.put_nowait((priority, self.sequence, command, self.clock() + 2.0))
            # Terminal receipts are bounded; expired CAS revisions still prevent replay.
            for ident in list(self.receipts):
                if len(self.receipts) <= 256:
                    break
                if self.receipts[ident]["status"] not in {"queued", "executing"}:
                    del self.receipts[ident]
            return dict(receipt)

    def get(self, ident: str) -> dict:
        with self.lock:
            if ident not in self.receipts:
                raise ValueError("Unknown or expired command receipt")
            return dict(self.receipts[ident])

    def publish(self, session) -> None:
        session._owned()
        with self.lock:
            self.state = session.snapshot()
            if session.mode in {"faulted", "closed"}:
                for receipt in self.receipts.values():
                    if receipt["status"] in {"queued", "executing"}:
                        receipt.update(
                            {
                                "status": "rejected",
                                "error": "Execution did not complete; motion outcome is unverified",
                            }
                        )
            applied = session.last_applied_command_id
            if applied in self.receipts and self.receipts[applied]["status"] == "executing":
                self.receipts[applied].update(
                    {
                        "status": "acknowledged",
                        "state": dict(self.state),
                        "acknowledged_monotonic_ns": time.monotonic_ns(),
                        "applied_step": session.last_applied_step,
                    }
                )
            self.frame = session.frame_snapshot
            self.frame_published_monotonic_ns = time.monotonic_ns()

    def frame_payload(self) -> dict:
        """One locked response binds frozen acquisition context and current executor state."""
        with self.lock:
            if self.frame is None:
                return {"available": False, "schema_version": 1}
            current = {
                "session_id": self.state["session_id"],
                "revision": self.state["revision"],
                "active_episode_id": self.state["episode_id"],
                "mode": self.state["mode"],
            }
            return self.frame.payload(
                current, self.frame_published_monotonic_ns, time.monotonic_ns()
            )

    def drain(self, session) -> None:
        # One command per simulator turn; a command never causes multiple physics ticks.
        try:
            _, _, command, deadline = self.pending.get_nowait()
        except queue.Empty:
            return
        ident = command.command_id
        with self.lock:
            self.receipts[ident]["status"] = "executing"
        try:
            if self.clock() >= deadline:
                raise ValueError("Command expired before execution; no operation applied")
            result = session.execute(command)
        except Exception as error:
            update = {"status": "rejected", "error": str(error)}
        else:
            update = {
                "status": "executing" if command.operation == "correct" else "acknowledged",
                "state": result,
            }
            if command.operation != "correct":
                update["acknowledged_monotonic_ns"] = time.monotonic_ns()
        # Publish invalidation before acknowledging a changed command context.
        # Waiting for the next tick could otherwise expose old pixels as current.
        self.publish(session)
        with self.lock:
            self.receipts[ident].update(update)


def server(mailbox: Mailbox, token: str, port: int = 0) -> ThreadingHTTPServer:
    if not isinstance(token, str) or len(token) < 32:
        raise ValueError("Control token must contain at least 32 characters")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _reply(self, status, value):
            raw = canonical(value)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def _authorized(self):
            if self.headers.get("Origin") is not None or not hmac.compare_digest(
                self.headers.get("Authorization", ""), "Bearer " + token
            ):
                self._reply(401, {"error": "Unauthorized control request"})
                return False
            return True

        def do_GET(self):
            if not self._authorized():
                return
            try:
                if self.path == "/state":
                    with mailbox.lock:
                        value = dict(mailbox.state)
                elif self.path == "/frame":
                    value = mailbox.frame_payload()
                elif self.path.startswith("/commands/"):
                    value = mailbox.get(self.path.removeprefix("/commands/"))
                else:
                    raise ValueError("Unknown teaching endpoint")
                self._reply(200, value)
            except ValueError as error:
                self._reply(400, {"error": str(error)})

        def do_POST(self):
            if not self._authorized():
                return
            try:
                if self.path != "/commands" or self.headers.get("Transfer-Encoding"):
                    raise ValueError("Unknown endpoint or unsupported transfer encoding")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_JSON:
                    raise ValueError("Invalid control request size")
                self.connection.settimeout(2)
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("Incomplete control request")
                self._reply(202, mailbox.submit(decode(raw)))
            except (ValueError, OSError) as error:
                self._reply(400, {"error": str(error)})

    result = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    result.daemon_threads = True
    return result


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Control redirects are forbidden")


class Client:
    def __init__(self, origin: str, token: str):
        url = urlsplit(origin)
        if (
            url.scheme != "http"
            or url.hostname != "127.0.0.1"
            or url.path not in {"", "/"}
            or url.query
            or url.fragment
            or url.username
            or url.password
            or not url.port
        ):
            raise ValueError("Teaching control must use an explicit http://127.0.0.1:PORT origin")
        if len(token) < 32:
            raise ValueError("Missing teaching control token")
        self.opener = build_opener(ProxyHandler({}), NoRedirect())
        self.origin, self.token = origin.rstrip("/"), token

    def request(self, path: str, body: dict | None = None) -> dict:
        request = Request(
            self.origin + path,
            data=None if body is None else canonical(body),
            headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"},
        )
        with self.opener.open(request, timeout=3) as response:
            raw = response.read(16 * 1024**2 + 1)
        if len(raw) > 16 * 1024**2:
            raise ValueError("Teaching response exceeds limit")
        return decode(raw, limit=16 * 1024**2)

    def command(self, operation: str, arguments: dict, *, wait_seconds: float = 3.0) -> dict:
        state = self.request("/state")
        ident = uuid.uuid4().hex
        receipt = self.request(
            "/commands",
            {
                "command_id": ident,
                "session_id": state["session_id"],
                "episode_id": state["episode_id"],
                "expected_revision": state["revision"],
                "operation": operation,
                "arguments": arguments,
            },
        )
        deadline = time.monotonic() + wait_seconds
        while receipt["status"] in {"queued", "executing"} and time.monotonic() < deadline:
            time.sleep(0.02)
            receipt = self.request("/commands/" + ident)
        # Unacknowledged is observable; never report timeout as successful execution.
        return receipt
