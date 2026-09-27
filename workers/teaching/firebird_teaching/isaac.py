"""Run interactive teaching in Isaac's existing environment; no LeRobot/voice imports."""

from __future__ import annotations

import argparse
import json
import signal
import threading
import time
from pathlib import Path

from .contracts import Settings
from .control import Mailbox, server
from .credentials import control_token
from .journal import Journal
from .session import Session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--max-seconds", type=int, default=300)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    settings = Settings.load(args.settings)
    if not 1 <= args.max_seconds <= 3600 or not 1024 <= args.port <= 65535:
        parser.error("Use a 1..3600 second session and unprivileged control port")
    if args.validate_only:
        print(json.dumps({"status": "settings_valid", "scope": settings.metadata()["scope"]}))
        return
    token = control_token()
    mailbox = Mailbox()
    http = server(mailbox, token, args.port)
    stop = threading.Event()
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in handlers:
        signal.signal(sig, lambda *_: stop.set())
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    session = None
    journal = None
    deadline = time.monotonic() + args.max_seconds
    try:
        # Import SDK adapter only after config/auth preflight.
        from sim_worker.rollout.isaac import IsaacSim

        journal = Journal(args.output, settings)
        with IsaacSim(settings.sim) as simulation:
            session = Session(settings, simulation, journal)
            session.prepare()
            mailbox.publish(session)
            while not stop.is_set() and time.monotonic() < deadline and session.mode != "faulted":
                started = time.monotonic()
                mailbox.drain(session)
                try:
                    session.tick()
                finally:
                    mailbox.publish(session)
                stop.wait(max(0, 1 / settings.sim.fps - (time.monotonic() - started)))
            session.close()
            mailbox.publish(session)
    finally:
        if journal is not None:
            journal.abort()
        http.shutdown()
        http.server_close()
        thread.join(timeout=2)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
