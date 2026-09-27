"""Fixed local supervisor and raw-capture verifier. Never allocates cloud resources.

The supervisor is a new POSIX process-group leader. Its terminal receipt describes
its Isaac child's exit, not the supervisor's own exit: final SIGKILL ends the whole
owned group (including this leader). The application must reap and prove that group
absent before considering any receipt. Parent EOF never authorizes publication.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import select
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

from .contracts import Settings, canonical
from .journal import atomic_new

LOG_LIMIT = 16 * 1024**2
ENTRY_LIMIT = 50000
STOP_SECONDS = 10


def capture_size(root: Path, maximum: int) -> None:
    """Bound live disk growth without following links or reading frame contents."""
    total = count = 0
    if not root.exists():
        return
    for directory, folders, files in os.walk(root, followlinks=False):
        for name in folders + files:
            count += 1
            if count > ENTRY_LIMIT:
                raise ValueError("Capture entry limit exceeded")
            info = (Path(directory) / name).lstat()
            if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                raise ValueError("Unexpected node in owned capture")
            total += info.st_size if stat.S_ISREG(info.st_mode) else 0
            if total > maximum:
                raise ValueError("Capture byte limit exceeded")


def verify_capture(root: Path, result: Path, maximum: int) -> None:
    from .dataset import _inventory, inspect_capture

    capture_size(root, maximum)
    episodes = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not episodes or len(episodes) > 100:
        raise ValueError("Capture must contain 1..100 finalized episodes")
    # Every episode is checked, not just the subset that happens to have a receipt.
    meta, selected, inventory = inspect_capture(root, episodes)
    if _inventory(root) != inventory:
        raise ValueError("Capture changed during final verification")
    atomic_new(
        result,
        {
            "schema_version": 1,
            "session_id": meta["session_id"],
            "session_sha256": inventory["session.json"],
            "inventory": inventory,
            "inventory_sha256": hashlib.sha256(canonical(inventory)).hexdigest(),
            "episodes": [
                {
                    "episode_id": item["id"],
                    "receipt_sha256": inventory[item["id"] + "/episode.json"],
                    "frames": item["receipt"]["frames"],
                    "termination": item["receipt"]["termination"],
                    "outcome": item["receipt"]["outcome"],
                }
                for item in selected
            ],
            "origin": meta["origin"],
            "lineage_group": meta["lineage_group"],
        },
    )


def supervise(
    argv: list[str],
    output: Path,
    terminal: Path,
    log: Path,
    seconds: int,
    maximum: int,
    lease_fd: int,
) -> None:
    """Run one fixed child. Caller creates the new group and supplies its lease FD."""
    if os.name != "posix" or os.getpid() != os.getpgrp():
        raise ValueError("Managed teaching requires its own POSIX process group")
    os.fstat(lease_fd)
    stopping = False

    def stop_signal(_signum, _frame):
        nonlocal stopping
        stopping = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop_signal)
    reason = "worker_exit"
    code = None
    process = None
    started = time.monotonic()
    last_inventory = 0.0
    try:
        ready, _, _ = select.select([sys.stdin.buffer], [], [], 0)
        if stopping or ready:
            # Parent loss or stop before admission never launches an SDK process.
            return
        with log.open("xb", buffering=0) as stream:
            # Same group, inherited kernel lease: API death cannot release the GPU
            # lease while this supervisor/worker is still alive.
            process = subprocess.Popen(
                argv, stdin=subprocess.DEVNULL, stdout=stream, stderr=stream, pass_fds=(lease_fd,)
            )
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if stopping:
                    reason = "cancelled"
                    break
                if elapsed >= seconds:
                    reason = "deadline"
                    break
                ready, _, _ = select.select([sys.stdin.buffer], [], [], 0.1)
                if ready:
                    instruction = os.read(sys.stdin.fileno(), 32)
                    if instruction == b"stop\n":
                        reason = "requested_stop"
                    else:
                        reason = "parent_lost" if not instruction else "invalid_control"
                    break
                if stream.tell() > LOG_LIMIT:
                    reason = "log_limit"
                    break
                if elapsed - last_inventory >= 1:
                    capture_size(output, maximum)
                    last_inventory = elapsed
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=STOP_SECONDS)
                except subprocess.TimeoutExpired:
                    reason = "cleanup_timeout"
                    process.kill()
                    process.wait(timeout=2)
            code = process.returncode
        # A group cancellation may also finish the child before the loop body
        # observes the signal. Preserve that cancellation over worker_exit or
        # requested_stop, while retaining a more specific cleanup failure.
        if stopping and reason != "cleanup_timeout":
            reason = "cancelled"
        atomic_new(
            terminal,
            {
                "schema_version": 1,
                "reason": reason,
                "worker_exit_code": code,
                "supervisor_pid": os.getpid(),
                "group_id": os.getpgrp(),
                "group_kill_required": True,
            },
        )
    finally:
        # This fixed supervisor never returns normally. This also eliminates late
        # SDK descendants when the parent is gone; a receipt alone proves no cleanup.
        os.killpg(os.getpgrp(), signal.SIGKILL)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--capture", type=Path, required=True)
    verify.add_argument("--result", type=Path, required=True)
    verify.add_argument("--max-bytes", type=int, required=True)
    run = sub.add_parser("run")
    for name in ("isaac-python", "settings", "output", "terminal", "log", "preflight"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--port", type=int, required=True)
    run.add_argument("--seconds", type=int, required=True)
    run.add_argument("--max-bytes", type=int, required=True)
    run.add_argument("--lease-fd", type=int, required=True)
    args = parser.parse_args()
    if not 1024 <= args.max_bytes <= 8 * 1024**3:
        parser.error("Invalid capture byte limit")
    if args.operation == "verify":
        verify_capture(args.capture, args.result, args.max_bytes)
        return
    if not 1 <= args.seconds <= 3600 or not 1024 <= args.port <= 65535:
        parser.error("Invalid teaching time or port bound")
    settings = Settings.load(args.settings)
    atomic_new(
        args.preflight,
        {
            "schema_version": 1,
            "metadata": settings.metadata(),
            "settings_sha256": hashlib.sha256(args.settings.read_bytes()).hexdigest(),
        },
    )
    command = [
        str(args.isaac_python),
        "--no-ros-env",
        "-m",
        "firebird_teaching.isaac",
        "--settings",
        str(args.settings),
        "--output",
        str(args.output),
        "--port",
        str(args.port),
        "--max-seconds",
        str(args.seconds),
    ]
    supervise(
        command, args.output, args.terminal, args.log, args.seconds, args.max_bytes, args.lease_fd
    )


if __name__ == "__main__":
    main()
