"""Time a fresh Python process through its first valid complete CPU action chunk.

Checkpoint assets are cached. Parent timing begins immediately before Popen,
therefore interpreter/backend imports, runtime setup and first inference count.
The child's monotonic clock marks completion before teardown or artifact writes.
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

CLEANUP_SECONDS = 5


def group_exists(process):
    """Inspect only the newly owned session; an exited leader does not end its group."""
    process.poll()  # Reap the direct child while checking its remaining descendants.
    try:
        os.killpg(process.pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError as original:
        # Darwin can transiently report EPERM while an exiting group disappears.
        # Only a later kernel ESRCH resolves it; permission denial is never success.
        deadline = time.monotonic() + 0.25
        while time.monotonic() < deadline:
            time.sleep(0.01)
            process.poll()
            try:
                os.killpg(process.pid, 0)
                return True
            except ProcessLookupError:
                return False
            except PermissionError:
                continue
        raise original


def cleanup_group(process):
    """Bound TERM, KILL and reap for this Popen-created POSIX session, not just its leader."""

    def send(value):
        try:
            os.killpg(process.pid, value)
        except ProcessLookupError:
            pass
        except PermissionError:
            if group_exists(process):
                raise

    if group_exists(process):
        send(signal.SIGTERM)
        deadline = time.monotonic() + CLEANUP_SECONDS
        while group_exists(process) and time.monotonic() < deadline:
            time.sleep(0.02)
        if group_exists(process):
            send(signal.SIGKILL)
    process.wait(timeout=CLEANUP_SECONDS)
    deadline = time.monotonic() + CLEANUP_SECONDS
    while group_exists(process) and time.monotonic() < deadline:
        time.sleep(0.02)
    if group_exists(process):
        raise RuntimeError("Startup worker group cleanup remains unknown")


def worker(args):
    # Keep GPU/framework imports inside the timed child process.
    import numpy as np
    import torch
    from smolvla_cross_stack import Runtime, load_fixture
    from smolvla_gpu_probe import MODEL, REVISION

    torch.set_num_threads(4)
    if args.backend.startswith(("vllm-", "trtllm-")):
        from llm_policy_runtime import LLMRuntime

        runtime_class = LLMRuntime
    else:
        runtime_class = Runtime
    runtime = runtime_class(args.backend, args.root, args.output)
    try:
        raw, noise = load_fixture(args.fixture)
        actions, inference_ms = runtime.predict(raw, noise)
        if actions.shape != (1, 50, 7) or actions.device.type != "cpu":
            raise ValueError("First action must be a complete CPU action chunk")
        if not torch.isfinite(actions).all():
            raise ValueError("First action contains nonfinite values")
        completed = time.monotonic()
        elapsed = completed - args.started_at
        if elapsed <= 0:
            raise ValueError("Invalid parent/child monotonic clock interval")
        result = {
            "status": "passed",
            "backend": args.backend,
            "checkpoint": MODEL,
            "revision": REVISION,
            "fixture": args.fixture.name,
            "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
            "process_to_first_action_s": elapsed,
            "cached_runtime_init_s": runtime.startup_s,
            "first_inference_ms": inference_ms,
            "action_shape": list(actions.shape),
            "child_pid": os.getpid(),
            "contract": (
                "Fresh Python process, cached assets; includes interpreter and backend "
                "imports, runtime initialization, fixture read, preprocessing, all ten "
                "denoising steps and unnormalized CPU actions. Excludes teardown and "
                "result writes. OS/file and driver caches are not flushed."
            ),
        }
        np.save(args.output / "first.actions.npy", actions.float().numpy())
        (args.output / "first-action.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        runtime.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--started-at", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.started_at is None:
            parser.error("Internal worker requires the parent's monotonic start")
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        "--backend",
        args.backend,
        "--root",
        str(args.root.resolve()),
        "--fixture",
        str(args.fixture.resolve()),
        "--output",
        str(args.output.resolve()),
    ]
    with (args.output / "worker.log").open("w") as log:
        started = time.monotonic()
        command.extend(["--started-at", str(started)])
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            process.wait(timeout=300)
        finally:
            cleanup_group(process)
    if process.returncode:
        raise RuntimeError(f"Startup worker failed with exit code {process.returncode}")
    result = json.loads((args.output / "first-action.json").read_text())
    result["process_wall_s"] = time.monotonic() - started
    result["command"] = command
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert result["child_pid"] == process.pid
    assert result["process_to_first_action_s"] <= result["process_wall_s"]
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
