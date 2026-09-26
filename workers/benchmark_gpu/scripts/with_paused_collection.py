"""Bounded, authorized collection pause with a detached resume watchdog.

Run as root on the L4. Resumes only the exact processes it stopped, verified by
PID and kernel start time. The benchmark runs as the original non-root user.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil


def resume(manifest):
    record = json.loads(manifest.read_text())
    for entry in record["processes"]:
        try:
            p = psutil.Process(entry["pid"])
            if p.create_time() == entry["created"]:
                p.send_signal(signal.SIGCONT)
        except psutil.NoSuchProcess:
            pass
    manifest.with_suffix(".resumed").touch()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--watchdog", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--seconds", type=int, default=900)
    p.add_argument("--collector-user")
    p.add_argument("--collector-script")
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    if args.watchdog:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        record = json.loads(args.watchdog.read_text())
        while time.time() < record["deadline"]:
            if args.watchdog.with_suffix(".resumed").exists():
                return
            time.sleep(1)
        resume(args.watchdog)
        return
    if os.geteuid() != 0 or not args.manifest or not args.command:
        p.error("Root, manifest and benchmark command required")
    if not args.collector_user or not args.collector_script:
        p.error("Explicit collector user and script basename are required")
    if not 60 <= args.seconds <= 900:
        p.error("Pause window must be between 60 and 900 seconds")
    roots = []
    for proc in psutil.process_iter(["pid", "username", "cmdline"]):
        words = proc.info["cmdline"] or []
        if (
            proc.info["username"] == args.collector_user
            and any(Path(word).name == args.collector_script for word in words)
            and any("python" in Path(word).name for word in words[:1])
        ):
            roots.append(proc)
    if len(roots) != 1:
        raise RuntimeError(f"Expected one collection parent, found {len(roots)}")
    processes = [roots[0], *roots[0].children(recursive=True)]
    record = {
        "started": time.time(),
        "deadline": time.time() + args.seconds,
        "processes": [{"pid": x.pid, "created": x.create_time()} for x in processes],
        "command": args.command,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    if args.manifest.exists():
        raise FileExistsError(args.manifest)
    args.manifest.write_text(json.dumps(record, indent=2))
    with args.manifest.with_suffix(".watchdog.log").open("w") as log:
        watchdog = subprocess.Popen(
            [sys.executable, __file__, "--watchdog", str(args.manifest)],
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
        )

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Pause interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    benchmark = None
    try:
        for process in processes:
            process.send_signal(signal.SIGSTOP)
        time.sleep(2)
        states = [{"pid": x.pid, "status": x.status()} for x in processes]
        if any(x["status"] != psutil.STATUS_STOPPED for x in states):
            raise RuntimeError("Collection processes did not all stop")
        record["stopped"] = states
        record["idle_samples"] = []
        for _ in range(10):
            value = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
            record["idle_samples"].append(value)
            time.sleep(1)
        if any(float(x.split(",")[0]) > 2 for x in record["idle_samples"][-5:]):
            raise RuntimeError("GPU remained active after pausing collection")
        args.manifest.write_text(json.dumps(record, indent=2))
        command = args.command[1:] if args.command[0] == "--" else args.command
        benchmark = subprocess.Popen(command, start_new_session=True)
        try:
            record["returncode"] = benchmark.wait(timeout=max(1, args.seconds - 30))
        finally:
            if benchmark.poll() is None:
                os.killpg(benchmark.pid, signal.SIGTERM)
                try:
                    benchmark.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(benchmark.pid, signal.SIGKILL)
                    benchmark.wait()
    finally:
        resume(args.manifest)
        record["resumed_at"] = time.time()
        record["post_resume_states"] = []
        for entry in record["processes"]:
            try:
                process = psutil.Process(entry["pid"])
                record["post_resume_states"].append({"pid": process.pid, "state": process.status()})
            except psutil.NoSuchProcess:
                record["post_resume_states"].append({"pid": entry["pid"], "state": "exited"})
        args.manifest.write_text(json.dumps(record, indent=2))
        watchdog.wait(timeout=5)
    if record["returncode"]:
        raise SystemExit(record["returncode"])


if __name__ == "__main__":
    main()
