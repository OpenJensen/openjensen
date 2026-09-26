"""Sequential fresh-process comparison with process-scoped NVIDIA memory samples."""

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument(
        "--backends",
        nargs="+",
        default=[
            "native-bf16",
            "cpp-bf16",
            "native-fp16",
            "cpp-Q8_0",
            "native-int8",
            "cpp-Q4_0",
            "native-nf4",
            "cpp-Q8_0-vision",
        ],
    )
    p.add_argument("--repeat-backend", default="native-bf16")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    candidates = [(backend, backend) for backend in args.backends]
    candidates.append((args.repeat_backend, args.repeat_backend + "-repeat"))
    if len({name for _, name in candidates}) != len(candidates):
        p.error("Duplicate candidate names")
    for backend, name in candidates:
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).with_name("smolvla_cross_stack.py")),
            "--backend",
            backend,
            "--output",
            str(args.output / name),
            "--fixtures",
            str(args.fixtures),
            "--samples",
            str(args.samples),
            "--warmup",
            "1",
        ]
        started = time.time()
        with (
            (args.output / f"{name}.log").open("w") as log,
            (args.output / f"{name}.gpu.csv").open("w") as mem,
        ):
            sampler = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-compute-apps=timestamp,pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                    "-lms",
                    "100",
                ],
                stdout=mem,
                stderr=subprocess.DEVNULL,
            )
            proc = subprocess.Popen(
                command,
                start_new_session=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={
                    **os.environ,
                    "HF_HUB_OFFLINE": "1",
                    "OMP_NUM_THREADS": "4",
                    "VLA_N_THREADS": "4",
                },
            )
            pids = {proc.pid}
            try:
                while proc.poll() is None:
                    try:
                        pids.update(
                            x.pid for x in psutil.Process(proc.pid).children(recursive=True)
                        )
                    except psutil.NoSuchProcess:
                        pass
                    if time.time() - started > 300:
                        raise TimeoutError(name)
                    time.sleep(0.5)
            finally:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                sampler.terminate()
                sampler.wait(timeout=5)
            row = {
                "name": name,
                "backend": backend,
                "command": command,
                "pids": sorted(pids),
                "returncode": proc.returncode,
                "wall_s": time.time() - started,
            }
        samples = {}
        for line in (args.output / f"{name}.gpu.csv").read_text().splitlines():
            try:
                timestamp, pid, memory = [s.strip() for s in line.split(",")]
                if int(pid) in pids:
                    samples[timestamp] = samples.get(timestamp, 0) + float(memory)
            except (ValueError, IndexError):
                pass
        peak = max(samples.values()) if samples else 0
        row["sampled_process_peak_mib"] = peak if peak > 0 else None
        rows.append(row)
        (args.output / "runs.json").write_text(json.dumps(rows, indent=2))
        print(name, proc.returncode, row["sampled_process_peak_mib"], flush=True)
    if any(r["returncode"] for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
