"""Application-owned jobs for the native SmolVLA worker (no scheduler or database)."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .cuda_bench import measure, percentile
from .cuda_common import hardware_fingerprint
from .worker import atomic_json, code_hash, describe_runtime, sha256
from .worker import run as quantize_job


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    actual = {
        p.relative_to(directory).as_posix()
        for p in directory.rglob("*")
        if p.is_file() and p != directory / "manifest.json"
    }
    if actual != set(manifest["files"]):
        raise ValueError("Artifact inventory changed")
    for name, expected in manifest["files"].items():
        path = directory / name
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(directory.resolve())
            or sha256(path) != expected
        ):
            raise ValueError("Artifact identity changed: " + name)
    return manifest


def publish(directory, metadata, label, kind="gguf"):
    files = {
        p.relative_to(directory).as_posix(): sha256(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p != directory / "manifest.json"
    }
    atomic_json(
        directory / "manifest.json", {"schema_version": 1, "metadata": metadata, "files": files}
    )
    return {"path": str(directory), "label": label, "format": kind}


def copied(source, destination):
    source = Path(source)
    verify(source)
    shutil.copytree(source, destination)
    (destination / "manifest.json").unlink()
    return destination


def runtime_identity(runtime):
    build = Path(runtime["build"])
    binaries = {name: sha256(build / name) for name in ["vla-bench", "tests/vla_predict_check"]}
    if (build / "vla-server").is_file():
        binaries["vla-server"] = sha256(build / "vla-server")
    identity = {
        "binaries": binaries,
        "device": runtime["device"],
        "hardware": hardware_fingerprint(),
        "worker_sha256": code_hash(),
    }
    if runtime["device"] == "cuda":
        identity["gpu"] = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                "0",
                "--query-gpu=uuid,name,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    return identity


def import_policy(job):
    output = Path(job["output_dir"]) / "bundle"
    output.mkdir()
    if job["artifact"]:
        artifact = job["artifact"]
        source = Path(artifact["path"])
        original = verify(source)
        if artifact["format"] != "native_checkpoint":
            raise ValueError("Only a native floating checkpoint can be converted to GGUF")
        converter = (
            Path(job["runtime"].get("conversion_vendor") or job["runtime"]["vendor"])
            / "scripts/convert_smolvla_to_gguf.py"
        )
        subprocess.run(
            [
                sys.executable,
                str(converter),
                "--ckpt",
                str(source / "policy"),
                "--out",
                str(output / "model.gguf"),
            ],
            check=True,
        )
        metadata = {**original["metadata"], "converter_sha256": sha256(converter)}
    else:
        source = job["source"]
        model = Path(source["path"])
        if sha256(model) != source["sha256"]:
            raise ValueError("Configured policy source hash mismatch")
        shutil.copyfile(model, output / "model.gguf")
        if sha256(output / "model.gguf") != source["sha256"]:
            raise ValueError("Policy source changed during import")
        metadata = {
            "task": source["task"],
            "action_dim": source["action_dim"],
            "source": source["provenance"],
            "source_sha256": source["sha256"],
        }
    import gguf

    from .quantization import validate_master

    reader = gguf.GGUFReader(output / "model.gguf")
    if reader.fields["general.architecture"].contents() != "smolvla":
        raise ValueError("Only SmolVLA is supported by this worker")
    validate_master(reader)
    tokenizer = job["runtime"].get("tokenizer")
    if tokenizer:
        source = Path(tokenizer)
        if not source.is_dir():
            raise ValueError("Configure a local pinned tokenizer directory")
        dest = output / "tokenizer"
        dest.mkdir()
        for file in source.iterdir():
            if file.is_file() and file.suffix in {".json", ".txt", ".model", ".jinja"}:
                shutil.copyfile(file, dest / file.name)
    metadata.update(
        architecture="smolvla",
        precision="float",
        weight_bytes=sum(t.data.nbytes for t in reader.tensors),
        deployment_verified=False,
    )
    return {
        "artifact": publish(output, metadata, "SmolVLA floating reference"),
        "report": {"scope": "policy_import", "model_sha256": sha256(output / "model.gguf")},
    }


def quantize_policy(job):
    artifact = job["artifact"]
    if artifact["format"] != "gguf":
        raise ValueError("Quantization requires a floating GGUF artifact")
    source = Path(artifact["path"])
    manifest = verify(source)
    if manifest["metadata"].get("precision") != "float":
        raise ValueError("Already packed artifacts cannot be requantized")
    runtime = describe_runtime(
        Path(job["runtime"].get("conversion_vendor") or job["runtime"]["vendor"])
    )
    precision = job["parameters"]["precision"]
    recipe = {
        "schema_version": 1,
        "operation": "policy.quantize",
        "backend": "vla_cpp_smolvla",
        "policy": {"architecture": "smolvla", "lineage": artifact["id"]},
        "source": {"path": str(source / "model.gguf"), "sha256": sha256(source / "model.gguf")},
        "runtime": runtime,
        "target": job["runtime"]["device"],
        **precision,
    }
    import hashlib

    from .worker import canonical

    out = Path(job["output_dir"]) / "conversion"
    out.mkdir()
    envelope = {
        "schema_version": 1,
        "run_id": job["job_id"],
        "attempt_id": Path(job["output_dir"]).name,
        "output_dir": str(out),
        "recipe": recipe,
        "recipe_sha256": hashlib.sha256(canonical(recipe).encode()).hexdigest(),
    }
    atomic_json(out / "job.json", envelope)
    if quantize_job(out / "job.json"):
        raise ValueError(json.loads((out / "result.json").read_text()).get("error"))
    bundle = out / "bundle"
    converted = json.loads((bundle / "manifest.json").read_text())
    (bundle / "manifest.json").rename(bundle / "conversion-manifest.json")
    if (source / "tokenizer").exists():
        shutil.copytree(source / "tokenizer", bundle / "tokenizer")
    metadata = {
        **manifest["metadata"],
        "precision": precision,
        "weight_bytes": converted["deployed_weight_bytes"],
        "deployment_verified": False,
    }
    return {
        "artifact": publish(
            bundle,
            metadata,
            precision["language"] + (" + vision Q8" if precision["vision"] else ""),
        ),
        "report": {"scope": "conversion_only", "seconds": converted["quantize_seconds"]},
    }


def cpu_measure(command, log, runtime):
    peak = 0
    start = time.monotonic()
    with log.open("w") as stream:
        process = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env={**os.environ, "VLA_N_THREADS": "4", "OMP_NUM_THREADS": "4", "VLA_IMG_SIZE": "512"},
        )
        try:
            while process.poll() is None:
                pending, seen, rss = [process.pid], set(), 0
                while pending:
                    pid = pending.pop()
                    if pid in seen:
                        continue
                    seen.add(pid)
                    try:
                        status = Path(f"/proc/{pid}/status").read_text()
                        match = re.search(r"VmRSS:\s+(\d+)", status)
                        rss += int(match[1]) if match else 0
                        pending.extend(
                            map(int, Path(f"/proc/{pid}/task/{pid}/children").read_text().split())
                        )
                    except (FileNotFoundError, ProcessLookupError):
                        pass
                peak = max(peak, rss)
                if time.monotonic() - start > 1800:
                    raise TimeoutError("Native prediction timed out")
                time.sleep(0.05)
            if process.returncode:
                raise RuntimeError("Native process failed; inspect " + log.name)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    return {
        "sampled_peak_rss_mib": peak / 1024 if peak else None,
        "memory_scope": (
            "Process-tree RSS sum sampled at 50ms on Linux (shared pages counted per process); "
            "unavailable on other hosts"
        ),
        "wall_seconds": time.monotonic() - start,
    }


def engine(job):
    artifact = job["artifact"]
    if artifact["format"] not in {"gguf", "deployment_package"}:
        raise ValueError("Native evaluation requires a GGUF artifact")
    source = Path(artifact["path"])
    manifest = verify(source)
    runtime = job["runtime"]
    identity = runtime_identity(runtime)
    model = source / "model.gguf"
    build = Path(runtime["build"])
    output = Path(job["output_dir"])
    evaluation = job["parameters"]["evaluation"]

    def measured(argv, name):
        log = output / name
        memory = (
            measure(argv, log, timeout=1800)
            if runtime["device"] == "cuda"
            else cpu_measure(argv, log, runtime)
        )
        text = log.read_text()
        if runtime["device"] == "cuda" and (
            "backend = CUDA" not in text or "falling back to CPU" in text
        ):
            raise ValueError("Requested CUDA execution was not established")
        if runtime["device"] == "cpu" and "backend = CUDA" in text:
            raise ValueError("Requested CPU execution used CUDA")
        return text, memory

    text, probe_memory = measured(
        [str(build / "tests/vla_predict_check"), str(model)], "reload.log"
    )
    match = re.search(r"action_len=(\d+)\n", text)
    if not match or int(match[1]) != 1600:
        raise ValueError("Native probe must emit all 50 x 32 action values")
    values = [float(x) for x in text[match.end() :].splitlines()[:1600]]
    if len(values) != 1600 or not all(math.isfinite(x) for x in values):
        raise ValueError("Native probe produced invalid actions")
    packed = re.search(r"packed resident matrices: lm=(\d+) vision=(\d+)", text)
    precision = manifest["metadata"].get("precision")
    expected = (0, 0) if precision == "float" else (224, 72 if precision.get("vision") else 0)
    if packed is None or tuple(map(int, packed.groups())) != expected:
        raise ValueError("Native resident packing does not match the artifact recipe")
    atomic_json(output / "actions.json", {"values": values})
    text, bench_memory = measured(
        [
            str(build / "vla-bench"),
            "--ckpt",
            str(model),
            "--images",
            "2",
            "--size",
            "512",
            "--warmup",
            str(evaluation["warmups"]),
            "--reps",
            str(evaluation["repetitions"]),
            "--markdown",
        ],
        "timing.log",
    )
    match = re.search(r"^vla-bench: samples_ms=([^\n]+)", text, re.M)
    samples = [] if match is None else [float(x) for x in match[1].split(",")]
    if len(samples) != evaluation["repetitions"] or not all(
        math.isfinite(x) and x > 0 for x in samples
    ):
        raise ValueError("Native benchmark requires the per-call timing instrumentation")
    key = "sampled_peak_device_used_mib" if runtime["device"] == "cuda" else "sampled_peak_rss_mib"
    memory_values = [x[key] for x in (probe_memory, bench_memory) if x.get(key) is not None]
    if runtime_identity(runtime) != identity:
        raise ValueError("Runtime identity changed during measurement")
    return {
        "scope": "engine_diagnostics",
        "model_sha256": sha256(model),
        "runtime": identity,
        "fresh_reload_verified": True,
        "finite_action_values": len(values),
        "p50_ms": percentile(samples, 50),
        "p95_ms": percentile(samples, 95),
        "samples_ms": samples,
        "peak_device_mib": max(memory_values) if memory_values else None,
        "memory_scope": bench_memory["memory_scope"],
        "measurement": bench_memory,
        "success_rate": None,
        "complete_episodes": 0,
    }


def evaluate_policy(job):
    report = engine(job)
    evaluation = job["parameters"]["evaluation"]
    if evaluation["mode"] == "libero":
        source = Path(job["artifact"]["path"])
        manifest = verify(source)
        if (
            manifest["metadata"].get("task") != "libero_object"
            or manifest["metadata"].get("action_dim") != 7
        ):
            raise ValueError(
                "This policy has no LIBERO observation/action compatibility declaration"
            )
        runtime = job["runtime"]
        if not runtime.get("simulator_lane") or not (source / "tokenizer").is_dir():
            raise ValueError("LIBERO evaluation needs its prepared simulator and pinned tokenizer")
        output = Path(job["output_dir"])
        candidate = {
            "preset": "candidate",
            "status": "engine_verified",
            "artifact": str(source / "model.gguf"),
            "sha256": sha256(source / "model.gguf"),
        }
        atomic_json(output / "engine.json", {"results": [candidate]})
        episodes = []
        states = evaluation["final_states"] if job.get("final") else evaluation["initial_states"]
        for state in states:
            command = [
                sys.executable,
                "-m",
                "policykit.cuda_rollout",
                "--preset",
                "candidate",
                "--manifest",
                str(output / "engine.json"),
                "--runtime-build",
                runtime["build"],
                "--lane",
                runtime["simulator_lane"],
                "--output-root",
                str(output / "episodes"),
                "--run",
                "evaluation",
                "--task-id",
                str(evaluation["task_id"]),
                "--init-state-id",
                str(state),
                "--seed",
                str(evaluation["seed"]),
                "--steps",
                str(evaluation["steps"]),
                "--tokenizer",
                str(source / "tokenizer"),
            ]
            if runtime["device"] == "cuda":
                command += ["--require-cuda"]
            log = output / f"episode-{state}.log"
            resource = (
                measure(command, log, timeout=1800)
                if runtime["device"] == "cuda"
                else cpu_measure(command, log, runtime)
            )
            memory_key = (
                "sampled_peak_device_used_mib"
                if runtime["device"] == "cuda"
                else "sampled_peak_rss_mib"
            )
            if resource.get(memory_key) is not None:
                report["peak_device_mib"] = max(
                    report.get("peak_device_mib") or 0, resource[memory_key]
                )
            report.setdefault("rollout_measurements", []).append(resource)
            path = (
                output
                / "episodes/evaluation/candidate"
                / (
                    f"task{evaluation['task_id']}-init{state}-seed{evaluation['seed']}"
                    f"-steps{evaluation['steps']}"
                )
                / "result.json"
            )
            episodes.append(json.loads(path.read_text()))
        complete = [x for x in episodes if x["status"] == "episode_complete"]
        report.update(
            scope="libero_development" if not job.get("final") else "libero_final",
            episodes=episodes,
            complete_episodes=len(complete),
            requested_episodes=len(states),
            success_rate=sum(x["task_success"] for x in complete) / len(episodes)
            if len(complete) == len(episodes)
            else None,
            protocol=evaluation,
            training_overlap="not audited",
        )
    return {"report": report}


def run_policy(job):
    report = evaluate_policy(job)["report"]
    limits = job["parameters"].get("limits")
    if (
        job.get("final")
        and limits
        and (
            report.get("success_rate") is None
            or report["success_rate"] < limits["min_success_rate"]
            or report["p95_ms"] > limits["max_p95_ms"]
            or report.get("peak_device_mib") is None
            or report["peak_device_mib"] > limits["max_peak_device_mib"]
        )
    ):
        raise ValueError("Final package rerun failed acceptance constraints")
    if job.get("final") and limits:
        reference = next(
            (x for x in job.get("prior_reports", []) if x.get("stage") == "final-reference"), None
        )
        if (
            not reference
            or report["runtime"] != reference["runtime"]
            or report["complete_episodes"] != len(job["parameters"]["evaluation"]["final_states"])
            or report["success_rate"] < reference["success_rate"] - limits["max_success_drop"]
        ):
            raise ValueError("Final package rerun no longer matches the paired reference")
    source = Path(job["artifact"]["path"])
    manifest = verify(source)
    output = copied(source, Path(job["output_dir"]) / "package")
    atomic_json(output / "reload-verification.json", report)
    atomic_json(
        output / "workflow-evidence.json",
        {
            "reports": job.get("prior_reports", []),
            "acceptance_limits": limits,
            "search_and_final_states": job["parameters"]["evaluation"],
        },
    )
    atomic_json(
        output / "lineage.json", {"source_artifact": job["artifact"], "job_id": job["job_id"]}
    )
    metadata = {
        **manifest["metadata"],
        "fresh_reload_verified": True,
        "runtime": report["runtime"],
        "deployment_verified": bool(job.get("final") and limits),
        "required_runtime": "Pinned vla.cpp binaries recorded in reload-verification.json",
    }
    return {
        "artifact": publish(
            output, metadata, "Reload-verified SmolVLA package", "deployment_package"
        ),
        "report": report,
    }


def main():
    request, result = map(Path, sys.argv[1:])
    job = json.loads(request.read_text())
    response = {"schema_version": 1, "job_id": job["job_id"]}
    try:
        if job.get("schema_version") != 1:
            raise ValueError("Unsupported application envelope")
        handlers = {
            "policy.import": import_policy,
            "policy.quantize": quantize_policy,
            "policy.evaluate": evaluate_policy,
            "policy.run": run_policy,
        }
        response.update(handlers[job["operation"]](job))
    except Exception as exc:
        response["error"] = f"{type(exc).__name__}: {exc}"
    atomic_json(result, response)
    if response.get("error"):
        print(response["error"], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
