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
from importlib.util import find_spec
from pathlib import Path

from .acceptance import artifact_contract, memory_coverage, sanitize_measurement, satisfies_limits
from .cuda_bench import measure, percentile
from .provenance import runtime_identity
from .worker import atomic_json, describe_runtime, sha256
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
    original = verify(source)
    shutil.copytree(source, destination)
    if verify(destination) != original:
        raise ValueError("Artifact changed during materialization")
    return destination


def import_policy(job):
    output = Path(job["output_dir"]) / "bundle"
    output.mkdir()
    if job["artifact"]:
        artifact = job["artifact"]
        source = Path(artifact["path"])
        original = verify(source)
        if artifact["format"] != "native_checkpoint":
            raise ValueError("Only a native floating checkpoint can be converted to GGUF")
        if any(find_spec(name) is None for name in ("torch", "safetensors")):
            raise ValueError(
                "Install the vla_cpp convert extra in the configured conversion environment"
            )
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
    peak = samples = 0
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
                if rss > 0:
                    samples += 1
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
        "rss_samples": samples,
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
    contract = artifact_contract(source, manifest)
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
        memory = sanitize_measurement(memory)
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
    action_length = contract["action_length"]
    if not match or int(match[1]) != action_length:
        raise ValueError("Native probe action length disagrees with validated GGUF metadata")
    values = [float(x) for x in text[match.end() :].splitlines()[:action_length]]
    if len(values) != action_length or not all(math.isfinite(x) for x in values):
        raise ValueError("Native probe produced invalid actions")
    packed = re.search(r"packed resident matrices: lm=(\d+) vision=(\d+)", text)
    expected = tuple(contract["packed_matrices"][name] for name in ("language", "vision"))
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
    measurements = {"reload": probe_memory, "timing": bench_memory}
    if runtime_identity(runtime) != identity:
        raise ValueError("Runtime identity changed during measurement")
    if verify(source) != manifest:
        raise ValueError("Artifact changed during measurement")
    return {
        "scope": "engine_diagnostics",
        "model_sha256": sha256(model),
        "artifact_path": str(source.resolve()),
        "artifact_manifest_sha256": sha256(source / "manifest.json"),
        "artifact_contract": contract,
        "worker_process_id": os.getpid(),
        "runtime": identity,
        "fresh_reload_verified": True,
        "finite_action_values": len(values),
        "p50_ms": percentile(samples, 50),
        "p95_ms": percentile(samples, 95),
        "samples_ms": samples,
        **memory_coverage(measurements, runtime["device"]),
        "measurements": measurements,
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
            resource = sanitize_measurement(resource)
            report.setdefault("measurements", {})[f"rollout-{state}"] = resource
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
        report.update(memory_coverage(report.get("measurements", {}), runtime["device"]))
        if runtime_identity(runtime) != report["runtime"]:
            raise ValueError("Runtime identity changed during simulator evaluation")
        complete = [
            x
            for x in episodes
            if x["status"] == "episode_complete" and type(x.get("task_success")) is bool
        ]
        report.update(
            scope="libero_development" if not job.get("final") else "libero_final",
            episodes=episodes,
            complete_episodes=len(complete),
            requested_episodes=len(states),
            success_rate=sum(x["task_success"] for x in complete) / len(episodes)
            if episodes and len(complete) == len(episodes)
            else None,
            protocol=evaluation,
            training_overlap="not audited",
        )
    return {"report": report}


def evaluate_package(job, package):
    """Run the materialized package in an independent, offline worker process."""
    output = Path(job["output_dir"]) / "package-verification"
    output.mkdir()
    request, result = output / "request.json", output / "result.json"
    child = {
        **job,
        "operation": "policy.evaluate",
        "artifact": {
            **job["artifact"],
            "path": str(package.resolve()),
            "format": "deployment_package",
        },
        "output_dir": str(output.resolve()),
        "source": None,
        "prior_reports": [],
    }
    atomic_json(request, child)
    manifest_sha = sha256(package / "manifest.json")
    model_sha = sha256(package / "model.gguf")
    # The fallback path supports direct local invocation without making the
    # current working directory or the original policy directory import roots.
    bootstrap = (
        "import sys; sys.path.append("
        + repr(str(Path(__file__).resolve().parents[1]))
        + "); from policykit.application import main; raise SystemExit(main())"
    )
    with (output / "worker.log").open("w") as log:
        completed = subprocess.run(
            [sys.executable, "-c", bootstrap, str(request.resolve()), str(result.resolve())],
            cwd=output,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=43200,
        )
    if completed.returncode or not result.is_file() or result.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Fresh package worker failed; inspect package-verification/worker.log")
    response = json.loads(result.read_text())
    report = response.get("report", {})
    if (
        response.get("schema_version") != 1
        or response.get("job_id") != job["job_id"]
        or response.get("error")
        or report.get("worker_process_id") in {None, os.getpid()}
        or report.get("artifact_path") != str(package.resolve())
        or report.get("artifact_manifest_sha256") != manifest_sha
        or report.get("model_sha256") != model_sha
        or report.get("fresh_reload_verified") is not True
    ):
        raise ValueError("Fresh package worker did not verify the exact materialized package")
    return report


def run_policy(job):
    source = Path(job["artifact"]["path"])
    manifest = verify(source)
    reserved = {
        "reload-verification.json",
        "runtime-lock.json",
        "tested-payload.json",
        "workflow-evidence.json",
        "lineage.json",
    }
    if reserved.intersection(manifest["files"]):
        raise ValueError("Artifact contains reserved export evidence names")
    output = copied(source, Path(job["output_dir"]) / "package-pending")
    tested_manifest_sha = sha256(output / "manifest.json")
    report = evaluate_package(job, output)
    # The evaluator may add reports outside the package, never mutate its input.
    if verify(output) != manifest or sha256(output / "manifest.json") != tested_manifest_sha:
        raise ValueError("Materialized package changed during fresh-process evaluation")
    limits = job["parameters"].get("limits")
    if job.get("final") and limits and not satisfies_limits(report, limits):
        raise ValueError("Final package rerun failed acceptance constraints")
    if job.get("final") and limits:
        reference = next(
            (x for x in job.get("prior_reports", []) if x.get("stage") == "final-reference"), None
        )
        if (
            not reference
            or not isinstance(reference.get("success_rate"), (int, float))
            or isinstance(reference.get("success_rate"), bool)
            or not math.isfinite(reference["success_rate"])
            or not 0 <= reference["success_rate"] <= 1
            or report["runtime"] != reference["runtime"]
            or report["complete_episodes"] != len(job["parameters"]["evaluation"]["final_states"])
            or report["success_rate"] < reference["success_rate"] - limits["max_success_drop"]
        ):
            raise ValueError("Final package rerun no longer matches the paired reference")
    atomic_json(output / "reload-verification.json", report)
    atomic_json(output / "runtime-lock.json", report["runtime"])
    atomic_json(
        output / "tested-payload.json",
        {
            "manifest_sha256": tested_manifest_sha,
            "files": manifest["files"],
            "scope": "Exact pre-export payload evaluated in the fresh package worker",
        },
    )
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
        "required_runtime": "External prepared Linux runtime; see runtime-lock.json",
    }
    artifact = publish(output, metadata, "Reload-verified SmolVLA package", "deployment_package")
    final = Path(job["output_dir"]) / "package"
    if final.exists():
        raise ValueError("Preserve the existing package before exporting again")
    output.rename(final)
    artifact["path"] = str(final)
    return {"artifact": artifact, "report": report}


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
