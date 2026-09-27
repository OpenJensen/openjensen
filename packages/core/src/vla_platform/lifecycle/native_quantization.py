"""Byte-bound local ACT packing adapter. ML lives only in the isolated worker."""

import asyncio
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path

from .contracts import LifecycleResult, PolicyArtifact
from .runtime import native_quantization_ready
from .simulation import finish_owned, strict_json

TOTAL_LIMIT = 768 * 1024**2
WEIGHT_LIMIT = 512 * 1024**2
JSON_LIMIT = 1024**2
NAME = re.compile(r"[A-Za-z0-9_.-]{1,120}\Z")
RUNTIME = {"lerobot": "0.6.1", "torch": "2.11.0", "torchvision": "0.26.0", "safetensors": "0.8.0"}
SCOPE = "generated observations; no calibration or task-quality acceptance"
UNITS = "saved processor output coordinates; physical units unverified"


def checked_path(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Native policy must be an owned absolute path")
    if any(p.is_symlink() or p.is_junction() for p in [path, *path.parents]):
        raise ValueError("Native policy path links are unsupported")
    return path


def hash_file(path):
    """Bounded streaming hash, refusing special files and mutation during the read."""
    path = checked_path(path)
    parent = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent)
            parent = child
        fd = os.open(path.name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=parent)
    finally:
        os.close(parent)
    limit = WEIGHT_LIMIT if path.name in {"model.safetensors", "model.fbq"} else JSON_LIMIT
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit:
            raise ValueError("Native policy has an empty, oversized or nonregular file")
        digest, size = hashlib.sha256(), 0
        while chunk := stream.read(min(1024**2, before.st_size + 1 - size)):
            size += len(chunk)
            digest.update(chunk)
            if size > before.st_size:
                raise ValueError("Native policy grew while reading")
        after = os.fstat(stream.fileno())
        if size != before.st_size or (before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("Native policy changed while reading")
    return {"sha256": digest.hexdigest(), "bytes": size}


def inventory(root):
    root = checked_path(root)
    result, total, entries = {}, 0, 0
    pending = [root]
    while pending:
        directory = checked_path(pending.pop())
        with os.scandir(directory) as paths:
            for entry in paths:
                entries += 1
                path = Path(entry.path)
                relative = path.relative_to(root)
                if entries > 64 or len(relative.parts) > 6 or not NAME.fullmatch(path.name):
                    raise ValueError("Native policy inventory exceeds supported bounds")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                else:
                    item = hash_file(path)
                    total += item["bytes"]
                    if total > TOTAL_LIMIT:
                        raise ValueError("Native policy exceeds its aggregate byte bound")
                    result[relative.as_posix()] = item
    if not result:
        raise ValueError("Native policy is empty")
    return result


def bundle(root):
    files = inventory(root)
    manifest = strict_json(root / "manifest.json", JSON_LIMIT)
    declared = manifest.get("files") if isinstance(manifest, dict) else None
    actual = {name: row["sha256"] for name, row in files.items() if name != "manifest.json"}
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or declared != actual
    ):
        raise ValueError("Native artifact inventory differs from its manifest")
    return manifest, files


def temporal_info(config, policy=None, files=None):
    """Check saved ACT dimensions and optional exact training sampling provenance."""
    if not isinstance(config, dict):
        raise ValueError("ACT config must be an object")
    prediction, execution = config.get("chunk_size"), config.get("n_action_steps")
    if (
        type(prediction) is not int
        or type(execution) is not int
        or not 1 <= execution <= prediction <= 1024
    ):
        raise ValueError("ACT horizons require 1 <= execution <= prediction <= 1024")
    observation_steps = config.get(
        "n_obs_steps", 1 if (prediction, execution) == (100, 100) else None
    )
    if type(observation_steps) is not int or observation_steps != 1:
        raise ValueError("ACT temporal admission requires one saved observation step")
    if config.get("temporal_ensemble_coeff") is not None:
        raise ValueError("ACT temporal admission does not support temporal ensembling")
    result = {"prediction_horizon": prediction, "execution_horizon": execution}
    name = "temporal-contract.json"
    if files is not None and name in files:
        from types import SimpleNamespace

        from .temporal import resolved_temporal

        record = strict_json(policy / name, JSON_LIMIT)
        if not isinstance(record, dict):
            raise ValueError("ACT temporal contract must be an object")
        actual = SimpleNamespace(
            chunk_size=prediction,
            n_action_steps=execution,
            n_obs_steps=observation_steps,
            action_delta_indices=list(range(prediction)),
            observation_delta_indices=record.get("observation_delta_indices"),
        )
        expected = resolved_temporal(actual, record.get("action_fps"), "act")
        if not exact_json(record, expected):
            raise ValueError("Temporal contract differs from saved ACT config")
        result["temporal_contract_sha256"] = files[name]["sha256"]
    else:
        result["temporal_contract_sha256"] = None
    return result


def temporal_claims(metadata, admitted):
    expected = {
        key: admitted.get(key, 100 if key != "temporal_contract_sha256" else None)
        for key in ("prediction_horizon", "execution_horizon", "temporal_contract_sha256")
    }
    # Historical 100/100 artifacts did not declare these additive fields.
    if any(key in metadata for key in expected) or expected != {
        "prediction_horizon": 100,
        "execution_horizon": 100,
        "temporal_contract_sha256": None,
    }:
        return expected
    return {}


def source_info(artifact, data_dir, *, require_inference=True):
    """Verify an owned complete ACT policy; packing additionally requires VAE removal."""
    if (
        artifact.format not in {"native_checkpoint", "inference_export"}
        or artifact.metadata.get("architecture") != "act"
    ):
        raise ValueError(
            "Native quantization requires an imported ACT policy or ACT inference export"
        )
    if artifact.metadata.get("storage") == "gcs" or artifact.metadata.get("remote"):
        raise ValueError("Native quantization requires an owned local complete ACT policy")
    root = checked_path(data_dir.absolute() / artifact.path)
    owned = (data_dir / "jobs" / artifact.job_id).absolute()
    if not root.is_relative_to(owned):
        raise ValueError("Native policy must remain inside its owning application job")
    manifest, outer = bundle(root)
    if (
        outer["manifest.json"]["sha256"] != artifact.manifest_sha256
        or manifest.get("metadata") != artifact.metadata
    ):
        raise ValueError("Registered ACT artifact identity or metadata changed")
    if any(Path(name).name in {"remote.json", "remote-checkpoint.json"} for name in outer):
        raise ValueError("Materialize and export the complete checkpoint before quantization")
    models = [root / name for name in outer if Path(name).name == "model.safetensors"]
    if len(models) != 1:
        raise ValueError("Expected exactly one complete ACT inference policy")
    policy = models[0].parent
    files = {name: row for name, row in inventory(policy).items()}
    if len(files) > 16 or any("/" in name for name in files):
        raise ValueError("Expected one flat, complete ACT policy directory")
    required = {
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    }
    if not required <= files.keys():
        raise ValueError("ACT policy requires config, weights and both saved processors")
    config = strict_json(policy / "config.json", JSON_LIMIT)
    if (
        config.get("type") != "act"
        or type(config.get("use_vae")) is not bool
        or (require_inference and config["use_vae"] is not False)
    ):
        raise ValueError("First export ACT to inference-only FP32: use_vae=false is required")
    temporal = temporal_info(config, policy, files)
    features = config.get("input_features", {})
    if not isinstance(features, dict) or not isinstance(config.get("output_features"), dict):
        raise ValueError("ACT input/output features must be saved mappings")
    state = features.get("observation.state")
    action = config["output_features"].get("action")
    if not isinstance(state, dict) or not isinstance(action, dict):
        raise ValueError("ACT state and action features must be saved objects")
    cameras = [
        value.get("shape")
        for key, value in features.items()
        if key.startswith("observation.images.") and isinstance(value, dict)
    ]
    if len(cameras) != 1 or state.get("shape") != [6] or action.get("shape") != [6]:
        raise ValueError(
            "Native ACT packing currently requires one camera and six state/action coordinates"
        )
    shape = cameras[0]
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or any(type(n) is not int for n in shape)
        or shape[0] != 3
        or not all(32 <= n <= 2048 for n in shape[1:])
        or shape[1] * shape[2] > 1920 * 1080
    ):
        raise ValueError("ACT camera shape exceeds the supported bounded image profile")
    for processor in ("policy_preprocessor.json", "policy_postprocessor.json"):
        value = strict_json(policy / processor, JSON_LIMIT)
        steps = value.get("steps")
        if (
            not isinstance(steps, list)
            or not steps
            or any(not isinstance(step, dict) for step in steps)
        ):
            raise ValueError("Saved ACT processors and normalization statistics are required")
        references = [
            step.get("state_file")
            for step in steps
            if isinstance(step, dict) and step.get("state_file")
        ]
        if not references or any(
            not isinstance(name, str) or not NAME.fullmatch(name) or name not in files
            for name in references
        ):
            raise ValueError("Saved ACT normalization statistics are missing")
    return {
        "root": root,
        "path": policy,
        "files": files,
        "outer": outer,
        "image_shape": cameras[0],
        **temporal,
        "manifest_sha256": files.get("manifest.json", {}).get("sha256"),
    }


def command(runtime, request, result):
    if not native_quantization_ready(runtime):
        raise ValueError("Configure the isolated native ACT quantization CPU environment first")
    root = Path(runtime.native_quantization_root).resolve()
    env = {
        key: os.environ[key]
        for key in ("PATH", "TMPDIR", "SYSTEMROOT", "WINDIR")
        if key in os.environ
    }
    env.update(
        PYTHONPATH=os.pathsep.join([str(root / "src"), str(root.parent / "act_optimizer/src")]),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    return (
        [
            runtime.native_quantization_python,
            "-m",
            "firebird_quant.native_application",
            str(request),
            str(result),
        ],
        str(root),
        env,
    )


async def validate(lifecycle, project_id, request):
    runtime = lifecycle.runtime(request.runtime_id)
    if (
        runtime is None
        or runtime.execution != "native"
        or runtime.provider != "local"
        or runtime.device != "cpu"
        or not native_quantization_ready(runtime)
    ):
        raise ValueError("Configure a ready local CPU native ACT quantization runtime first")
    lifecycle.compute.require_enabled(runtime)
    artifact = await lifecycle.artifact(project_id, request.artifact_id)
    try:
        await finish_owned(asyncio.to_thread(source_info, artifact, lifecycle.settings.data_dir))
    except OSError as exc:
        raise ValueError("The complete local ACT source is missing or unreadable") from exc
    return None


def model_identity(files):
    digest = hashlib.sha256(b"firebird-native-packed-policy-v1\0")
    for name, item in sorted(files.items()):
        encoded = name.encode("utf-8")
        digest.update(
            len(encoded).to_bytes(8, "big")
            + encoded
            + item["bytes"].to_bytes(8, "big")
            + bytes.fromhex(item["sha256"])
        )
    return "sha256:" + digest.hexdigest()


def exact_json(value, expected):
    """JSON equality with bool/int/float distinctions preserved, including nested fields."""
    return json.dumps(value, sort_keys=True, allow_nan=False) == json.dumps(
        expected, sort_keys=True, allow_nan=False
    )


def check_result(response, job, source, admitted, directory):
    if (
        not isinstance(response, dict)
        or type(response.get("schema_version")) is not int
        or response["schema_version"] != 1
        or response.get("job_id") != job.id
        or response.get("operation") != "policy.quantize"
    ):
        raise ValueError("Native quantization response identity mismatch")
    info, report = response.get("artifact"), response.get("report")
    if (
        not isinstance(info, dict)
        or info.get("format") != "native_quantized"
        or not isinstance(report, dict)
        or info.get("path") != str(directory)
    ):
        raise ValueError("Native quantization must return its exact owned packed package")
    manifest, files = bundle(directory)
    policy = {
        name.removeprefix("policy/"): item
        for name, item in files.items()
        if name.startswith("policy/")
    }
    expected_policy = {
        "config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "model.fbq",
        "encoding.json",
    }
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        for step in strict_json(admitted["path"] / name, JSON_LIMIT)["steps"]:
            if step.get("state_file"):
                expected_policy.add(step["state_file"])
    if "temporal-contract.json" in admitted["files"]:
        expected_policy.add("temporal-contract.json")
    if set(policy) != expected_policy:
        raise ValueError("Packed policy must contain exact processors and no floating master")
    for name in expected_policy - {"model.fbq", "encoding.json"}:
        if policy[name] != admitted["files"][name]:
            raise ValueError("Packed policy config or processor bytes changed")
    bits = job.request.native_quantization.bits
    encoding = strict_json(directory / "policy/encoding.json", JSON_LIMIT)
    if not exact_json(
        encoding,
        {
            "schema_version": 1,
            "format": "firebird_quant",
            "format_version": 1,
            "policy_family": "act",
            "weights": "model.fbq",
            "compute_dtype": "float32",
            "recipe": {
                "bits": bits,
                "group_size": 64,
                "min_elements": 128,
                "min_ndim": 2,
                "include": [],
                "exclude": [],
            },
            "runtime": RUNTIME,
        },
    ):
        raise ValueError("Packed encoding differs from the requested recipe/runtime")
    expected = {
        "architecture": "act",
        "format": "firebird_quant",
        "format_version": 1,
        "model_id": model_identity(policy),
        "precision": f"int{bits}",
        "policy_subdirectory": "policy",
        "inference_only": True,
        "training_resume_supported": False,
        "fresh_reload_verified": True,
        "cpu_reload_verified": True,
        "runtime_verified": False,
        "isaac_runtime_verified": False,
        "quality_verified": False,
        "calibration_verified": False,
        "speedup_verified": False,
        "task_success": None,
        "source_artifact_id": source.id,
        "source_artifact_manifest_sha256": source.manifest_sha256,
    }
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Packed metadata must be an object")
    expected.update(temporal_claims(metadata, admitted))
    if (
        metadata != expected
        or any(type(metadata.get(key)) is not type(value) for key, value in expected.items())
        or any(
            report.get(key) != value or type(report.get(key)) is not type(value)
            for key, value in expected.items()
        )
    ):
        raise ValueError("Packed policy claims or source identity do not match")
    lineage = strict_json(directory / "lineage.json", JSON_LIMIT)
    if not exact_json(
        lineage,
        {
            "schema_version": 1,
            "source_artifact_id": source.id,
            "source_artifact_manifest_sha256": source.manifest_sha256,
            "source_files": admitted["files"],
            "source_manifest_sha256": admitted["manifest_sha256"],
            "source_registry_binding": "caller-owned; worker verifies supplied exact bytes",
        },
    ):
        raise ValueError("Packed policy lineage differs from the admitted source")
    expected_files = {"manifest.json", "lineage.json", "verification.json"} | {
        "policy/" + name for name in policy
    }
    if admitted["manifest_sha256"]:
        expected_files.add("source-manifest.json")
        if files.get("source-manifest.json") != admitted["files"]["manifest.json"]:
            raise ValueError("Packed source manifest changed")
    if set(files) != expected_files:
        raise ValueError("Unexpected packed package payload")
    proof = strict_json(directory / "verification.json", JSON_LIMIT)
    check_proof(
        proof,
        expected["model_id"],
        admitted["image_shape"],
        admitted.get("prediction_horizon", 100),
        admitted.get("execution_horizon", 100),
    )
    sizes = {
        "source_weight_bytes": admitted["files"]["model.safetensors"]["bytes"],
        "packed_weight_bytes": policy["model.fbq"]["bytes"],
        "policy_package_bytes": sum(row["bytes"] for row in policy.values()),
    }
    if set(report) != set(expected) | set(sizes) | {
        "drift_from_fp32",
        "fixture_scope",
        "units",
        "gpu_memory_bytes",
        "inference_speedup",
    }:
        raise ValueError("Unexpected native packed report claims")
    if (
        any(
            type(report.get(key)) is not int or report[key] != value for key, value in sizes.items()
        )
        or not exact_json(report.get("drift_from_fp32"), proof["drift_from_fp32"])
        or report.get("fixture_scope") != SCOPE
        or report.get("units") != UNITS
        or any(
            report.get(key, "missing") is not None
            for key in ("gpu_memory_bytes", "inference_speedup")
        )
    ):
        raise ValueError("Packed report sizes, drift or unmeasured claims differ")
    if inventory(directory) != files:
        raise ValueError("Packed package changed while verifying its receipt")
    return manifest, files


def check_proof(proof, model_id, shape, prediction=100, execution=100):
    expected = {
        "schema_version": 1,
        "model_id": model_id,
        "scope": SCOPE,
        "units": UNITS,
        "fresh_packed_reload_exact": True,
        "full_chunk_queue_reset_verified": True,
        "floating_master_reads_blocked": True,
        "network_disabled": True,
        "source_read_protection": "Python open audit hook; not an OS filesystem sandbox",
        "quality_verified": False,
        "calibration_verified": False,
        "speedup_verified": False,
        "task_success": None,
        "gpu_memory_bytes": None,
        "inference_speedup": None,
    }
    if (
        not isinstance(proof, dict)
        or any(
            proof.get(key) != value or type(proof.get(key)) is not type(value)
            for key, value in expected.items()
        )
        or not isinstance(proof.get("versions"), dict)
        or {k: str(v).split("+")[0] for k, v in proof["versions"].items()} != RUNTIME
    ):
        raise ValueError("Incomplete native CPU reload evidence")
    for key, value in (("prediction_horizon", prediction), ("execution_horizon", execution)):
        if key in proof or (prediction, execution) != (100, 100):
            if type(proof.get(key)) is not int or proof[key] != value:
                raise ValueError("Proof temporal dimensions differ from saved policy")
    rows = []
    floating, packed = proof.get("floating"), proof.get("packed")
    if (
        not isinstance(floating, list)
        or not isinstance(packed, list)
        or len(floating) != 2
        or len(packed) != 2
    ):
        raise ValueError("Two complete generated fixture measurements are required")
    for seed, baseline, candidate in zip((171, 902), floating, packed, strict=True):
        for row in (baseline, candidate):
            if (
                not isinstance(row, dict)
                or type(row.get("seed")) is not int
                or row["seed"] != seed
                or row.get("image_shape") != shape
                or row.get("queue_and_reset_exact") is not True
                or not isinstance(row.get("input_sha256"), str)
                or not re.fullmatch(r"[a-f0-9]{64}", row["input_sha256"])
            ):
                raise ValueError("Generated fixture identity differs")
            for key in ("raw", "postprocessed"):
                chunk = row.get(key)
                if (
                    not isinstance(chunk, list)
                    or len(chunk) != prediction
                    or any(
                        not isinstance(action, list)
                        or len(action) != 6
                        or any(type(v) not in {int, float} or not math.isfinite(v) for v in action)
                        for action in chunk
                    )
                ):
                    raise ValueError("Expected finite complete prediction-horizon x 6 actions")
        if baseline["input_sha256"] != candidate["input_sha256"]:
            raise ValueError("Floating and packed fixtures differ")
        drift = {"seed": seed, "input_sha256": candidate["input_sha256"]}
        for key in ("raw", "postprocessed"):
            delta = [
                abs(a - b)
                for first, second in zip(baseline[key], candidate[key], strict=True)
                for a, b in zip(first, second, strict=True)
            ]
            drift[key] = {
                "rmse": math.sqrt(math.fsum(v * v for v in delta) / (prediction * 6)),
                "maximum_absolute_difference": max(delta),
                "coordinates": prediction * 6,
            }
        rows.append(drift)
    if not exact_json(proof.get("drift_from_fp32"), rows):
        raise ValueError("Reported drift does not match the measured full chunks")


def unchanged(source, data_dir, admitted):
    if source_info(source, data_dir) != admitted:
        raise ValueError("Source artifact changed during native quantization")


async def bounded_error(stream):
    captured = bytearray()
    while chunk := await stream.read(4096):
        captured.extend(chunk[: max(0, 4096 - len(captured))])
    return "".join(
        char
        for char in captured.decode("utf-8", errors="replace")
        if char.isprintable() or char == "\n"
    ).strip()


async def run(lifecycle, job):
    await validate(lifecycle, job.project_id, job.request)
    source = await lifecycle.artifact(job.project_id, job.request.artifact_id)
    admitted = await finish_owned(
        asyncio.to_thread(source_info, source, lifecycle.settings.data_dir)
    )
    runtime = lifecycle.runtime(job.request.runtime_id)
    stage = (lifecycle.settings.data_dir / "jobs" / job.id / "operation").absolute()
    stage.mkdir(parents=True, exist_ok=False)
    request_path, response_path = stage / "request.json", stage / "result.json"
    payload = {
        "schema_version": 1,
        "job_id": job.id,
        "operation": "policy.quantize",
        "source": {
            "path": str(admitted["path"]),
            "files": admitted["files"],
            "manifest_sha256": admitted["manifest_sha256"],
            "artifact_id": source.id,
            "artifact_manifest_sha256": source.manifest_sha256,
        },
        "output_dir": str(stage),
        "native_quantization": job.request.native_quantization.model_dump(),
        "timeout_seconds": job.request.timeout_seconds,
    }
    request_path.write_text(json.dumps(payload, allow_nan=False))
    argv, cwd, env = command(runtime, request_path, response_path)
    await lifecycle.event(
        job,
        "operation",
        "Packing ACT weights and checking fresh CPU reload; task quality remains unverified",
    )
    process, error_task, code = None, None, None
    try:
        process = await lifecycle.act_spawn_owned(
            *argv,
            cwd=cwd,
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            grace_seconds=8,
        )
        error_task = asyncio.create_task(bounded_error(process.stderr))
        code = await process.wait()
    finally:
        try:
            if process is not None:
                await lifecycle.act_stop_owned(process, grace_seconds=8)
        finally:
            try:
                if error_task is not None:
                    try:
                        await finish_owned(asyncio.wait_for(error_task, 2))
                    except TimeoutError:
                        pass
            finally:
                await finish_owned(
                    asyncio.to_thread(unchanged, source, lifecycle.settings.data_dir, admitted)
                )
    if code:
        detail = (
            error_task.result() if error_task is not None and not error_task.cancelled() else ""
        )
        raise ValueError(
            "Native ACT quantization failed; no package was registered. " + detail[:4096]
        )
    response = strict_json(response_path, JSON_LIMIT)
    destination = stage / "native-quantized"
    manifest, files = await finish_owned(
        asyncio.to_thread(check_result, response, job, source, admitted, destination)
    )
    artifact = PolicyArtifact(
        id=f"{job.id}:operation",
        project_id=job.project_id,
        job_id=job.id,
        label=f"ACT native INT{job.request.native_quantization.bits} (CPU reload checked)",
        format="native_quantized",
        path=destination.relative_to(lifecycle.settings.data_dir.absolute()).as_posix(),
        manifest_sha256=files["manifest.json"]["sha256"],
        file_bytes=sum(row["bytes"] for name, row in files.items() if name != "manifest.json"),
        parent_ids=[source.id],
        metadata=manifest["metadata"],
    )
    report = {
        **response["report"],
        "stage": "operation",
        "operation": "policy.quantize",
        "artifact_id": source.id,
        "final": False,
    }
    result = LifecycleResult(artifacts=[artifact], reports=[report])
    await lifecycle.publish(job, result)
    await lifecycle.event(
        job,
        "operation",
        "Packed ACT package ready to download; simulation and task quality are unverified",
    )
    return result
