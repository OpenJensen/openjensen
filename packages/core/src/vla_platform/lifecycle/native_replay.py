"""Registered packed ACT + explicit immutable observations → bounded CPU replay only."""

import asyncio
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
from types import SimpleNamespace

from vla_platform.contracts import DatasetProfile
from vla_platform.datasets.local_preview import _open_beneath
from vla_platform.datasets.snapshots import resolve_snapshot, stage_snapshot, verify_snapshot

from .contracts import LifecycleResult, PolicyArtifact
from .native_quantization import (
    JSON_LIMIT,
    RUNTIME,
    bundle,
    checked_path,
    exact_json,
    hash_file,
    inventory,
    model_identity,
)
from .simulation import finish_owned, strict_json

MAX_RGB = 1920 * 1920 * 3
MAX_TOTAL = 128 * 1024**2
RESULT_LIMIT = 8 * 1024**2
FLAGS = {
    "quality_verified": False,
    "calibration_verified": False,
    "speedup_verified": False,
    "isaac_runtime_verified": False,
    "task_success": None,
}
SCOPE = (
    "Predictions on immutable observed inputs; actions were not applied "
    "to any robot or simulator. No task-quality acceptance."
)


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def equal(value, expected, message):
    if not exact_json(value, expected):
        raise ValueError(message)


def keys(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("Unexpected or missing native replay fields")


def integer(value, low=0, high=2**53 - 1):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("Invalid replay integer")
    return value


def number(value, low=-3.4e38, high=3.4e38):
    if type(value) not in (int, float) or not low <= value <= high or not math.isfinite(value):
        raise ValueError("Invalid finite replay measurement")
    return value


def text(value, limit=256):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError("Invalid bounded replay text")
    return value


def file_digest(root, name, maximum):
    with _open_beneath(checked_path(root), Path(name)) as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise ValueError("Replay payload is empty, nonregular or oversized")
        digest, size = hashlib.sha256(), 0
        while chunk := stream.read(min(1024**2, before.st_size + 1 - size)):
            size += len(chunk)
            digest.update(chunk)
            if size > before.st_size:
                raise ValueError("Replay payload grew during verification")
        after = os.fstat(stream.fileno())
        if size != before.st_size or (before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("Replay payload changed during verification")
    return {"sha256": digest.hexdigest(), "bytes": size}


def source_info(artifact, data_dir):
    if artifact.format != "native_quantized" or artifact.metadata.get("architecture") != "act":
        raise ValueError("CPU observation replay requires a registered complete packed ACT policy")
    if artifact.metadata.get("remote") or artifact.metadata.get("storage") == "gcs":
        raise ValueError("CPU replay requires local immutable packed policy bytes")
    root = checked_path(data_dir.absolute() / artifact.path)
    if not root.is_relative_to((data_dir / "jobs" / artifact.job_id).absolute()):
        raise ValueError("Packed policy must remain within its owning application job")
    manifest, outer = bundle(root)
    equal(outer["manifest.json"]["sha256"], artifact.manifest_sha256, "Packed manifest changed")
    equal(manifest.get("metadata"), artifact.metadata, "Registered packed metadata changed")
    model = checked_path(root / "policy")
    files = inventory(model)
    if len(files) > 16 or any("/" in name for name in files):
        raise ValueError("Replay requires one flat complete packed policy")
    required = {
        "config.json",
        "model.fbq",
        "encoding.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    }
    if not required <= set(files):
        raise ValueError("Replay policy is missing weights, encoding or saved processors")
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        processor = strict_json(model / name, JSON_LIMIT)
        if not isinstance(processor.get("steps"), list):
            raise ValueError("Saved ACT processors must contain steps")
        for step in processor["steps"]:
            if not isinstance(step, dict):
                raise ValueError("Invalid ACT processor step")
            state_file = step.get("state_file")
            if state_file:
                if not isinstance(state_file, str) or state_file not in files or "/" in state_file:
                    raise ValueError("ACT processor state must be within the packed policy")
                required.add(state_file)
    if "temporal-contract.json" in files:
        required.add("temporal-contract.json")
    if set(files) != required:
        raise ValueError("Packed replay cannot contain floating masters or extra policy files")
    encoding = strict_json(model / "encoding.json", JSON_LIMIT)
    bits = (
        encoding.get("recipe", {}).get("bits") if isinstance(encoding.get("recipe"), dict) else None
    )
    if type(bits) is not int or bits not in (4, 8):
        raise ValueError("Only reviewed INT8/INT4 ACT encodings are supported")
    equal(
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
        "Packed encoding differs from the supported CPU recipe",
    )
    metadata = artifact.metadata
    identity = model_identity(files)
    for name, value in {
        "model_id": identity,
        "format": "firebird_quant",
        "format_version": 1,
        "precision": f"int{bits}",
        "policy_subdirectory": "policy",
        "inference_only": True,
        "training_resume_supported": False,
        **FLAGS,
    }.items():
        equal(metadata.get(name), value, "Packed metadata cannot promote unsupported claims")
    config = strict_json(model / "config.json", JSON_LIMIT)
    for name, value in {
        "type": "act",
        "use_vae": False,
    }.items():
        equal(config.get(name), value, "Replay requires the reviewed ACT inference configuration")
    from .native_quantization import temporal_claims, temporal_info

    temporal = temporal_info(config, model, files)
    for key, value in temporal_claims(metadata, temporal).items():
        equal(metadata.get(key), value, "Packed temporal metadata differs from config")
    features = config.get("input_features")
    outputs = config.get("output_features")
    if not isinstance(features, dict) or not isinstance(outputs, dict):
        raise ValueError("ACT feature configuration is missing")
    cameras = [k for k in features if k.startswith("observation.images.")]
    if len(cameras) != 1:
        raise ValueError("Replay requires one exact camera")
    equal(
        features.get("observation.state", {}).get("shape"),
        [6],
        "Six ACT state coordinates required",
    )
    equal(outputs.get("action", {}).get("shape"), [6], "Six ACT action coordinates required")
    shape = features[cameras[0]].get("shape")
    if not isinstance(shape, list) or len(shape) != 3 or shape[0] != 3:
        raise ValueError("Expected original RGB input shape")
    for dimension in shape:
        integer(dimension, 2, 1920)
    if inventory(model) != files or inventory(root) != outer:
        raise ValueError("Packed source changed during admission")
    return {
        "path": model,
        "files": files,
        "model_id": identity,
        "outer": outer,
        "camera": cameras[0],
        "image_shape": shape,
        **temporal,
    }


def dataset_info(profile, store, admitted, recipe):
    descriptor = profile.snapshot.model_dump()
    root = resolve_snapshot(store, descriptor)
    manifest = verify_snapshot(root, descriptor["manifest_sha256"])
    features = manifest["features"]
    shape = admitted["image_shape"]
    equal(
        features.get(admitted["camera"], {}).get("shape"),
        [shape[1], shape[2], 3],
        "Snapshot camera must match the packed policy exactly; no resizing",
    )
    state, action = features.get("observation.state", {}), features.get("action", {})
    names = state.get("names")
    if (
        state.get("shape") != [6]
        or action.get("shape") != [6]
        or not isinstance(names, list)
        or len(names) != 6
        or any(not isinstance(n, str) or not n.strip() or len(n) > 256 for n in names)
        or len(set(names)) != 6
        or action.get("names") != names
    ):
        raise ValueError("Six matching explicit state/action names and order are required")
    selected = [row.model_dump() for row in recipe.selection]
    pairs = [
        (
            integer(row["episode_index"], 0, manifest["total_episodes"] - 1),
            integer(row["frame_index"], 0, min(10**7, manifest["total_frames"] - 1)),
        )
        for row in selected
    ]
    if not 1 <= len(pairs) <= 32 or len(set(pairs)) != len(pairs):
        raise ValueError("Select1..32 unique explicit episode/frame observations")
    if len(pairs) * shape[1] * shape[2] * 3 > MAX_TOTAL:
        raise ValueError("Selected replay observations exceed128MiB before decoding")
    lineage = {row["episode_index"]: row for row in manifest.get("lineage", [])}
    origins = {lineage.get(e, {}).get("origin", "imported") for e, _ in pairs}
    generated = "synthetic" in origins
    if generated and origins != {"synthetic"}:
        raise ValueError("Generated and recorded replay inputs cannot be mixed")
    expected = "generated_fixture" if generated else "policy_recorded_coordinates"
    if recipe.coordinate_attestation != expected:
        raise ValueError("Explicit coordinate attestation must match dataset provenance")
    units = recipe.units
    if not isinstance(units, list) or len(units) != 6:
        raise ValueError("Six explicit coordinate units are required")
    for unit in units:
        text(unit, 80)
    return {
        "root": root,
        "descriptor": descriptor,
        "manifest": manifest,
        "selection": selected,
        "lineage": lineage,
        "source": {
            "kind": "generated_fixture" if generated else "lerobot_snapshot",
            "identity": descriptor["id"],
            "manifest_sha256": descriptor["manifest_sha256"],
        },
        "semantics": {
            "state_names": names,
            "action_names": names,
            "units": units,
            "compatibility": "generated_fixture"
            if generated
            else "operator_attested_policy_recorded_coordinates",
        },
    }


async def admission(lifecycle, project_id, request):
    from .runtime import native_replay_ready

    runtime = lifecycle.runtime(request.runtime_id)
    if (
        runtime is None
        or runtime.execution != "native"
        or runtime.provider != "local"
        or runtime.device != "cpu"
        or not native_replay_ready(runtime)
    ):
        raise ValueError("Configure a ready local CPU packed ACT replay and dataset-reader runtime")
    lifecycle.compute.require_enabled(runtime)
    source = await lifecycle.artifact(project_id, request.artifact_id)
    dataset = await lifecycle.execution.get(request.dataset_job_id)
    if (
        dataset is None
        or dataset.project_id != project_id
        or dataset.status != "succeeded"
        or not isinstance(dataset.result, DatasetProfile)
        or dataset.result.snapshot is None
        or dataset.result.inspection_scope != "complete_snapshot"
    ):
        raise ValueError("Replay requires a completed immutable dataset snapshot in this project")
    try:
        admitted = await finish_owned(
            asyncio.to_thread(source_info, source, lifecycle.settings.data_dir)
        )
        data = await finish_owned(
            asyncio.to_thread(
                dataset_info,
                dataset.result,
                lifecycle.settings.data_dir / "dataset-snapshots",
                admitted,
                request.native_replay,
            )
        )
    except (OSError, KeyError, TypeError, AttributeError) as error:
        raise ValueError(
            "Complete packed policy or dataset snapshot is missing/unreadable"
        ) from error
    return runtime, source, admitted, data


async def validate(lifecycle, project_id, request):
    await admission(lifecycle, project_id, request)


def command(runtime, mode, request, result):
    from .runtime import native_replay_ready

    if mode not in {"prepare", "run"} or not native_replay_ready(runtime):
        raise ValueError("Fixed CPU replay worker is not configured")
    root = Path(runtime.native_replay_root).resolve()
    env = {k: os.environ[k] for k in ("PATH", "TMPDIR", "SYSTEMROOT", "WINDIR") if k in os.environ}
    env.update(
        PYTHONPATH=os.pathsep.join(
            [
                str(root),
                str(root.parent / "firebird_quant/src"),
                str(root.parent / "act_optimizer/src"),
                str(root.parent / "smolvla_qlora/src"),
            ]
        ),
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
        HF_HOME=str(result.parent / "offline-cache"),
        HF_DATASETS_CACHE=str(result.parent / "offline-cache/datasets"),
    )
    python = (
        runtime.native_replay_dataset_python if mode == "prepare" else runtime.native_replay_python
    )
    module = "native_replay_prepare" if mode == "prepare" else "native_replay"
    return [python, "-m", "sim_worker.rollout." + module, str(request), str(result)], str(root), env


def implementation_identity(runtime):
    root = Path(runtime.native_replay_root).resolve()
    replay = root / "sim_worker/rollout"
    result = {
        "replay/" + name: hash_file(replay / name)["sha256"]
        for name in (
            "native_replay.py",
            "native_replay_contracts.py",
            "native_replay_runtime.py",
            "native_replay_prepare.py",
        )
    }
    for package, directory, names in (
        (
            "firebird_quant",
            root.parent / "firebird_quant/src/firebird_quant",
            ("native_application", "native_consumer", "native_package"),
        ),
        ("sim_worker.rollout", replay, ("backend", "client", "server")),
    ):
        for name in names:
            result[package + "." + name] = hash_file(directory / (name + ".py"))["sha256"]
    return result


def check_corpus(directory, response, data, admitted):
    keys(
        response,
        {
            "schema_version",
            "path",
            "manifest_sha256",
            "files",
            "observations",
            "source",
            "scope",
            "episode_lengths",
        },
    )
    equal(response["schema_version"], 1, "Invalid preparation schema")
    equal(response["path"], str(directory), "Preparation must return its exact owned directory")
    manifest_file = file_digest(directory, "manifest.json", JSON_LIMIT)
    equal(response["manifest_sha256"], manifest_file["sha256"], "Prepared manifest changed")
    doc = strict_json(directory / "manifest.json", JSON_LIMIT)
    keys(doc, {"schema_version", "format", "source", "camera_key", "semantics", "samples"})
    for key, expected in {
        "schema_version": 1,
        "format": "native-policy-observations-v1",
        "source": data["source"],
        "camera_key": admitted["camera"],
        "semantics": data["semantics"],
    }.items():
        equal(doc[key], expected, "Prepared input identity or coordinates differ")
    samples = doc["samples"]
    if not isinstance(samples, list) or len(samples) != len(data["selection"]):
        raise ValueError("Prepared observations omit explicit selected frames")
    equal(response["observations"], len(samples), "Prepared observation count differs")
    equal(response["source"], data["source"], "Prepared source differs")
    lengths = response["episode_lengths"]
    if not isinstance(lengths, list):
        raise ValueError("Native episode lengths are missing")
    expected_episodes = sorted({r["episode_index"] for r in data["selection"]})
    if len(lengths) != len(expected_episodes):
        raise ValueError("Native episode lengths do not cover the selection")
    native_lengths = {}
    for row, episode in zip(lengths, expected_episodes, strict=True):
        keys(row, {"episode_index", "length"})
        equal(row["episode_index"], episode, "Native episode length identity differs")
        native_lengths[episode] = integer(row["length"], 1, data["manifest"]["total_frames"])
    if sum(native_lengths.values()) > data["manifest"]["total_frames"]:
        raise ValueError("Selected native episode lengths exceed immutable snapshot size")
    for row in data["selection"]:
        if row["frame_index"] >= native_lengths[row["episode_index"]]:
            raise ValueError("Selected frame exceeds native episode length")
    files = {"manifest.json": manifest_file}
    width, height = admitted["image_shape"][2], admitted["image_shape"][1]
    if len(samples) * width * height * 3 > MAX_TOTAL:
        raise ValueError("Prepared RGB exceeds128MiB")
    for index, (row, selected) in enumerate(zip(samples, data["selection"], strict=True)):
        keys(
            row,
            {
                "episode_index",
                "frame_index",
                "timestamp_seconds",
                "task",
                "state",
                "origin",
                "lineage_group",
                "image",
            },
        )
        for key, expected in selected.items():
            equal(row[key], expected, "Prepared native frame selection differs")
        ancestry = data["lineage"].get(row["episode_index"], {})
        equal(row["origin"], ancestry.get("origin", "imported"), "Observation origin changed")
        equal(row["lineage_group"], ancestry.get("lineage_group"), "Observation lineage changed")
        number(row["timestamp_seconds"], 0, 1e7)
        text(row["task"], 4096)
        if not isinstance(row["state"], list) or len(row["state"]) != 6:
            raise ValueError("Six original state coordinates required")
        for coordinate in row["state"]:
            number(coordinate)
        image = row["image"]
        keys(image, {"file", "width", "height", "sha256", "bytes"})
        for key, expected in {
            "file": f"frame-{index:06d}.rgb",
            "width": width,
            "height": height,
            "bytes": width * height * 3,
        }.items():
            equal(image[key], expected, "Prepared original RGB shape/path differs")
        item = file_digest(directory, image["file"], MAX_RGB)
        equal(
            item, {"bytes": image["bytes"], "sha256": image["sha256"]}, "Prepared RGB bytes changed"
        )
        files[image["file"]] = item
    if {p.name for p in directory.iterdir()} != set(files):
        raise ValueError("Unexpected observation payload")
    equal(response["files"], files, "Prepared file inventory differs")
    return doc, files


def predictions(value, admitted, doc):
    expected = {
        "schema_version": 1,
        "model_id": admitted["model_id"],
        "device": "cpu",
        "mode": "independent_observation_replay",
        "server_closed": True,
        "external_network_disabled": True,
        "floating_master_reads_blocked": True,
        **FLAGS,
    }
    keys(value, {*expected, "versions", "records"})
    for key, item in expected.items():
        equal(value[key], item, "Replay prediction identity/lifecycle/claims differ")
    keys(value["versions"], RUNTIME)
    for name, pin in RUNTIME.items():
        actual = value["versions"][name]
        if not isinstance(actual, str) or actual.split("+")[0] != pin:
            raise ValueError("Replay runtime differs from reviewed CPU pins")
    records = value["records"]
    if not isinstance(records, list) or len(records) != len(doc["samples"]):
        raise ValueError("Replay predictions omit selected inputs")
    for index, (row, sample) in enumerate(zip(records, doc["samples"], strict=True)):
        keys(
            row,
            {
                "sample_index",
                "episode_index",
                "frame_index",
                "timestamp_seconds",
                "input_rgb_sha256",
                "input_state_sha256",
                "actions",
                "reset_repeat_exact",
                "requests",
            },
        )
        for name, item in {
            "sample_index": index,
            "episode_index": sample["episode_index"],
            "frame_index": sample["frame_index"],
            "timestamp_seconds": sample["timestamp_seconds"],
            "input_rgb_sha256": sample["image"]["sha256"],
            "input_state_sha256": hashlib.sha256(canonical(sample["state"])).hexdigest(),
            "reset_repeat_exact": True,
        }.items():
            equal(row[name], item, "Replay predictions differ from exact selected observation")
        actions = row["actions"]
        if not isinstance(actions, list) or len(actions) != admitted.get("prediction_horizon", 100):
            raise ValueError("Replay requires full prediction-horizon x 6 actions")
        for action in actions:
            if not isinstance(action, list) or len(action) != 6:
                raise ValueError("Replay requires full prediction-horizon x 6 actions")
            for coordinate in action:
                number(coordinate)
        if not isinstance(row["requests"], list) or len(row["requests"]) != 2:
            raise ValueError("Two reset comparisons are required")
        for request in row["requests"]:
            keys(request, {"seconds", "actions_sha256"})
            number(request["seconds"], 0, 600)
            equal(
                request["actions_sha256"],
                hashlib.sha256(canonical(actions)).hexdigest(),
                "Replay reset outputs differ",
            )


def check_result(response, job, source, admitted, data, doc, corpus_files, destination):
    keys(response, {"schema_version", "job_id", "operation", "artifact", "report"})
    for key, expected in {"schema_version": 1, "job_id": job.id, "operation": "policy.run"}.items():
        equal(response[key], expected, "Replay response identity differs")
    equal(
        response["artifact"],
        {
            "path": str(destination),
            "format": "native_run_record",
            "label": "CPU observation replay",
        },
        "Replay must return its exact owned report artifact",
    )
    # The complete prediction payload is larger than ordinary JSON metadata at32observations.
    names = {"manifest.json", "predictions.json", "lineage.json", "report.json"}
    if {p.name for p in destination.iterdir()} != names:
        raise ValueError("Unexpected replay report payload")
    files = {name: file_digest(destination, name, RESULT_LIMIT) for name in names}
    manifest = strict_json(destination / "manifest.json", JSON_LIMIT)
    expected_metadata = {
        "architecture": "act",
        "model_id": admitted["model_id"],
        "recipe": "native-observation-replay-v1",
        "device": "cpu",
        **FLAGS,
    }
    equal(
        manifest,
        {
            "schema_version": 1,
            "metadata": expected_metadata,
            "files": {
                name: row["sha256"] for name, row in files.items() if name != "manifest.json"
            },
        },
        "Replay report manifest differs",
    )
    value = strict_json(destination / "predictions.json", RESULT_LIMIT)
    predictions(value, admitted, doc)
    equal(
        strict_json(destination / "lineage.json", RESULT_LIMIT),
        {
            "schema_version": 1,
            "source": {
                "files": admitted["files"],
                "model_id": admitted["model_id"],
                "artifact_id": source.id,
                "artifact_manifest_sha256": source.manifest_sha256,
            },
            "observation_manifest_sha256": corpus_files["manifest.json"]["sha256"],
            "observation_files": corpus_files,
            "observations": doc,
            "implementation_sha256": data["implementation"],
        },
        "Replay lineage differs from exact admitted inputs and worker",
    )
    report = response["report"]
    expected_report = {
        "stage": "native_replay",
        "mode": "independent_observation_replay",
        "model_id": admitted["model_id"],
        "device": "cpu",
        "versions": value["versions"],
        "observation_source": data["source"],
        "observations": len(doc["samples"]),
        "action_shape": [admitted.get("prediction_horizon", 100), 6],
        "reset_repeat_exact": True,
        "coordinate_semantics": data["semantics"],
        "server_closed": True,
        **FLAGS,
        "scope": SCOPE,
    }
    keys(report, {*expected_report, "elapsed_seconds"})
    for key, expected in expected_report.items():
        equal(
            report[key],
            expected,
            "Replay report cannot promote unmeasured quality or change identity",
        )
    number(report["elapsed_seconds"], 0, job.request.timeout_seconds)
    equal(
        strict_json(destination / "report.json", JSON_LIMIT),
        report,
        "Returned report differs from saved report",
    )
    if {name: file_digest(destination, name, RESULT_LIMIT) for name in names} != files:
        raise ValueError("Replay output changed during verification")
    return manifest, files


async def error_tail(stream):
    result = bytearray()
    while part := await stream.read(4096):
        result.extend(part)
        del result[:-4096]
    return "".join(
        c for c in result.decode("utf-8", errors="replace") if c.isprintable() or c == "\n"
    )


async def execute(lifecycle, runtime, mode, request, result):
    argv, cwd, env = command(runtime, mode, request, result)
    process, drain, code = None, None, None
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
        drain = asyncio.create_task(error_tail(process.stderr))
        code = await process.wait()
    finally:
        try:
            if process is not None:
                await lifecycle.act_stop_owned(process, grace_seconds=8)
        finally:
            if drain is not None:
                try:
                    await finish_owned(asyncio.wait_for(drain, 2))
                except TimeoutError:
                    pass
    if code:
        detail = drain.result() if drain is not None and not drain.cancelled() else ""
        raise ValueError("Offline CPU replay stage failed: " + detail[-4096:])
    return strict_json(result, RESULT_LIMIT)


def unchanged(source, admitted, data, data_dir):
    if source_info(source, data_dir) != admitted:
        raise ValueError("Original packed source changed during replay")
    equal(
        verify_snapshot(data["root"], data["descriptor"]["manifest_sha256"]),
        data["manifest"],
        "Original dataset snapshot changed during replay",
    )


async def run(lifecycle, job):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + job.request.timeout_seconds
    async with asyncio.timeout(job.request.timeout_seconds):
        runtime, source, admitted, data = await admission(lifecycle, job.project_id, job.request)
        stage = checked_path(
            (lifecycle.settings.data_dir / "jobs" / job.id / "operation").absolute()
        )
        stage.mkdir(parents=True, exist_ok=False)
        data["implementation"] = await finish_owned(
            asyncio.to_thread(implementation_identity, runtime)
        )
        exact_source = {
            "path": str(admitted["path"]),
            "files": admitted["files"],
            "model_id": admitted["model_id"],
            "artifact_id": source.id,
            "artifact_manifest_sha256": source.manifest_sha256,
        }
        try:
            snapshot = await finish_owned(
                asyncio.to_thread(
                    stage_snapshot,
                    data["root"],
                    stage / "snapshot",
                    data["descriptor"]["manifest_sha256"],
                )
            )
            prepare = {
                "schema_version": 1,
                "source": exact_source,
                "dataset_snapshot": {
                    "path": str(snapshot),
                    "id": data["descriptor"]["id"],
                    "manifest_sha256": data["descriptor"]["manifest_sha256"],
                },
                "selection": data["selection"],
                "semantics": data["semantics"],
                "output_dir": str(stage / "observations"),
            }
            prepare_path, prepared_path = stage / "prepare-request.json", stage / "prepared.json"
            prepare_path.write_bytes(canonical(prepare))
            await lifecycle.event(
                job,
                "replaying",
                "Reading explicitly selected immutable observations; no actions are applied",
            )
            prepared = await execute(lifecycle, runtime, "prepare", prepare_path, prepared_path)
            doc, corpus_files = await finish_owned(
                asyncio.to_thread(check_corpus, stage / "observations", prepared, data, admitted)
            )
            remaining = math.floor(deadline - loop.time())
            if remaining < 1:
                raise TimeoutError("Replay deadline exceeded during observation preparation")
            payload = {
                "schema_version": 1,
                "job_id": job.id,
                "operation": "policy.run",
                "source": exact_source,
                "observations": {
                    "path": str(stage / "observations"),
                    "manifest_sha256": corpus_files["manifest.json"]["sha256"],
                },
                "output_dir": str(stage / "worker-output"),
                "timeout_seconds": min(remaining, 600),
            }
            request_path, result_path = stage / "request.json", stage / "result.json"
            request_path.write_bytes(canonical(payload))
            await lifecycle.event(
                job, "replaying", "Computing full ACT chunks on CPU; actions remain unexecuted"
            )
            response = await execute(lifecycle, runtime, "run", request_path, result_path)
            destination = stage / "worker-output/native-run"
            manifest, files = await finish_owned(
                asyncio.to_thread(
                    check_result,
                    response,
                    job,
                    source,
                    admitted,
                    data,
                    doc,
                    corpus_files,
                    destination,
                )
            )
            await finish_owned(
                asyncio.to_thread(check_corpus, stage / "observations", prepared, data, admitted)
            )
        finally:
            try:
                equal(
                    await finish_owned(asyncio.to_thread(implementation_identity, runtime)),
                    data["implementation"],
                    "Replay worker source changed",
                )
            finally:
                await finish_owned(
                    asyncio.to_thread(
                        unchanged, source, admitted, data, lifecycle.settings.data_dir
                    )
                )
        artifact = PolicyArtifact(
            id=f"{job.id}:operation",
            project_id=job.project_id,
            job_id=job.id,
            label="CPU observation replay (actions not executed)",
            format="native_run_record",
            path=destination.relative_to(lifecycle.settings.data_dir.resolve()).as_posix(),
            manifest_sha256=files["manifest.json"]["sha256"],
            file_bytes=sum(v["bytes"] for k, v in files.items() if k != "manifest.json"),
            parent_ids=[source.id],
            metadata=manifest["metadata"],
        )
        report = {
            **response["report"],
            "operation": "policy.run",
            "source_artifact_id": source.id,
            "source_artifact_manifest_sha256": source.manifest_sha256,
            "dataset_job_id": job.request.dataset_job_id,
            "dataset_snapshot_id": data["descriptor"]["id"],
        }
        result = LifecycleResult(artifacts=[artifact], reports=[report])
        await lifecycle.publish(job, result)
        await lifecycle.event(
            job, "replaying", "CPU replay report verified; robot task quality is unverified"
        )
        return result


def historical_record(data_dir, artifact, job):
    """Verify a completed saved record without depending on current model availability."""
    directory = checked_path(data_dir.absolute() / artifact.path)
    expected = (data_dir / "jobs" / job.id / "operation/worker-output/native-run").absolute()
    if directory != expected:
        raise ValueError("Replay record is outside its exact owning job directory")
    names = {"manifest.json", "report.json", "lineage.json", "predictions.json"}
    if {p.name for p in directory.iterdir()} != names:
        raise ValueError("Replay record has unexpected payload")
    files = {name: file_digest(directory, name, RESULT_LIMIT) for name in names}
    equal(
        files["manifest.json"]["sha256"],
        artifact.manifest_sha256,
        "Registered replay manifest changed",
    )
    if (
        sum(item["bytes"] for name, item in files.items() if name != "manifest.json")
        != artifact.file_bytes
    ):
        raise ValueError("Registered replay byte count changed")
    manifest = strict_json(directory / "manifest.json", JSON_LIMIT)
    equal(manifest.get("metadata"), artifact.metadata, "Registered replay metadata changed")
    report = strict_json(directory / "report.json", JSON_LIMIT)
    lineage = strict_json(directory / "lineage.json", RESULT_LIMIT)
    keys(
        lineage,
        {
            "schema_version",
            "source",
            "observation_manifest_sha256",
            "observation_files",
            "observations",
            "implementation_sha256",
        },
    )
    keys(lineage["source"], {"files", "model_id", "artifact_id", "artifact_manifest_sha256"})
    model_id = artifact.metadata.get("model_id")
    if not isinstance(model_id, str) or not re.fullmatch(r"sha256:[a-f0-9]{64}", model_id):
        raise ValueError("Invalid historical replay model identity")
    matches = [r for r in job.result.reports if r.get("stage") == "native_replay"]
    if len(matches) != 1:
        raise ValueError("Completed replay job has no unique saved report")
    saved = matches[0]
    source_sha = saved.get("source_artifact_manifest_sha256")
    if not isinstance(source_sha, str) or not re.fullmatch(r"[a-f0-9]{64}", source_sha):
        raise ValueError("Invalid saved packed source binding")
    equal(
        saved,
        {
            **report,
            "operation": "policy.run",
            "source_artifact_id": job.request.artifact_id,
            "source_artifact_manifest_sha256": source_sha,
            "dataset_job_id": job.request.dataset_job_id,
            "dataset_snapshot_id": report.get("observation_source", {}).get("identity"),
        },
        "Saved job report differs from immutable replay artifact",
    )
    equal(artifact.parent_ids, [job.request.artifact_id], "Replay source artifact link changed")
    doc = lineage["observations"]
    keys(doc, {"schema_version", "format", "source", "camera_key", "semantics", "samples"})
    equal(doc["schema_version"], 1, "Invalid historical observation schema")
    equal(doc["format"], "native-policy-observations-v1", "Invalid historical observation format")
    keys(doc["source"], {"kind", "identity", "manifest_sha256"})
    if doc["source"]["kind"] not in {"lerobot_snapshot", "generated_fixture"}:
        raise ValueError("Invalid historical observation origin")
    for name in ("identity", "manifest_sha256"):
        text(doc["source"][name])
    if not re.fullmatch(r"[a-f0-9]{64}", doc["source"]["manifest_sha256"]):
        raise ValueError("Invalid historical snapshot fingerprint")
    equal(doc["source"], report["observation_source"], "Historical snapshot identity differs")
    semantics = doc["semantics"]
    keys(semantics, {"state_names", "action_names", "units", "compatibility"})
    for field in ("state_names", "action_names", "units"):
        if not isinstance(semantics[field], list) or len(semantics[field]) != 6:
            raise ValueError("Six historical coordinate names and units required")
        for item in semantics[field]:
            text(item)
    if (
        len(set(semantics["state_names"])) != 6
        or semantics["state_names"] != semantics["action_names"]
    ):
        raise ValueError("Historical coordinate ordering differs")
    equal(semantics["units"], job.request.native_replay.units, "Requested coordinate units changed")
    generated = job.request.native_replay.coordinate_attestation == "generated_fixture"
    equal(
        doc["source"]["kind"],
        "generated_fixture" if generated else "lerobot_snapshot",
        "Observation provenance changed",
    )
    equal(
        semantics["compatibility"],
        "generated_fixture" if generated else "operator_attested_policy_recorded_coordinates",
        "Coordinate attestation changed",
    )
    samples = doc["samples"]
    selection = job.request.native_replay.selection
    if (
        not isinstance(samples, list)
        or len(samples) != len(selection)
        or not 1 <= len(samples) <= 32
    ):
        raise ValueError("Historical selected observations are incomplete")
    expected_files = {
        "manifest.json": {
            "sha256": hashlib.sha256(canonical(doc)).hexdigest(),
            "bytes": len(canonical(doc)),
        }
    }
    total = 0
    for index, (row, selected) in enumerate(zip(samples, selection, strict=True)):
        keys(
            row,
            {
                "episode_index",
                "frame_index",
                "timestamp_seconds",
                "task",
                "state",
                "origin",
                "lineage_group",
                "image",
            },
        )
        equal(row["episode_index"], selected.episode_index, "Historical episode selection changed")
        equal(row["frame_index"], selected.frame_index, "Historical frame selection changed")
        if (
            row["origin"] not in {"recorded", "imported", "augmented", "synthetic"}
            or (row["origin"] == "synthetic") != generated
        ):
            raise ValueError("Historical origin changed")
        if row["lineage_group"] is not None:
            text(row["lineage_group"], 160)
        number(row["timestamp_seconds"], 0, 1e7)
        text(row["task"], 4096)
        if not isinstance(row["state"], list) or len(row["state"]) != 6:
            raise ValueError("Invalid historical state shape")
        for value in row["state"]:
            number(value)
        image = row["image"]
        keys(image, {"file", "width", "height", "sha256", "bytes"})
        equal(image["file"], f"frame-{index:06d}.rgb", "Historical image path differs")
        width, height = integer(image["width"], 2, 1920), integer(image["height"], 2, 1920)
        equal(image["bytes"], width * height * 3, "Historical RGB size differs")
        if not isinstance(image["sha256"], str) or not re.fullmatch(
            r"[a-f0-9]{64}", image["sha256"]
        ):
            raise ValueError("Invalid historical image fingerprint")
        total += image["bytes"]
        if total > MAX_TOTAL:
            raise ValueError("Historical RGB budget exceeded")
        expected_files[image["file"]] = {"bytes": image["bytes"], "sha256": image["sha256"]}
    equal(lineage["observation_files"], expected_files, "Historical observation inventory differs")
    equal(
        lineage["observation_manifest_sha256"],
        expected_files["manifest.json"]["sha256"],
        "Historical observation fingerprint differs",
    )
    admitted = {"model_id": model_id, "files": lineage["source"]["files"]}
    data = {
        "source": doc["source"],
        "semantics": semantics,
        "implementation": lineage["implementation_sha256"],
    }
    source = SimpleNamespace(id=job.request.artifact_id, manifest_sha256=source_sha)
    response = {
        "schema_version": 1,
        "job_id": job.id,
        "operation": "policy.run",
        "artifact": {
            "path": str(directory),
            "format": "native_run_record",
            "label": "CPU observation replay",
        },
        "report": report,
    }
    _, checked = check_result(response, job, source, admitted, data, doc, expected_files, directory)
    if checked != files:
        raise ValueError("Historical replay changed while reading")
    records = strict_json(directory / "predictions.json", RESULT_LIMIT)
    # Bind bytes once more after the final parse; no original model or dataset access.
    equal(
        file_digest(directory, "predictions.json", RESULT_LIMIT),
        files["predictions.json"],
        "Historical predictions changed",
    )
    return {
        "artifact_id": artifact.id,
        "job_id": job.id,
        "model_id": model_id,
        "source_kind": doc["source"]["kind"],
        "coordinate_names": semantics["action_names"],
        "units": semantics["units"],
        "records": records["records"],
    }


async def read_record(lifecycle, project_id, artifact_id):
    artifact = await lifecycle.artifact(project_id, artifact_id)
    job = await lifecycle.execution.get(artifact.job_id)
    if (
        artifact.format != "native_run_record"
        or artifact.project_id != project_id
        or job is None
        or job.project_id != project_id
        or job.status != "succeeded"
        or job.kind != "policy.run"
        or job.request.native_replay is None
        or not isinstance(job.result, LifecycleResult)
    ):
        raise ValueError("A succeeded CPU replay record in this project is required")
    recorded = [a for a in job.result.artifacts if a.id == artifact.id]
    if len(recorded) != 1 or recorded[0] != artifact:
        raise ValueError("Replay artifact differs from its completed job registration")
    try:
        return await finish_owned(
            asyncio.to_thread(historical_record, lifecycle.settings.data_dir, artifact, job)
        )
    except (OSError, KeyError, TypeError, AttributeError) as error:
        raise ValueError("Saved CPU replay record is unavailable or invalid") from error
