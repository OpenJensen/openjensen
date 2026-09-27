"""Verified snapshot → offline ACT student, with ML confined to fixed worker processes."""

import asyncio
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path

from vla_platform.contracts import DatasetProfile
from vla_platform.datasets.local_preview import _open_beneath
from vla_platform.datasets.snapshots import (
    resolve_snapshot,
    stage_snapshot,
    verify_snapshot,
)

from .contracts import LifecycleResult, PolicyArtifact
from .native_quantization import (
    JSON_LIMIT,
    RUNTIME,
    bundle,
    checked_path,
    exact_json,
    hash_file,
    source_info,
)
from .runtime import native_distillation_ready
from .simulation import finish_owned, strict_json

SHA = re.compile(r"[a-f0-9]{64}\Z")


def canonical(value):
    """Match the isolated ACT worker byte format, including indentation/newline."""
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


RECIPE_KEYS = {"adapter", "student", "steps", "learning_rate", "seed"}
SPLITS = {"train", "validation", "final"}


def equal(actual, expected, message):
    if not exact_json(actual, expected):
        raise ValueError(message)


def integer(value, low=1, high=2**63 - 1):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("Invalid integer in distillation evidence")
    return value


def finite(value, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Invalid finite measurement in distillation evidence")
    if positive and value == 0:
        raise ValueError("Distillation measurement must be positive")
    return value


def sha(value):
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise ValueError("Invalid distillation SHA256")
    return value


def file_digest(root, name, maximum):
    """Do not allow prepared samples to inherit the smaller JSON metadata limit."""
    root = checked_path(root)
    with _open_beneath(root, Path(name)) as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise ValueError("Prepared corpus file is empty, nonregular or oversized")
        digest, size = hashlib.sha256(), 0
        while chunk := stream.read(min(1024**2, before.st_size + 1 - size)):
            size += len(chunk)
            digest.update(chunk)
            if size > before.st_size:
                raise ValueError("Prepared corpus grew during verification")
        after = os.fstat(stream.fileno())
        if size != before.st_size or (before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("Prepared corpus changed during verification")
    return {"sha256": digest.hexdigest(), "bytes": size}


def dataset_info(profile, store, admitted, recipe):
    descriptor = profile.snapshot.model_dump()
    root = resolve_snapshot(store, descriptor)
    manifest = verify_snapshot(root, descriptor["manifest_sha256"])
    if manifest.get("lineage_validated") is not True:
        raise ValueError("Distillation requires explicit episode lineage")
    config = strict_json(admitted["path"] / "config.json", JSON_LIMIT)
    if integer(config.get("dim_model"), 1) <= 256 or integer(config.get("n_encoder_layers")) < 2:
        raise ValueError("ACT teacher must be larger than the fixed ACT256 student")
    camera = next(k for k in config["input_features"] if k.startswith("observation.images."))
    shape = admitted["image_shape"]
    features = manifest["features"]
    equal(
        features.get(camera, {}).get("shape"),
        [shape[1], shape[2], 3],
        "Snapshot camera must match the teacher exactly; no implicit resize",
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
        raise ValueError("Six matching explicit state/action coordinate names are required")
    lineage = {row["episode_index"]: row for row in manifest["lineage"]}
    selected, groups, origins = {}, {}, set()
    for split, episodes in recipe.splits.model_dump().items():
        for episode in episodes:
            if episode not in lineage or episode in selected:
                raise ValueError("Selected distillation episode is absent or repeated")
            row = lineage[episode]
            group = row["lineage_group"]
            if groups.setdefault(group, split) != split:
                raise ValueError("A lineage group cannot cross distillation splits")
            selected[episode] = (split, group)
            origins.add(row["origin"])
    if len(groups) < 3 or set(groups.values()) != SPLITS:
        raise ValueError("Three explicit lineage groups are required for train/validation/final")
    generated = "synthetic" in origins
    if generated and origins != {"synthetic"}:
        raise ValueError("Mixed generated and recorded episodes are not supported")
    expected = "generated_fixture" if generated else "teacher_recorded_coordinates"
    if recipe.coordinate_attestation != expected:
        raise ValueError("Coordinate attestation must match the selected dataset provenance")
    processor_names = {"policy_preprocessor.json", "policy_postprocessor.json"}
    for name in tuple(processor_names):
        for step in strict_json(admitted["path"] / name, JSON_LIMIT)["steps"]:
            if step.get("state_file"):
                processor_names.add(step["state_file"])
    processor_identity = {name: admitted["files"][name] for name in sorted(processor_names)}
    return {
        "root": root,
        "descriptor": descriptor,
        "manifest": manifest,
        "selected": selected,
        "camera": camera,
        "image_shape": shape,
        "source": {
            "kind": "generated_fixture" if generated else "lerobot",
            "identity": descriptor["id"],
            "revision": descriptor["manifest_sha256"],
            "inventory_sha256": hashlib.sha256(canonical(manifest["files"])).hexdigest(),
        },
        "semantics": {
            "state_names": names,
            "action_names": names,
            "units": recipe.units,
            "compatibility": "generated_fixture"
            if generated
            else "operator_attested_teacher_recorded_coordinates",
            "teacher_processors_sha256": hashlib.sha256(canonical(processor_identity)).hexdigest(),
        },
    }


def implementation_identity(runtime):
    root = Path(runtime.native_distillation_root).resolve()
    names = ("__init__.py", "application.py", "contracts.py", "prepare.py", "runtime.py")
    result = {
        "distillation/" + name: hash_file(root / "src/firebird_distill" / name)["sha256"]
        for name in names
    }
    for name in ("bundle.py", "probe.py"):
        result["act/" + name] = hash_file(root.parent / "act_optimizer/src/firebird_act" / name)[
            "sha256"
        ]
    return result


async def admission(lifecycle, project_id, request):
    runtime = lifecycle.runtime(request.runtime_id)
    if (
        runtime is None
        or runtime.execution != "native"
        or runtime.provider != "local"
        or runtime.device != "cpu"
        or not native_distillation_ready(runtime)
    ):
        raise ValueError("Configure a ready local CPU ACT distillation and dataset-reader runtime")
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
        raise ValueError(
            "Distillation requires a completed immutable dataset snapshot in this project"
        )
    try:
        admitted = await finish_owned(
            asyncio.to_thread(
                source_info, source, lifecycle.settings.data_dir, require_inference=False
            )
        )
        data = await finish_owned(
            asyncio.to_thread(
                dataset_info,
                dataset.result,
                lifecycle.settings.data_dir / "dataset-snapshots",
                admitted,
                request.native_distillation,
            )
        )
    except OSError as exc:
        raise ValueError(
            "The complete teacher or dataset snapshot is missing or unreadable"
        ) from exc
    return runtime, source, admitted, data


async def validate(lifecycle, project_id, request):
    await admission(lifecycle, project_id, request)


def command(runtime, mode, request, result):
    if mode not in {"prepare", "application", "metadata"} or not native_distillation_ready(runtime):
        raise ValueError("Fixed ACT distillation worker is not configured")
    root = Path(runtime.native_distillation_root).resolve()
    env = {k: os.environ[k] for k in ("PATH", "TMPDIR", "SYSTEMROOT", "WINDIR") if k in os.environ}
    env.update(
        PYTHONPATH=os.pathsep.join(
            [
                str(root / "src"),
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
    )
    python = (
        runtime.native_distillation_dataset_python
        if mode in {"prepare", "metadata"}
        else runtime.native_distillation_python
    )
    module = "prepare" if mode == "metadata" else mode
    argv = [python, "-m", "firebird_distill." + module, str(request), str(result)]
    if mode == "metadata":
        argv.append("--metadata-only")
    return argv, str(root), env


def check_lengths(response, data):
    if not isinstance(response, dict) or set(response) != {
        "schema_version",
        "snapshot_id",
        "snapshot_manifest_sha256",
        "episode_lengths",
    }:
        raise ValueError("Invalid native episode metadata receipt")
    for key, expected in {
        "schema_version": 1,
        "snapshot_id": data["descriptor"]["id"],
        "snapshot_manifest_sha256": data["descriptor"]["manifest_sha256"],
    }.items():
        equal(response.get(key), expected, "Native episode metadata source identity differs")
    rows = response["episode_lengths"]
    if not isinstance(rows, list) or len(rows) != len(data["selected"]):
        raise ValueError("Native episode metadata selection is incomplete")
    found = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"episode_id", "length"}:
            raise ValueError("Invalid native episode length row")
        episode = integer(row["episode_id"], 0)
        if episode in found or episode not in data["selected"]:
            raise ValueError("Repeated or unselected native episode metadata")
        found[episode] = integer(row["length"], 1, data["manifest"]["total_frames"])
    if sum(found.values()) > data["manifest"]["total_frames"]:
        raise ValueError("Selected episode lengths exceed immutable snapshot bounds")
    return found


def check_corpus(directory, response, data, recipe):
    if not isinstance(response, dict) or response.get("path") != str(directory):
        raise ValueError("Preparation must return its exact owned corpus")
    digest = file_digest(directory, "manifest.json", JSON_LIMIT)["sha256"]
    equal(response.get("manifest_sha256"), digest, "Prepared corpus manifest changed")
    doc = strict_json(directory / "manifest.json", JSON_LIMIT)
    if set(doc) != {
        "schema_version",
        "format",
        "source",
        "semantics",
        "camera",
        "image_shape",
        "chunk_size",
        "samples",
    }:
        raise ValueError("Invalid prepared corpus fields")
    for key, expected in {
        "schema_version": 1,
        "format": "act-observation-corpus-v1",
        "source": data["source"],
        "semantics": data["semantics"],
        "camera": data["camera"],
        "image_shape": data["image_shape"],
        "chunk_size": 100,
    }.items():
        equal(doc.get(key), expected, "Prepared corpus source/coordinate identity mismatch")
    samples = doc.get("samples")
    if not isinstance(samples, list) or not 3 <= len(samples) <= 256:
        raise ValueError("Prepared corpus must contain3..256 observations")
    equal(response.get("samples"), len(samples), "Preparation count differs")
    equal(response.get("source"), data["source"], "Preparation source differs")
    actual, observed, lengths, total = {"manifest.json"}, {}, {}, 0
    for i, row in enumerate(samples):
        if not isinstance(row, dict) or set(row) != {
            "file",
            "sha256",
            "bytes",
            "episode_id",
            "lineage_group",
            "frame_index",
            "episode_length",
            "split",
        }:
            raise ValueError("Invalid prepared observation fields")
        episode = integer(row["episode_id"], 0)
        frame, length = integer(row["frame_index"], 0), integer(row["episode_length"])
        if (
            row["file"] != f"sample-{i:06d}.safetensors"
            or frame >= length
            or frame % recipe.frame_stride
            or episode not in data["selected"]
            or data["selected"][episode] != (row["split"], row["lineage_group"])
            or data["episode_lengths"].get(episode) != length
            or lengths.setdefault(episode, length) != length
            or frame in observed.setdefault(episode, set())
        ):
            raise ValueError("Prepared observation differs from explicit episode/split selection")
        observed[episode].add(frame)
        expected = {"sha256": sha(row["sha256"]), "bytes": integer(row["bytes"], 1, 8 * 1024**2)}
        equal(
            file_digest(directory, row["file"], 8 * 1024**2),
            expected,
            "Prepared observation bytes changed",
        )
        total += row["bytes"]
        actual.add(row["file"])
    if total > 2 * 1024**3 or set(observed) != set(data["selected"]):
        raise ValueError("Prepared corpus omitted selected episodes or exceeded byte limit")
    for episode, frames in observed.items():
        if len(range(0, lengths[episode], recipe.frame_stride)) != len(frames):
            raise ValueError("Prepared corpus omitted selected frames")
    if {p.name for p in directory.iterdir()} != actual:
        raise ValueError("Prepared corpus has unexpected files")
    return doc, digest


def model_id(directory, files):
    value = hashlib.sha256(b"sim-policy-checkpoint-v1\0")
    for name, item in sorted(files.items()):
        encoded = name.encode("utf-8")
        value.update(len(encoded).to_bytes(8, "big") + encoded + item["bytes"].to_bytes(8, "big"))
        with _open_beneath(directory, Path(name)) as stream:
            before, digest, size = os.fstat(stream.fileno()), hashlib.sha256(), 0
            while chunk := stream.read(min(1024**2, item["bytes"] + 1 - size)):
                size += len(chunk)
                if size > item["bytes"]:
                    raise ValueError("Student grew while computing native identity")
                digest.update(chunk)
                value.update(chunk)
            after = os.fstat(stream.fileno())
            if (
                size != item["bytes"]
                or digest.hexdigest() != item["sha256"]
                or (before.st_mtime_ns, before.st_ctime_ns)
                != (after.st_mtime_ns, after.st_ctime_ns)
            ):
                raise ValueError("Student changed while computing native identity")
    return "sha256:" + value.hexdigest()


def check_result(response, job, source, admitted, data, doc, corpus_sha, destination):
    if not isinstance(response, dict):
        raise ValueError("Expected distillation response object")
    for key, expected in {
        "schema_version": 1,
        "job_id": job.id,
        "operation": "policy.distill",
    }.items():
        equal(response.get(key), expected, "Distillation response identity mismatch")
    info, report = response.get("artifact"), response.get("report")
    if (
        not isinstance(info, dict)
        or info.get("path") != str(destination)
        or info.get("format") != "native_checkpoint"
        or not isinstance(report, dict)
    ):
        raise ValueError("Distillation must return the exact owned native student package")
    manifest, files = bundle(destination)
    policy = {name[7:]: row for name, row in files.items() if name.startswith("policy/")}
    processors = set()
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        processors.add(name)
        for step in strict_json(admitted["path"] / name, JSON_LIMIT)["steps"]:
            if step.get("state_file"):
                processors.add(step["state_file"])
    if set(policy) != processors | {"config.json", "model.safetensors"} or set(files) != {
        "manifest.json",
        "training.json",
        "verification.json",
        "lineage.json",
    } | {"policy/" + name for name in policy}:
        raise ValueError("Student package inventory is incomplete or contains unexpected files")
    for name in processors:
        equal(policy[name], admitted["files"][name], "Student changed teacher processor bytes")
    teacher_config = strict_json(admitted["path"] / "config.json", JSON_LIMIT)
    student_config = strict_json(destination / "policy/config.json", JSON_LIMIT)
    for key, expected in {
        "type": "act",
        "use_vae": False,
        "chunk_size": 100,
        "n_action_steps": 100,
        "dim_model": 256,
        "dim_feedforward": 1024,
        "n_encoder_layers": 2,
        "n_decoder_layers": 1,
        "n_heads": 4,
        "dropout": 0.0,
        "replace_final_stride_with_dilation": False,
    }.items():
        equal(
            student_config.get(key), expected, "Student architecture differs from approved recipe"
        )
    for key in ("input_features", "output_features", "normalization_mapping", "vision_backbone"):
        equal(
            student_config.get(key), teacher_config.get(key), "Student changed teacher conventions"
        )
    metadata = {
        "architecture": "act",
        "recipe": "act-action-distillation-v1",
        "model_id": model_id(destination / "policy", policy),
        "policy_inventory_sha256": hashlib.sha256(canonical(policy)).hexdigest(),
        "precision": "fp32",
        "policy_subdirectory": "policy",
        "inference_only": True,
        "training_resume_supported": False,
        "fresh_reload_verified": True,
        "cpu_reload_verified": True,
        "quality_verified": False,
        "calibration_verified": False,
        "speedup_verified": False,
        "isaac_runtime_verified": False,
        "task_success": None,
        "dataset_kind": data["source"]["kind"],
        "teacher_artifact_id": source.id,
        "teacher_artifact_manifest_sha256": source.manifest_sha256,
    }
    equal(manifest.get("metadata"), metadata, "Student metadata identity or claim mismatch")
    recipe = job.request.native_distillation.model_dump(include=RECIPE_KEYS)
    lineage = strict_json(destination / "lineage.json", JSON_LIMIT)
    for key, expected in {
        "schema_version": 1,
        "teacher": {
            "files": admitted["files"],
            "artifact_id": source.id,
            "artifact_manifest_sha256": source.manifest_sha256,
        },
        "corpus_manifest_sha256": corpus_sha,
        "corpus": doc,
        "recipe": recipe,
        "teacher_training_overlap": "unknown",
        "student_splits_disjoint": True,
    }.items():
        equal(lineage.get(key), expected, "Student lineage differs from the requested teacher/data")
    implementation = lineage.get("implementation_sha256")
    if not isinstance(implementation, dict) or not implementation:
        raise ValueError("Missing distillation implementation identity")
    for value in implementation.values():
        sha(value)
    equal(implementation, data["implementation"], "Distillation implementation source changed")
    train = strict_json(destination / "training.json", JSON_LIMIT)
    verified = strict_json(destination / "verification.json", JSON_LIMIT)
    equal(
        {
            k: v
            for k, v in report.items()
            if k not in {"fresh_reload_verified", "elapsed_seconds", "scope"}
        },
        train,
        "Distillation report differs from saved training evidence",
    )
    for key, expected in {
        "schema_version": 1,
        "adapter": "act-act-v1",
        "device": "cpu",
        "dtype": "float32",
        "dataset_kind": data["source"]["kind"],
        "steps": recipe["steps"],
        "policy_files": policy,
        "fresh_reload_verified": True,
        "teacher_gradients_absent": True,
        "copied_backbone_unchanged": True,
        "action_head_changed": True,
        "quality_verified": False,
        "calibration_verified": False,
        "speedup_verified": False,
        "task_success": None,
        "training_resume_supported": False,
        "teacher_weights_bytes": admitted["files"]["model.safetensors"]["bytes"],
        "student_weights_bytes": policy["model.safetensors"]["bytes"],
    }.items():
        equal(
            report.get(key), expected, "Distillation report contains missing or unsupported claims"
        )
    equal(
        report.get("selection"),
        "fixed last step; final split assessed only after saving immutable student",
        "Student selection protocol differs",
    )
    equal(
        report.get("teacher_training_overlap"),
        "unknown; student-only held-out split",
        "Teacher overlap must remain unknown",
    )
    for key in ("training_losses", "gradient_norms"):
        values = report.get(key)
        if not isinstance(values, list) or len(values) != recipe["steps"]:
            raise ValueError("Incomplete optimizer evidence")
        for value in values:
            finite(value)
    for key in ("student_parameters", "trainable_parameters", "teacher_inference_tensor_bytes"):
        integer(report.get(key))
    if (
        report["trainable_parameters"] > report["student_parameters"]
        or report["student_weights_bytes"] >= report["teacher_inference_tensor_bytes"]
    ):
        raise ValueError("Student size or trainable-parameter accounting is invalid")
    finite(report.get("elapsed_seconds"), positive=True)
    for key, splits in (("untuned_student", {"train", "validation"}), ("trained_student", SPLITS)):
        measures = report.get(key)
        if not isinstance(measures, dict) or set(measures) != splits:
            raise ValueError("Incomplete held-out imitation measurements")
        for split, values in measures.items():
            count = sum(
                min(100, s["episode_length"] - s["frame_index"]) * 6
                for s in doc["samples"]
                if s["split"] == split
            )
            if not isinstance(values, dict):
                raise ValueError("Invalid imitation measurements")
            equal(values.get("valid_action_coordinates"), count, "Incomplete masked measurements")
            finite(values.get("teacher_normalized_l1"))
            finite(values.get("teacher_normalized_rmse"))
            coords = values.get("demonstration_per_coordinate_l1")
            if not isinstance(coords, list) or len(coords) != 6:
                raise ValueError("Six separate coordinate diagnostics are required")
            for value in coords:
                finite(value)
    predictions = report.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != len(doc["samples"]):
        raise ValueError("Incomplete full-chunk reload predictions")
    for predicted, sample in zip(predictions, doc["samples"], strict=True):
        if not isinstance(predicted, dict):
            raise ValueError("Invalid reload prediction")
        for key, expected in {
            "sample_sha256": sample["sha256"],
            "split": sample["split"],
            "queue_and_reset_exact": True,
        }.items():
            equal(predicted.get(key), expected, "Reload sample identity mismatch")
        sha(predicted.get("raw_sha256"))
        sha(predicted.get("postprocessed_sha256"))
    actual_versions = report.get("versions")
    if (
        not isinstance(actual_versions, dict)
        or set(actual_versions) != set(RUNTIME)
        or any(
            not isinstance(actual_versions[k], str) or actual_versions[k].split("+")[0] != v
            for k, v in RUNTIME.items()
        )
    ):
        raise ValueError("Distillation runtime versions differ from approved base pins")
    equal(
        verified,
        {
            "schema_version": 1,
            "versions": actual_versions,
            "policy_files": policy,
            "predictions": predictions,
        },
        "Fresh-process verification differs",
    )
    targets = report.get("teacher_target_sha256")
    if not isinstance(targets, dict) or set(targets) != {s["file"] for s in doc["samples"]}:
        raise ValueError("Missing teacher-target provenance")
    for value in targets.values():
        sha(value)
    return manifest, files


async def error_tail(stream):
    result = bytearray()
    while chunk := await stream.read(4096):
        result.extend(chunk)
        del result[:-4096]
    return "".join(
        c for c in result.decode("utf-8", errors="replace") if c.isprintable() or c == "\n"
    )


async def execute(lifecycle, runtime, mode, request_path, result_path):
    argv, cwd, env = command(runtime, mode, request_path, result_path)
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
        raise ValueError("Offline ACT distillation stage failed: " + detail[-4096:])
    return strict_json(result_path, JSON_LIMIT)


def unchanged(source, admitted, data, data_dir):
    if source_info(source, data_dir, require_inference=False) != admitted:
        raise ValueError("Original teacher changed during distillation")
    equal(
        verify_snapshot(data["root"], data["descriptor"]["manifest_sha256"]),
        data["manifest"],
        "Original dataset snapshot changed during distillation",
    )


async def run(lifecycle, job):
    # The outer job supervisor also owns this deadline. Cleanup is drained even on repeated cancel.
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
        recipe = job.request.native_distillation
        try:
            snapshot = await finish_owned(
                asyncio.to_thread(
                    stage_snapshot,
                    data["root"],
                    stage / "snapshot",
                    data["descriptor"]["manifest_sha256"],
                )
            )
            metadata_path, metadata_result = (
                stage / "metadata-request.json",
                stage / "metadata.json",
            )
            metadata_path.write_bytes(
                canonical(
                    {
                        "schema_version": 1,
                        "dataset_snapshot": {
                            "path": str(snapshot),
                            "id": data["descriptor"]["id"],
                            "manifest_sha256": data["descriptor"]["manifest_sha256"],
                        },
                        "episodes": sorted(data["selected"]),
                    }
                )
            )
            lengths = await execute(lifecycle, runtime, "metadata", metadata_path, metadata_result)
            data["episode_lengths"] = check_lengths(lengths, data)
            prepare = {
                "schema_version": 1,
                "teacher": str(admitted["path"]),
                "dataset_snapshot": {
                    "path": str(snapshot),
                    "id": data["descriptor"]["id"],
                    "manifest_sha256": data["descriptor"]["manifest_sha256"],
                },
                "splits": recipe.splits.model_dump(),
                "frame_stride": recipe.frame_stride,
                "semantics": {
                    k: v for k, v in data["semantics"].items() if k != "teacher_processors_sha256"
                },
                "output_dir": str(stage / "corpus"),
            }
            prepare_path, prepared_path = stage / "prepare-request.json", stage / "prepared.json"
            prepare_path.write_bytes(canonical(prepare))
            await lifecycle.event(
                job, "distilling", "Decoding the selected immutable observation splits"
            )
            prepared = await execute(lifecycle, runtime, "prepare", prepare_path, prepared_path)
            doc, corpus_sha = await finish_owned(
                asyncio.to_thread(check_corpus, stage / "corpus", prepared, data, recipe)
            )
            remaining = math.floor(deadline - loop.time())
            if remaining < 1:
                raise TimeoutError("Distillation deadline exceeded during data preparation")
            payload = {
                "schema_version": 1,
                "job_id": job.id,
                "operation": "policy.distill",
                "teacher": {
                    "path": str(admitted["path"]),
                    "files": admitted["files"],
                    "artifact_id": source.id,
                    "artifact_manifest_sha256": source.manifest_sha256,
                },
                "dataset": {"path": str(stage / "corpus"), "manifest_sha256": corpus_sha},
                "recipe": recipe.model_dump(include=RECIPE_KEYS),
                "output_dir": str(stage / "worker-output"),
                "timeout_seconds": remaining,
            }
            request_path, result_path = stage / "request.json", stage / "result.json"
            request_path.write_bytes(canonical(payload))
            await lifecycle.event(
                job,
                "distilling",
                "Training ACT256 on teacher actions; task quality remains unverified",
            )
            response = await execute(lifecycle, runtime, "application", request_path, result_path)
            destination = stage / "worker-output/distilled-policy"
            manifest, files = await finish_owned(
                asyncio.to_thread(
                    check_result,
                    response,
                    job,
                    source,
                    admitted,
                    data,
                    doc,
                    corpus_sha,
                    destination,
                )
            )
            # Recheck the exact prepared corpus and originals before registration.
            await finish_owned(
                asyncio.to_thread(check_corpus, stage / "corpus", prepared, data, recipe)
            )
        finally:
            try:
                equal(
                    await finish_owned(asyncio.to_thread(implementation_identity, runtime)),
                    data["implementation"],
                    "Worker source changed during execution",
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
            label="ACT256 student (offline imitation only)",
            format="native_checkpoint",
            path=destination.relative_to(lifecycle.settings.data_dir.resolve()).as_posix(),
            manifest_sha256=files["manifest.json"]["sha256"],
            file_bytes=sum(v["bytes"] for k, v in files.items() if k != "manifest.json"),
            parent_ids=[source.id],
            metadata=manifest["metadata"],
        )
        report = {
            **response["report"],
            "stage": "distilling",
            "operation": "policy.distill",
            "teacher_artifact_id": source.id,
            "teacher_artifact_manifest_sha256": source.manifest_sha256,
            "dataset_job_id": job.request.dataset_job_id,
            "dataset_snapshot_id": data["descriptor"]["id"],
        }
        result = LifecycleResult(artifacts=[artifact], reports=[report])
        await lifecycle.publish(job, result)
        await lifecycle.event(
            job,
            "distilling",
            "Student package verified; closed-loop task quality remains unverified",
        )
        return result
