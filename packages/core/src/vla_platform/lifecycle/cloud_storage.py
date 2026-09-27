"""Private GCS artifact storage shared by the host and isolated GPU workers.

Only bounded JSON descriptors cross back to the application server. Model files
are fetched by a subsequent cloud worker, checked against the committed manifest.
"""

import hashlib
import json
import os
import re
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

MAX_METADATA = 8 * 1024**2
MAX_FILES = 10000
MAX_BYTES = 100 * 1024**3
STREAM_CHUNK_BYTES = 1024**2
GCS_READ_AHEAD_BYTES = 16 * 1024**2

INFERENCE_EVIDENCE_BYTES = 256 * 1024
INFERENCE_EVIDENCE_FILES = (
    "inference-report.json",
    "actions.json",
    "reload.log",
    "timing.log",
    "reload.gpu.csv",
    "timing.gpu.csv",
    "package-verification/result.json",
    "package-verification/worker.log",
    "package-verification/actions.json",
    "package-verification/reload.log",
    "package-verification/timing.log",
    "package-verification/reload.gpu.csv",
    "package-verification/timing.gpu.csv",
)
TRAINING_SUMMARY_FILES = (
    "recipe.json",
    "training/environment.json",
    "training/metrics.jsonl",
    "training/splits.json",
    "resource-preflight.json",
)
SUMMARY_FILES = frozenset((*TRAINING_SUMMARY_FILES, *INFERENCE_EVIDENCE_FILES))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_bytes(encoded(value))
    temporary.replace(path)


def split_uri(uri):
    match = re.fullmatch(r"gs://([a-z0-9][a-z0-9._-]{1,61}[a-z0-9])/(.+)", uri)
    if not match or any(x in {"", ".", ".."} for x in match[2].split("/")):
        raise ValueError("Invalid cloud artifact URI")
    return match[1], match[2].rstrip("/")


def client():
    from google.cloud import storage

    return storage.Client()


def json_blob(bucket, name):
    blob = bucket.get_blob(name)
    if blob is None:
        return None
    if not blob.size or blob.size > MAX_METADATA:
        raise ValueError("Cloud metadata exceeds the supported size")
    return json.loads(blob.download_as_bytes())


def ensure_bucket(project, region):
    from google.api_core.exceptions import Conflict
    from google.cloud import storage

    service = storage.Client(project=project)
    name = "firebird-artifacts-" + project
    bucket = service.lookup_bucket(name)
    if bucket is None:
        bucket = service.bucket(name)
        bucket.iam_configuration.uniform_bucket_level_access_enabled = True
        bucket.iam_configuration.public_access_prevention = "enforced"
        bucket.labels = {"application": "firebird", "firebird_project": project}
        try:
            bucket = service.create_bucket(bucket, location=region)
        except Conflict:
            bucket = service.get_bucket(name)
    if bucket.labels.get("firebird_project") != project:
        raise ValueError("Cloud bucket is not owned by this Firebird project")
    if (
        not bucket.iam_configuration.uniform_bucket_level_access_enabled
        or bucket.iam_configuration.public_access_prevention != "enforced"
    ):
        raise ValueError("Firebird artifact bucket must use private uniform access")
    return "gs://" + name


def checked_files(manifest):
    if not isinstance(manifest, dict):
        raise ValueError("Invalid cloud artifact manifest")
    files = manifest.get("files")
    if not isinstance(files, dict) or not 1 <= len(files) <= MAX_FILES:
        raise ValueError("Invalid cloud artifact inventory")
    for name, sha in files.items():
        if not isinstance(name, str):
            raise ValueError("Invalid cloud artifact filename")
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or not path.parts
            or ".." in path.parts
            or "\\" in name
            or "\x00" in name
            or path.as_posix() != name
            or name in {"manifest.json", "remote.json"}
            or not isinstance(sha, str)
            or not re.fullmatch(r"[a-f0-9]{64}", sha)
        ):
            raise ValueError("Invalid cloud artifact manifest entry")
    return files


def upload_artifact(directory, uri, *, metadata=None, label=None, kind=None):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    files = checked_files(manifest)
    bucket_name, prefix = split_uri(uri)
    bucket = client().bucket(bucket_name)
    total = 0
    for name, sha in files.items():
        path = directory / name
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(directory.resolve())
            or not path.is_file()
            or digest(path) != sha
        ):
            raise ValueError("Cloud upload source differs from its manifest")
        total += path.stat().st_size
        if total > MAX_BYTES:
            raise ValueError("Cloud artifact exceeds storage limit")
        bucket.blob(prefix + "/" + name).upload_from_filename(str(path), timeout=600)
    manifest["metadata"] = {
        **manifest.get("metadata", {}),
        **(metadata or {}),
        "storage": "gcs",
        "remote_uri": uri,
    }
    content = encoded(manifest)
    bucket.blob(prefix + "/manifest.json").upload_from_string(
        content, content_type="application/json"
    )
    return {
        "uri": uri,
        "manifest": manifest,
        "manifest_sha256": hashlib.sha256(content).hexdigest(),
        "file_bytes": total,
        "label": label or "Cloud checkpoint",
        "format": kind or "training_checkpoint",
    }


def publish_training_checkpoint(checkpoint):
    """Commit each checkpoint to durable storage before reporting it as selectable."""
    checkpoint = Path(checkpoint)
    source = json.loads((checkpoint / "manifest.json").read_text())
    step = source["step"]
    if not re.fullmatch(r"checkpoint-\d{6,}", checkpoint.name) or step != int(checkpoint.name[11:]):
        raise ValueError("Invalid checkpoint step")
    files = checked_files(source)
    request = json.loads(Path(os.environ["FIREBIRD_CLOUD_REQUEST"]).read_text())
    recipe = json.loads((checkpoint / "recipe.json").read_text())
    data = request.get("dataset", {})
    model = recipe.get("policy_type") or recipe.get("architecture") or "smolvla"
    uri = os.environ["FIREBIRD_GCS_PREFIX"] + "/checkpoints/" + checkpoint.name
    bucket_name, prefix = split_uri(uri)
    bucket = client().bucket(bucket_name)
    wrapper_files, total = {}, 0
    for name in [*files, "manifest.json"]:
        path = checkpoint / name
        if path.is_symlink() or not path.resolve().is_relative_to(checkpoint.resolve()):
            raise ValueError("Unsafe checkpoint source")
        sha = digest(path)
        if name in files and sha != files[name]:
            raise ValueError("Checkpoint checksum mismatch before upload")
        wrapper_files["checkpoint/" + name] = sha
        total += path.stat().st_size
        if total > MAX_BYTES:
            raise ValueError("Checkpoint exceeds storage limit")
        bucket.blob(prefix + "/checkpoint/" + name).upload_from_filename(str(path), timeout=600)
    metadata = {
        "storage": "gcs",
        "remote_uri": uri,
        "step": step,
        "architecture": model,
        "method": recipe.get("method", request.get("parameters", {}).get("training_method")),
        "base_model": {
            "repository": recipe.get("model_id"),
            "revision": recipe.get("model_revision"),
        },
        "dataset": data,
        "action_dim": data.get("features", {}).get("action", {}).get("shape", [0])[0],
        "camera_keys": recipe.get("camera_keys") or [recipe.get("camera_key")],
        "reload_verified": False,
        "task_success": None,
    }
    manifest = {"schema_version": 1, "metadata": metadata, "files": wrapper_files}
    content = encoded(manifest)
    bucket.blob(prefix + "/manifest.json").upload_from_string(
        content, content_type="application/json"
    )
    descriptor = {
        "name": checkpoint.name,
        "step": step,
        "uri": uri,
        "manifest": manifest,
        "manifest_sha256": hashlib.sha256(content).hexdigest(),
        "file_bytes": total,
        "format": "training_checkpoint",
        "label": f"{model} · step {step}",
        "recipe": recipe,
        "checkpoint_manifest": source,
    }
    run_prefix = prefix.rsplit("/checkpoints/", 1)[0]
    index = json_blob(bucket, run_prefix + "/checkpoints.json") or {"checkpoints": []}
    index["checkpoints"] = [x for x in index["checkpoints"] if x["name"] != checkpoint.name] + [
        descriptor
    ]
    if len(index["checkpoints"]) > MAX_FILES or len(encoded(index)) > MAX_METADATA:
        raise ValueError("Cloud checkpoint index exceeds supported size")
    bucket.blob(run_prefix + "/checkpoints.json").upload_from_string(
        encoded(index), content_type="application/json"
    )
    # GCS is authoritative. Keep only the latest two working copies on the GPU disk.
    for previous in sorted(index["checkpoints"], key=lambda x: x["step"])[:-2]:
        path = checkpoint.parent / previous["name"]
        if path.exists() and not path.is_symlink() and path.parent == checkpoint.parent:
            import shutil

            shutil.rmtree(path)
    print(
        json.dumps(
            {
                "event": "checkpoint",
                "phase": "checkpoint",
                "step": step,
                "checkpoint_saved": checkpoint.name,
                "message": f"Checkpoint {step} saved on Google Cloud",
            }
        ),
        flush=True,
    )


def install_descriptor(directory, descriptor):
    directory = Path(directory)
    manifest = descriptor["manifest"]
    checked_files(manifest)
    split_uri(descriptor["uri"])
    if hashlib.sha256(encoded(manifest)).hexdigest() != descriptor["manifest_sha256"]:
        raise ValueError("Cloud manifest checksum mismatch")
    if manifest.get("metadata", {}).get("remote_uri") != descriptor["uri"]:
        raise ValueError("Cloud manifest location mismatch")
    if type(descriptor["file_bytes"]) is not int or not 0 <= descriptor["file_bytes"] <= MAX_BYTES:
        raise ValueError("Invalid cloud artifact size")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_bytes(encoded(manifest))
    write_json(
        directory / "remote.json",
        {k: descriptor[k] for k in ("uri", "manifest_sha256", "file_bytes")},
    )


def read_descriptor(directory):
    directory = Path(directory)
    pointer = json.loads((directory / "remote.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    checked_files(manifest)
    if (
        not isinstance(pointer, dict)
        or not {"uri", "manifest_sha256", "file_bytes"}.issubset(pointer)
        or not isinstance(pointer["uri"], str)
        or not isinstance(manifest.get("metadata", {}), dict)
    ):
        raise ValueError("Invalid stored cloud artifact descriptor")
    split_uri(pointer["uri"])
    if digest(directory / "manifest.json") != pointer["manifest_sha256"]:
        raise ValueError("Stored cloud manifest changed")
    if manifest.get("metadata", {}).get("remote_uri") != pointer["uri"]:
        raise ValueError("Stored cloud artifact location mismatch")
    if type(pointer["file_bytes"]) is not int or not 0 <= pointer["file_bytes"] <= MAX_BYTES:
        raise ValueError("Invalid cloud artifact size")
    return pointer, manifest


def materialize(directory):
    directory = Path(directory)
    _, objects = archive_plan(directory)
    for name, blob, sha in objects:
        path = directory / name
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Cloud artifact destination escapes its directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            blob.download_to_filename(str(temporary), timeout=600)
            if temporary.stat().st_size != blob.size or digest(temporary) != sha:
                raise ValueError("Cloud artifact checksum mismatch")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    (directory / "remote.json").unlink()


def archive_plan(directory):
    """Check remote metadata/size before download, pinning each object generation."""
    pointer, manifest = read_descriptor(directory)
    bucket_name, prefix = split_uri(pointer["uri"])
    bucket = client().bucket(bucket_name)
    manifest_blob = bucket.get_blob(prefix + "/manifest.json")
    if manifest_blob is None or not manifest_blob.size or manifest_blob.size > MAX_METADATA:
        raise ValueError("Cloud artifact manifest is missing or exceeds the size limit")
    remote = manifest_blob.download_as_bytes()
    if hashlib.sha256(remote).hexdigest() != pointer["manifest_sha256"]:
        raise ValueError("Cloud artifact manifest changed")
    objects, total = [], 0
    for name, sha in sorted(checked_files(manifest).items()):
        blob = bucket.get_blob(prefix + "/" + name)
        if blob is None or type(blob.size) is not int or blob.size < 0:
            raise ValueError("Cloud artifact file is missing")
        total += blob.size
        if total > MAX_BYTES or total > pointer["file_bytes"]:
            raise ValueError("Cloud artifact exceeds committed size")
        objects.append((name, blob, sha))
    if total != pointer["file_bytes"]:
        raise ValueError("Cloud artifact size mismatch")
    return remote, objects


def archive_chunks(manifest_bytes, objects, *, chunk_size=STREAM_CHUNK_BYTES):
    """Stream a checksum-verified TAR with bounded buffers and no weight files on disk."""
    if type(chunk_size) is not int or not 1 <= chunk_size <= STREAM_CHUNK_BYTES:
        raise ValueError("Archive output chunks must be between 1 byte and 1 MiB")
    header = tarfile.TarInfo("policy/manifest.json")
    header.mode, header.size = 0o644, len(manifest_bytes)
    yield header.tobuf(format=tarfile.PAX_FORMAT)
    for offset in range(0, len(manifest_bytes), chunk_size):
        yield manifest_bytes[offset : offset + chunk_size]
    yield b"\0" * (-len(manifest_bytes) % 512)
    for name, blob, expected in objects:
        header = tarfile.TarInfo("policy/" + name)
        header.mode, header.size = 0o644, blob.size
        yield header.tobuf(format=tarfile.PAX_FORMAT)
        checksum, remaining = hashlib.sha256(), blob.size
        # BlobReader's chunk_size controls each blocking GCS range request, not
        # the size returned by read(). Amortize request latency with a bounded
        # read-ahead window while keeping pipe/HTTP backpressure at 1 MiB.
        with blob.open(
            "rb", chunk_size=GCS_READ_AHEAD_BYTES, timeout=60, raw_download=True
        ) as stream:
            while remaining:
                block = stream.read(min(chunk_size, remaining))
                if not block:
                    raise ValueError("Cloud artifact download ended before its committed size")
                remaining -= len(block)
                checksum.update(block)
                # Withhold the last block until its full-file checksum passes.
                if not remaining and checksum.hexdigest() != expected:
                    raise ValueError("Cloud artifact checksum mismatch")
                yield block
        if checksum.hexdigest() != expected:
            raise ValueError("Cloud artifact checksum mismatch")
        yield b"\0" * (-blob.size % 512)
    yield b"\0" * 1024


def publish_result(output, prefix):
    output = Path(output)
    result = json.loads((output / "result.json").read_text())
    bucket_name, object_prefix = split_uri(prefix)
    bucket = client().bucket(bucket_name)
    if result.get("artifact"):
        artifact = result["artifact"]
        artifact_path = Path(artifact["path"])
        if not artifact_path.resolve().is_relative_to(output.resolve()):
            raise ValueError("Output artifact escaped job directory")
        recipe_path = output / "recipe.json"
        recipe = json.loads(recipe_path.read_text()) if recipe_path.exists() else {}
        result["cloud_artifact"] = upload_artifact(
            artifact_path,
            prefix + "/artifact",
            metadata={"step": recipe["steps"]} if recipe.get("steps") else {},
            label=artifact["label"],
            kind=artifact["format"],
        )
    # Persist only fixed, bounded metadata paths. No wildcard can include model weights.
    summaries, evidence = {}, []
    for name in (*TRAINING_SUMMARY_FILES, *INFERENCE_EVIDENCE_FILES):
        path = output / name
        if not path.is_file():
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(output.resolve()):
            raise ValueError("Unsafe result evidence path")
        size = path.stat().st_size
        limit = INFERENCE_EVIDENCE_BYTES if name in INFERENCE_EVIDENCE_FILES else MAX_METADATA
        if size > limit and path.suffix == ".json":
            continue  # Never publish invalid partial JSON as a complete report.
        with path.open("rb") as stream:
            if size > limit:
                stream.seek(size - limit)
            raw = stream.read(limit)
        value = raw.decode("utf-8", errors="replace")
        value = value.encode("utf-8")[:limit].decode("utf-8", errors="ignore")
        captured = value.encode("utf-8")
        candidate = {**summaries, name: value}
        # Leave room for the evidence index added below and JSON escaping.
        if len(encoded({**result, "summaries": candidate})) > MAX_METADATA - 64 * 1024:
            continue
        summaries = candidate
        if name in INFERENCE_EVIDENCE_FILES:
            uri = prefix + "/evidence/" + name
            bucket.blob(object_prefix + "/evidence/" + name).upload_from_string(
                captured,
                content_type="application/json" if path.suffix == ".json" else "text/plain",
            )
            evidence.append(
                {
                    "path": name,
                    "uri": uri,
                    "sha256": hashlib.sha256(captured).hexdigest(),
                    "bytes": len(captured),
                    "original_bytes": size,
                    "truncated": size > len(captured),
                }
            )
    if evidence:
        report = result.setdefault("report", {})
        report["retained_evidence"] = evidence
        report["evidence_scope"] = (
            "Paths are relative to the application job operation directory; URI fields "
            "identify durable GCS copies. Raw reports may retain original VM execution paths."
        )
        retained = {entry["path"]: entry for entry in evidence}
        measurements = [report.get("measurement"), *report.get("measurements", {}).values()]
        for measurement in measurements:
            if not isinstance(measurement, dict) or not measurement.get("gpu_telemetry"):
                continue
            original = Path(measurement["gpu_telemetry"])
            if original.is_absolute() and original.is_relative_to(output.resolve()):
                name = original.relative_to(output.resolve()).as_posix()
                measurement["gpu_telemetry"] = name
                measurement["gpu_telemetry_uri"] = retained.get(name, {}).get("uri")
    result["summaries"] = summaries
    document = encoded(result)
    if len(document) > MAX_METADATA:
        raise ValueError("Result metadata exceeds the supported size")
    bucket.blob(object_prefix + "/result.json").upload_from_string(
        document, content_type="application/json"
    )


def sync(prefix, stage):
    stage = Path(stage)
    bucket_name, object_prefix = split_uri(prefix)
    bucket = client().bucket(bucket_name)
    index = json_blob(bucket, object_prefix + "/checkpoints.json") or {"checkpoints": []}
    for item in sorted(index["checkpoints"], key=lambda item: item["step"]):
        if not re.fullmatch(r"checkpoint-\d{6,}", item["name"]) or item["step"] != int(
            item["name"][11:]
        ):
            raise ValueError("Invalid cloud checkpoint identity")
        if item["uri"] != prefix + "/checkpoints/" + item["name"]:
            raise ValueError("Cloud checkpoint escaped job prefix")
        directory = stage / "cloud-checkpoints" / item["name"]
        created = not directory.exists()
        install_descriptor(directory, item)
        # Keep recipe evidence available for resume/progress without weight downloads.
        training = stage / "training" / item["name"]
        training.mkdir(parents=True, exist_ok=True)
        write_json(training / "manifest.json", item["checkpoint_manifest"])
        write_json(training / "recipe.json", item["recipe"])
        write_json(training / "remote-checkpoint.json", {"bundle": str(directory.resolve())})
        write_json(
            stage / "training/latest.json", {"checkpoint": item["name"], "step": item["step"]}
        )
        if created:
            print("FIREBIRD_CLOUD_CHECKPOINT=" + str(item["step"]), flush=True)
    write_json(stage / "cloud-checkpoints.json", index)
    result = json_blob(bucket, object_prefix + "/result.json")
    if result is None:
        return False
    descriptor = result.pop("cloud_artifact", None)
    if descriptor:
        if descriptor["uri"] != prefix + "/artifact":
            raise ValueError("Cloud output escaped job prefix")
        install_descriptor(stage / "bundle", descriptor)
        result["artifact"]["path"] = str((stage / "bundle").resolve())
    evidence = {
        entry["path"]: entry for entry in result.get("report", {}).get("retained_evidence", [])
    }
    for name, value in result.pop("summaries", {}).items():
        if name not in SUMMARY_FILES:
            continue
        if not isinstance(value, str) or len(value.encode("utf-8")) > MAX_METADATA:
            raise ValueError("Invalid result summary")
        if name in INFERENCE_EVIDENCE_FILES:
            item = evidence.get(name)
            raw = value.encode("utf-8")
            if (
                not item
                or item.get("uri") != prefix + "/evidence/" + name
                or len(raw) > INFERENCE_EVIDENCE_BYTES
                or item.get("bytes") != len(raw)
                or item.get("sha256") != hashlib.sha256(raw).hexdigest()
            ):
                raise ValueError("Inference evidence differs from its published receipt")
        path = stage / name
        if path.is_symlink() or not path.resolve().is_relative_to(stage.resolve()):
            raise ValueError("Unsafe result summary path")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    write_json(stage / "result.json", result)
    return True


if __name__ == "__main__":
    if sys.argv[1] == "ensure":
        print(ensure_bucket(sys.argv[2], sys.argv[3]))
    elif sys.argv[1] == "sync":
        sync(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "stream":
        plan = archive_plan(sys.argv[2])
        sys.stdout.buffer.write(b"FIREBIRD_ARCHIVE_READY\n")
        sys.stdout.buffer.flush()
        for block in archive_chunks(*plan):
            sys.stdout.buffer.write(block)
        sys.stdout.buffer.flush()
    else:
        raise SystemExit("Unsupported cloud storage operation")
