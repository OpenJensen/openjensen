"""Cloud artifacts remain remote until an explicit verified worker/browser download."""

import hashlib
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from vla_platform.lifecycle import cloud_storage as storage


class Blob:
    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name

    @property
    def size(self):
        return len(self.bucket.objects[self.name])

    def upload_from_filename(self, path, **kwargs):
        self.bucket.objects[self.name] = Path(path).read_bytes()

    def upload_from_string(self, value, **kwargs):
        self.bucket.objects[self.name] = value.encode() if isinstance(value, str) else value

    def download_as_bytes(self):
        self.bucket.downloads.append(self.name)
        return self.bucket.objects[self.name]

    def download_to_filename(self, path, **kwargs):
        Path(path).write_bytes(self.download_as_bytes())

    def open(self, mode, **kwargs):
        assert mode == "rb"
        self.bucket.open_options.append(kwargs)
        self.bucket.downloads.append(self.name)
        return io.BytesIO(self.bucket.objects[self.name])


class Bucket:
    def __init__(self):
        self.objects, self.downloads = {}, []
        self.open_options = []

    def blob(self, name):
        return Blob(self, name)

    def get_blob(self, name):
        return self.blob(name) if name in self.objects else None


@pytest.fixture
def cloud(monkeypatch):
    bucket = Bucket()
    monkeypatch.setattr(storage, "client", lambda: SimpleNamespace(bucket=lambda _: bucket))
    return bucket


def upload(tmp_path, content=b"trained weights"):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.gguf").write_bytes(content)
    storage.write_json(
        source / "manifest.json",
        {
            "metadata": {"architecture": "smolvla", "precision": {"language": "Q4_0"}},
            "files": {"model.gguf": hashlib.sha256(content).hexdigest()},
        },
    )
    descriptor = storage.upload_artifact(source, "gs://test-bucket/jobs/one/artifact", kind="gguf")
    destination = tmp_path / "descriptor"
    storage.install_descriptor(destination, descriptor)
    return destination, descriptor


def test_upload_and_descriptor_do_not_download_weights(tmp_path, cloud):
    directory, descriptor = upload(tmp_path)
    assert cloud.downloads == []
    assert {path.name for path in directory.iterdir()} == {"manifest.json", "remote.json"}
    assert descriptor["file_bytes"] == len(b"trained weights")
    storage.materialize(directory)
    assert (directory / "model.gguf").read_bytes() == b"trained weights"
    assert not (directory / "remote.json").exists()


@pytest.mark.parametrize("damage", ["manifest", "weights", "size", "missing"])
def test_materialize_refuses_corruption_and_preserves_descriptor(tmp_path, cloud, damage):
    directory, _ = upload(tmp_path)
    key = "jobs/one/artifact/model.gguf"
    if damage == "manifest":
        cloud.objects["jobs/one/artifact/manifest.json"] += b" "
    elif damage == "weights":
        cloud.objects[key] = b"corrupt weights"
    elif damage == "size":
        cloud.objects[key] += b"extra"
    else:
        del cloud.objects[key]
    with pytest.raises(ValueError):
        storage.materialize(directory)
    assert (directory / "remote.json").exists()
    assert not (directory / "model.gguf").exists()
    assert {path.name for path in directory.iterdir()} == {"manifest.json", "remote.json"}


@pytest.mark.parametrize(
    "name", ["../escape", "/absolute", "a/../escape", "a//b", "a/./b", "remote.json", "null\x00"]
)
def test_cloud_inventory_refuses_ambiguous_or_unsafe_paths(name):
    with pytest.raises(ValueError):
        storage.checked_files({"files": {name: "a" * 64}})


def test_materialize_cannot_follow_destination_symlinks(tmp_path, cloud):
    directory, _ = upload(tmp_path)
    outside = tmp_path / "outside"
    outside.write_bytes(b"preserve")
    (directory / "model.gguf").symlink_to(outside)
    with pytest.raises(ValueError, match="destination"):
        storage.materialize(directory)
    assert outside.read_bytes() == b"preserve"


def test_browser_archive_streams_actual_weights_without_materializing(tmp_path, cloud):
    weights = b"x" * (2 * 1024**2 + 17)
    directory, _ = upload(tmp_path, weights)
    plan = storage.archive_plan(directory)
    assert all(name.endswith("manifest.json") for name in cloud.downloads)
    chunks = list(storage.archive_chunks(*plan))
    assert max(map(len, chunks)) <= 1024**2
    assert cloud.open_options[0]["chunk_size"] == 16 * 1024**2
    with tarfile.open(fileobj=io.BytesIO(b"".join(chunks))) as archive:
        assert archive.extractfile("policy/model.gguf").read() == weights
        manifest = json.load(archive.extractfile("policy/manifest.json"))
        assert manifest["files"]["model.gguf"] == hashlib.sha256(weights).hexdigest()
        assert archive.getnames() == ["policy/manifest.json", "policy/model.gguf"]
    assert {path.name for path in directory.iterdir()} == {"manifest.json", "remote.json"}


@pytest.mark.parametrize("size", [0, -1, True, 2 * 1024**2])
def test_archive_output_buffer_is_bounded(tmp_path, cloud, size):
    directory, _ = upload(tmp_path)
    with pytest.raises(ValueError, match="1 MiB"):
        list(storage.archive_chunks(*storage.archive_plan(directory), chunk_size=size))


def test_archive_aborts_before_complete_file_when_hash_is_wrong(tmp_path, cloud):
    directory, _ = upload(tmp_path)
    cloud.objects["jobs/one/artifact/model.gguf"] = b"corrupt weights"
    plan = storage.archive_plan(directory)
    with pytest.raises(ValueError, match="checksum"):
        list(storage.archive_chunks(*plan))


def test_sync_uses_highest_step_even_if_index_was_reordered(tmp_path, cloud):
    _, descriptor = upload(tmp_path)
    entries = []
    for step in (20, 10):
        name = f"checkpoint-{step:06d}"
        uri = f"gs://test-bucket/jobs/one/checkpoints/{name}"
        manifest = {**descriptor["manifest"], "metadata": {"remote_uri": uri, "step": step}}
        entries.append(
            {
                **descriptor,
                "uri": uri,
                "manifest": manifest,
                "manifest_sha256": hashlib.sha256(storage.encoded(manifest)).hexdigest(),
                "name": name,
                "step": step,
                "recipe": {"steps": 20},
                "checkpoint_manifest": {"step": step},
            }
        )
    cloud.objects["jobs/one/checkpoints.json"] = storage.encoded({"checkpoints": entries})
    stage = tmp_path / "stage"
    assert storage.sync("gs://test-bucket/jobs/one", stage) is False
    assert json.loads((stage / "training/latest.json").read_text())["step"] == 20
    assert not list(stage.rglob("*.gguf"))


@pytest.mark.parametrize("manifest", [None, [], "manifest", 1])
def test_non_mapping_manifest_is_a_clear_integrity_error(manifest):
    with pytest.raises(ValueError, match="Invalid cloud artifact manifest"):
        storage.checked_files(manifest)


@pytest.mark.parametrize(
    "pointer", [None, [], {}, {"uri": None, "manifest_sha256": "a" * 64, "file_bytes": 1}]
)
def test_non_mapping_or_incomplete_descriptor_is_a_clear_integrity_error(tmp_path, pointer):
    storage.write_json(tmp_path / "manifest.json", {"files": {"weights.bin": "a" * 64}})
    storage.write_json(tmp_path / "remote.json", pointer)
    with pytest.raises(ValueError, match="Invalid stored cloud artifact descriptor"):
        storage.read_descriptor(tmp_path)
