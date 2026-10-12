"""Dataset persistence/retry regressions without models, cloud or reader installs."""

import asyncio
import io
import json
import zipfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from vla_platform.api import create_app
from vla_platform.contracts import IntakeRequest
from vla_platform.datasets import library as module
from vla_platform.datasets.library import DatasetLibrary, LibraryError
from vla_platform.settings import Settings


def zip_bytes(files):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return payload.getvalue()


async def upload(library, ident, name, data):
    async def chunks():
        yield data

    return await library.upload(ident, name, chunks())


@pytest.mark.parametrize(
    "damage", ["truncated", "missing", "fields", "identity", "duplicate", "nonfinite"]
)
def test_bad_record_isolated_on_real_app_restart(tmp_path, damage):
    settings = Settings(data_dir=tmp_path)
    with TestClient(create_app(settings)) as client:
        project = client.post("/api/v1/projects", json={"name": "Preserved"}).json()
        first = client.post(
            f"/api/v1/projects/{project['id']}/dataset-uploads", json={"name": "Damaged"}
        ).json()
        healthy = client.post(
            f"/api/v1/projects/{project['id']}/dataset-uploads", json={"name": "Healthy"}
        ).json()
    record = tmp_path / "dataset-library" / first["id"] / "record.json"
    if damage == "missing":
        record.unlink()
        original = None
    else:
        original = {
            "truncated": "{",
            "fields": json.dumps({"id": first["id"], "status": "ready"}),
            "identity": record.read_text().replace(first["id"], "a" * 32),
            "duplicate": '{"id":"one","id":"two"}',
            "nonfinite": record.read_text().replace(
                '"detection": null', '"detection": {"frames": 1e999}'
            ),
        }[damage]
        record.write_text(original)
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/v1/projects").json()[0]["id"] == project["id"]
        unavailable = client.get(f"/api/v1/datasets/{first['id']}").json()
        assert unavailable["status"] == "unavailable"
        assert unavailable["can_discard"] is False
        assert "preserved" in unavailable["error"]
        assert client.get(f"/api/v1/datasets/{healthy['id']}").json()["status"] == "uploading"
        rows = client.get("/api/v1/datasets").json()
        assert {row["id"] for row in rows} == {first["id"], healthy["id"]}
        scoped = client.get("/api/v1/datasets", params={"project_id": project["id"]}).json()
        assert {row["id"] for row in scoped} == {first["id"], healthy["id"]}
        diagnostic = next(row for row in scoped if row["id"] == first["id"])
        assert diagnostic["project_id"] is None and diagnostic["can_discard"] is False
        assert "source_directory" not in diagnostic and "files" not in diagnostic
        assert (
            client.delete(
                f"/api/v1/dataset-uploads/{first['id']}", params={"project_id": project["id"]}
            ).status_code
            == 404
        )
    assert record.read_text() == original if original is not None else not record.exists()


@pytest.mark.parametrize("alias", ["a//b.txt", "a/./b.txt", "A/B.txt"])
def test_upload_aliases_preserve_original_inventory(tmp_path, alias):
    async def check():
        library = DatasetLibrary(tmp_path)
        entry = library.begin("one", "Aliases")
        await upload(library, entry["id"], "a/b.txt", b"ORIGINAL")
        with pytest.raises(LibraryError) as error:
            await upload(library, entry["id"], alias, b"REPLACEMENT")
        assert error.value.status in {409, 422}
        record = library.get(entry["id"])
        assert record["files"] == {"a/b.txt": 8}
        assert record["total_bytes"] == 8
        assert (library.directory(entry["id"]) / "source/a/b.txt").read_bytes() == b"ORIGINAL"

    asyncio.run(check())


def test_unicode_alias_preserves_uploaded_bytes(tmp_path):
    async def check():
        library = DatasetLibrary(tmp_path)
        entry = library.begin("one", "Unicode")
        await upload(library, entry["id"], "café.txt", b"original")
        with pytest.raises(LibraryError, match="already uploaded"):
            await upload(library, entry["id"], "cafe\u0301.txt", b"replacement")
        assert library.get(entry["id"])["total_bytes"] == 8

    asyncio.run(check())


def test_zip_finalize_retries_after_reader_failure_without_reextracting(tmp_path):
    async def check():
        library = DatasetLibrary(tmp_path)
        entry = library.begin("one", "ZIP")
        await upload(library, entry["id"], "dataset.zip", zip_bytes({"dataset/frames": b"data"}))
        library.worker = AsyncMock(side_effect=LibraryError("Reader unavailable", 503))
        with pytest.raises(LibraryError, match="unavailable"):
            await library.finalize(entry["id"])
        extracted = library.directory(entry["id"]) / "extracted/dataset/frames"
        before = extracted.stat().st_mtime_ns
        library.worker = AsyncMock(
            return_value={
                "format": "unknown",
                "sha256": "a" * 64,
                "convertible": False,
                "cameras": [],
                "mapping": {},
            }
        )
        finished = await library.finalize(entry["id"])
        assert finished["status"] == "detected"
        assert finished["error"] is None
        assert extracted.read_bytes() == b"data"
        assert extracted.stat().st_mtime_ns == before

    asyncio.run(check())


def test_incomplete_legacy_extraction_preserved_and_retry_publishes_complete_copy(tmp_path):
    async def check():
        library = DatasetLibrary(tmp_path)
        entry = library.begin("one", "ZIP")
        await upload(library, entry["id"], "dataset.zip", zip_bytes({"folder/frames": b"complete"}))
        old = library.directory(entry["id"]) / "extracted/folder"
        old.mkdir(parents=True)
        (old / "frames").write_bytes(b"partial")
        library.worker = AsyncMock(
            return_value={
                "format": "unknown",
                "sha256": "a" * 64,
                "convertible": False,
                "cameras": [],
                "mapping": {},
            }
        )
        await library.finalize(entry["id"])
        value = library.get(entry["id"])
        assert value["source_directory"].startswith("extracted-")
        assert (library.source(value) / "frames").read_bytes() == b"complete"
        assert (old / "frames").read_bytes() == b"partial"

    asyncio.run(check())


def test_zip_low_disk_preserves_source_and_remains_retryable(tmp_path, monkeypatch):
    async def check():
        library = DatasetLibrary(tmp_path)
        entry = library.begin("one", "ZIP")
        raw = zip_bytes({"folder/frames": b"data"})
        await upload(library, entry["id"], "dataset.zip", raw)
        monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
        with pytest.raises(LibraryError) as error:
            await library.finalize(entry["id"])
        assert error.value.status == 507
        assert library.get(entry["id"])["status"] == "uploading"
        assert (library.directory(entry["id"]) / "source/dataset.zip").read_bytes() == raw
        assert not list(library.directory(entry["id"]).glob(".extract-*"))
        assert not (library.directory(entry["id"]) / "extracted").exists()

    asyncio.run(check())


def test_failed_durable_save_keeps_previous_record(tmp_path, monkeypatch):
    library = DatasetLibrary(tmp_path)
    entry = library.begin("one", "Keep")
    path = library.directory(entry["id"]) / "record.json"
    original = path.read_bytes()
    value = library.get(entry["id"])
    value["error"] = "Changed"
    monkeypatch.setattr(module.os, "fsync", lambda _: (_ for _ in ()).throw(OSError("disk error")))
    with pytest.raises(OSError, match="disk error"):
        library.save(value)
    assert path.read_bytes() == original
    assert not list(path.parent.glob(".record-*"))


def test_interrupted_upload_visible_and_discard_preserves_other_project_and_accepted_data(tmp_path):
    app = create_app(Settings(data_dir=tmp_path))
    with TestClient(app) as client:
        first = client.post("/api/v1/projects", json={"name": "First"}).json()["id"]
        second = client.post("/api/v1/projects", json={"name": "Second"}).json()["id"]
        entry = client.post(
            f"/api/v1/projects/{first}/dataset-uploads", json={"name": "Unfinished"}
        ).json()
        client.put(
            f"/api/v1/dataset-uploads/{entry['id']}/file",
            params={"path": "frames"},
            content=b"kept",
        )
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        rows = client.get("/api/v1/datasets", params={"project_id": first}).json()
        assert len(rows) == 1 and rows[0]["can_discard"] is True
        assert rows[0]["uploaded_files"] == 1 and rows[0]["total_bytes"] == 4
        assert "unfinished" in rows[0]["error"]
        assert (
            client.delete(
                f"/api/v1/dataset-uploads/{entry['id']}", params={"project_id": second}
            ).status_code
            == 404
        )
        assert (
            tmp_path / "dataset-library" / entry["id"] / "source/frames"
        ).read_bytes() == b"kept"
        response = client.delete(
            f"/api/v1/dataset-uploads/{entry['id']}", params={"project_id": first}
        )
        assert response.json() == {"discarded": True, "id": entry["id"]}
        assert not (tmp_path / "dataset-library" / entry["id"]).exists()


def test_discard_rejects_detected_upload(tmp_path):
    async def check():
        library = DatasetLibrary(tmp_path)
        entry = library.begin("one", "Saved")
        execution = SimpleNamespace(list=AsyncMock(return_value=[]))
        value = library.get(entry["id"])
        value.update(
            status="detected",
            detection={
                "sha256": "a" * 64,
                "format": "unknown",
                "convertible": False,
                "cameras": [],
                "mapping": {},
            },
        )
        library.save(value)
        with pytest.raises(LibraryError) as error:
            await library.discard(entry["id"], "one", execution)
        assert error.value.status == 409
        assert library.directory(entry["id"]).exists()

    asyncio.run(check())


def test_discard_preserves_upload_referenced_by_saved_job(tmp_path):
    async def check():
        library = DatasetLibrary(tmp_path)
        entry = library.begin("one", "Referenced")
        job = SimpleNamespace(request=IntakeRequest(source="local", library_id=entry["id"]))
        execution = SimpleNamespace(list=AsyncMock(return_value=[job]))
        with pytest.raises(LibraryError, match="referenced") as error:
            await library.discard(entry["id"], "one", execution)
        assert error.value.status == 409
        assert library.directory(entry["id"]).exists()

    asyncio.run(check())


def test_stream_disconnect_preserves_completed_files_and_removes_owned_partial(tmp_path):
    async def check():
        library = DatasetLibrary(tmp_path)
        entry = library.begin("one", "Disconnected")
        await upload(library, entry["id"], "complete", b"preserved")

        async def interrupted():
            yield b"partial"
            raise ConnectionError("Client disconnected")

        with pytest.raises(ConnectionError, match="disconnected"):
            await library.upload(entry["id"], "unfinished", interrupted())
        record = library.get(entry["id"])
        assert record["files"] == {"complete": 9}
        assert record["total_bytes"] == 9
        source = library.directory(entry["id"]) / "source"
        assert (source / "complete").read_bytes() == b"preserved"
        assert not (source / "unfinished").exists()
        assert not list(source.glob(".upload-*"))

    asyncio.run(check())


@pytest.mark.parametrize("damaged", ["detection", "converted"])
def test_malformed_saved_camera_shape_is_diagnosed_before_labeling(tmp_path, damaged):
    library = DatasetLibrary(tmp_path)
    entry = library.begin("one", "Previously saved")
    value = library.get(entry["id"])
    value.update(
        status="ready",
        normalized="source",
        detection={
            "sha256": "a" * 64,
            "format": "lerobot_v3",
            "convertible": True,
            "cameras": [],
            "mapping": {},
        },
    )
    value[damaged] = {
        "sha256": "a" * 64,
        "format": "lerobot_v3",
        "convertible": True,
        "mapping": {},
    }
    record = library.directory(entry["id"]) / "record.json"
    original = json.dumps(value)
    record.write_text(original)
    restored = DatasetLibrary(tmp_path)
    assert restored.get(entry["id"])["status"] == "unavailable"
    with pytest.raises(LibraryError, match="conversion"):
        restored.annotate(entry["id"], {"revision": 0, "views": {}, "frames": {}})
    assert record.read_text() == original


@pytest.mark.parametrize(
    "mapping",
    [
        None,
        [],
        {"action": []},
        {"table": 7},
        {"state": "state"},
        {"state": [None]},
        {"cameras": "camera"},
        {"cameras": [{}]},
    ],
)
def test_malformed_saved_mapping_is_unavailable_and_preserved(tmp_path, mapping):
    library = DatasetLibrary(tmp_path)
    entry = library.begin("one", "Previously saved")
    value = library.get(entry["id"])
    value.update(
        status="detected",
        detection={
            "sha256": "a" * 64,
            "format": "image_records",
            "convertible": True,
            "cameras": [],
        },
    )
    if mapping is not None:
        value["detection"]["mapping"] = mapping
    record = library.directory(entry["id"]) / "record.json"
    original = json.dumps(value).encode()
    record.write_bytes(original)
    source = record.parent / "source/original"
    source.write_bytes(b"preserved")

    restored = DatasetLibrary(tmp_path)
    diagnostic = restored.get(entry["id"])
    assert diagnostic["status"] == "unavailable"
    assert diagnostic["can_discard"] is False
    assert "detection" not in diagnostic
    projects = SimpleNamespace(list=AsyncMock(return_value=[]))
    execution = SimpleNamespace(list=AsyncMock(return_value=[]))
    assert asyncio.run(restored.list(projects, execution, project_id="one"))[0] == diagnostic
    assert record.read_bytes() == original
    assert source.read_bytes() == b"preserved"


@pytest.mark.parametrize("format_name", ["unknown", "rlds_tfrecord", "ros_recording", "lerobot_v3"])
def test_empty_worker_mapping_remains_available(tmp_path, format_name):
    library = DatasetLibrary(tmp_path)
    entry = library.begin("one", "Supported empty mapping")
    value = library.get(entry["id"])
    value.update(
        status="detected",
        detection={
            "sha256": "a" * 64,
            "format": format_name,
            "convertible": format_name == "lerobot_v3",
            "cameras": [],
            "mapping": {},
        },
    )
    library.save(value)
    assert DatasetLibrary(tmp_path).get(entry["id"])["detection"]["mapping"] == {}
