"""Separate dataset operation history using synthetic in-process workers."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from vla_platform.datasets.library import DatasetLibrary, LibraryError
from vla_platform.workspace_jobs import dataset_job

DETECTION = {
    "sha256": "a" * 64,
    "format": "image_records",
    "convertible": True,
    "cameras": ["front"],
    "mapping": {},
}
CONVERTED = {"cameras": ["front"], "frames": 2, "episodes": 1}


async def source(library):
    entry = library.begin("project", "Original dataset")

    async def chunks():
        yield b"ORIGINAL"

    await library.upload(entry["id"], "source.txt", chunks())
    value = library.get(entry["id"])
    value.update(status="detected", detection=dict(DETECTION))
    library.save(value)
    return entry["id"]


def test_import_has_its_own_identity_and_private_links_are_not_public(tmp_path):
    library = DatasetLibrary(tmp_path)
    entry = library.begin("project", "An upload")
    operations = library.operation_records("project")
    assert len(operations) == 1
    job = operations[0]
    assert job["id"] != entry["id"] and job["dataset_id"] == entry["id"]
    assert job["operation"] == "dataset.import" and job["status"] == "waiting"
    assert not any(key.startswith("_") for key in entry)
    assert library.get(entry["id"])["_import_job_id"] == job["id"]
    assert DatasetLibrary(tmp_path, reconcile=False).operation_records("project") == operations


def test_detected_import_is_complete_before_conversion_starts(tmp_path):
    async def exercise():
        library = DatasetLibrary(tmp_path)
        ident = await source(library)
        imported = library.operation_records("project")[0]
        assert imported["status"] == "succeeded"
        assert imported["operation"] == "dataset.import"
        assert library.get(ident)["status"] == "detected"

    asyncio.run(exercise())


def test_conversion_retries_preserve_each_attempt_and_source_bytes(tmp_path, monkeypatch):
    async def exercise():
        library = DatasetLibrary(tmp_path)
        ident = await source(library)

        async def failed(payload, timeout=300):
            raise LibraryError("Synthetic conversion failure")

        monkeypatch.setattr(library, "worker", failed)
        await library.start_conversion(ident, {"fps": 10})
        await library.tasks[ident]
        first = next(
            row for row in library.operation_records() if row["operation"] == "dataset.convert"
        )
        assert first["status"] == "failed" and first["inputs"]["conversion"] == {"fps": 10}

        async def successful(payload, timeout=300):
            Path(payload["output"]).mkdir()
            return dict(CONVERTED)

        monkeypatch.setattr(library, "worker", successful)
        await library.start_conversion(ident, {"fps": 20})
        await library.tasks[ident]
        conversions = [
            row for row in library.operation_records() if row["operation"] == "dataset.convert"
        ]
        assert len(conversions) == 2 and conversions[0]["id"] != conversions[1]["id"]
        assert {row["status"] for row in conversions} == {"failed", "succeeded"}
        assert library.operations.get(first["id"]) == first
        assert (library.directory(ident) / "source/source.txt").read_bytes() == b"ORIGINAL"
        completed = next(row for row in conversions if row["status"] == "succeeded")
        projected = dataset_job(completed)
        assert projected.source_id == completed["id"]
        assert projected.inputs["dataset_id"] == ident and projected.parent_ids
        assert projected.outputs[0].url.endswith(ident + "/download")
        await library.close()

    asyncio.run(exercise())


def test_cancellation_waits_for_owned_worker_and_does_not_cancel_later_retry(tmp_path, monkeypatch):
    async def exercise():
        library = DatasetLibrary(tmp_path)
        ident = await source(library)
        started, cleaned = asyncio.Event(), asyncio.Event()

        async def blocked(payload, timeout=300):
            Path(payload["output"]).mkdir()
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        monkeypatch.setattr(library, "worker", blocked)
        await library.start_conversion(ident, {"fps": 10})
        await started.wait()
        first = library.get(ident)["_conversion_job_id"]
        await library.cancel_conversion(ident, expected_job_id=first)
        assert cleaned.is_set()
        assert library.operations.get(first)["status"] == "cancelled"
        assert not (library.directory(ident) / "converted.pending").exists()
        started.clear()
        await library.start_conversion(ident, {"fps": 20})
        await started.wait()
        second = library.get(ident)["_conversion_job_id"]
        assert second != first
        with pytest.raises(LibraryError) as error:
            await library.cancel_conversion(ident, expected_job_id=first)
        assert error.value.status == 409
        assert library.get(ident)["status"] == "converting"
        await library.cancel_conversion(ident, expected_job_id=second)
        assert library.operations.get(first)["status"] == "cancelled"
        assert library.operations.get(second)["status"] == "cancelled"
        assert (library.directory(ident) / "source/source.txt").read_bytes() == b"ORIGINAL"
        await library.close()

    asyncio.run(exercise())


def test_cancel_before_worker_entry_still_records_terminal_operation(tmp_path):
    async def exercise():
        library = DatasetLibrary(tmp_path)
        ident = await source(library)
        await library.start_conversion(ident, {"fps": 10})
        job = library.get(ident)["_conversion_job_id"]
        await library.cancel_conversion(ident, expected_job_id=job)
        assert library.operations.get(job)["status"] == "cancelled"
        await library.close()

    asyncio.run(exercise())


def test_restart_reconciles_abandoned_conversion_without_reexecuting_it(tmp_path):
    async def exercise():
        library = DatasetLibrary(tmp_path)
        ident = await source(library)
        await library.start_conversion(ident, {"fps": 10})
        job = library.get(ident)["_conversion_job_id"]
        task = library.tasks[ident]
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        restored = DatasetLibrary(tmp_path)
        assert restored.operations.get(job)["status"] == "interrupted"
        assert restored.get(ident)["status"] == "failed" and not restored.tasks
        assert (restored.directory(ident) / "source/source.txt").read_bytes() == b"ORIGINAL"

    asyncio.run(exercise())


def test_duplicate_import_preserves_successful_redirect_history(tmp_path, monkeypatch):
    async def exercise():
        library = DatasetLibrary(tmp_path)
        original = await source(library)
        duplicate = library.begin("project", "Duplicate upload")

        async def chunks():
            yield b"ORIGINAL"

        await library.upload(duplicate["id"], "source.txt", chunks())
        monkeypatch.setattr(library, "extract", lambda ident: None)

        async def detected(payload, timeout=300):
            return dict(DETECTION)

        monkeypatch.setattr(library, "worker", detected)
        reused = await library.finalize(duplicate["id"])
        assert reused["id"] == original
        jobs = library.operation_records("project")
        assert len(jobs) == 2 and all(row["status"] == "succeeded" for row in jobs)
        redirected = next(row for row in jobs if row["dataset_id"] == duplicate["id"])
        assert redirected["redirect_dataset_id"] == original
        assert dataset_job(redirected).inputs["dataset_id"] == original
        assert not library.directory(duplicate["id"]).exists()
        assert library.directory(original).is_dir()

    asyncio.run(exercise())


def test_discard_preserves_the_cancelled_import_job(tmp_path):
    async def exercise():
        library = DatasetLibrary(tmp_path)
        entry = library.begin("project", "Discarded upload")

        async def jobs(project):
            return []

        await library.discard(entry["id"], "project", SimpleNamespace(list=jobs))
        assert not library.directory(entry["id"]).exists()
        history = library.operation_records("project")
        assert len(history) == 1 and history[0]["status"] == "cancelled"

    asyncio.run(exercise())


def test_journal_paths_cannot_escape_storage(tmp_path):
    library = DatasetLibrary(tmp_path)
    with pytest.raises(ValueError):
        library.operations.get("../record")


def test_mismatched_journal_identity_is_not_projected_as_another_job(tmp_path):
    library = DatasetLibrary(tmp_path)
    library.begin("project", "Upload")
    record = library.operation_records()[0]
    path = library.operations.path(record["id"])
    path.write_text(path.read_text().replace(record["id"], "b" * 32))
    with pytest.raises(ValueError):
        library.operations.get(record["id"])
    assert library.operation_records() == []
