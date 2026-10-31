"""Dataset listing prefers an exact project-owned library snapshot without hiding fallbacks."""

import asyncio
import copy
from types import SimpleNamespace

import pytest
from vla_platform.contracts import DatasetProfile, DatasetSnapshot, IntakeRequest
from vla_platform.datasets.library import DatasetLibrary

LIBRARY_ID = "a" * 32
METADATA = "1" * 64


def profile(snapshot="2" * 64):
    return DatasetProfile(
        source="local",
        revision="metadata-sha256:" + METADATA,
        format="lerobot_v3",
        total_episodes=2,
        total_frames=4,
        fps=13,
        features={"action": {"dtype": "float32", "shape": [6]}},
        metadata_sha256=METADATA,
        inspected_at="2026-01-01T00:00:00Z",
        warnings=[],
        inspection_scope="complete_snapshot" if snapshot else "metadata_only",
        snapshot=DatasetSnapshot(
            id="sha256:" + snapshot,
            manifest_sha256=snapshot,
            total_bytes=1,
            file_count=1,
            total_episodes=2,
            total_frames=4,
            lineage_validated=False,
            warnings=[],
        )
        if snapshot
        else None,
    )


def job(ident, result, *, library_id=None, created="2026-01-01T00:00:00Z"):
    return SimpleNamespace(
        id=ident,
        result=result,
        created_at=created,
        request=IntakeRequest(source="local", library_id=library_id)
        if library_id
        else IntakeRequest(source="local", path="/synthetic/extracted"),
    )


def row(project="one", status="ready"):
    return {
        "id": LIBRARY_ID,
        "project_id": project,
        "name": "Pen demonstration",
        "status": status,
        "source": "local",
        "created_at": "2026-01-01T00:00:00Z",
        "normalized": "converted",
        "files": {},
    }


async def listing(library, records, jobs, project_id=None):
    library.records = lambda: iter(copy.deepcopy(records))

    class Projects:
        async def list(self):
            return [SimpleNamespace(id=ident) for ident in jobs]

    class Execution:
        async def list(self, ident):
            return jobs[ident]

    return await library.list(Projects(), Execution(), project_id)


def test_named_ready_snapshot_suppresses_older_and_newer_repeated_inspections(tmp_path):
    library = DatasetLibrary(tmp_path)
    records = [row()]
    original = copy.deepcopy(records)
    jobs = {
        "one": [
            job("old", profile(), created="2026-01-01T00:00:00Z"),
            job("new", profile(), created="2026-01-04T00:00:00Z"),
            job("linked", profile(), library_id=LIBRARY_ID, created="2026-01-03T00:00:00Z"),
        ]
    }
    result = asyncio.run(listing(library, records, jobs))
    assert len(result) == 1
    assert result[0]["id"] == LIBRARY_ID and result[0]["job_id"] == "linked"
    assert result[0]["training_copy"] is True and result[0]["can_discard"] is False
    assert records == original


def test_different_snapshots_and_metadata_only_rows_remain_visible(tmp_path):
    library = DatasetLibrary(tmp_path)
    jobs = {
        "one": [
            job("linked", profile(), library_id=LIBRARY_ID),
            job("different", profile("3" * 64), created="2026-01-02T00:00:00Z"),
            job("metadata-only", profile(None), created="2026-01-03T00:00:00Z"),
        ]
    }
    result = asyncio.run(listing(library, [row()], jobs))
    assert {entry["id"] for entry in result} == {
        LIBRARY_ID,
        "inspection:different",
        "inspection:metadata-only",
    }


def test_distinct_snapshot_identity_is_not_collapsed_by_equal_metadata(tmp_path):
    library = DatasetLibrary(tmp_path)
    jobs = {
        "one": [
            job("first", profile()),
            job("second", profile("3" * 64), created="2026-01-02T00:00:00Z"),
        ]
    }
    result = asyncio.run(listing(library, [], jobs))
    assert {entry["id"] for entry in result} == {"inspection:first", "inspection:second"}


def test_same_snapshot_in_another_project_is_not_suppressed_or_relinked(tmp_path):
    library = DatasetLibrary(tmp_path)
    jobs = {
        "one": [job("owner-linked", profile(), library_id=LIBRARY_ID)],
        "two": [job("other-linked", profile(), library_id=LIBRARY_ID)],
    }
    result = asyncio.run(listing(library, [row()], jobs))
    assert {entry["id"] for entry in result} == {LIBRARY_ID, "inspection:other-linked"}
    owned = next(entry for entry in result if entry["id"] == LIBRARY_ID)
    assert owned["job_id"] == "owner-linked"
    scoped = asyncio.run(listing(library, [row()], jobs, "two"))
    assert [entry["id"] for entry in scoped] == ["inspection:other-linked"]


@pytest.mark.parametrize("records", [[], [row(None, "unavailable")], [row("one", "failed")]])
def test_missing_or_unavailable_library_preserves_the_linked_intake_fallback(tmp_path, records):
    library = DatasetLibrary(tmp_path)
    jobs = {"one": [job("accepted", profile(), library_id=LIBRARY_ID)]}
    result = asyncio.run(listing(library, records, jobs, "one"))
    fallback = next(entry for entry in result if entry["id"] == "inspection:accepted")
    assert fallback["job_id"] == "accepted" and fallback["training_copy"] is True
    assert fallback["status"] == "ready"


def test_metadata_only_library_link_does_not_claim_a_complete_snapshot(tmp_path):
    library = DatasetLibrary(tmp_path)
    jobs = {
        "one": [
            job("metadata-link", profile(None), library_id=LIBRARY_ID),
            job("complete", profile(), created="2026-01-02T00:00:00Z"),
        ]
    }
    result = asyncio.run(listing(library, [row()], jobs))
    assert {entry["id"] for entry in result} == {LIBRARY_ID, "inspection:complete"}


def test_one_library_represents_its_latest_snapshot_and_keeps_distinct_history(tmp_path):
    library = DatasetLibrary(tmp_path)
    jobs = {
        "one": [
            job("old-linked", profile("3" * 64), library_id=LIBRARY_ID),
            job("latest-linked", profile(), library_id=LIBRARY_ID, created="2026-01-02T00:00:00Z"),
            job("new-repeat", profile(), created="2026-01-03T00:00:00Z"),
        ]
    }
    result = asyncio.run(listing(library, [row()], jobs))
    assert {entry["id"] for entry in result} == {LIBRARY_ID, "inspection:old-linked"}
    owned = next(entry for entry in result if entry["id"] == LIBRARY_ID)
    assert owned["job_id"] == "latest-linked"
    previous = next(entry for entry in result if entry["id"] == "inspection:old-linked")
    assert previous["profile"]["snapshot"]["id"] == "sha256:" + "3" * 64
