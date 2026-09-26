import asyncio
import json

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient
from vla_platform import api
from vla_platform.contracts import IntakeRequest, Job, now
from vla_platform.datasets.explore import (
    Budget,
    DatasetExplorer,
    ExplorationError,
    HubReader,
    allowed_url,
    frame_samples,
    render_path,
    safe_path,
    source_profile,
)
from vla_platform.datasets.inspect import profile
from vla_platform.settings import Settings

SHA = "a" * 40
CAMERA = "observation.images.wrist"


def parquet(rows):
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(rows), sink)
    return sink.getvalue().to_pybytes()


@pytest.fixture(params=["v2.1", "v3.0"])
def dataset(request):
    version = request.param
    v3 = version.startswith("v3")
    info = {
        "codebase_version": version,
        "total_episodes": 2,
        "total_frames": 12,
        "fps": 5,
        "chunks_size": 1,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        if v3
        else ("data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"),
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        if v3
        else "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [2], "names": ["shoulder", "gripper"]},
            "observation.state": {
                "dtype": "float32",
                "shape": [2],
                "names": {"motors": ["shoulder", "gripper"]},
            },
            CAMERA: {
                "dtype": "video",
                "shape": [3, 240, 320],
                "names": ["channels", "height", "width"],
                "info": {"video.fps": 5},
            },
        },
    }
    raw = json.dumps(info).encode()
    req = IntakeRequest(repo_id="fixture/robot", revision="main")
    job = Job(
        id="done",
        project_id="p",
        request=req,
        status="succeeded",
        created_at=now(),
        updated_at=now(),
        result=profile(raw, req, SHA),
    )
    rows = [{"episode_index": i, "length": 6, "tasks": [f"Task {i}"]} for i in range(2)]
    for row in rows:
        row.update(
            {
                "data/chunk_index": 2,
                "data/file_index": 4,
                f"videos/{CAMERA}/chunk_index": 3,
                f"videos/{CAMERA}/file_index": 7,
                f"videos/{CAMERA}/from_timestamp": 4 + row["episode_index"] * 1.2,
                f"videos/{CAMERA}/to_timestamp": 5.2 + row["episode_index"] * 1.2,
            }
        )
    frames = [
        {
            "episode_index": episode,
            "frame_index": frame,
            "timestamp": frame / 5,
            "action": [episode * 100 + frame, 0.5],
            "observation.state": [frame, 0.1],
        }
        for episode in range(2)
        for frame in range(6)
    ]
    data_path = (
        "data/chunk-002/file-004.parquet" if v3 else ("data/chunk-001/episode_000001.parquet")
    )
    video_path = (
        f"videos/{CAMERA}/chunk-003/file-007.mp4"
        if v3
        else (f"videos/chunk-001/{CAMERA}/episode_000001.mp4")
    )
    files = {"meta/info.json": raw, data_path: parquet(frames), video_path: b"video-not-downloaded"}
    if v3:
        files["meta/episodes/chunk-000/file-000.parquet"] = parquet(rows)
    else:
        files["meta/episodes.jsonl"] = b"\n".join(json.dumps(row).encode() for row in rows)
    calls = []

    def respond(request):
        calls.append((request.method, str(request.url)))
        assert SHA in request.url.path, "Every request must use the inspected SHA, never main"
        if "/tree/" in request.url.path:
            return httpx.Response(
                200, json=[{"type": "file", "path": "meta/episodes/chunk-000/file-000.parquet"}]
            )
        key = request.url.path.split(f"/resolve/{SHA}/", 1)[1]
        if key in files:
            assert not key.endswith(".mp4") or request.method == "HEAD"
            return httpx.Response(200, content=files[key])
        return httpx.Response(404)

    return job, files, calls, respond, data_path, video_path


def test_preview_uses_versioned_paths_selected_episode_rows_and_video_offsets(dataset):
    job, _, calls, respond, data_path, video_path = dataset

    async def run():
        explorer = DatasetExplorer(httpx.AsyncClient(transport=httpx.MockTransport(respond)))
        try:
            page = await explorer.page(job, offset=1, limit=1)
            assert page.total_episodes == 2
            assert [episode.episode_index for episode in page.episodes] == [1]
            assert page.episodes[0].duration_seconds == 1.2
            preview = await explorer.preview(job, 1)
            assert preview.tasks == ["Task 1"]
            assert len(preview.samples) == 5
            assert preview.samples[0].action == [100, 0.5]
            assert preview.samples[-1].frame_index == 4
            assert preview.samples[-1].timestamp == 0.8
            assert preview.action_names == preview.state_names == ["shoulder", "gripper"]
            camera = preview.cameras[0]
            assert camera.url.endswith(video_path)
            assert (camera.width, camera.height, camera.fps) == (320, 240, 5)
            assert camera.start_seconds == pytest.approx(
                5.2 if job.result.format == "lerobot_v3" else 0
            )
            assert camera.end_seconds == pytest.approx(
                6.4 if job.result.format == "lerobot_v3" else 1.2
            )
            assert preview.warnings == []
            assert any(url.endswith(data_path) for _, url in calls)
            assert (await explorer.page(job, offset=100)).episodes == []
        finally:
            await explorer.close()

    asyncio.run(run())


def test_missing_video_and_corrupt_parquet_return_honest_partial_preview(dataset):
    job, files, _, respond, data_path, video_path = dataset
    del files[video_path]
    files[data_path] = b"not parquet"

    async def run():
        explorer = DatasetExplorer(httpx.AsyncClient(transport=httpx.MockTransport(respond)))
        try:
            preview = await explorer.preview(job, 1)
            assert preview.cameras == [] and preview.samples == []
            assert any("not found" in message for message in preview.warnings)
            assert any("Sample rows unavailable" in message for message in preview.warnings)
        finally:
            await explorer.close()

    asyncio.run(run())


def test_metadata_snapshot_must_match_inspection(dataset):
    job, files, _, respond, _, _ = dataset
    files["meta/info.json"] += b" "

    async def run():
        explorer = DatasetExplorer(httpx.AsyncClient(transport=httpx.MockTransport(respond)))
        try:
            with pytest.raises(ExplorationError, match="snapshot"):
                await explorer.page(job)
        finally:
            await explorer.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "path",
    [
        "../secret",
        "/absolute",
        "data/../secret",
        "https://evil.test/file",
        "data//file",
        "data/%2e%2e/file",
        "data/file?x=1",
        "data\\file",
        "data/\x00file",
    ],
)
def test_unsafe_paths_rejected(path):
    with pytest.raises(ExplorationError, match="unsafe"):
        safe_path(path)


@pytest.mark.parametrize(
    "template",
    [
        "data/{chunk_index.__class__}.parquet",
        "data/{chunk_index[0]}.parquet",
        "data/{chunk_index:999999999999d}.parquet",
        "data/{unknown}.parquet",
        "data/{chunk_index!r}.parquet",
        "../{chunk_index}.parquet",
    ],
)
def test_unsafe_templates_rejected(template):
    with pytest.raises(ExplorationError):
        render_path(template, chunk_index=0)


def test_hub_hosts_and_redirects_are_restricted():
    assert allowed_url("https://cas-bridge.xethub.hf.co/file")
    assert not allowed_url("https://huggingface.co.attacker.test/file")
    assert not allowed_url("http://huggingface.co/file")
    assert not allowed_url("https://user:password@huggingface.co/file")
    assert not allowed_url("https://huggingface.co:8443/file")

    async def run():
        calls = []

        def respond(request):
            calls.append(str(request.url))
            return httpx.Response(302, headers={"location": "https://attacker.test/file"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            with pytest.raises(ExplorationError, match="outside"):
                await HubReader(client).read("https://huggingface.co/file", 1024, Budget())
        assert len(calls) == 1

    asyncio.run(run())


def test_download_limits_apply_even_without_content_length():
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, content=b"123456789", headers={"content-length": "0"})
            )
        ) as client:
            with pytest.raises(ExplorationError, match="size limit"):
                await HubReader(client).read("https://huggingface.co/file", 8, Budget())

    asyncio.run(run())


def test_samples_do_not_fabricate_rows_and_reject_missing_index():
    raw = parquet([{"episode_index": 0, "frame_index": 0, "timestamp": 0.0, "action": [1]}])
    assert frame_samples(raw, 1) == []
    with pytest.raises(ExplorationError, match="missing"):
        frame_samples(parquet([{"action": [1]}]), 0)


def test_preview_routes_validate_state_bounds_and_serialize_contract(
    tmp_path, monkeypatch, dataset
):
    job, _, _, respond, _, _ = dataset

    async def get(_, job_id):
        if job_id == "done":
            return job
        if job_id == "pending":
            return job.model_copy(update={"status": "queued", "result": None})
        if job_id == "local":
            return job.model_copy(
                update={"result": job.result.model_copy(update={"source": "local"})}
            )
        return None

    monkeypatch.setattr(api.Execution, "get", get)
    monkeypatch.setattr(
        api,
        "DatasetExplorer",
        lambda: DatasetExplorer(httpx.AsyncClient(transport=httpx.MockTransport(respond))),
    )
    with TestClient(api.create_app(Settings(data_dir=tmp_path))) as client:
        base = "/api/v1/jobs"
        assert client.get(f"{base}/missing/episodes").status_code == 404
        assert client.get(f"{base}/pending/episodes").status_code == 409
        assert client.get(f"{base}/local/episodes").status_code == 422
        assert client.get(f"{base}/done/episodes?offset=-1").status_code == 422
        assert client.get(f"{base}/done/episodes?limit=25").status_code == 422
        assert client.get(f"{base}/done/episodes/2").status_code == 404
        assert client.get(f"{base}/done/episodes/-1").status_code == 404
        page = client.get(f"{base}/done/episodes?offset=1&limit=1")
        assert page.status_code == 200, page.text
        assert page.json()["episodes"][0]["episode_index"] == 1
        preview = client.get(f"{base}/done/episodes/1")
        assert preview.status_code == 200, preview.text
        assert preview.json()["samples"][0]["action"] == [100, 0.5]
        assert preview.json()["revision"] == SHA


def test_mutable_revision_rejected(dataset):
    job = dataset[0]
    job.result.revision = "main"
    with pytest.raises(ExplorationError, match="immutable"):
        source_profile(job)


def test_malformed_camera_metadata_preserves_other_episode_details(dataset):
    job, files, _, respond, _, _ = dataset
    info = json.loads(files["meta/info.json"])
    info["features"][CAMERA]["info"] = ["malformed"]
    files["meta/info.json"] = json.dumps(info).encode()
    job.result = profile(files["meta/info.json"], job.request, SHA)

    async def run():
        explorer = DatasetExplorer(httpx.AsyncClient(transport=httpx.MockTransport(respond)))
        try:
            preview = await explorer.preview(job, 1)
            assert preview.cameras == []
            assert len(preview.samples) == 5
            assert any("Camera metadata" in warning for warning in preview.warnings)
        finally:
            await explorer.close()

    asyncio.run(run())


def test_v3_invalid_video_segment_is_not_exposed(dataset):
    job, files, _, respond, _, _ = dataset
    if job.result.format != "lerobot_v3":
        return
    key = "meta/episodes/chunk-000/file-000.parquet"
    rows = pq.read_table(pa.BufferReader(files[key])).to_pylist()
    rows[1][f"videos/{CAMERA}/to_timestamp"] = 1.0
    files[key] = parquet(rows)

    async def run():
        explorer = DatasetExplorer(httpx.AsyncClient(transport=httpx.MockTransport(respond)))
        try:
            preview = await explorer.preview(job, 1)
            assert preview.cameras == []
            assert any("time boundaries" in warning for warning in preview.warnings)
        finally:
            await explorer.close()

    asyncio.run(run())


def test_concurrent_cache_replacement_preserves_exact_size_and_eviction_budget(monkeypatch):
    monkeypatch.setattr("vla_platform.datasets.explore.MAX_PARQUET_BYTES", 10)

    async def run():
        requests_started = 0
        both_started = asyncio.Event()
        files = {"/same": b"123456", "/second": b"7890", "/third": b"abcdef"}

        async def respond(request):
            nonlocal requests_started
            if request.url.path == "/same":
                requests_started += 1
                if requests_started == 2:
                    both_started.set()
                await both_started.wait()
            return httpx.Response(200, content=files[request.url.path])

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            reader = HubReader(client)
            url = "https://huggingface.co/same"
            budgets = [Budget(), Budget()]
            results = await asyncio.wait_for(
                asyncio.gather(*(reader.read(url, 10, budget) for budget in budgets)), timeout=2
            )
            assert results == [b"123456", b"123456"]
            assert requests_started == 2
            assert [budget.used for budget in budgets] == [6, 6]
            assert list(reader.cache) == [url]
            assert reader.cache_size == sum(map(len, reader.cache.values())) == 6
            await reader.read("https://huggingface.co/second", 10, Budget())
            assert len(reader.cache) == 2
            assert reader.cache_size == sum(map(len, reader.cache.values())) == 10
            await reader.read("https://huggingface.co/third", 10, Budget())
            assert list(reader.cache) == [
                "https://huggingface.co/second",
                "https://huggingface.co/third",
            ]
            assert reader.cache_size == sum(map(len, reader.cache.values())) == 10

    asyncio.run(run())


def test_dictionary_encoded_tasks_are_preserved_and_materialization_is_bounded():
    from vla_platform.datasets.explore import index_rows, parquet_file

    task = "x" * 4096
    rows = [{"episode_index": i, "length": 1, "tasks": [task]} for i in range(1000)]
    raw = parquet(rows)
    assert len(raw) < 100_000
    field = parquet_file(raw).schema_arrow.field("tasks")
    assert pa.types.is_dictionary(field.type.value_type)
    # The small encoded input would otherwise expand all repeated task strings.
    with pytest.raises(ExplorationError, match="materialized preview size limit"):
        index_rows(raw)
    assert index_rows(parquet(rows[:2])) == rows[:2]


def test_task_and_vector_lengths_are_checked_before_python_conversion():
    from vla_platform.datasets.explore import index_rows

    for tasks in [["x" * 4097], ["short"] * 21]:
        with pytest.raises(ExplorationError, match="length limit"):
            index_rows(parquet([{"episode_index": 0, "length": 1, "tasks": tasks}]))
    with pytest.raises(ExplorationError, match="sample vector.*length limit"):
        frame_samples(
            parquet(
                [
                    {
                        "episode_index": 0,
                        "frame_index": 0,
                        "timestamp": 0.0,
                        "action": [1.0] * 1025,
                    }
                ]
            ),
            0,
        )


def test_projected_leaf_counts_reject_encoded_list_expansion_before_iteration(monkeypatch):
    from vla_platform.datasets.explore import index_rows

    monkeypatch.setattr("vla_platform.datasets.explore.MAX_PROJECTED_VALUES", 10)
    raw = parquet([{"episode_index": 0, "length": 1, "tasks": ["same"] * 9}])
    with pytest.raises(ExplorationError, match="decoded preview value limit"):
        index_rows(raw)
    raw = parquet(
        [
            {
                "episode_index": 0,
                "frame_index": 0,
                "timestamp": 0.0,
                "action": [1.0] * 9,
            }
        ]
    )
    with pytest.raises(ExplorationError, match="decoded preview value limit"):
        frame_samples(raw, 0)


@pytest.mark.parametrize("row_group_size", [1, 2])
def test_episode_index_handles_distinct_task_dictionaries_across_row_groups(row_group_size):
    from vla_platform.datasets.explore import index_rows

    rows = [{"episode_index": i, "length": 6, "tasks": [f"Task {i}"]} for i in range(5)]
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(rows), sink, row_group_size=row_group_size)
    assert index_rows(sink.getvalue().to_pybytes()) == rows


def test_episode_index_materialization_budget_spans_row_groups(monkeypatch):
    from vla_platform.datasets.explore import index_rows

    monkeypatch.setattr("vla_platform.datasets.explore.MAX_INDEX_MATERIALIZED_BYTES", 2000)
    rows = [{"episode_index": i, "length": 6, "tasks": ["x" * 200]} for i in range(3)]
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(rows), sink, row_group_size=1)
    with pytest.raises(ExplorationError, match="materialized preview size limit"):
        index_rows(sink.getvalue().to_pybytes())
