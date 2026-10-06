import asyncio
import csv
import io
import json
import shutil
import zipfile

from test_dataset_library import run_worker, terminal
from vla_platform.datasets.library import DatasetLibrary


def test_csv_records_and_valid_zip_reuse_the_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    run_worker(source, "demo")
    records = [json.loads(line) for line in (source / "frames.jsonl").read_text().splitlines()]
    with (source / "frames.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(
            {k: json.dumps(v) if isinstance(v, list) else v for k, v in r.items()} for r in records
        )
    (source / "frames.jsonl").unlink()
    detected = run_worker(source, "detect")
    assert detected["format"] == "image_records"

    async def check():
        library = DatasetLibrary(tmp_path / "workspace")
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    archive.write(path, "my-dataset/" + path.relative_to(source).as_posix())

        async def chunks():
            yield payload.getvalue()

        first = library.begin("one", "CSV source")
        await library.upload(first["id"], "dataset.zip", chunks())
        finalized = await library.finalize(first["id"])
        assert finalized["detection"]["sha256"] == detected["sha256"]
        duplicate = library.begin("one", "same source")
        await library.upload(duplicate["id"], "dataset.zip", chunks())
        assert (await library.finalize(duplicate["id"]))["id"] == first["id"]
        await library.start_conversion(first["id"], {"fps": 6, "task": "CSV example"})
        converted = await terminal(library, first["id"])
        assert converted["status"] == "ready", converted
        assert converted["converted"]["frames"] == 24
        await library.close()

    asyncio.run(check())


def test_lerobot_v2_conversion_and_v3_import(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    detected = run_worker(source, "demo")
    v3 = tmp_path / "v3"
    run_worker(
        source,
        "convert",
        output=str(v3),
        sha256=detected["sha256"],
        settings={"fps": 6, "task": "Synthetic task"},
    )
    assert run_worker(v3, "detect")["format"] == "lerobot_v3"
    v2 = tmp_path / "v2"
    shutil.copytree(v3, v2)
    info = json.loads((v2 / "meta/info.json").read_text())
    info["codebase_version"] = "v2.1"
    info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    info["video_path"] = (
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    )
    (v2 / "meta/info.json").write_text(json.dumps(info))
    (v2 / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "Original task"}) + "\n"
    )
    for index in range(2):
        old = v2 / f"data/chunk-000/file-{index:03d}.parquet"
        old.rename(v2 / f"data/chunk-000/episode_{index:06d}.parquet")
        for camera in ("observation.images.front", "observation.images.wrist"):
            target = v2 / f"videos/chunk-000/{camera}/episode_{index:06d}.mp4"
            target.parent.mkdir(parents=True, exist_ok=True)
            (v2 / f"videos/{camera}/chunk-000/file-{index:03d}.mp4").rename(target)
    detected = run_worker(v2, "detect")
    assert detected["format"] == "lerobot_v2"
    result = run_worker(
        v2,
        "convert",
        output=str(tmp_path / "normalized"),
        sha256=detected["sha256"],
        settings={"fps": 6, "task": "Fallback task"},
    )
    assert result["frames"] == 24 and len(result["cameras"]) == 2

    async def check():
        library = DatasetLibrary(tmp_path / "workspace")
        entry = library.begin("one", "Imported LeRobot")
        for path in sorted(v3.rglob("*")):
            if path.is_file():

                async def chunks(path=path):
                    yield path.read_bytes()

                await library.upload(entry["id"], path.relative_to(v3).as_posix(), chunks())
        imported = await library.finalize(entry["id"])
        assert imported["status"] == "ready", imported
        assert imported["detection"]["format"] == "lerobot_v3"
        assert len(library.samples(entry["id"])) == 4
        await library.close()

    asyncio.run(check())
