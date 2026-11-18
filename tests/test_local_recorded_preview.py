"""Recorded PNG frames remain intact while a bounded playback cache is published."""

import hashlib
import json
import subprocess
from pathlib import Path

from test_dataset_library import run_worker
from vla_platform.datasets import import_worker
from vla_platform.datasets.local_preview import reader_python

FIXTURE = r"""
import io,json,sys
from pathlib import Path
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
root=Path(sys.argv[1]);count=int(sys.argv[2]);bad=sys.argv[3]=='bad'
(root/'meta').mkdir(parents=True);(root/'data/chunk-000').mkdir(parents=True)
info={'codebase_version':'v3.0','fps':10,'total_frames':count,'total_episodes':2 if count==6 else 1,
'features':{'observation.images.front':{'dtype':'image','shape':[16,16,3]},'observation.state':{'dtype':'float32','shape':[6]},'action':{'dtype':'float32','shape':[6]}}}
(root/'meta/info.json').write_text(json.dumps(info))
rows=[]
for i in range(count):
 b=io.BytesIO();Image.new('RGB',(16,16),((i*70)%255,10,100)).save(b,format='PNG')
 ep=i//3 if count==6 else 0;frame=i%3 if count==6 else i
 rows.append({'episode_index':ep,'frame_index':frame,'timestamp':float('nan') if bad else frame/10,
 'observation.images.front':{'bytes':b.getvalue(),'path':None},'observation.state':[0.,1.,2.,3.,4.,5.],
 'action':[1.,2.,3.,4.,5.,6.],'task':'Move the test object'})
pq.write_table(pa.Table.from_pylist(rows),root/'data/chunk-000/file-000.parquet')
"""


def fixture(root, count=6, bad=False):
    subprocess.run(
        [str(reader_python()), "-I", "-c", FIXTURE, str(root), str(count), "bad" if bad else "ok"],
        check=True,
        capture_output=True,
    )


def hashes(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def test_recorded_frames_play_in_order_with_timestamps_and_joint_values(tmp_path):
    source, output = tmp_path / "recording", tmp_path / "previews"
    fixture(source)
    before = hashes(source)
    result = run_worker(source, "preview", output=str(output))
    samples = json.loads((tmp_path / "samples.json").read_text())
    assert result["samples"] == 6 and not result["truncated"]
    assert [(s["episode_index"], s["frame_index"]) for s in samples] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    ]
    assert [s["timestamp"] for s in samples[:3]] == [0, 0.1, 0.2]
    assert samples[0]["state"] == [0, 1, 2, 3, 4, 5]
    assert samples[0]["action"] == [1, 2, 3, 4, 5, 6]
    assert (output / samples[0]["path"]).read_bytes() != (output / samples[1]["path"]).read_bytes()
    assert hashes(source) == before


def test_preview_cache_stops_at_its_frame_budget(tmp_path):
    source, output = tmp_path / "recording", tmp_path / "previews"
    fixture(source, 2401)
    result = run_worker(source, "preview", output=str(output))
    samples = json.loads((tmp_path / "samples.json").read_text())
    assert result["truncated"] and result["samples"] == 2400
    assert samples[-1]["frame_index"] == 2399 and samples[-1]["preview_truncated"]
    assert len(list(output.glob("*.jpg"))) == 2400


def test_failed_preview_keeps_the_previous_published_manifest(tmp_path):
    source, output = tmp_path / "recording", tmp_path / "previews"
    fixture(source, 1, bad=True)
    prior = b'[{"path":"saved.jpg"}]'
    manifest = tmp_path / "samples.json"
    manifest.write_bytes(prior)
    result = subprocess.run(
        [str(reader_python()), "-I", str(Path(import_worker.__file__))],
        input=json.dumps(
            {"operation": "preview", "root": str(source), "output": str(output)}
        ).encode(),
        capture_output=True,
    )
    assert result.returncode != 0 and "error" in json.loads(result.stdout)
    assert manifest.read_bytes() == prior


def test_preview_reads_large_recording_in_bounded_row_groups(tmp_path):
    source, output = tmp_path / "recording", tmp_path / "previews"
    fixture(source, 10)
    script = r"""
import sys
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
path=Path(sys.argv[1])/"data/chunk-000/file-000.parquet"
table=pq.read_table(path)
padding=b"x"*(60*1024**2)
schema=table.schema.append(pa.field("padding",pa.binary()))
with pq.ParquetWriter(path,schema,compression="zstd",use_dictionary=False) as writer:
    for i in range(len(table)):
        writer.write_table(table.slice(i,1).append_column("padding",pa.array([padding])))
metadata=pq.ParquetFile(path).metadata
assert sum(metadata.row_group(i).total_byte_size for i in range(10))>512*1024**2
"""
    subprocess.run([str(reader_python()), "-I", "-c", script, str(source)], check=True)
    before = hashes(source)
    result = run_worker(source, "preview", output=str(output))
    assert result["samples"] == 10
    assert hashes(source) == before


def test_first_thumbnail_survives_a_later_preview_error(tmp_path):
    source, output = tmp_path / "recording", tmp_path / "previews"
    fixture(source)
    script = r"""
import sys
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
p=Path(sys.argv[1])/"data/chunk-000/file-000.parquet"
table=pq.read_table(p); rows=table.to_pylist(); rows[-1]["timestamp"]=float("nan")
pq.write_table(pa.Table.from_pylist(rows,schema=table.schema),p)
"""
    subprocess.run([str(reader_python()), "-I", "-c", script, str(source)], check=True)
    result = subprocess.run(
        [str(reader_python()), "-I", str(Path(import_worker.__file__))],
        input=json.dumps(
            {"operation": "preview", "root": str(source), "output": str(output)}
        ).encode(),
        capture_output=True,
    )
    assert result.returncode != 0
    samples = json.loads((tmp_path / "samples.json").read_text())
    assert samples[0]["frame_index"] == 0
    assert (output / samples[0]["path"]).is_file()


def test_samples_request_repairs_an_old_missing_preview_cache(tmp_path):
    import asyncio
    import shutil

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from vla_platform.dataset_library_api import router
    from vla_platform.datasets.library import DatasetLibrary

    source = tmp_path / "recording"
    fixture(source)
    before = hashes(source)
    library = DatasetLibrary(tmp_path / "workspace")

    async def upload():
        entry = library.begin("project-one", "Saved recording")
        for path in source.rglob("*"):
            if path.is_file():

                async def stream(path=path):
                    yield path.read_bytes()

                await library.upload(entry["id"], path.relative_to(source).as_posix(), stream())
        return await library.finalize(entry["id"])

    entry = asyncio.run(upload())
    directory = library.directory(entry["id"])
    (directory / "samples.json").unlink()
    shutil.rmtree(directory / "previews")
    app = FastAPI()
    app.include_router(router)
    app.state.dataset_library = library
    with TestClient(app) as client:
        response = client.get("/api/v1/datasets/" + entry["id"] + "/samples")
        assert response.status_code == 200
        samples = response.json()
        assert samples
        image = client.get("/api/v1/datasets/" + entry["id"] + "/preview/" + samples[0]["path"])
        assert image.status_code == 200 and image.headers["content-type"] == "image/jpeg"
        modified = (directory / "samples.json").stat().st_mtime_ns
        assert client.get("/api/v1/datasets/" + entry["id"] + "/samples").json() == samples
        assert (directory / "samples.json").stat().st_mtime_ns == modified
        client.portal.call(library.close)
    assert hashes(source) == before
