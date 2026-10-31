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
