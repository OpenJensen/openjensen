"""Admit real finalized Robot Lab image recordings without changing originals."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from vla_platform.datasets.local_preview import reader_python
from vla_platform.datasets.snapshots import SnapshotError, create_snapshot, verify_snapshot


@pytest.fixture
def recording(tmp_path):
    source = tmp_path / "recordings"
    source.mkdir()
    script = """
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(sys.argv[2]) / 'workers/robot_lab'))
from dataset import DatasetWriter
from hardware import JOINTS
writer = DatasetWriter(Path(sys.argv[1]), 'pen-demo', 'Pick up the pen', 10, (16,16,3))
for ep in range(2):
    writer.start_episode()
    for frame in range(3):
        values = {joint: float(frame + ep) for joint in JOINTS}
        writer.add_frame(values, values, np.full((16,16,3), 80 + frame, dtype=np.uint8),
                         captured_at=ep + frame / 10)
    writer.save_episode()
writer.finalize()
"""
    subprocess.run(
        [
            str(reader_python()),
            "-I",
            "-c",
            script,
            str(source),
            str(Path(__file__).resolve().parents[1]),
        ],
        check=True,
        timeout=30,
    )
    return source / "pen-demo"


def snapshot(recording, tmp_path):
    return create_snapshot(recording.parent, recording, tmp_path / "snapshots")


def test_finalized_robot_image_recording_is_imported_without_mutation(recording, tmp_path):
    before = {
        str(p.relative_to(recording)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in recording.rglob("*")
        if p.is_file()
    }
    result = snapshot(recording, tmp_path)
    manifest = verify_snapshot(
        tmp_path / "snapshots" / result["manifest_sha256"], result["manifest_sha256"]
    )
    assert result["total_episodes"] == 2 and result["total_frames"] == 6
    assert manifest["features"]["observation.images.front"]["dtype"] == "image"
    assert any(f["path"] == "openjensen.json" for f in manifest["files"])
    assert before == {
        str(p.relative_to(recording)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in recording.rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize("field,value", [("finalized", False), ("abandoned", True), ("frames", 7)])
def test_unfinished_or_inconsistent_recording_is_rejected(recording, tmp_path, field, value):
    path = recording / "openjensen.json"
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(SnapshotError, match="recording"):
        snapshot(recording, tmp_path)
    assert not list((tmp_path / "snapshots").iterdir())


@pytest.mark.parametrize("change", ["external_path", "corrupt", "wrong_dimensions"])
def test_embedded_images_are_fully_validated(recording, tmp_path, change):
    script = """
import io,sys
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
path = Path(sys.argv[1]) / 'data/chunk-000/file-000.parquet'
table = pq.read_table(path)
rows = table.to_pylist()
image = rows[0]['observation.images.front']
if sys.argv[2] == 'external_path': image['path'] = '/etc/passwd'
elif sys.argv[2] == 'corrupt': image['bytes'] = b'not an image'
else:
    output = io.BytesIO()
    Image.new('RGB', (8,8)).save(output, format='PNG')
    image['bytes'] = output.getvalue()
pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
"""
    subprocess.run(
        [str(reader_python()), "-I", "-c", script, str(recording), change], check=True, timeout=20
    )
    with pytest.raises(SnapshotError, match="Dataset validation failed"):
        snapshot(recording, tmp_path)
    assert not list((tmp_path / "snapshots").iterdir())
