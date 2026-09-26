"""Real CPU Parquet fixtures, not live robotics or video-decoding evidence."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from vla_platform.contracts import IntakeRequest
from vla_platform.datasets import local, preview
from vla_platform.datasets.inspect import inspect_local


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "datasets"
    directory = root / "fixture"
    (directory / "meta").mkdir(parents=True)
    (directory / "data").mkdir()
    info = {
        "codebase_version": "v3.0",
        "robot_type": "cpu_test_fixture",
        "total_episodes": 2,
        "total_frames": 20,
        "fps": 10,
        "data_path": "data/file-{file_index:03d}.parquet",
        "features": {
            "action": {"dtype": "float32", "shape": [2]},
            "observation.state": {"dtype": "float32", "shape": [2]},
            "observation.images.front": {"dtype": "video", "shape": [8, 8, 3]},
        },
    }
    (directory / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    table = pa.table(
        {
            "episode_index": [0] * 10 + [1] * 10,
            "frame_index": list(range(10)) * 2,
            "timestamp": [i / 10 for i in range(10)] * 2,
            "action": pa.array([[i * 0.5, -i * 0.5] for i in range(20)], pa.list_(pa.float32())),
            "observation.state": [[float(i), float(i + 1)] for i in range(20)],
            "image_payload": [b"fixture payload, never a decoded video"] * 20,
        }
    )
    pq.write_table(table, directory / "data/file-000.parquet", row_group_size=10)
    return directory


def run_preview(dataset, **kwargs):
    return preview.preview_local("fixture", str(dataset.parent), "data/file-000.parquet", **kwargs)


def update_metadata(dataset, **kwargs):
    target = dataset / "meta/info.json"
    info = json.loads(target.read_text(encoding="utf-8"))
    info.update(kwargs)
    target.write_text(json.dumps(info), encoding="utf-8")


def test_actual_parquet_values_provenance_bounds_and_no_copy(dataset):
    before = {path: path.stat().st_mtime_ns for path in dataset.rglob("*") if path.is_file()}
    result = run_preview(dataset, max_rows=3)
    assert result["status"] == "previewed"
    assert result["rows"][2]["action"] == [1.0, -1.0]
    assert result["rows"][2]["observation.state"] == [2.0, 3.0]
    assert result["rows"][2]["frame_index"] == 2
    assert result["episode_indices"] == [0] and result["episode_identity"] == "sample_column"
    assert result["sample_truncated"] and result["file_rows"] == 20
    assert result["row_group"] == 0 and result["row_group_rows"] == 10
    assert (
        result["metadata_sha256"]
        == hashlib.sha256((dataset / "meta/info.json").read_bytes()).hexdigest()
    )
    assert (
        result["file_sha256"]
        == hashlib.sha256((dataset / "data/file-000.parquet").read_bytes()).hexdigest()
    )
    assert result["omitted_columns"] == ["image_payload"]
    assert result["video_decode"] == "unsupported"
    assert result["declared_file_checks"][1]["status"] == "unverified"
    assert "unverified" in result["semantics"]
    assert len(preview.encode_preview(result).encode()) <= preview.MAX_JSON_BYTES
    assert before == {
        path: path.stat().st_mtime_ns for path in dataset.rglob("*") if path.is_file()
    }


def test_v2_metadata_and_changed_file_identity(dataset):
    before = run_preview(dataset)
    update_metadata(dataset, codebase_version="v2.1")
    pq.write_table(
        pa.table({"action": [[9.0]], "observation.state": [[8.0]], "episode_index": [7]}),
        dataset / "data/file-000.parquet",
    )
    after = run_preview(dataset)
    assert after["format"] == "lerobot_v2" and after["rows"][0]["action"] == [9.0]
    assert after["episode_indices"] == [7]
    assert after["file_sha256"] != before["file_sha256"]
    assert after["metadata_sha256"] != before["metadata_sha256"]


def test_metadata_stays_metadata_only_with_missing_file_and_semantic_notes(dataset):
    update_metadata(dataset, video_path="videos/{video_key}/file-{file_index:03d}.mp4")
    (dataset / "data/file-000.parquet").unlink()
    result = inspect_local(IntakeRequest(source="local", path="fixture"), str(dataset.parent))
    assert result.inspection_scope == "metadata_only"
    assert any("Missing initial declared data file" in warning for warning in result.warnings)
    assert any("Missing initial declared video file" in warning for warning in result.warnings)
    assert any("controller semantics" in warning for warning in result.warnings)
    sample = run_preview(dataset)
    assert sample["status"] == "missing" and sample["rows"] == []
    assert sample["video_decode"] == "unsupported" and "file_sha256" not in sample


def test_missing_templates_and_actual_columns_are_explicit(dataset):
    update_metadata(dataset, data_path=None)
    metadata = inspect_local(IntakeRequest(source="local", path="fixture"), str(dataset.parent))
    assert any("data file presence is unverified" in item for item in metadata.warnings)
    assert any("video file presence is unverified" in item for item in metadata.warnings)
    pq.write_table(pa.table({"action": [[1.0]]}), dataset / "data/file-000.parquet")
    sample = run_preview(dataset)
    assert sample["status"] == "incomplete"
    assert sample["missing_feature_columns"] == ["observation.state"]
    assert sample["episode_identity"] == "unavailable"


@pytest.mark.parametrize(
    "path",
    [
        "../missing/outside",
        "data/../../missing",
        "//server/share/missing",
        r"\\server\share\missing",
        r"\\?\C:\missing",
        "C:relative",
        "data/file.parquet:payload",
        "data/A:/file.parquet",
        "data/NUL.parquet",
        "data/trailing. /file.parquet",
    ],
)
def test_unsafe_paths_rejected_before_filesystem_access(tmp_path, monkeypatch, path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Unsafe lexical path accessed filesystem")

    monkeypatch.setattr(Path, "lstat", forbidden)
    with pytest.raises(ValueError):
        local.confined_path(tmp_path, path)


def test_absolute_missing_escape_is_rejected_before_missing_path(dataset):
    with pytest.raises(ValueError, match="within"):
        preview.preview_local(str(dataset.parent.parent / "missing"), str(dataset.parent), "x")
    with pytest.raises(ValueError, match="within"):
        preview.preview_local(
            "fixture", str(dataset.parent), str(dataset.parent / "sibling.parquet")
        )


def test_junction_or_symlink_escape_is_refused(dataset, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = dataset / "linked"
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
            capture_output=True,
            text=True,
            check=True,
        )
    else:
        link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="junctions"):
        preview.preview_local("fixture", str(dataset.parent), "linked/missing.parquet")
    with pytest.raises(ValueError, match="junctions"):
        inspect_local(IntakeRequest(source="local", path="fixture/linked"), str(dataset.parent))


@pytest.mark.parametrize(
    "template",
    [
        "../escape.parquet",
        "//server/share/file.parquet",
        "data/{file_index.__class__}.parquet",
        "data/{file_index[0]}.parquet",
        "data/{file_index!r}.parquet",
        "data/{file_index:99999999d}.parquet",
        "data/{unexpected}.parquet",
        "data/{file_index:{file_index}}.parquet",
    ],
)
def test_unsafe_metadata_templates_refused(dataset, template):
    update_metadata(dataset, data_path=template)
    with pytest.raises(ValueError):
        run_preview(dataset)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"features":{},"features":{}}',
        b'{"features":{},"fps":NaN}',
        b'{"features":{},"fps":1e999}',
        b'{"features":{},"x":' + b"[" * 30 + b"0" + b"]" * 30 + b"}",
        b'{"features":{},"x":"' + b"x" * 16_385 + b'"}',
        b"x" * (local.MAX_METADATA_BYTES + 1),
    ],
    ids=["duplicate", "nan", "overflow", "deep", "long-string", "over-2mib"],
)
def test_unsafe_or_oversized_json_refused(dataset, raw):
    (dataset / "meta/info.json").write_bytes(raw)
    with pytest.raises(ValueError):
        run_preview(dataset)


def test_declared_path_escape_inside_dataset_is_refused(dataset, tmp_path):
    update_metadata(dataset, data_path=str(tmp_path / "outside.parquet"))
    with pytest.raises(ValueError):
        inspect_local(IntakeRequest(source="local", path="fixture"), str(dataset.parent))


def test_file_and_footer_budgets_checked_before_native_decoder(dataset, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Native decoder must not open oversized files")

    monkeypatch.setattr(pq, "ParquetFile", forbidden)
    target = dataset / "data/file-000.parquet"
    with target.open("wb") as handle:
        handle.truncate(preview.MAX_FILE_BYTES + 1)
    with pytest.raises(ValueError, match="32 MiB"):
        run_preview(dataset)
    target.write_bytes(
        b"PAR1" + b"x" * 16 + (preview.MAX_FOOTER_BYTES + 1).to_bytes(4, "little") + b"PAR1"
    )
    with pytest.raises(ValueError, match="footer"):
        run_preview(dataset)
    target.write_bytes(b"not parquet at all")
    with pytest.raises(ValueError, match="header"):
        run_preview(dataset)


def test_compressed_large_group_and_many_rows_refused_before_decode(dataset, monkeypatch):
    # Repetitive binary data compresses well, but its uncompressed group is over budget.
    target = dataset / "data/file-000.parquet"
    pq.write_table(
        pa.table({"action": [[1.0]], "payload": [b"a" * (preview.MAX_ROW_GROUP_BYTES + 1)]}),
        target,
        compression="gzip",
        use_dictionary=False,
    )
    assert target.stat().st_size < preview.MAX_FILE_BYTES

    def forbidden(*args, **kwargs):
        raise AssertionError("Row-group budget must be checked before decoding")

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", forbidden)
    with pytest.raises(ValueError, match="decoder budget"):
        run_preview(dataset)
    pq.write_table(pa.table({"action": [0] * (preview.MAX_ROW_GROUP_ROWS + 1)}), target)
    with pytest.raises(ValueError, match="decoder budget"):
        run_preview(dataset)


def test_column_value_budget_rejects_nested_allocation(dataset, monkeypatch):
    pq.write_table(
        pa.table({"action": [[0] * (preview.MAX_COLUMN_VALUES + 1)]}),
        dataset / "data/file-000.parquet",
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Column budget must be checked before decoding")

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", forbidden)
    with pytest.raises(ValueError, match="column chunk"):
        run_preview(dataset)


def test_file_growth_read_is_bounded(dataset, monkeypatch):
    target = dataset / "data/file-000.parquet"
    maximum = target.stat().st_size + 16
    monkeypatch.setattr(preview, "MAX_FILE_BYTES", maximum)
    original_open = Path.open
    reads = []

    class GrowingFile:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def fileno(self):
            return self.handle.fileno()

        def read(self, size):
            reads.append(size)
            with original_open(target, "ab") as writer:
                writer.write(b"x" * 32)
            return self.handle.read(size)

    def controlled_open(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        return GrowingFile(handle) if path == target and args == ("rb",) else handle

    monkeypatch.setattr(Path, "open", controlled_open)
    with pytest.raises(ValueError, match="32 MiB"):
        run_preview(dataset)
    assert reads == [maximum + 1]


def test_decoder_and_hash_use_same_snapshot_after_source_mutation(dataset, monkeypatch):
    target = dataset / "data/file-000.parquet"
    original_bytes = target.read_bytes()
    original_decoder = pq.ParquetFile

    def mutate_then_decode(source, *args, **kwargs):
        assert isinstance(source, pa.BufferReader)
        target.write_bytes(b"source was replaced after the bounded snapshot")
        return original_decoder(source, *args, **kwargs)

    monkeypatch.setattr(pq, "ParquetFile", mutate_then_decode)
    result = run_preview(dataset, max_rows=2)
    assert result["file_sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert result["rows"][1]["action"] == [0.5, -0.5]


@pytest.mark.parametrize("values", [[[1.0] * 129], [[float("inf")]], ["x" * 513]])
def test_oversized_or_nonfinite_values_refused(dataset, values):
    pq.write_table(pa.table({"action": values}), dataset / "data/file-000.parquet")
    with pytest.raises(ValueError, match="preview value"):
        run_preview(dataset)


def test_binary_columns_and_invalid_limits_are_refused(dataset):
    with pytest.raises(ValueError, match="unsupported preview column"):
        run_preview(dataset, columns=["image_payload"])
    with pytest.raises(ValueError, match="unsupported preview column"):
        run_preview(dataset, columns=["missing"])
    for rows in [0, 17, True]:
        with pytest.raises(ValueError, match="row limit"):
            run_preview(dataset, max_rows=rows)
    with pytest.raises(ValueError, match="unique"):
        run_preview(dataset, columns=["action", "action"])
    with pytest.raises(ValueError, match="eight"):
        run_preview(dataset, columns=["action"] * 9)
    with pytest.raises(ValueError, match="96 KiB"):
        preview.encode_preview({"payload": "x" * preview.MAX_JSON_BYTES})


def test_runnable_cli_actual_rows_and_missing_exit_code(dataset):
    command = [
        sys.executable,
        "-m",
        "vla_platform.datasets.preview",
        "fixture",
        "--root",
        str(dataset.parent),
        "--parquet",
        "data/file-000.parquet",
        "--rows",
        "2",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    body = json.loads(result.stdout)
    assert len(body["rows"]) == 2 and body["rows"][1]["action"] == [0.5, -0.5]
    (dataset / "data/file-000.parquet").unlink()
    missing = subprocess.run(command, capture_output=True, text=True, check=False)
    assert missing.returncode == 2 and json.loads(missing.stdout)["status"] == "missing"
    update_metadata(dataset, data_path="../escape.parquet")
    refused = subprocess.run(command, capture_output=True, text=True, check=False)
    assert refused.returncode == 2 and json.loads(refused.stderr)["status"] == "refused"


def test_metadata_module_does_not_import_arrow_or_ml():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from vla_platform.datasets.inspect import inspect_local; "
            "assert not {'torch','lerobot','transformers','openvla','sky','pyarrow'}"
            ".intersection(name.split('.')[0] for name in sys.modules); print('isolated')",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "isolated"


def test_missing_required_metadata_cli_has_bounded_refusal(dataset):
    (dataset / "meta/info.json").write_text('{"features":{}}', encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vla_platform.datasets.preview",
            "fixture",
            "--root",
            str(dataset.parent),
            "--parquet",
            "data/file-000.parquet",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2 and not result.stdout
    refusal = json.loads(result.stderr)
    assert refusal["status"] == "refused" and "Missing required" in refusal["error"]
    assert len(result.stderr) < 600 and "Traceback" not in result.stderr
