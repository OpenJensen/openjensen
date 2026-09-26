import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from vla_platform.contracts import LocalDatasetPreview
from vla_platform.datasets import local_preview as preview
from vla_platform.datasets.local_preview import PreviewError, PreviewLimits, preview_local

FIXTURE = Path(__file__).parent / "fixtures/lerobot_v3_preview"
FRAMES = "data/chunk-000/file-000.parquet"
EPISODES = "meta/episodes/chunk-000/file-000.parquet"
requires_secure_reader = pytest.mark.skipif(
    os.open not in os.supports_dir_fd or not hasattr(os, "O_NOFOLLOW"),
    reason="Actual row preview requires POSIX no-follow opens; Windows must fail closed",
)


@pytest.fixture
def dataset(tmp_path):
    # Test-only copies allow corruption without modifying the committed fixture.
    return Path(shutil.copytree(FIXTURE, tmp_path / "fixture")).resolve()


def read(dataset=FIXTURE, **kwargs):
    dataset = dataset.absolute()
    return preview_local(dataset.name, dataset.parent, **kwargs)


def failure(code, dataset=FIXTURE, **kwargs):
    with pytest.raises(PreviewError) as error:
        read(dataset, **kwargs)
    assert error.value.code == code, str(error.value)


def rewrite_frames(dataset, expression):
    # Only trusted test code; no native reader dependencies enter the core.
    subprocess.run(
        [
            str(preview.reader_python()),
            "-I",
            "-c",
            "import sys, pyarrow as pa, pyarrow.parquet as pq\n"
            "path = sys.argv[1]\ntable = pq.read_table(path)\n"
            + expression
            + "\npq.write_table(table, path)\n",
            str(dataset / FRAMES),
        ],
        check=True,
        capture_output=True,
        timeout=10,
    )


@requires_secure_reader
def test_fixture_identity_known_values_and_no_writes():
    manifest = json.loads((FIXTURE / "manifest.json").read_text())["files"]
    before = {
        path.relative_to(FIXTURE).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in FIXTURE.rglob("*")
        if path.is_file()
    }
    for name, expected in manifest.items():
        raw = (FIXTURE / name).read_bytes()
        assert len(raw) == expected["bytes"]
        assert hashlib.sha256(raw).hexdigest() == expected["sha256"]
    result = read(limits=PreviewLimits(max_rows=2))
    assert LocalDatasetPreview.model_validate(result).model_dump(mode="json") == result
    assert result["file"] == {
        "path": FRAMES,
        "size_bytes": 3667,
        "sha256": "6d217d5ff9d9ac1e079fbf9073a4d169a72e4407a110b0845c97cc760c60aee3",
    }
    assert result["metadata_sha256"] == manifest["meta/info.json"]["sha256"]
    assert result["total_file_rows"] == 6 and result["returned_rows"] == 2
    assert result["rows"][0]["action"] == [0.0, 0.0]
    assert result["rows"][1]["action"] == [1.0, -1.0]
    assert result["rows"][1]["observation.state"] == [1.25, 1.5]
    assert result["rows"][1]["timestamp"] == pytest.approx(0.1)
    assert result["truncated"] is True
    assert result["read_bytes"] <= result["limits"]["max_read_bytes"]
    assert len(preview._json_bytes(result)) <= result["limits"]["max_output_bytes"]
    assert any("semantics" in warning for warning in result["warnings"])
    assert any("not established" in warning for warning in result["warnings"])
    after = {
        path.relative_to(FIXTURE).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in FIXTURE.rglob("*")
        if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize("maximum, expected", [(1, 1), (4, 4), (100, 6)])
@requires_secure_reader
def test_row_budget_across_row_groups(maximum, expected):
    result = read(limits=PreviewLimits(max_rows=maximum))
    assert result["returned_rows"] == len(result["rows"]) == expected
    assert result["truncated"] is (expected < 6)
    assert result["rows"][-1]["action"] == [float(expected - 1), float(1 - expected)]


@requires_secure_reader
def test_episode_preview_and_absolute_dataset_path():
    result = preview_local(FIXTURE.resolve(), FIXTURE.parent.resolve(), kind="episodes")
    assert LocalDatasetPreview.model_validate(result).model_dump(mode="json") == result
    assert result["file"]["path"] == EPISODES
    assert result["file"]["sha256"] == hashlib.sha256((FIXTURE / EPISODES).read_bytes()).hexdigest()
    assert result["total_file_rows"] == result["returned_rows"] == 2
    assert result["truncated"] is False
    assert result["rows"] == [
        {
            "episode_index": 0,
            "tasks": ["synthetic reach"],
            "length": 3,
            "dataset_from_index": 0,
            "dataset_to_index": 3,
            "data/chunk_index": 0,
            "data/file_index": 0,
        },
        {
            "episode_index": 1,
            "tasks": ["synthetic reset"],
            "length": 3,
            "dataset_from_index": 3,
            "dataset_to_index": 6,
            "data/chunk_index": 0,
            "data/file_index": 0,
        },
    ]


@pytest.mark.parametrize(
    "limits, code",
    [
        (PreviewLimits(max_file_bytes=3666), "file_limit"),
        (PreviewLimits(max_read_bytes=1), "read_limit"),
        (PreviewLimits(max_decoded_bytes=1), "decoded_limit"),
        (PreviewLimits(max_output_bytes=1024), "output_limit"),
    ],
)
@requires_secure_reader
def test_resource_limits_fail_without_partial_preview(limits, code):
    failure(code, limits=limits)


@requires_secure_reader
def test_repeated_parquet_reads_count_towards_budget():
    size = (FIXTURE / "meta/info.json").stat().st_size + (FIXTURE / FRAMES).stat().st_size
    failure("read_limit", limits=PreviewLimits(max_read_bytes=size))


@requires_secure_reader
def test_timeout_reaps_real_reader(monkeypatch):
    spawned = []
    original = subprocess.Popen

    def spawn(*args, **kwargs):
        process = original(*args, **kwargs)
        spawned.append(process)
        return process

    monkeypatch.setattr(preview.subprocess, "Popen", spawn)
    started = time.monotonic()
    failure("timeout", limits=PreviewLimits(timeout_seconds=0.001))
    assert time.monotonic() - started < 2
    assert len(spawned) == 1 and spawned[0].poll() is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_rows": 0},
        {"max_rows": 101},
        {"max_rows": True},
        {"max_file_bytes": 0},
        {"max_read_bytes": 129 * 1024 * 1024},
        {"max_output_bytes": 1023},
        {"max_decoded_bytes": 65 * 1024 * 1024},
        {"timeout_seconds": 0},
        {"timeout_seconds": 31},
        {"timeout_seconds": float("nan")},
        {"timeout_seconds": float("inf")},
    ],
)
def test_invalid_budgets(kwargs):
    with pytest.raises(PreviewError, match="must be") as error:
        PreviewLimits(**kwargs)
    assert error.value.code == "invalid_limits"


@pytest.mark.parametrize("relative", ["meta/info.json", FRAMES, EPISODES])
@requires_secure_reader
def test_missing_files(dataset, relative):
    (dataset / relative).unlink()
    failure("missing_file", dataset, kind="episodes" if relative == EPISODES else "frames")


@pytest.mark.parametrize(
    "relative, contents, code",
    [
        ("meta/info.json", b"not JSON", "invalid_metadata"),
        ("meta/info.json", b'{"codebase_version":"v2.1","features":{}}', "invalid_metadata"),
        (FRAMES, b"PAR1brokenPAR1", "corrupt_payload"),
        (EPISODES, b"", "corrupt_payload"),
    ],
)
@requires_secure_reader
def test_corrupt_and_unsupported_inputs(dataset, relative, contents, code):
    (dataset / relative).write_bytes(contents)
    failure(code, dataset, kind="episodes" if relative == EPISODES else "frames")


@pytest.mark.parametrize(
    "dataset_path, parquet_path",
    [
        ("../lerobot_v3_preview", None),
        ("lerobot_v3_preview/../lerobot_v3_preview", None),
        ("/outside/local-root", None),
        ("lerobot_v3_preview", "data/../../outside.parquet"),
        ("lerobot_v3_preview", "/data/chunk-000/file-000.parquet"),
        ("lerobot_v3_preview", "data\\outside.parquet"),
        ("lerobot_v3_preview", "meta/info.json"),
    ],
)
@requires_secure_reader
def test_traversal_and_invalid_selectors(dataset_path, parquet_path):
    with pytest.raises(PreviewError) as error:
        preview_local(dataset_path, FIXTURE.parent.resolve(), parquet_path=parquet_path)
    assert error.value.code == "unsafe_path"


@pytest.mark.parametrize("relative", ["meta", "meta/info.json", "data", FRAMES])
@pytest.mark.parametrize("outside", [False, True])
@requires_secure_reader
def test_symlinks_rejected_even_when_target_is_inside_root(dataset, relative, outside):
    path = dataset / relative
    destination = (dataset.parent if outside else dataset) / "relocated"
    path.rename(destination)
    path.symlink_to(destination, target_is_directory=destination.is_dir())
    failure("unsafe_path", dataset)


@requires_secure_reader
def test_dataset_and_configured_root_symlinks(dataset):
    link = dataset.parent / "linked"
    link.symlink_to(dataset, target_is_directory=True)
    failure("unsafe_path", link)
    with pytest.raises(PreviewError) as error:
        preview_local(".", link)
    assert error.value.code == "unsafe_path"


@requires_secure_reader
def test_nonregular_payload_does_not_block(dataset):
    path = dataset / FRAMES
    path.unlink()
    os.mkfifo(path)
    failure("unsafe_path", dataset)


@requires_secure_reader
def test_symlink_swap_at_open_is_rejected(dataset, monkeypatch):
    path = dataset / FRAMES
    target = dataset / "relocated.parquet"
    original = os.open

    def swap(file, flags, *args, **kwargs):
        if file == "file-000.parquet":
            path.rename(target)
            path.symlink_to(target)
        return original(file, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", swap)
    monkeypatch.setattr(os, "supports_dir_fd", {*os.supports_dir_fd, swap})
    with pytest.raises(PreviewError) as error:
        preview._open_beneath(dataset.parent, Path(dataset.name) / FRAMES)
    assert error.value.code == "unsafe_path"


@requires_secure_reader
def test_binary_image_column_is_omitted_without_decode(dataset):
    rewrite_frames(
        dataset,
        "table = table.append_column('observation.images.front', "
        "pa.array([b'not an image or video'] * 6))",
    )
    result = read(dataset)
    assert result["omitted_columns"] == ["observation.images.front"]
    assert all("observation.images.front" not in row for row in result["rows"])
    assert result["returned_rows"] == 6


@requires_secure_reader
def test_unsupported_action_and_nonfinite_values_fail(dataset):
    rewrite_frames(
        dataset,
        "table = table.set_column(table.schema.get_field_index('action'), "
        "'action', pa.array([b'opaque'] * 6))",
    )
    failure("unsupported_columns", dataset)
    rewrite_frames(
        dataset,
        "table = table.set_column(table.schema.get_field_index('action'), "
        "'action', pa.array([float('nan')] * 6))",
    )
    failure("corrupt_payload", dataset)


@requires_secure_reader
def test_metadata_templates_are_not_followed(dataset):
    path = dataset / "meta/info.json"
    info = json.loads(path.read_text())
    info["data_path"] = "../../outside.parquet"
    info["video_path"] = "/outside/video.mp4"
    path.write_text(json.dumps(info))
    assert read(dataset)["file"]["path"] == FRAMES


@pytest.mark.parametrize(
    "field, value",
    [
        ("features", {"action": {}, "observation.state": {}}),
        ("total_frames", -1),
        ("fps", float("nan")),
    ],
)
@requires_secure_reader
def test_invalid_metadata_declarations(dataset, field, value):
    path = dataset / "meta/info.json"
    info = json.loads(path.read_text())
    info[field] = value
    path.write_text(json.dumps(info))
    failure("invalid_metadata", dataset)


def test_path_safety_fails_closed_without_platform_support(monkeypatch):
    monkeypatch.setattr(os, "supports_dir_fd", set())
    with pytest.raises(PreviewError) as error:
        preview._open_beneath(FIXTURE.parent, Path(FIXTURE.name) / FRAMES)
    assert error.value.code == "unsupported_platform"


@requires_secure_reader
def test_reader_isolation_and_missing_reader(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
class BlockArrow:
    def find_spec(self, fullname, *args):
        if fullname == 'pyarrow' or fullname.startswith('pyarrow.'):
            raise AssertionError('Core must not import PyArrow')
sys.meta_path.insert(0, BlockArrow())
from vla_platform.datasets.local_preview import preview_local
result = preview_local('lerobot_v3_preview', sys.argv[1])
assert result['returned_rows'] == 6
assert 'pyarrow' not in sys.modules
print('Core PyArrow imports blocked; isolated preview succeeded')
""",
            str(FIXTURE.parent.resolve()),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "isolated preview succeeded" in result.stdout
    failure("reader_unavailable", python=tmp_path / "missing-python")
    assert preview.reader_python().is_relative_to(Path(__file__).resolve().parents[1] / "workers")


def test_disabled_preview():
    with pytest.raises(PreviewError) as error:
        preview_local("fixture", None)
    assert error.value.code == "local_disabled"


def test_unsupported_host_rejects_before_reader_launch(monkeypatch):
    def unexpected_launch(*args, **kwargs):
        pytest.fail("Unsupported hosts must not launch a native reader")

    monkeypatch.setattr(os, "supports_dir_fd", set())
    monkeypatch.setattr(preview.subprocess, "Popen", unexpected_launch)
    failure("unsupported_platform")


@pytest.mark.skipif(os.name != "nt", reason="Actual Windows rejection check")
def test_windows_preview_fails_closed_without_reader_installation():
    failure("unsupported_platform")
