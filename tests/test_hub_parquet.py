"""Real subprocess boundary failures, without cloud or GPU access."""

import asyncio
import subprocess
import sys
import time
from pathlib import Path

import pytest
from vla_platform.datasets import hub_parquet as reader

FIXTURE = Path(__file__).parent / "fixtures/lerobot_v3_preview/data/chunk-000/file-000.parquet"


def worker(tmp_path, monkeypatch, code):
    script = tmp_path / "trusted_test_reader.py"
    script.write_text(code, encoding="utf-8")
    monkeypatch.setattr(reader, "WORKER_SCRIPT", script)
    launched = []
    original = asyncio.create_subprocess_exec

    async def capture(*args, **kwargs):
        process = await original(*args, **kwargs)
        launched.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture)
    return launched


@pytest.mark.parametrize(
    "raw", [b"PAR1", b"x" * (32 * 1024 * 1024)], ids=["small", "blocked-stdin"]
)
def test_hung_reader_is_killed_and_reaped_including_blocked_stdin(tmp_path, monkeypatch, raw):
    launched = worker(tmp_path, monkeypatch, "import time; time.sleep(60)")
    started = time.monotonic()
    with pytest.raises(reader.ReaderError) as error:
        asyncio.run(reader.read_parquet(raw, "frames", python=sys.executable, timeout_seconds=0.5))
    assert error.value.code == "timeout"
    assert time.monotonic() - started < 5
    assert len(launched) == 1 and launched[0].returncode is not None


def test_crashed_reader_does_not_crash_core_and_cannot_claim_success(tmp_path, monkeypatch):
    launched = worker(
        tmp_path,
        monkeypatch,
        'import os; os.write(1, b\'{"schema_version":1,"rows":[]}\'); os._exit(23)',
    )
    with pytest.raises(reader.ReaderError) as error:
        asyncio.run(reader.read_parquet(b"PAR1", "index", python=sys.executable))
    assert error.value.code == "reader_failed"
    assert launched[0].returncode == 23


@pytest.mark.parametrize("descriptor", [1, 2])
def test_reader_output_and_diagnostics_are_bounded_and_process_reaped(
    tmp_path, monkeypatch, descriptor
):
    launched = worker(
        tmp_path,
        monkeypatch,
        f"import os\nwhile True: os.write({descriptor}, b'x' * 65536)",
    )
    with pytest.raises(reader.ReaderError) as error:
        asyncio.run(reader.read_parquet(b"PAR1", "index", python=sys.executable))
    assert error.value.code == "output_limit"
    assert len(str(error.value)) < 100
    assert launched[0].returncode is not None


@pytest.mark.parametrize("repeat_cancel", [False, True])
def test_cancelled_reader_is_reaped_before_task_returns(tmp_path, monkeypatch, repeat_cancel):
    launched = worker(tmp_path, monkeypatch, "import time; time.sleep(60)")

    async def run():
        task = asyncio.create_task(reader.read_parquet(b"PAR1", "index", python=sys.executable))
        async with asyncio.timeout(5):
            while not launched:
                await asyncio.sleep(0.01)
        # The event loop remains available while the native subprocess blocks.
        await asyncio.sleep(0.05)
        task.cancel()
        if repeat_cancel:
            async with asyncio.timeout(5):
                while not task.done():
                    task.cancel()
                    await asyncio.sleep(0.001)
        with pytest.raises(asyncio.CancelledError):
            await task
        assert launched[0].returncode is not None

    asyncio.run(run())


@pytest.mark.parametrize(
    "response",
    ["not json", "[]", '{"schema_version":9,"rows":[]}', '{"schema_version":1,"rows":[1]}'],
)
def test_invalid_worker_results_fail_without_partial_rows(tmp_path, monkeypatch, response):
    worker(tmp_path, monkeypatch, f"print({response!r})")
    with pytest.raises(reader.ReaderError) as error:
        asyncio.run(reader.read_parquet(b"PAR1", "frames", python=sys.executable))
    assert error.value.code == "reader_failed"


def test_invalid_input_never_launches_a_reader(monkeypatch):
    async def fail(*args, **kwargs):
        pytest.fail("Oversized or unsupported requests must be rejected before launch")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail)
    for raw, operation, extra in [
        (b"x" * (8 * 1024 * 1024 + 1), "index", {}),
        (b"x" * (32 * 1024 * 1024 + 1), "frames", {}),
        (b"x", "dataset_code", {}),
        (b"x", "frames", {"episode_index": -1}),
        (b"x", "frames", {"timeout_seconds": float("nan")}),
    ]:
        with pytest.raises(reader.ReaderError):
            asyncio.run(reader.read_parquet(raw, operation, **extra))


def test_missing_reader_has_stable_error(tmp_path, monkeypatch):
    monkeypatch.setenv("FIREBIRD_CPU_READER_PYTHON", str(tmp_path / "missing-python"))
    with pytest.raises(reader.ReaderError) as error:
        asyncio.run(reader.read_parquet(b"PAR1", "index"))
    assert error.value.code == "reader_unavailable"


def test_reader_override_is_trusted_configuration(tmp_path, monkeypatch):
    worker(tmp_path, monkeypatch, 'print(\'{"schema_version":1,"rows":[]}\')')
    monkeypatch.setenv("FIREBIRD_CPU_READER_PYTHON", sys.executable)
    assert asyncio.run(reader.read_parquet(b"PAR1", "index")) == []


def test_api_and_hub_preview_work_with_native_imports_blocked_in_core():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import asyncio, sys
from pathlib import Path
class BlockNative:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'pyarrow', 'torch', 'numpy'}:
            raise AssertionError('Core imported a native ML/reader dependency')
sys.meta_path.insert(0, BlockNative())
from vla_platform import api
from vla_platform.datasets.explore import frame_samples
rows = asyncio.run(frame_samples(Path(sys.argv[1]).read_bytes(), 0))
assert len(rows) == 3
assert rows[1].action == [1., -1.]
assert not any(k in sys.modules for k in ['pyarrow', 'torch', 'numpy'])
print('Isolated native decode succeeded with core native imports blocked')
""",
            str(FIXTURE),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "native imports blocked" in result.stdout
