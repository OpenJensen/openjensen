"""Bounded subprocess framing and diagnostics with in-memory process fixtures."""

import asyncio
from types import SimpleNamespace

import pytest
from vla_platform.compute import providers
from vla_platform.compute.providers import LocalAdapter, WorkerInvocation
from vla_platform.compute.streams import BoundedLines, output_lines
from vla_platform.execution import Execution
from vla_platform.job_observability import JobObservability


class Chunks:
    def __init__(self, chunks):
        self.chunks = list(chunks)

    async def read(self, size):
        assert size == 8192
        await asyncio.sleep(0)
        return self.chunks.pop(0) if self.chunks else b""


def test_utf8_characters_split_across_chunks_remain_intact():
    wire = "Training 😃 étape\nfinal partial".encode()
    framing = BoundedLines()
    actual = []
    for byte in wire:
        actual.extend(framing.feed(bytes([byte])))
    actual.extend(framing.finish())
    assert actual == ["Training 😃 étape", "final partial"]
    assert framing.finish() == []


def test_oversized_line_is_discarded_until_its_terminator():
    framing = BoundedLines(maximum=16)
    assert framing.feed(b"token=private-") == []
    assert framing.feed(b"value" * 30) == []
    assert framing.dropping and len(framing.pending) == 0
    result = framing.feed(b"more-private-data\nsafe next line\n")
    assert result == ["[oversized worker line omitted]", "safe next line"]
    assert not framing.dropping and not framing.pending


def test_oversized_unterminated_output_has_one_explicit_marker():
    framing = BoundedLines(maximum=8)
    assert framing.feed(b"x" * 10000) == []
    assert len(framing.pending) == 0
    assert framing.finish() == ["[oversized worker line omitted]"]
    assert framing.finish() == []


def test_output_lines_reassembles_split_credentials_before_redaction(tmp_path):
    async def exercise():
        store = JobObservability(tmp_path)
        stream = Chunks(
            [b"password=pri", b"vate-value\nnormal ", "😀".encode()[:2], "😀".encode()[2:] + b"\n"]
        )
        async for line in output_lines(stream):
            store.record("job-1", line, kind="log", stream="stdout")
        rows = store.page("job-1").records
        assert rows[0].message == "password=[redacted]"
        assert rows[1].message == "normal 😀"

    asyncio.run(exercise())


def test_metadata_output_observation_preserves_stream_identity(tmp_path):
    async def exercise():
        owner = SimpleNamespace(observability=JobObservability(tmp_path))
        await Execution.observe_output(
            owner, "job-1", Chunks([b"stdout line\n"]), "stdout", "inspection"
        )
        await Execution.observe_output(
            owner, "job-1", Chunks([b"stderr line\n"]), "stderr", "inspection"
        )
        rows = owner.observability.page("job-1").records
        assert [(row.stream, row.message) for row in rows] == [
            ("stdout", "stdout line"),
            ("stderr", "stderr line"),
        ]
        assert all(row.component == "worker" and row.stage == "inspection" for row in rows)

    asyncio.run(exercise())


def test_native_adapter_persists_only_redacted_separate_streams(tmp_path, monkeypatch):
    async def exercise():
        store = JobObservability(tmp_path)
        state = {"stopped": False}
        stdout = [b"ordinary output\npassword=split-", b"private-value\n"]
        stderr = [
            b"-----BEGIN PRIVATE KEY-----\n",
            b"base64-private-material\n",
            b"-----END PRIVATE KEY-----\n",
            b"after key\n",
        ]

        async def waited():
            return 0

        process = SimpleNamespace(
            stdout=Chunks(stdout), stderr=Chunks(stderr), wait=waited, returncode=0
        )

        async def spawn(*args, **kwargs):
            assert kwargs["stdout"] == asyncio.subprocess.PIPE
            assert kwargs["stderr"] == asyncio.subprocess.PIPE
            return process

        async def stop(owned, container):
            assert owned is process and container is None
            state["stopped"] = True

        async def event(*args, **kwargs):
            return None

        owner = SimpleNamespace(
            settings=SimpleNamespace(data_dir=tmp_path),
            execution=SimpleNamespace(observability=store),
            require_compute=lambda *args: None,
            event=event,
            act_spawn_owned=spawn,
            stop=stop,
        )
        runtime = SimpleNamespace(
            label="Configured worker", image=None, training_image=None, conversion_python=None
        )
        job = SimpleNamespace(id="job-1", request=SimpleNamespace(), execution=None)
        invocation = WorkerInvocation(
            owner, job, runtime, "policy.import", {}, tmp_path, "operation", False, False
        )
        monkeypatch.setattr(
            providers, "command", lambda *args, **kwargs: (["fixture-worker"], None, None)
        )
        assert await LocalAdapter().run(invocation) == 0 and state["stopped"]
        rows = store.page(job.id).records
        stdout_rows = [row for row in rows if row.stream == "stdout"]
        stderr_rows = [row for row in rows if row.stream == "stderr"]
        assert [row.message for row in stdout_rows] == ["ordinary output", "password=[redacted]"]
        assert [row.message for row in stderr_rows][-1] == "after key"
        persisted = (
            (tmp_path / "worker.log").read_text()
            + (tmp_path / "worker-stderr.log").read_text()
            + " ".join(row.message for row in rows)
        )
        assert "split-private-value" not in persisted
        assert "base64-private-material" not in persisted
        assert "ordinary output" in (tmp_path / "worker.log").read_text()
        assert "after key" in (tmp_path / "worker-stderr.log").read_text()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "chunks", [[b"one\ntwo\n"], [b"one\r\ntwo\r\n"], [b"one\r", b"\ntwo\r", b"\n"]]
)
def test_chunk_boundaries_do_not_create_extra_log_records(chunks):
    framing = BoundedLines()
    lines = []
    for chunk in chunks:
        lines.extend(framing.feed(chunk))
    lines.extend(framing.finish())
    assert lines == ["one", "two"]
