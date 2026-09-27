"""SkyPilot secret delivery and log redaction without subprocesses or network calls."""

import asyncio
import json
import os
import tarfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from vla_platform.lifecycle import sky_runner

TOKEN = "hf_FixtureCredentialNeverUseThisInProduction123456789"
SECOND_TOKEN = "hf_AnotherFixtureCredentialForConcurrentRun987654321"
ENDPOINT = "http://127.0.0.1:46580"


class ChunkStream:
    def __init__(self, chunks):
        self.chunks = iter(chunk for chunk in chunks if chunk)

    async def read(self, size):
        assert size == 8192
        await asyncio.sleep(0)
        return next(self.chunks, b"")


def test_exact_secret_redaction_at_every_stream_boundary_and_single_byte_reads():
    body = ("prefix 🚀 " + TOKEN + "\n" + TOKEN + SECOND_TOKEN + " final " + TOKEN).encode()
    expected = body.replace(TOKEN.encode(), b"[REDACTED]").replace(
        SECOND_TOKEN.encode(), b"[REDACTED]"
    )

    async def run():
        partitions = [[body[:cut], body[cut:]] for cut in range(1, len(body))]
        partitions += [[body[index : index + 1] for index in range(len(body))], [body]]
        for chunks in partitions:
            result = [
                chunk
                async for chunk in sky_runner._safe_output(
                    ChunkStream(chunks), (TOKEN, SECOND_TOKEN, TOKEN)
                )
            ]
            assert b"".join(result) == expected
            assert all(
                TOKEN.encode() not in chunk and SECOND_TOKEN.encode() not in chunk
                for chunk in result
            )

    asyncio.run(run())


@pytest.mark.parametrize("cut", range(1, len(TOKEN)))
def test_command_redacts_split_token_before_capture_log_lines_and_telemetry(
    tmp_path, monkeypatch, cut
):
    state = {"target": {"sky_api_endpoint": ENDPOINT}}
    (tmp_path / sky_runner.STATE_NAME).write_text(json.dumps(state))
    prefix = b'{"step":1,"note":"'
    suffix = b'"}\ntrailing token: ' + TOKEN.encode() + b"\n"
    chunks = [prefix + TOKEN[:cut].encode(), TOKEN[cut:].encode() + suffix]
    recorded = []

    class Process:
        returncode = 0
        stdout = ChunkStream(chunks)

        async def wait(self):
            return self.returncode

    async def create(*argv, **kwargs):
        recorded.append((argv, kwargs["env"].get("HF_TOKEN")))
        assert TOKEN not in " ".join(argv)
        assert kwargs["env"]["SKYPILOT_API_SERVER_ENDPOINT"] == ENDPOINT
        return Process()

    monkeypatch.setattr(sky_runner.asyncio, "create_subprocess_exec", create)
    monkeypatch.setattr(sky_runner, "_stop_process", AsyncMock())
    monkeypatch.delenv("HF_TOKEN", raising=False)
    lines, events = [], []

    async def run():
        context = sky_runner._RUN_SECRETS.set((TOKEN,))
        try:
            return await sky_runner._command(
                ["/fixture/sky", "launch", "--secret", "HF_TOKEN"],
                tmp_path,
                timeout=2,
                env_overrides={"HF_TOKEN": TOKEN},
                secrets=(TOKEN,),
                on_line=AsyncMock(side_effect=lines.append),
                on_event=AsyncMock(side_effect=events.append),
            )
        finally:
            sky_runner._RUN_SECRETS.reset(context)

    code, captured = asyncio.run(run())
    assert code == 0
    assert recorded == [(("/fixture/sky", "launch", "--secret", "HF_TOKEN"), TOKEN)]
    assert TOKEN not in captured
    assert captured.count("[REDACTED]") == 2
    assert TOKEN not in (tmp_path / "worker.log").read_text()
    assert (tmp_path / "worker.log").read_text() == captured
    assert len(lines) == 2 and all(TOKEN not in line for line in lines)
    assert events and all(TOKEN not in event for event in events)
    assert "[REDACTED]" in events[0]
    assert "HF_TOKEN" not in os.environ
    assert sky_runner._RUN_SECRETS.get() == ()


@pytest.fixture
def fixture_dispatch(tmp_path, monkeypatch):
    worker = tmp_path / "worker"
    (worker / "src/firebird_vla").mkdir(parents=True)
    (worker / "src/firebird_vla/application.py").write_text("# fixture only\n")
    (worker / "pyproject.toml").write_text("[project]\nname='fixture'\nversion='1'\n")
    (worker / "requirements-smolvla-linux.txt").write_text("setuptools==80.10.2\n")
    monkeypatch.setattr(sky_runner, "worker_root", lambda: worker)
    monkeypatch.setattr(sky_runner, "executable", lambda: "/fixture/sky")

    async def storage(sky, target):
        return "gs://firebird-fixture"

    monkeypatch.setattr(sky_runner, "ensure_cloud_storage", storage)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    payload = {
        "schema_version": 1,
        "operation": "policy.finetune",
        "job_id": "secret-fixture-job",
        "parameters": {
            "timeout_seconds": 30,
            "training_method": "qlora",
            "training": {"steps": 10, "save_every": 5},
        },
        "runtime": {},
        "dataset": {"repo_id": "fixture/robot", "revision": "a" * 40},
        "artifact": None,
        "source": None,
    }
    target = {
        "project_id": "robotics-demo",
        "region": "us-central1",
        "accelerator": "A100",
        "gpu_count": 1,
        "disk_size_gb": 200,
        "idle_minutes": 10,
        "instance_type": "a2-highgpu-1g",
        "workspace": "firebird-gcp-fixture",
        "sky_api_endpoint": ENDPOINT,
    }

    async def verify(executable, project, *, endpoint):
        assert (executable, project, endpoint) == ("/fixture/sky", "robotics-demo", ENDPOINT)
        return {"workspace": target["workspace"], "sky_api_endpoint": ENDPOINT}

    monkeypatch.setattr(sky_runner.cloud_compute_catalog, "verify_sky_target", verify)
    return payload, target


@pytest.mark.parametrize("token", [None, TOKEN])
@pytest.mark.parametrize("fail_launch", [False, True])
def test_only_launch_receives_secret_and_no_job_task_bundle_or_archive_contains_it(
    tmp_path, monkeypatch, fixture_dispatch, token, fail_launch
):
    payload, target = fixture_dispatch
    before = json.dumps({"payload": payload, "target": target}, sort_keys=True)
    stage = tmp_path / "stage"
    calls = []
    context_values = []

    async def command(argv, directory, **kwargs):
        context_values.append(sky_runner._RUN_SECRETS.get())
        calls.append((list(argv), kwargs.get("env_overrides"), kwargs.get("secrets")))
        assert TOKEN not in " ".join(argv)
        if argv[1] == "launch":
            return (
                (1, "fixture failure")
                if fail_launch
                else (0, "Submitted sky.launch request: abcdef12-12345678")
            )
        return 0, "fixture output"

    async def follow(sky, directory, state, timeout, event):
        await sky_runner._command([sky, "logs", state["cluster"], "1"], directory, timeout=1)

    async def sync(sky, directory, state, event, *, final):
        assert final is True
        await sky_runner._command(["/fixture/python", "checkpoint-sync.py"], directory, timeout=1)

    async def cleanup(directory, state, **kwargs):
        await sky_runner._command(["/fixture/sky", "down", state["cluster"]], directory, timeout=1)
        return True

    def collect(directory, cluster):
        (directory / "result.json").write_text(
            json.dumps({"job_id": payload["job_id"], "result": "fixture"})
        )

    monkeypatch.setattr(sky_runner, "_command", command)
    monkeypatch.setattr(sky_runner, "follow_training", follow)
    monkeypatch.setattr(sky_runner, "sync_checkpoints", sync)
    monkeypatch.setattr(sky_runner, "cleanup", cleanup)
    monkeypatch.setattr(sky_runner, "collect", collect)

    async def run():
        if fail_launch:
            with pytest.raises(RuntimeError, match="could not submit"):
                await sky_runner.run(payload, stage, target, AsyncMock(), hf_token=token)
        else:
            assert await sky_runner.run(payload, stage, target, AsyncMock(), hf_token=token) == 0
        assert sky_runner._RUN_SECRETS.get() == ()

    asyncio.run(run())
    assert context_values and all(value == ((token,) if token else ()) for value in context_values)
    launch = next(call for call in calls if call[0][1] == "launch")
    if token:
        position = launch[0].index("--secret")
        assert launch[0][position + 1] == "HF_TOKEN"
        assert launch[1:] == ({"HF_TOKEN": TOKEN}, (TOKEN,))
    else:
        assert "--secret" not in launch[0]
        assert launch[1:] == (None, None)
    assert all(
        env is None and secrets is None for argv, env, secrets in calls if argv[1] != "launch"
    )
    assert calls[-1][0][1] == "down"
    assert "HF_TOKEN" not in os.environ
    assert json.dumps({"payload": payload, "target": target}, sort_keys=True) == before
    assert TOKEN not in before
    with tarfile.open(stage / "reviewed-bundle.tar", "w") as archive:
        archive.add(stage / "sky-bundle", arcname="bundle")
    for path in stage.rglob("*"):
        if path.is_file():
            assert TOKEN.encode() not in path.read_bytes(), path.name


@pytest.mark.parametrize("failure", ["none", "exception", "cancel"])
def test_secret_context_restored_before_next_run_even_on_failure(monkeypatch, failure):
    seen = []

    async def inner(payload, stage, target, on_event, *, hf_token):
        seen.append(sky_runner._RUN_SECRETS.get())
        await asyncio.sleep(0)
        if payload.get("fail"):
            if failure == "exception":
                raise RuntimeError("fixture failure")
            if failure == "cancel":
                raise asyncio.CancelledError
        return 0

    monkeypatch.setattr(sky_runner, "_run", inner)

    async def run():
        try:
            await sky_runner.run({"fail": True}, Path("unused"), {}, AsyncMock(), hf_token=TOKEN)
        except RuntimeError, asyncio.CancelledError:
            pass
        assert sky_runner._RUN_SECRETS.get() == ()
        assert await sky_runner.run({}, Path("unused"), {}, AsyncMock()) == 0
        assert sky_runner._RUN_SECRETS.get() == ()

    asyncio.run(run())
    assert seen == [(TOKEN,), ()]


def test_concurrent_runs_do_not_share_secret_context(monkeypatch):
    async def inner(payload, stage, target, on_event, *, hf_token):
        assert sky_runner._RUN_SECRETS.get() == (hf_token,)
        await asyncio.sleep(0)
        assert sky_runner._RUN_SECRETS.get() == (hf_token,)
        return 0

    monkeypatch.setattr(sky_runner, "_run", inner)

    async def run():
        await asyncio.gather(
            *(
                sky_runner.run({}, Path("unused"), {}, AsyncMock(), hf_token=token)
                for token in (TOKEN, SECOND_TOKEN)
            )
        )
        assert sky_runner._RUN_SECRETS.get() == ()

    asyncio.run(run())
