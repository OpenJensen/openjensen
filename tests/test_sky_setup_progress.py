"""Bounded setup observations from captured SkyPilot stdout; no cloud calls."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from vla_platform.lifecycle import sky_runner


def test_setup_progress_reports_fixed_dependency_and_native_build_milestones():
    parser = sky_runner.SetupProgress()
    lines = [
        "(setup) Reading package lists... Done",
        "(setup) Setting up git (1:2.43.0-1) ...",
        "(setup) Resolved 20 packages in 1.1s",
        "(setup) Installed 20 packages in 2.1s",
        "(setup) Setting up cuda-nvcc-12-8 (12.8.93) ...",
        "(setup) -- The CUDA compiler identification is NVIDIA 12.8.93",
        "(setup) -- Configuring done (1.2s)",
        "\x1b[32m(setup) [  1%] Building CXX object private/path.cpp.o\x1b[0m",
        "(setup) [  4%] Building CUDA object private/path.cu.o",
        "(setup) [  5%] Building CUDA object private/path.cu.o",
        "(setup) [  9%] Linking CXX static library private/lib.a",
        "(setup) [ 10%] Built target ggml-base",
        "(setup) [  8%] Built target cached-target",
        "(setup) [100%] Built target vla-bench",
        "(setup) [ 98%] Built target cached-target",
        "(setup) [100%] Built target vla_predict_check",
        "(setup) Installed 1 package in 0.1s",
    ]
    emitted = [item for line in lines if (item := parser.observe(line))]
    assert [item["message"] for item in emitted[:4]] == [
        "Installing system dependencies",
        "Installing Python dependencies",
        "Installing CUDA dependencies",
        "Configuring native engine",
    ]
    assert [item["build_percent"] for item in emitted[4:]] == [0, 5, 10, 100]
    assert all(item["phase"] == "compiling" for item in emitted[3:])
    assert all(item["scope"] == "native_build" for item in emitted[4:])
    assert all("build progress" in item["message"] for item in emitted[4:])
    assert "private" not in json.dumps(emitted)
    assert "percent" not in emitted[-1]  # Build progress is never overall job progress.


def test_setup_progress_noise_secrets_and_ansi_titles_never_become_public_log_text():
    parser = sky_runner.SetupProgress()
    noise = [
        "Authorization: Bearer hf_not_for_public_events",
        "https://user:password@example.test/private?token=private",
        "[50%] Downloading data",
        "[101%] Building CUDA object bad.cu.o",
        "[-1%] Building CUDA object bad.cu.o",
        "50% Building CUDA object without-cmake-marker",
        "\x1b]0;[90%] Building CUDA object hidden-title\x07",
        "\x1b]0;[90%] Building CUDA object hidden-title\x1b\\",
        "x" * 16384 + "[90%] Building CUDA object unbounded-tail.cu.o",
    ]
    assert all(parser.observe(line) is None for line in noise)
    secret = "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"
    emitted = parser.observe(f"[ 25%] Building CUDA object {secret}/file.cu.o")
    assert emitted == {
        "phase": "compiling",
        "message": "Compiling native engine · build progress 25%",
        "scope": "native_build",
        "build_percent": 25,
    }
    assert secret not in json.dumps(emitted)


def test_setup_progress_emits_at_most_one_event_per_five_percent():
    parser = sky_runner.SetupProgress()
    events = []
    for _ in range(5):
        for percentage in range(101):
            if event := parser.observe(f"[ {percentage}%] Building CUDA object bounded.cu.o"):
                events.append(event)
    assert [item["build_percent"] for item in events] == list(range(0, 101, 5))


@pytest.mark.parametrize("chunk_size", [1, 2, 9, 8192])
def test_command_handles_partial_ansi_crlf_and_final_setup_lines(tmp_path, monkeypatch, chunk_size):
    (tmp_path / sky_runner.STATE_NAME).write_text(
        json.dumps(
            {
                "target": {"sky_api_endpoint": "http://127.0.0.1:46580"},
            }
        )
    )
    body = (
        b"\x1b[32m(setup) Reading package lists...\x1b[0m\r\n"
        b"noise credentials: hf_NotAProgressEvent\n"
        b"(setup) [ 5%] Building CUDA object private.cu.o\r"
        b"(setup) [ 9%] Building CUDA object private.cu.o\r\n"
        b"(setup) [10%] Building CXX object private.cpp.o"
    )
    chunks = iter(body[index : index + chunk_size] for index in range(0, len(body), chunk_size))

    class Stream:
        async def read(self, size):
            assert size == 8192
            return next(chunks, b"")

    class Process:
        returncode = 0
        stdout = Stream()

        async def wait(self):
            return 0

    monkeypatch.setattr(
        sky_runner.asyncio, "create_subprocess_exec", AsyncMock(return_value=Process())
    )
    monkeypatch.setattr(sky_runner, "_stop_process", AsyncMock())
    parser, emitted = sky_runner.SetupProgress(), []

    async def line(value):
        if item := parser.observe(value):
            emitted.append(item)

    code, _ = asyncio.run(
        sky_runner._command(
            ["/fixture/sky", "api", "logs", "fixture"],
            tmp_path,
            timeout=2,
            on_line=line,
        )
    )
    assert code == 0
    assert [item["phase"] for item in emitted] == ["setup", "compiling", "compiling"]
    assert [item.get("build_percent") for item in emitted] == [None, 5, 10]
    assert "hf_NotAProgressEvent" not in json.dumps(emitted)
