"""Private SDK stderr must never become a browser-visible setup error."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from vla_platform.lifecycle import sky_runner


@pytest.mark.parametrize("code,stdout", [(1, b""), (0, b"invalid bucket")])
def test_storage_prepare_failure_does_not_reflect_private_sdk_output(monkeypatch, code, stdout):
    process = AsyncMock()
    process.returncode = code
    process.communicate.return_value = (
        stdout,
        b"Traceback /Users/private/credentials.json token=secret-token account@example.com",
    )
    monkeypatch.setattr(sky_runner.cloud_compute_catalog, "sky_python", lambda _: "/private/sdk")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    stop = AsyncMock()
    monkeypatch.setattr(sky_runner, "_stop_process", stop)
    with pytest.raises(ValueError) as caught:
        asyncio.run(
            sky_runner.ensure_cloud_storage(
                "sky", {"project_id": "fixture", "region": "us-central1"}
            )
        )
    message = str(caught.value)
    assert message == (
        "Checkpoint storage cannot be prepared. Verify bucket ownership, private access, "
        "and bucket metadata/object permissions for the connected Google Cloud account."
    )
    assert "credentials.json" not in message and "secret-token" not in message
    assert "example.com" not in message and "Traceback" not in message
    stop.assert_awaited_once_with(process)
