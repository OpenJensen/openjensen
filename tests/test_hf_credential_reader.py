"""Real malformed-file startup boundary; no real credentials or provider calls."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(os.name != "posix", reason="POSIX FIFO filesystem boundary")
def test_credential_fifo_is_rejected_without_waiting_for_writer(tmp_path):
    os.mkfifo(tmp_path / "huggingface-credential.json", 0o600)
    source = Path(__file__).resolve().parents[1] / "packages/core/src"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; "
            "from vla_platform.huggingface_connection import HuggingFaceConnection; "
            "c=HuggingFaceConnection(Path(sys.argv[1])); "
            "assert c.token() is None; assert not c.status().configured; "
            "assert 'securely' in c.status().message",
            str(tmp_path),
        ],
        env={**os.environ, "PYTHONPATH": str(source)},
        capture_output=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr.decode()
