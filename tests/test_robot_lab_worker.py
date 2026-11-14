"""Run the hardware companion safety and real-format tests in the isolated CPU reader."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

# The companion suite runs in its own interpreter, outside tests/conftest.py. Give it the
# same guarantee: results must not depend on how full this machine's disk happens to be.
# Tests about low disk space still patch the value themselves.
HERMETIC_SUITE = """
import shutil, unittest
_real = shutil.disk_usage
def _roomy(path):
    usage = _real(path)
    return usage._replace(free=max(usage.free, 1 << 40))
shutil.disk_usage = _roomy
unittest.main(module=None, argv=["unittest", "discover", "-s", "workers/robot_lab/tests", "-v"])
"""


@pytest.mark.deadline(240)
def test_robot_lab_safety_and_recording_suite():
    root = Path(__file__).resolve().parents[1]
    reader = root / "workers/_cpu_readers/.venv"
    interpreter = reader / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not interpreter.is_file():
        interpreter = Path(sys.executable)
    environment = os.environ.copy()
    # The optional official Torch/LeRobot proof has its own measured native lane.
    environment.pop("OPENJENSEN_VERIFY_LEROBOT", None)
    result = subprocess.run(
        [str(interpreter), "-c", HERMETIC_SUITE],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        # About 50 s alone on a laptop; leave room for a loaded machine.
        timeout=200,
    )
    assert result.returncode == 0, result.stdout + result.stderr
