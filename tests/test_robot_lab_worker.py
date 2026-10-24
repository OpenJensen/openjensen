"""Run the hardware companion safety and real-format tests in the isolated CPU reader."""

import os
import subprocess
import sys
from pathlib import Path


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
        [str(interpreter), "-m", "unittest", "discover", "-s", "workers/robot_lab/tests", "-v"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
