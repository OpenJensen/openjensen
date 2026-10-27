"""Run isolated transport and fake-simulator invariants in the core CI lane."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("worker", ["provider", "simulation"])
def test_agent_worker_contracts_without_model_or_physics(worker):
    environment = os.environ.copy()
    environment.update(
        HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"
    )
    if worker == "provider":
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT / "workers/agent_control/src"), environment.get("PYTHONPATH", "")]
        )
        command = [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "workers/agent_control/tests",
            "-q",
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "workers/teaching/tests/test_agent_control.py",
            "workers/teaching/tests/test_frame_snapshot.py",
        ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
