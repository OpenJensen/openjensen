"""Exercise runtime resource lookup from a real wheel outside the source tree."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def test_installed_wheel_can_describe_runtime(tmp_path):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("wheel installation smoke test requires uv")
    source = Path(__file__).resolve().parents[1]
    project = tmp_path / "project"
    project.mkdir()
    for name in ("pyproject.toml", "README.md"):
        shutil.copy2(source / name, project / name)
    shutil.copytree(source / "policykit", project / "policykit",
                    ignore=shutil.ignore_patterns("__pycache__"))
    subprocess.run([uv, "build", "--wheel", "--out-dir", str(tmp_path / "dist")],
                   cwd=project, check=True, capture_output=True, text=True)
    wheel, = (tmp_path / "dist").glob("*.whl")
    environment = tmp_path / "installed"
    subprocess.run([uv, "venv", "--python", sys.executable, str(environment)],
                   check=True, capture_output=True, text=True)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run([uv, "pip", "install", "--python", str(python), "--no-deps", str(wheel)],
                   check=True, capture_output=True, text=True)
    # No native checkout or NumPy is needed to test packaging. Only the runtime
    # identity verifier and dependency versions are stubbed; resource access and
    # the installed CLI's argument parsing/JSON output execute normally.
    vendor = tmp_path / "vendor"
    (vendor / "scripts").mkdir(parents=True)
    (vendor / "scripts/quantize_gguf.py").write_text("# synthetic runtime\n")
    code = """
import sys
from pathlib import Path
from unittest.mock import patch
from policykit import worker
assert Path(worker.__file__).is_relative_to(Path(sys.prefix))
with patch.object(worker, 'check_runtime'), patch.object(worker, 'version', return_value='test'):
    sys.argv = ['policykit-worker', '--describe-runtime', sys.argv[1]]
    raise SystemExit(worker.main())
"""
    result = subprocess.run([str(python), "-I", "-c", code, str(vendor)],
                            cwd=tmp_path, check=True, capture_output=True, text=True)
    runtime = json.loads(result.stdout)
    expected_patch = source / "policykit/patches/vla-cpp-smolvla-packed.patch"
    assert runtime["patch_sha256"] == hashlib.sha256(expected_patch.read_bytes()).hexdigest()
    assert runtime["vendor_path"] == str(vendor.resolve())
    # Experimental entry points must be inspectable from the installed wheel
    # without importing optional CUDA/PyTorch/simulator dependencies.
    for module in ("cuda_bench", "cuda_rollout", "modelopt_pilot"):
        help_result = subprocess.run(
            [str(python), "-I", "-m", f"policykit.{module}", "--help"],
            cwd=tmp_path, check=True, capture_output=True, text=True,
        )
        assert "usage:" in help_result.stdout
