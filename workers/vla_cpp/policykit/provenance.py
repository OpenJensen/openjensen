"""Identity of the supported Linux native execution environment.

This is a reproducibility fingerprint, not a portable runtime or an attestation
of arbitrary third-party code. Native libraries are hashed, Python distributions
are recorded by version, and prepared simulator/adapter source is hashed.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

from .cuda_common import hardware_fingerprint
from .worker import canonical, code_hash, sha256


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def source_identity(root):
    """Hash executable simulator source/configuration, excluding caches/assets."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError("Prepared runtime source is unavailable: " + str(root))
    files = {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not any(
            part.startswith(".") or part == "__pycache__" for part in path.relative_to(root).parts
        )
        and path.suffix in {".py", ".proto", ".yaml", ".yml", ".toml", ".json"}
    }
    if not files:
        raise ValueError("Prepared runtime source has no auditable code: " + str(root))
    return {"root": str(root.resolve()), "files": files}


def native_libraries(build, executables):
    """Resolve ELF dependencies plus build-local libraries loaded via dlopen."""
    libraries = {}
    targets = set(executables)
    targets.update(path for path in build.rglob("*") if path.is_file() and ".so" in path.name)
    for target in sorted(targets):
        if target not in executables:
            libraries[str(target.resolve())] = sha256(target)
        try:
            output = subprocess.check_output(
                ["ldd", str(target)], text=True, stderr=subprocess.STDOUT
            )
        except subprocess.CalledProcessError as exc:
            if "statically linked" in (exc.output or "") or "not a dynamic executable" in (
                exc.output or ""
            ):
                continue
            raise ValueError("Unable to inspect native linked libraries: " + str(target)) from exc
        for line in output.splitlines():
            line = line.strip()
            if not line or "linux-vdso" in line or line == "statically linked":
                continue
            match = re.search(r"(?:=>\s+)?(/\S+)\s+\(", line)
            if match is None:
                raise ValueError("Unresolved native linked library: " + line)
            path = Path(match[1]).resolve()
            if not path.is_file():
                raise ValueError("Native linked library disappeared: " + str(path))
            libraries[str(path)] = sha256(path)
    return libraries


def runtime_identity(runtime):
    if platform.system() != "Linux":
        raise ValueError(
            "Verified native runtime identity requires Linux ELF; this host is unsupported"
        )
    build = Path(runtime["build"]).resolve()
    names = ["vla-bench", "tests/vla_predict_check"]
    if (build / "vla-server").is_file():
        names.append("vla-server")
    executables = [build / name for name in names]
    worker = Path(runtime.get("worker_root") or Path(__file__).resolve().parents[1])
    locks = {name: sha256(worker / name) for name in ("pyproject.toml", "uv.lock")}
    sources = {}
    if runtime.get("vendor"):
        sources["native_adapters"] = source_identity(Path(runtime["vendor"]) / "eval")
    lane = runtime.get("simulator_lane")
    if lane:
        lane = Path(lane)
        if not (build / "vla-server").is_file():
            raise ValueError("Simulator runtime requires vla-server")
        sources["rollout_adapters"] = source_identity(lane / "vendor/vla.cpp/eval")
        sources["simulator"] = source_identity(lane / "LIBERO/libero/libero")
        # cuda_rollout generates its configuration from these roots, so fingerprint
        # the derived values rather than a config file it creates during execution.
        sources["simulator_configuration"] = {
            "base": str((lane / "LIBERO/libero/libero").resolve()),
            "datasets": str((lane / "LIBERO/libero/datasets").resolve()),
            "renderer": "osmesa",
        }
    environment_names = {
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_HOME",
        "PYTHONPATH",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VLA_N_THREADS",
        "VLA_IMG_SIZE",
        "MUJOCO_GL",
        "PYOPENGL_PLATFORM",
        "LP_NUM_THREADS",
    }
    identity = {
        "schema_version": 2,
        "scope": (
            "Linux ELF libraries, worker and prepared adapter source, "
            "Python package versions and runtime configuration"
        ),
        "binaries": {name: sha256(build / name) for name in names},
        "native_libraries": native_libraries(build, executables),
        "device": runtime["device"],
        "hardware": hardware_fingerprint(),
        "worker_sha256": code_hash(),
        "dependency_locks": locks,
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "sha256": sha256(Path(sys.executable).resolve()),
            "version": sys.version,
            "distributions": sorted(
                (distribution.metadata.get("Name", "unknown"), distribution.version)
                for distribution in importlib.metadata.distributions()
            ),
        },
        "prepared_sources": sources,
        # Never put credentials carried by the runtime record into exported reports.
        "configuration_sha256": digest(runtime),
        "environment_sha256": digest(
            {name: os.environ.get(name) for name in sorted(environment_names)}
        ),
    }
    if runtime["device"] == "cuda":
        identity["gpu"] = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                "0",
                "--query-gpu=uuid,name,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    return identity
