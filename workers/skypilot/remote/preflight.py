"""Check the baked GPU host; do not change drivers on an allocated VM."""

import json
import os
from pathlib import Path
import subprocess


_MIN_DRIVER = (580, 95, 5)
_ICD_DIRS = (Path("/etc/vulkan/icd.d"), Path("/usr/share/vulkan/icd.d"))
_GRAPHICS_LIBS = ("libGLX_nvidia.so.0", "libEGL_nvidia.so.0", "libvulkan.so.1")
_HOST_ABI = "x86-64"
_CHECK_TIMEOUT = 60


def _check():
    result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
                            check=True, text=True, capture_output=True, timeout=_CHECK_TIMEOUT)
    rows = result.stdout.strip().splitlines()
    if not rows:
        raise RuntimeError("No NVIDIA GPU detected")
    for row in rows:
        version = row.split(",", 1)[0].strip()
        if tuple(map(int, version.split("."))) < _MIN_DRIVER:
            raise RuntimeError(f"NVIDIA driver {version} is older than 580.95.05; use the pinned GPU host image")
        print(f"GPU: {row}", flush=True)

    descriptors = [path for directory in _ICD_DIRS for path in directory.glob("*nvidia*.json")]
    if not descriptors:
        raise RuntimeError("NVIDIA Vulkan ICD is missing; use a host image with NVIDIA graphics drivers")
    libraries = list(_GRAPHICS_LIBS)
    for path in descriptors:
        library = json.loads(path.read_text())["ICD"]["library_path"]
        if "/" in library and not Path(library).is_absolute():
            library = str(path.parent / library)
        libraries.append(library)

    # Inspect trusted host libraries without running NVIDIA teardown in Python.
    cache = _library_cache()
    for library in dict.fromkeys(libraries):
        _check_library(library, cache)
    subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL, timeout=_CHECK_TIMEOUT)
    print("Host driver, graphics dependencies, and Docker checks passed; rendering remains unverified.", flush=True)


def _library_cache():
    result = subprocess.run(["/sbin/ldconfig", "-p"], check=True, text=True,
                            capture_output=True, timeout=_CHECK_TIMEOUT)
    libraries = {}
    for line in result.stdout.splitlines():
        description, separator, path = line.partition(" => ")
        if not separator or _HOST_ABI not in description:
            continue
        libraries.setdefault(description.split()[0], Path(path.strip()))
    return libraries


def _check_library(library, cache):
    path = Path(library) if Path(library).is_absolute() else cache.get(library)
    if path is None or not path.is_file():
        raise RuntimeError(f"Graphics library is missing: {library}")

    result = subprocess.run(["/usr/bin/ldd", str(path)], check=False, text=True,
                            capture_output=True, timeout=_CHECK_TIMEOUT,
                            env=os.environ | {"LC_ALL": "C"})
    diagnostics = (result.stdout + result.stderr).strip()
    if result.returncode or "not found" in diagnostics:
        raise RuntimeError(f"Graphics dependencies unavailable for {library}: {diagnostics}")


if __name__ == "__main__":
    _check()
