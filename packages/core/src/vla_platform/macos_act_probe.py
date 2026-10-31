"""Fixed offline Python 3.12 readiness probe for the existing native ACT/MPS runtime."""

import contextlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

REVISION = "e595b7902714ba51f91e47523f66f89c5181b649"


def probe(root):
    sys.dont_write_bytecode = True

    def offline(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.bind"}:
            raise RuntimeError("The ACT readiness check is offline")

    sys.addaudithook(offline)
    if (
        platform.system() != "Darwin"
        or platform.machine() != "arm64"
        or sys.version_info[:2] != (3, 12)
        or sys.prefix == sys.base_prefix
    ):
        raise ValueError("ACT/MPS requires the existing Apple Silicon Python 3.12 environment")
    prefix = Path(sys.prefix).resolve()
    for name, expected in {"torch": "2.11.0", "torchvision": "0.26.0"}.items():
        distribution = importlib.metadata.distribution(name)
        if distribution.version.split("+")[0] != expected:
            raise ValueError("The existing ACT runtime has incompatible dependencies")
        if not Path(str(distribution.locate_file(""))).resolve().is_relative_to(prefix):
            raise ValueError("ACT dependencies must belong to the configured environment")
    distribution = importlib.metadata.distribution("lerobot")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != REVISION or not Path(
        str(distribution.locate_file(""))
    ).resolve().is_relative_to(prefix):
        raise ValueError("The existing ACT runtime is not pinned to the reviewed native revision")
    if not (root / "src/jensen_vla/lerobot_application.py").is_file():
        raise ValueError("The native ACT worker sources are unavailable")
    import torch

    importlib.import_module("torchvision")
    importlib.import_module("lerobot.policies.act.modeling_act")

    if not torch.backends.mps.is_available():
        raise ValueError("The existing ACT runtime cannot access the Apple GPU")
    device = torch.ones(1, device="mps")
    if float((device + 1).cpu()) != 2.0:
        raise ValueError("The Apple GPU readiness operation failed")
    torch.mps.synchronize()
    # Unified memory is system RAM, not dedicated CUDA VRAM.
    import subprocess

    memory = int(subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.memsize"], timeout=3))
    if memory < 16 * 1024**3:
        raise ValueError("Initial native ACT/MPS support requires at least 16 GiB unified memory")
    return {
        "schema_version": 1,
        "status": "ready",
        "reason": None,
        "gpu": {
            "name": "Apple GPU (MPS, unified memory)",
            "memory_mib": memory // 1024**2,
            "uuid": "GPU-APPLE-MPS",
            "index": 0,
        },
    }


if __name__ == "__main__":
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = probe(Path(sys.argv[1]).resolve())
    except Exception:
        result = {
            "schema_version": 1,
            "status": "unavailable",
            "reason": "mps_unavailable",
            "gpu": None,
        }
    print(json.dumps(result), flush=True)
