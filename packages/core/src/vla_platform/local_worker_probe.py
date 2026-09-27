"""Fixed, offline readiness probe executed by an isolated worker Python, never the app."""

import contextlib
import importlib
import importlib.metadata
import json
import sys
import uuid
from pathlib import Path

PINS = {
    "lerobot": "0.4.4",
    "torch": "2.7.1",
    "torchvision": "0.22.1",
    "transformers": "4.57.1",
    "peft": "0.18.0",
    "bitsandbytes": "0.48.2",
}


def block_network(event, arguments):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.bind"}:
        raise RuntimeError("Readiness checks are offline")


def unavailable(reason):
    return {"schema_version": 1, "status": "unavailable", "reason": reason, "gpu": None}


def gpu_uuid(value):
    """Torch 2.7.1 exposes CUuuid as bare hex; CUDA visibility needs GPU-<uuid>."""
    if value is None:
        return None
    try:
        identifier = uuid.UUID(str(value).removeprefix("GPU-"))
    except ValueError:
        return None
    return "GPU-" + str(identifier) if identifier.int else None


def probe(root):
    sys.dont_write_bytecode = True
    sys.addaudithook(block_network)
    if sys.version_info[:2] not in {(3, 11), (3, 12)} or sys.prefix == sys.base_prefix:
        return unavailable("incompatible_environment")
    prefix = Path(sys.prefix).resolve()
    for name, expected in PINS.items():
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            return unavailable("missing_dependencies")
        if distribution.version.split("+")[0] != expected:
            return unavailable("incompatible_dependencies")
        if not Path(str(distribution.locate_file(""))).resolve().is_relative_to(prefix):
            return unavailable("incompatible_environment")
    try:
        import torch

        if not torch.version.cuda or not torch.cuda.is_available():
            return unavailable("cuda_unavailable")
        if torch.cuda.get_device_capability(0)[0] < 7:
            return unavailable("unsupported_gpu")
        for module in (
            "torchvision",
            "peft",
            "bitsandbytes",
            "lerobot.policies.smolvla.modeling_smolvla",
            "lerobot.policies.smolvla.processor_smolvla",
        ):
            importlib.import_module(module)
        sys.path.insert(0, str(root / "src"))
        worker = importlib.import_module("firebird_vla.application")
        if Path(worker.__file__).resolve() != root / "src/firebird_vla/application.py":
            return unavailable("incompatible_environment")
        properties = torch.cuda.get_device_properties(0)
        identifier = gpu_uuid(getattr(properties, "uuid", None))
        return {
            "schema_version": 1,
            "status": "ready",
            "reason": None,
            "gpu": {
                "name": properties.name,
                "memory_mib": int(properties.total_memory // (1024 * 1024)),
                "uuid": identifier,
                "index": 0,
            },
        }
    except Exception:
        return unavailable("dependency_import_failed")


if __name__ == "__main__":
    try:
        # Dependency imports can print diagnostics; reserve stdout for one
        # machine-readable result and let the parent discard private stderr.
        with contextlib.redirect_stdout(sys.stderr):
            result = probe(Path(sys.argv[1]).resolve())
    except Exception:
        result = unavailable("probe_failed")
    print(json.dumps(result), flush=True)
