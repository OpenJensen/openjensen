"""Offline import/source sanity probe, never a policy or robot-quality evaluation."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

PINS = {"torch": "2.11.0", "torchvision": "0.26.0", "safetensors": "0.8.0"}
READER_PINS = {
    "numpy": "2.2.6",
    "torchcodec": "0.11.1",
    "datasets": "4.8.5",
    "pandas": "2.3.3",
    "pyarrow": "25.0.1",
    "av": "15.1.0",
}
WRITER = {
    "datasets/lerobot_dataset.py": "8f6a83a46d82b4401f58e859844c9b265695368ee50af7772143d042e01a84ec",  # noqa: E501
    "datasets/dataset_writer.py": "b1ce93c7e668adfea081075a0fd02d44d50aae81c83096e105f6ce0346e35a78",  # noqa: E501
    "datasets/video_utils.py": "b7f0e94dc547a269198ef19f1fd364374c32f4be8eb5cf5cc8505243c9bb79ed",
    "configs/video.py": "8575fd0baabbc1c35acb7491871e4b146b9981c92686f4bc95721079f038f079",
}


def block_network(event: str, arguments: tuple[object, ...]) -> None:
    """Reject Python socket DNS/connect during readiness imports."""
    if event in {"socket.connect", "socket.getaddrinfo", "socket.bind"}:
        raise RuntimeError("Runtime verification is offline")


def installed_versions(pins: dict[str, str], prefix: Path) -> dict[str, str]:
    """Read each distribution's metadata once while retaining exact admission checks."""
    indexed: dict[str, list[tuple[importlib.metadata.Distribution, str | None]]] = {}
    for distribution in importlib.metadata.distributions():
        metadata = distribution.metadata
        name = metadata["Name"]
        if not isinstance(name, str):
            raise ValueError("Installed distribution metadata has no package name")
        normalized = name.lower().replace("_", "-")
        if normalized in pins:
            indexed.setdefault(normalized, []).append((distribution, metadata["Version"]))
    versions = {}
    prefix = prefix.resolve()
    for name, expected in pins.items():
        matches = indexed.get(name, [])
        if len(matches) != 1:
            raise ValueError(
                f"{name} has missing or duplicate metadata; no borrowed environment paths"
            )
        distribution, actual = matches[0]
        if not isinstance(actual, str) or actual.split("+")[0] != expected:
            raise ValueError(f"{name} requires {expected}, found {actual}")
        if not Path(str(distribution.locate_file(""))).resolve().is_relative_to(prefix):
            raise ValueError(f"{name} is borrowed from another environment")
        versions[name] = actual
    return versions


def distillation_source_identity(repo: Path) -> dict[str, str]:
    """Bind the worker's declared dependency identity to the selected checkout."""
    root = repo.resolve() / "workers"
    groups = (
        (
            "distillation",
            "firebird_distill",
            "policy_distillation/src/firebird_distill",
            ("__init__", "application", "contracts", "prepare", "runtime", "provenance"),
        ),
        (
            "act",
            "firebird_act",
            "act_optimizer/src/firebird_act",
            ("bundle", "probe", "control_schema"),
        ),
        (
            "data",
            "firebird_vla",
            "smolvla_qlora/src/firebird_vla",
            ("control_contract", "control_schema"),
        ),
    )
    expected = {}
    for prefix, package, folder, names in groups:
        for name in names:
            imported = package if name == "__init__" else f"{package}.{name}"
            module = importlib.import_module(imported)
            path = root / folder / f"{name}.py"
            if module.__file__ is None or Path(module.__file__).resolve() != path:
                raise ValueError(
                    f"Distillation dependency is not from selected checkout: {imported}"
                )
            with path.open("rb") as stream:
                raw = stream.read(1024 * 1024 + 1)
            if not raw or len(raw) > 1024 * 1024:
                raise ValueError(f"Invalid bounded worker source: {imported}")
            expected[f"{prefix}/{name}.py"] = hashlib.sha256(raw).hexdigest()
    application = importlib.import_module("firebird_distill.application")
    actual = application.implementation_identity()
    if actual != expected:
        raise ValueError("Distillation implementation identity differs from selected source files")
    return expected


def check(role: str, repo: Path) -> dict[str, object]:
    """Check an independent installed runtime and its fixed repository modules."""
    if role not in {"model", "reader"} or sys.version_info[:2] != (3, 12):
        raise ValueError("Use the model/reader role with Python3.12")
    if sys.prefix == sys.base_prefix:
        raise ValueError("An isolated virtual environment is required")
    pins = PINS | {"lerobot": "0.6.1" if role == "model" else "0.6.2"}
    if role == "reader":
        pins |= READER_PINS
    versions = installed_versions(pins, Path(sys.prefix))
    if platform.system() == "Linux" and any(
        not versions[n].endswith("+cpu") for n in ("torch", "torchvision")
    ):
        raise ValueError("Linux requires the pinned CPU Torch/torchvision wheels")
    sys.addaudithook(block_network)
    import torch
    import torchvision  # noqa: F401

    if torch.version.cuda is not None or str(torch.ones(1).device) != "cpu":
        raise ValueError("Expected CPU-only Torch; CUDA is not part of this setup")
    import lerobot

    package = Path(lerobot.__file__).parent
    writer_hashes = {}
    if role == "reader":
        for name, expected in WRITER.items():
            actual = hashlib.sha256((package / name).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"Reader source differs from the approved upstream pin: {name}")
            writer_hashes[name] = actual
        import av
        import pyarrow

        importlib.import_module("lerobot.datasets.lerobot_dataset")
        if av.codec.Codec("h264", "r").type != "video":
            raise ValueError("PyAV H.264 decoder is unavailable")
        if pyarrow.table({"frame": [0]}).num_rows != 1:
            raise ValueError("PyArrow import/runtime check failed")
    folders = [
        "isaac_sim",
        "firebird_quant/src",
        "act_optimizer/src",
        "policy_distillation/src",
        "smolvla_qlora/src",
    ]
    sys.path[:0] = [str(repo / "workers" / x) for x in folders]
    modules = (
        ["sim_worker.rollout.native_replay", "firebird_distill.application"]
        if role == "model"
        else ["sim_worker.rollout.native_replay_prepare", "firebird_distill.prepare"]
    )
    loaded = {}
    for name in modules:
        module = importlib.import_module(name)
        if module.__file__ is None:
            raise ValueError("Worker module has no source file")
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(repo.resolve()):
            raise ValueError("Worker import did not come from the selected checkout")
        loaded[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "role": role,
        "python": platform.python_version(),
        "versions": versions,
        "worker_sha256": loaded,
        "distillation_implementation_sha256": distillation_source_identity(repo),
        "reader_source_sha256": writer_hashes,
        "model_execution": False,
        "network": False,
        "device": "cpu",
    }


if __name__ == "__main__":
    try:
        print(json.dumps(check(sys.argv[1], Path(sys.argv[2]))))
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"Runtime verification: {error}", file=sys.stderr)
        raise SystemExit(1) from None
