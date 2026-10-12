"""Structured worker evidence, without importing an optional ML stack."""

import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .checkpoint import write_json


def emit(phase, message, *, step=None, total_steps=None, **details):
    record = {
        "phase": phase,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    if step is not None:
        record["step"] = step
    if total_steps is not None:
        record["total_steps"] = total_steps
    print(json.dumps(record, allow_nan=False), flush=True)
    return record


def report(output, phase, message, *, step, total_steps, **details):
    record = emit(phase, message, step=step, total_steps=total_steps, **details)
    state = phase if phase in {"completed", "failed", "interrupted"} else "running"
    write_json(Path(output) / "status.json", {"state": state, **record})
    return record


def environment_report(torch, compute_dtype):
    packages = {}
    for name in (
        "torch",
        "torchvision",
        "lerobot",
        "transformers",
        "peft",
        "bitsandbytes",
        "safetensors",
        "datasets",
        "accelerate",
        "huggingface-hub",
        "numpy",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    # Content hashes identify uncommitted source too; no host paths or env secrets.
    source_files = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    return {
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "packages": packages,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "compute_dtype": compute_dtype,
        "source_files_sha256": source_files,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "reproducibility_note": (
            "Pinned inputs, seeded sampling and saved RNG/optimizer states support replay. "
            "Bitwise equivalence across GPU hardware and library versions is not guaranteed."
        ),
    }
