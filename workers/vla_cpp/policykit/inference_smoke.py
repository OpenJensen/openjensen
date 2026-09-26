"""One bounded native prediction on explicit synthetic inputs, not task evaluation."""

from __future__ import annotations

import math
import os
import re
import subprocess
import time
from pathlib import Path

from .worker import atomic_json, sha256


def model_contract(model):
    import gguf

    reader = gguf.GGUFReader(model)
    if reader.fields["general.architecture"].contents() != "smolvla":
        raise ValueError("Native smoke verification supports SmolVLA")
    dimensions = {}
    for name in ("chunk_size", "max_action_dim", "real_action_dim", "image_size"):
        value = int(reader.fields["smolvla." + name].contents())
        if not 1 <= value <= (2048 if name == "image_size" else 1000):
            raise ValueError("Invalid GGUF smoke-test dimensions")
        dimensions[name] = value
    if dimensions["real_action_dim"] > dimensions["max_action_dim"]:
        raise ValueError("GGUF action dimensions are inconsistent")
    dimensions["packed_language_tensors"] = sum(
        tensor.name.startswith("vlm.blk.") and tensor.tensor_type.name in {"Q4_0", "Q8_0"}
        for tensor in reader.tensors
    )
    dimensions["packed_vision_tensors"] = sum(
        tensor.name.startswith("vit.blk.") and tensor.tensor_type.name in {"Q4_0", "Q8_0"}
        for tensor in reader.tensors
    )
    return dimensions


def parse_prediction(text, contract):
    if "backend = CPU" not in text or "backend = CUDA" in text:
        raise ValueError("CPU smoke inference did not establish its requested backend")
    marker = re.search(r"^action_len=(\d+)\s*$", text, re.M)
    expected = contract["chunk_size"] * contract["max_action_dim"]
    if marker is None or int(marker[1]) != expected:
        raise ValueError("Native action output does not match the checkpoint dimensions")
    rows = text[marker.end() :].lstrip("\r\n").splitlines()[:expected]
    try:
        values = [float(row) for row in rows]
    except ValueError as exc:
        raise ValueError("Native action output contains invalid numeric values") from exc
    if len(values) != expected or not all(map(math.isfinite, values)):
        raise ValueError("Native smoke inference produced missing or non-finite actions")
    packed = re.search(r"packed resident matrices: lm=(\d+) vision=(\d+)", text)
    if packed is None or tuple(map(int, packed.groups())) != (
        contract["packed_language_tensors"],
        contract["packed_vision_tensors"],
    ):
        raise ValueError("Native resident packing differs from the quantized GGUF")
    return values


def verify_cpu_inference(model: Path, executable: Path, output: Path, *, cameras=2):
    if not executable.is_file():
        raise ValueError("The CPU inference verifier was not built")
    if type(cameras) is not int or not 1 <= cameras <= 8:
        raise ValueError("Native smoke inference needs 1 to 8 synthetic camera inputs")
    contract = model_contract(model)
    model_sha = sha256(model)
    output.mkdir(parents=True, exist_ok=True)
    log = output / "cpu-inference-smoke.log"
    command = [str(executable.resolve()), str(model.resolve()), "", str(cameras)]
    started = time.monotonic()
    with log.open("w") as stream:
        try:
            completed = subprocess.run(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=300,
                env={
                    **os.environ,
                    "VLA_N_THREADS": "2",
                    "OMP_NUM_THREADS": "2",
                    "VLA_IMG_SIZE": str(contract["image_size"]),
                    "VLA_BENCH_ITERS": "0",
                },
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError("Native CPU inference exceeded its 300-second limit") from exc
    if completed.returncode:
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 2000))
            diagnostic = stream.read().decode("utf-8", errors="replace")
        raise ValueError(
            "Quantized model failed native loading or inference: " + diagnostic
        )
    if log.stat().st_size > 4 * 1024**2:
        raise ValueError("Native smoke inference exceeded its output limit")
    values = parse_prediction(log.read_text(), contract)
    if sha256(model) != model_sha:
        raise ValueError("Quantized model changed during native smoke inference")
    report = {
        "scope": "synthetic_input_native_inference",
        "backend": "cpu",
        "calls": 1,
        "threads": 2,
        "synthetic_image_count": cameras,
        "image_size": contract["image_size"],
        "input_description": "Pinned vla_predict_check image patterns, token IDs, state and noise",
        "finite_action_values": len(values),
        "action_chunk_size": contract["chunk_size"],
        "max_action_dim": contract["max_action_dim"],
        "real_action_dim": contract["real_action_dim"],
        "packed_language_tensors": contract["packed_language_tensors"],
        "packed_vision_tensors": contract["packed_vision_tensors"],
        "wall_seconds": time.monotonic() - started,
        "model_sha256": model_sha,
        "executable_sha256": sha256(executable),
        "task_success": None,
        "deployment_verified": False,
    }
    atomic_json(output / "cpu-inference-smoke-actions.json", {"values": values})
    atomic_json(output / "cpu-inference-smoke.json", report)
    return report
