"""One bounded native prediction on explicit synthetic inputs, not task evaluation."""

from __future__ import annotations

import math
import hashlib
import json
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
                    **{key: value for key, value in os.environ.items()
                       if key not in {"VLA_EXTRA_TOKEN", "VLA_EXTRA_COUNT"}},
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
    executable_sha = sha256(executable)
    inputs = {
        "fixture": "vla_predict_check_fixed_inputs_v1",
        "executable_sha256": executable_sha,
        "synthetic_image_count": cameras,
        "image_size": contract["image_size"],
        "action_chunk_size": contract["chunk_size"],
        "max_action_dim": contract["max_action_dim"],
        "real_action_dim": contract["real_action_dim"],
        "extra_tokens": False,
    }
    actions = output / "cpu-inference-smoke-actions.json"
    atomic_json(actions, {"values": values})
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
        "executable_sha256": executable_sha,
        "input_sha256": hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
        "actions_sha256": sha256(actions),
        "task_success": None,
        "deployment_verified": False,
    }
    atomic_json(output / "cpu-inference-smoke.json", report)
    return report


def compare_cpu_predictions(reference, candidate, reference_output: Path, candidate_output: Path):
    """Compare paired native outputs, excluding padding; never claim held-out loss."""
    keys = ("input_sha256", "executable_sha256", "synthetic_image_count", "image_size",
            "action_chunk_size", "max_action_dim", "real_action_dim")
    if any(reference.get(key) is None or reference[key] != candidate.get(key) for key in keys):
        raise ValueError("Quantization comparison requires identical inputs, dimensions and verifier")
    if any(report.get("backend") != "cpu" or report.get("calls") != 1
           for report in (reference, candidate)):
        raise ValueError("Quantization comparison requires paired CPU predictions")
    chunk, width, real = (reference[key] for key in
                          ("action_chunk_size", "max_action_dim", "real_action_dim"))
    if any(type(value) is not int or not 1 <= value <= 1000 for value in (chunk, width, real)) or real > width:
        raise ValueError("Invalid quantization comparison dimensions")
    values = []
    for report, directory in ((reference, reference_output), (candidate, candidate_output)):
        path = directory / "cpu-inference-smoke-actions.json"
        if sha256(path) != report.get("actions_sha256"):
            raise ValueError("Quantization comparison action evidence changed")
        vector = json.loads(path.read_text())["values"]
        if not isinstance(vector, list) or len(vector) != chunk * width or any(
            type(value) not in (int, float) or not math.isfinite(value) for value in vector
        ):
            raise ValueError("Quantization comparison requires complete finite action chunks")
        values.append([vector[step * width + channel]
                       for step in range(chunk) for channel in range(real)])
    errors = [abs(left - right) for left, right in zip(*values)]
    mse = math.fsum(error * error for error in errors) / len(errors)
    metrics = {"action_rmse": math.sqrt(mse), "action_mse": mse,
               "action_mae": math.fsum(errors) / len(errors),
               "action_max_abs_difference": max(errors)}
    if not all(map(math.isfinite, metrics.values())):
        raise ValueError("Quantization comparison differences are non-finite")
    return {
        "schema_version": 1,
        "scope": "paired_synthetic_native_actions",
        "reference": "floating_gguf_before_quantization",
        "backend": "cpu",
        "samples": 1,
        "coordinates": len(errors),
        "action_chunk_size": chunk,
        "real_action_dim": real,
        "input_sha256": reference["input_sha256"],
        "executable_sha256": reference["executable_sha256"],
        "source_model_sha256": reference["model_sha256"],
        "quantized_model_sha256": candidate["model_sha256"],
        "validation_loss": None,
        "task_success": None,
        **metrics,
    }
