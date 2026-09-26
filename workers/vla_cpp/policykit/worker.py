"""Version-one core worker. Conversion only: no scheduler or deployment selector."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

from .quantization import quantize, tensor_quantization, validate_master


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def code_hash():
    root = Path(__file__).parent
    files = {p.name: sha256(p) for p in sorted(root.glob("*.py"))}
    return hashlib.sha256(canonical(files).encode()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def check_runtime(runtime):
    vendor = Path(runtime["vendor_path"])
    commit = subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"],
                                     text=True).strip()
    if commit != runtime["commit"]:
        raise ValueError("Runtime base commit mismatch")
    diff = subprocess.check_output(["git", "-C", str(vendor), "diff", "HEAD", "--binary"])
    if hashlib.sha256(diff).hexdigest() != runtime["patch_sha256"]:
        raise ValueError("Runtime patch mismatch; unexpected vendor edits")
    script = vendor / "scripts/quantize_gguf.py"
    if sha256(script) != runtime["quantizer_sha256"]:
        raise ValueError("Upstream quantizer hash mismatch")
    if code_hash() != runtime["worker_sha256"]:
        raise ValueError("Worker source changed; resolve a new recipe")
    installed = {name: version(name) for name in ("numpy", "gguf")}
    if installed != runtime["dependencies"]:
        raise ValueError(f"Worker dependency mismatch: installed {installed}")
    return script


def describe_runtime(vendor):
    patch = files("policykit").joinpath("patches/vla-cpp-smolvla-packed.patch")
    runtime = {
        "vendor_path": str(vendor.resolve()),
        "commit": "52439f7c6c362d7bee218b400b9080cc32d75cc3",
        "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
        "quantizer_sha256": sha256(vendor / "scripts/quantize_gguf.py"),
        "worker_sha256": code_hash(),
        "dependencies": {name: version(name) for name in ("numpy", "gguf")},
    }
    check_runtime(runtime)
    return runtime


def run(job_path):
    job = json.loads(job_path.read_text())
    output = job_path.parent.resolve()
    result = {key: job[key] for key in ("schema_version", "run_id", "attempt_id", "recipe_sha256")}
    result.update(status="failed", evidence_scope="conversion_only")
    sequence = 0

    def event(state, **fields):
        nonlocal sequence
        sequence += 1
        record = dict(schema_version=1, run_id=job["run_id"], attempt_id=job["attempt_id"],
                      sequence=sequence, timestamp=time.time(), state=state, **fields)
        with (output / "events.jsonl").open("a") as stream:
            stream.write(canonical(record) + "\n")
        print(canonical(record), flush=True)

    staging = None
    try:
        recipe = job["recipe"]
        if type(job["schema_version"]) is not int or job["schema_version"] != 1:
            raise ValueError("Unsupported job schema")
        if Path(job["output_dir"]).resolve() != output:
            raise ValueError("Job output directory does not match envelope location")
        if hashlib.sha256(canonical(recipe).encode()).hexdigest() != job["recipe_sha256"]:
            raise ValueError("Recipe digest mismatch")
        if (recipe["schema_version"] != 1 or recipe["operation"] != "policy.quantize"
                or recipe["backend"] != "vla_cpp_smolvla"
                or recipe["policy"]["architecture"] != "smolvla"
                or recipe["language"] not in {"Q4_0", "Q8_0"}
                or recipe["vision"] not in {None, "Q8_0"}):
            raise ValueError("Unsupported quantization recipe")
        event("preflight")
        source = Path(recipe["source"]["path"])
        if sha256(source) != recipe["source"]["sha256"]:
            raise ValueError("Source artifact hash mismatch")
        script = check_runtime(recipe["runtime"])
        import gguf

        reader = gguf.GGUFReader(source)
        if reader.fields["general.architecture"].contents() != "smolvla":
            raise ValueError("Source GGUF is not SmolVLA")
        validate_master(reader)
        staging = Path(tempfile.mkdtemp(prefix=".conversion-", dir=output))
        event("quantizing")
        started = time.perf_counter()
        audit = quantize(source, staging / "model.gguf", script,
                         recipe["language"], recipe["vision"])
        elapsed = time.perf_counter() - started
        # Verify input/runtime identities again before publishing any output.
        if sha256(source) != recipe["source"]["sha256"]:
            raise ValueError("Source changed during conversion")
        check_runtime(recipe["runtime"])
        packed = gguf.GGUFReader(staging / "model.gguf")
        original = {t.name: t for t in reader.tensors}
        actual = {t.name: t for t in packed.tensors}
        if set(original) != set(actual):
            raise ValueError("Conversion changed tensor inventory")
        changed = []
        for name, tensor in actual.items():
            before = original[name]
            if tuple(tensor.shape) != tuple(before.shape):
                raise ValueError(f"Conversion changed tensor shape: {name}")
            expected = tensor_quantization(name, before.shape, recipe["language"], recipe["vision"])
            if expected is not None and tensor.tensor_type.name != expected:
                raise ValueError(f"Incorrect packed precision: {name}")
            if expected is None and tensor.tensor_type != before.tensor_type:
                raise ValueError(f"Protected tensor precision changed: {name}")
            if tensor.tensor_type != before.tensor_type:
                changed.append(name)
            elif tensor.data.tobytes() != before.data.tobytes():
                raise ValueError(f"Protected tensor changed: {name}")
        if not changed:
            raise ValueError("No supported matrices were quantized")
        atomic_json(staging / "recipe.json", recipe)
        manifest = {
            "schema_version": 1, "kind": "quantized_weights", "format": "gguf",
            "run_id": job["run_id"], "attempt_id": job["attempt_id"],
            "recipe_sha256": job["recipe_sha256"], "source_sha256": recipe["source"]["sha256"],
            "policy": recipe["policy"], "runtime": recipe["runtime"],
            "target": recipe["target"], "host": {"system": platform.system(),
                                                  "machine": platform.machine()},
            "evidence_scope": "conversion_only", "task_success": None,
            "deployment_verified": False, "quantized_tensors": sorted(changed),
            "protected_tensors": sorted(set(original) - set(changed)),
            "tensor_types": audit["tensor_types"], "quantize_seconds": elapsed,
            "deployed_weight_bytes": sum(int(t.data.nbytes) for t in packed.tensors),
            "gguf_file_bytes": (staging / "model.gguf").stat().st_size,
            "files": {p.name: sha256(p) for p in sorted(staging.iterdir()) if p.is_file()},
        }
        atomic_json(staging / "manifest.json", manifest)
        if (output / "bundle").exists():
            raise FileExistsError("Refusing to overwrite an existing bundle")
        os.rename(staging, output / "bundle")
        staging = None
        result.update(status="succeeded", artifact_manifest="bundle/manifest.json")
        event("succeeded", evidence_scope="conversion_only")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        event("failed", error=result["error"])
    finally:
        if staging is not None:
            shutil.rmtree(staging)
        atomic_json(output / "result.json", result)
    return 0 if result["status"] == "succeeded" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--job", type=Path)
    mode.add_argument("--describe-runtime", type=Path, metavar="VENDOR")
    args = parser.parse_args()
    if args.describe_runtime:
        print(canonical(describe_runtime(args.describe_runtime)))
        return 0
    return run(args.job.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
