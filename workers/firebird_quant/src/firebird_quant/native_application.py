"""Fixed local ACT packed-policy operation, with bounded owned verification processes."""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from .native_package import (
    EVIDENCE_LIMIT,
    JSON_LIMIT,
    RUNTIME,
    SHA,
    admit_source,
    canonical,
    checked_path,
    inspect_policy,
    inventory,
    read,
    read_json,
    sha,
    validate_inventory,
    write_new,
)


def _signal_group(group, sig):
    try:
        os.killpg(group, sig)
    except ProcessLookupError:
        return
    except PermissionError as original:
        # Darwin can report EPERM while the last signalled member is exiting.
        # Only kernel-confirmed disappearance is success; persistent uncertainty fails.
        deadline = time.monotonic() + 0.25
        while time.monotonic() < deadline:
            time.sleep(0.01)
            try:
                os.killpg(group, 0)
            except ProcessLookupError:
                return
            except PermissionError:
                continue
            break
        raise original


class _Owner:
    """Signals only set a flag, so interruption cannot lose a just-spawned child."""

    def __init__(self, timeout):
        self.deadline = time.monotonic() + timeout
        self.interrupted = False
        self.previous = {}

    def check(self):
        if self.interrupted:
            raise InterruptedError("Native quantization interrupted; no success receipt")
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Native quantization exceeded its verification deadline")

    def __enter__(self):
        if os.name != "posix" or threading.current_thread() is not threading.main_thread():
            raise ValueError("Native quantization owner requires a POSIX main process")
        for sig in (signal.SIGINT, signal.SIGTERM):
            self.previous[sig] = signal.getsignal(sig)
            signal.signal(sig, self._signal)
        return self

    def _signal(self, _signum, _frame):
        self.interrupted = True

    def __exit__(self, *_):
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)

    def run(self, command, environment, log):
        self.check()
        process = None
        try:
            # The handler cannot raise during Popen's spawn/registration window.
            process = subprocess.Popen(
                command,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while process.poll() is None:
                self.check()
                try:
                    process.wait(timeout=min(0.05, max(0.001, self.deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    pass
            self.check()
            if process.returncode:
                raise ValueError("ACT worker failed; no quantized policy was published")
        finally:
            if process is not None:
                # Also remove descendants after a successful/error leader exit.
                _signal_group(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=0.25)
                except subprocess.TimeoutExpired:
                    pass
                _signal_group(process.pid, signal.SIGKILL)
                process.wait(timeout=5)


def _identity(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise ValueError("A bounded identity is required")
    return value


def validate_request(job):
    required = {
        "schema_version",
        "job_id",
        "operation",
        "source",
        "output_dir",
        "native_quantization",
        "timeout_seconds",
    }
    if (
        not isinstance(job, dict)
        or set(job) != required
        or type(job["schema_version"]) is not int
        or job["schema_version"] != 1
    ):
        raise ValueError("Unsupported native quantization request")
    _identity(job["job_id"])
    if job["operation"] != "policy.quantize":
        raise ValueError("Only policy.quantize is supported")
    source = job["source"]
    if not isinstance(source, dict) or set(source) != {
        "path",
        "files",
        "manifest_sha256",
        "artifact_id",
        "artifact_manifest_sha256",
    }:
        raise ValueError("Complete source identity and inventory are required")
    _identity(source["artifact_id"])
    if not isinstance(source["artifact_manifest_sha256"], str) or not SHA.fullmatch(
        source["artifact_manifest_sha256"]
    ):
        raise ValueError("Registered artifact manifest identity must be SHA256")
    validate_inventory(source["files"])
    value = source["manifest_sha256"]
    if value is not None and (not isinstance(value, str) or not SHA.fullmatch(value)):
        raise ValueError("Source manifest identity must be SHA256 or null")
    recipe = job["native_quantization"]
    if (
        not isinstance(recipe, dict)
        or set(recipe) != {"format", "bits", "group_size"}
        or recipe["format"] != "firebird_quant"
        or type(recipe["bits"]) is not int
        or recipe["bits"] not in {4, 8}
        or type(recipe["group_size"]) is not int
        or recipe["group_size"] != 64
    ):
        raise ValueError("Supported native recipe is signed 4/8-bit, group size 64")
    timeout = job["timeout_seconds"]
    if type(timeout) is not int or not 1 <= timeout <= 600:
        raise ValueError("Verification timeout must be integer 1..600 seconds")
    path, output = checked_path(source["path"]), checked_path(job["output_dir"])
    if output == path or output.is_relative_to(path) or path.is_relative_to(output):
        raise ValueError("Source and output must be separate directories")
    destination = output / "native-quantized"
    if os.path.lexists(destination):
        raise FileExistsError("Refusing to overwrite a native quantized artifact")
    return path, output, destination


def _environment():
    import firebird_act

    # No cloud/provider credentials are inherited by these offline CPU probes.
    environment = {
        key: os.environ[key]
        for key in ("PATH", "TMPDIR", "SYSTEMROOT", "WINDIR")
        if key in os.environ
    }
    environment.update(
        PYTHONPATH=os.pathsep.join(
            [
                str(Path(__file__).resolve().parents[1]),
                str(Path(firebird_act.__file__).resolve().parents[1]),
            ]
        ),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    return environment


def _probe(owner, mode, source, result, *, destination=None, bits=None):
    command = [sys.executable, "-m", "firebird_quant.native_probe", mode, str(source), str(result)]
    if destination is not None:
        command += ["--destination", str(destination), "--bits", str(bits)]
    with tempfile.TemporaryFile() as log:
        try:
            owner.run(command, _environment(), log)
        except ValueError as error:
            log.seek(0, os.SEEK_END)
            end = log.tell()
            log.seek(max(0, end - 4096))
            detail = log.read(4096).decode("utf-8", errors="replace")
            raise ValueError(
                f"{error}\nOffline worker diagnostic (last 4 KiB):\n{detail}"
            ) from error
    return read_json(result, limit=EVIDENCE_LIMIT)


def _fixtures(value, prediction=100):
    from .native_probe import SEEDS

    if not isinstance(value, list) or len(value) != len(SEEDS):
        raise ValueError("Complete generated fixture reports are required")
    for row, seed in zip(value, SEEDS, strict=True):
        if not isinstance(row, dict) or set(row) != {
            "seed",
            "input_sha256",
            "image_shape",
            "raw",
            "postprocessed",
            "queue_and_reset_exact",
        }:
            raise ValueError("Invalid generated fixture record")
        if (
            type(row["seed"]) is not int
            or row["seed"] != seed
            or not isinstance(row["input_sha256"], str)
            or not SHA.fullmatch(row["input_sha256"])
            or row["queue_and_reset_exact"] is not True
        ):
            raise ValueError("Invalid generated fixture identity/queue verification")
        image = row["image_shape"]
        if (
            not isinstance(image, list)
            or len(image) != 3
            or image[0] != 3
            or any(type(n) is not int or not 1 <= n <= 2048 for n in image)
        ):
            raise ValueError("Invalid generated image shape")
        for key in ("raw", "postprocessed"):
            chunk = row[key]
            if (
                not isinstance(chunk, list)
                or len(chunk) != prediction
                or any(not isinstance(action, list) or len(action) != 6 for action in chunk)
            ):
                raise ValueError("Expected full prediction-horizon x 6 chunk measurements")
            if any(
                type(v) not in {int, float} or not math.isfinite(v)
                for action in chunk
                for v in action
            ):
                raise ValueError("Nonfinite action measurement")
    return value


def verify_reports(conversion, reload, info):
    for report in (conversion, reload):
        versions = report.get("versions")
        if (
            type(report.get("schema_version")) is not int
            or report["schema_version"] != 1
            or not isinstance(versions, dict)
            or {k: str(v).split("+")[0] for k, v in versions.items()} != RUNTIME
            or report.get("model_id") != info["model_id"]
            or report.get("policy_files") != info["files"]
            or report.get("network_disabled") is not True
        ):
            raise ValueError("Probe runtime/package identity differs")
    prediction = info.get("prediction_horizon", 100)
    baseline = _fixtures(conversion.get("baseline"), prediction)
    candidate = _fixtures(conversion.get("packed"), prediction)
    restored = _fixtures(reload.get("packed"), prediction)
    if restored != candidate or reload.get("floating_master_reads_blocked") is not True:
        raise ValueError("Fresh packed-only replay differs from the tested candidate")
    differences = []
    for reference, packed in zip(baseline, candidate, strict=True):
        if any(reference[key] != packed[key] for key in ("seed", "input_sha256", "image_shape")):
            raise ValueError("Floating and packed probes used different inputs")
        delta = {"seed": packed["seed"], "input_sha256": packed["input_sha256"]}
        for key in ("raw", "postprocessed"):
            values = [
                abs(a - b)
                for first, second in zip(reference[key], packed[key], strict=True)
                for a, b in zip(first, second, strict=True)
            ]
            delta[key] = {
                "rmse": math.sqrt(math.fsum(v * v for v in values) / len(values)),
                "maximum_absolute_difference": max(values),
                "coordinates": prediction * 6,
            }
            if not all(math.isfinite(v) for v in delta[key].values()):
                raise ValueError("Nonfinite action drift")
        differences.append(delta)
    return {
        "schema_version": 1,
        "scope": "generated observations; no calibration or task-quality acceptance",
        "units": "saved processor output coordinates; physical units unverified",
        "model_id": info["model_id"],
        "prediction_horizon": prediction,
        "execution_horizon": info.get("execution_horizon", 100),
        "versions": conversion["versions"],
        "floating": baseline,
        "packed": candidate,
        "drift_from_fp32": differences,
        "fresh_packed_reload_exact": True,
        "full_chunk_queue_reset_verified": True,
        "floating_master_reads_blocked": True,
        "network_disabled": True,
        "source_read_protection": "Python open audit hook; not an OS filesystem sandbox",
        "quality_verified": False,
        "calibration_verified": False,
        "speedup_verified": False,
        "task_success": None,
        "gpu_memory_bytes": None,
        "inference_speedup": None,
    }


def run_job(job):
    from firebird_act.bundle import publish_new_directory

    source, output, destination = validate_request(job)
    expected = job["source"]["files"]
    with _Owner(job["timeout_seconds"]) as owner:
        admit_source(source, expected, job["source"]["manifest_sha256"])
        owner.check()
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".native-quant-", dir=output) as temporary:
            scratch = Path(temporary)
            snapshot = scratch / "source"
            snapshot.mkdir()
            for name, item in expected.items():
                raw = read(source / name, item["bytes"])
                if sha(raw) != item["sha256"]:
                    raise ValueError("Source changed during snapshot")
                write_new(snapshot / name, raw)
            admit_source(snapshot, expected, job["source"]["manifest_sha256"])
            bundle = scratch / "bundle"
            bundle.mkdir()
            policy = bundle / "policy"
            conversion = _probe(
                owner,
                "convert",
                snapshot,
                scratch / "conversion.json",
                destination=policy,
                bits=job["native_quantization"]["bits"],
            )
            info = inspect_policy(policy)
            reload = _probe(owner, "reload", policy, scratch / "reload.json")
            if conversion.get("source_files") != expected:
                raise ValueError("Conversion used a different source inventory")
            verification = verify_reports(conversion, reload, info)
            write_new(bundle / "verification.json", canonical(verification))
            lineage = {
                "schema_version": 1,
                "source_artifact_id": job["source"]["artifact_id"],
                "source_artifact_manifest_sha256": job["source"]["artifact_manifest_sha256"],
                "source_files": expected,
                "source_manifest_sha256": job["source"]["manifest_sha256"],
                "source_registry_binding": "caller-owned; worker verifies supplied exact bytes",
            }
            write_new(bundle / "lineage.json", canonical(lineage))
            if "manifest.json" in expected:
                write_new(bundle / "source-manifest.json", read(snapshot / "manifest.json"))
            metadata = {
                "architecture": "act",
                "prediction_horizon": info["prediction_horizon"],
                "execution_horizon": info["execution_horizon"],
                "temporal_contract_sha256": info["files"]
                .get("temporal-contract.json", {})
                .get("sha256"),
                "format": "firebird_quant",
                "format_version": 1,
                "model_id": info["model_id"],
                "precision": f"int{job['native_quantization']['bits']}",
                "policy_subdirectory": "policy",
                "inference_only": True,
                "training_resume_supported": False,
                "fresh_reload_verified": True,
                "runtime_verified": False,
                "cpu_reload_verified": True,
                "isaac_runtime_verified": False,
                "quality_verified": False,
                "calibration_verified": False,
                "speedup_verified": False,
                "task_success": None,
                "source_artifact_id": job["source"]["artifact_id"],
                "source_artifact_manifest_sha256": job["source"]["artifact_manifest_sha256"],
            }
            files = {"policy/" + name: item["sha256"] for name, item in info["files"].items()}
            for path in bundle.iterdir():
                if path.is_file():
                    limit = EVIDENCE_LIMIT if path.name == "verification.json" else JSON_LIMIT
                    files[path.name] = sha(read(path, limit))
            write_new(
                bundle / "manifest.json",
                canonical({"schema_version": 1, "metadata": metadata, "files": files}),
            )
            if inventory(source) != expected or inspect_policy(policy) != info:
                raise ValueError("Source or tested package changed before publication")
            owner.check()
            checked_path(str(output))
            publish_new_directory(bundle, destination)
            # A signal during publication must not become a successful job receipt.
            owner.check()
    return {
        "schema_version": 1,
        "job_id": job["job_id"],
        "operation": "policy.quantize",
        "artifact": {
            "path": str(destination),
            "format": "native_quantized",
            "label": f"ACT native INT{job['native_quantization']['bits']} (CPU reload checked)",
        },
        "report": {
            **metadata,
            "drift_from_fp32": verification["drift_from_fp32"],
            "fixture_scope": verification["scope"],
            "units": verification["units"],
            "source_weight_bytes": expected["model.safetensors"]["bytes"],
            "packed_weight_bytes": info["files"]["model.fbq"]["bytes"],
            "policy_package_bytes": sum(v["bytes"] for v in info["files"].values()),
            "gpu_memory_bytes": None,
            "inference_speedup": None,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args(argv)
    try:
        request = checked_path(str(args.request))
        result = checked_path(str(args.result))
        if os.path.lexists(result):
            raise FileExistsError("Result must be new")
        job = read_json(request)
        source, output, _ = validate_request(job)
        if (
            result == request
            or result.is_relative_to(source)
            or result.is_relative_to(output / "native-quantized")
        ):
            raise ValueError("Result path must not modify the request/source/artifact")
        value = run_job(job)
        write_new(result, canonical(value))
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        print(json.dumps({"stage": "failed", "message": str(error)}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
