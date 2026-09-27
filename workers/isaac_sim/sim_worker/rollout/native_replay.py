"""Owned local policy.run recorded-observation replay; no simulator or robot commands."""

import argparse
import copy
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from firebird_quant.native_application import _Owner
from firebird_quant.native_package import (
    canonical,
    checked_path,
    decode,
    read,
    read_json,
    sha,
    write_new,
)

from .native_replay_contracts import (
    MAX_RGB_BYTES,
    observations,
    policy,
    publish,
    request,
    validate_result,
)


class Owner(_Owner):
    def check(self):
        if self.interrupted:
            raise InterruptedError("Native replay interrupted; no successful result")
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Native replay exceeded its supervised deadline")


def environment(forbidden):
    import firebird_act
    import firebird_quant

    value = {
        k: os.environ[k] for k in ("PATH", "TMPDIR", "SYSTEMROOT", "WINDIR") if k in os.environ
    }
    value.update(
        PYTHONPATH=os.pathsep.join(
            [
                str(Path(__file__).resolve().parents[2]),
                str(Path(firebird_quant.__file__).resolve().parents[1]),
                str(Path(firebird_act.__file__).resolve().parents[1]),
            ]
        ),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
        FIREBIRD_REPLAY_FORBIDDEN=json.dumps([str(p) for p in forbidden]),
    )
    return value


def implementation_identity():
    from firebird_quant import native_application, native_consumer, native_package

    from . import backend, client, server

    files = {
        "replay/" + p.name: sha(read(p)) for p in Path(__file__).parent.glob("native_replay*.py")
    }
    for module in (native_application, native_consumer, native_package, backend, client, server):
        path = Path(module.__file__).resolve()
        files[module.__name__] = sha(read(path))
    return files


def run_job(job):
    source, inputs, output = request(job)
    implementation = implementation_identity()
    info = policy(source, job["source"])
    doc, input_files = observations(inputs, job["observations"]["manifest_sha256"], info["config"])
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with (
        Owner(job["timeout_seconds"]) as owner,
        tempfile.TemporaryDirectory(prefix=".native-replay-", dir=output) as temporary,
    ):
        temporary = Path(temporary).resolve()
        copies = [temporary / "policy", temporary / "observations"]
        try:
            for original, destination, files in (
                (source, copies[0], info["files"]),
                (inputs, copies[1], input_files),
            ):
                destination.mkdir()
                for name, identity in files.items():
                    limit = (
                        512 * 1024**2
                        if name == "model.fbq"
                        else MAX_RGB_BYTES
                        if name.endswith(".rgb")
                        else 1024**2
                    )
                    raw = read(original / name, limit)
                    if sha(raw) != identity["sha256"] or len(raw) != identity["bytes"]:
                        raise ValueError("Input changed during snapshot")
                    write_new(destination / name, raw)
            copied = copy.deepcopy(job)
            copied["source"]["path"], copied["observations"]["path"] = map(str, copies)
            policy(copies[0], copied["source"])
            observations(copies[1], copied["observations"]["manifest_sha256"], info["config"])
            job_file, child_result = temporary / "request.json", temporary / "result.json"
            write_new(job_file, canonical(copied))
            with tempfile.TemporaryFile() as log:
                try:
                    owner.run(
                        [
                            sys.executable,
                            "-m",
                            "sim_worker.rollout.native_replay_runtime",
                            str(job_file),
                            str(child_result),
                        ],
                        environment([source, inputs]),
                        log,
                    )
                except ValueError as error:
                    log.seek(0, 2)
                    log.seek(max(0, log.tell() - 4096))
                    raise ValueError(
                        "Offline native replay failed: "
                        + log.read(4096).decode("utf-8", errors="replace")
                    ) from error
            result = decode(read(child_result, 8 * 1024**2))
            validate_result(result, job["source"], doc, info["config"]["chunk_size"])
            stage = temporary / "artifact"
            stage.mkdir()
            write_new(stage / "predictions.json", canonical(result))
            lineage = {
                "schema_version": 1,
                "source": {k: v for k, v in job["source"].items() if k != "path"},
                "observation_manifest_sha256": job["observations"]["manifest_sha256"],
                "observation_files": input_files,
                "observations": doc,
                "implementation_sha256": implementation,
            }
            write_new(stage / "lineage.json", canonical(lineage))
            report = {
                "stage": "native_replay",
                "mode": "independent_observation_replay",
                "model_id": info["model_id"],
                "device": "cpu",
                "versions": result["versions"],
                "observation_source": doc["source"],
                "observations": len(doc["samples"]),
                "action_shape": [info["config"]["chunk_size"], 6],
                "reset_repeat_exact": all(r["reset_repeat_exact"] for r in result["records"]),
                "coordinate_semantics": doc["semantics"],
                "elapsed_seconds": time.monotonic() - started,
                "server_closed": True,
                "quality_verified": False,
                "calibration_verified": False,
                "speedup_verified": False,
                "isaac_runtime_verified": False,
                "task_success": None,
                "scope": (
                    "Predictions on immutable observed inputs; actions were not applied "
                    "to any robot or simulator. No task-quality acceptance."
                ),
            }
            write_new(stage / "report.json", canonical(report))
            files = {p.name: sha(read(p, 8 * 1024**2)) for p in stage.iterdir()}
            write_new(
                stage / "manifest.json",
                canonical(
                    {
                        "schema_version": 1,
                        "metadata": {
                            "architecture": "act",
                            "model_id": info["model_id"],
                            "recipe": "native-observation-replay-v1",
                            "device": "cpu",
                            "quality_verified": False,
                            "calibration_verified": False,
                            "speedup_verified": False,
                            "isaac_runtime_verified": False,
                            "task_success": None,
                        },
                        "files": files,
                    }
                ),
            )
            if (
                policy(source, job["source"]) != info
                or observations(inputs, job["observations"]["manifest_sha256"], info["config"])[1]
                != input_files
            ):
                raise ValueError("Original inputs changed during replay")
            if implementation_identity() != implementation:
                raise ValueError("Replay worker source changed during execution")
            owner.check()
            destination = output / "native-run"
            publish(stage, destination)
            owner.check()
            return {
                "schema_version": 1,
                "job_id": job["job_id"],
                "operation": "policy.run",
                "artifact": {
                    "path": str(destination),
                    "format": "native_run_record",
                    "label": "CPU observation replay",
                },
                "report": report,
            }
        finally:
            policy(source, job["source"])
            observations(inputs, job["observations"]["manifest_sha256"], info["config"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    checked_path(str(args.result))
    if os.path.lexists(args.result):
        parser.error("Refusing to overwrite result")
    job = read_json(args.request)
    roots = request(job)
    resolved = args.result.resolve()
    if any(resolved.is_relative_to(p.resolve()) for p in roots[:2]):
        parser.error("Result must not be inside source inputs")
    if resolved.is_relative_to((roots[2] / "native-run").resolve()):
        parser.error("Result must not be inside the published artifact")
    result = run_job(job)
    write_new(args.result, canonical(result))


if __name__ == "__main__":
    main()
