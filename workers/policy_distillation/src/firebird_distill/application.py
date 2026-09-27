"""Fixed offline policy.distill operation with a bounded, owned POSIX process group."""

import argparse
import copy
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from firebird_act.bundle import canonical, inventory, publish_new_directory, safe_file

from .contracts import (
    MAX_SAMPLE_BYTES,
    corpus,
    digest,
    read,
    policy_info,
    request,
    teacher_info,
)

from .provenance import action_fps, inherited_files, policy_metadata


def signal_owned_group(pid, sig):
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        return
    except PermissionError as original:
        # Darwin may report transient EPERM as the final group member exits.
        # Only kernel-confirmed disappearance is accepted; persistent failure remains visible.
        deadline = time.monotonic() + 0.25
        while time.monotonic() < deadline:
            time.sleep(0.01)
            try:
                os.killpg(pid, 0)
            except ProcessLookupError:
                return
            except PermissionError:
                continue
            break
        raise original


class Owner:
    """Signal handlers only set a flag, including inside Popen's creation window."""

    def __init__(self, timeout):
        self.deadline = time.monotonic() + timeout
        self.interrupted = False
        self.handlers = {}

    def check(self):
        if self.interrupted:
            raise InterruptedError("Distillation cancelled")
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Distillation deadline exceeded")

    def __enter__(self):
        if os.name != "posix" or threading.current_thread() is not threading.main_thread():
            raise ValueError("Distillation currently requires a POSIX main process")
        for sig in (signal.SIGINT, signal.SIGTERM):
            self.handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._signal)
        return self

    def _signal(self, _sig, _frame):
        self.interrupted = True

    def __exit__(self, *_):
        for sig, handler in self.handlers.items():
            signal.signal(sig, handler)

    def run(self, command, env):
        self.check()
        process, drain = None, None
        tail = bytearray()

        def collect(stream):
            while block := stream.read(4096):
                tail.extend(block)
                del tail[:-4096]

        try:
            process = subprocess.Popen(
                command,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            drain = threading.Thread(target=collect, args=(process.stdout,), daemon=True)
            drain.start()
            while process.poll() is None:
                self.check()
                try:
                    process.wait(timeout=0.05)
                except subprocess.TimeoutExpired:
                    pass
            self.check()
            if process.returncode != 0:
                drain.join(timeout=0.2)
                raise ValueError(
                    "Offline distillation process failed: " + tail.decode("utf-8", errors="replace")
                )
        finally:
            if process is not None:
                # Reap descendants even after leader exit; signals cannot lose this handle.
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    signal_owned_group(process.pid, sig)
                    if sig == signal.SIGTERM:
                        try:
                            process.wait(timeout=0.25)
                        except subprocess.TimeoutExpired:
                            pass
                process.wait(timeout=5)
                if drain:
                    drain.join(timeout=2)
                process.stdout.close()


def environment(forbidden):
    import firebird_act

    result = {
        k: os.environ[k] for k in ("PATH", "TMPDIR", "SYSTEMROOT", "WINDIR") if k in os.environ
    }
    result.update(
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
        FIREBIRD_DISTILL_FORBIDDEN=json.dumps([str(p) for p in forbidden]),
    )
    return result


def copy_corpus(root, destination, doc):
    destination.mkdir()
    for sample in doc["samples"]:
        raw = safe_file(root / sample["file"], MAX_SAMPLE_BYTES)
        if len(raw) != sample["bytes"] or digest(raw) != sample["sha256"]:
            raise ValueError("Corpus sample identity changed")
        (destination / sample["file"]).write_bytes(raw)
    (destination / "manifest.json").write_bytes(safe_file(root / "manifest.json", 1024**2))


def check_originals(teacher, data, job, doc):
    if inventory(teacher) != job["teacher"]["files"]:
        raise ValueError("Original teacher changed during distillation")
    if digest(safe_file(data / "manifest.json", 1024**2)) != job["dataset"]["manifest_sha256"]:
        raise ValueError("Original corpus manifest changed")
    for sample in doc["samples"]:
        raw = safe_file(data / sample["file"], MAX_SAMPLE_BYTES)
        if len(raw) != sample["bytes"] or digest(raw) != sample["sha256"]:
            raise ValueError("Original corpus sample changed")


def tree_inventory(root):
    files = {}
    for file in sorted(root.rglob("*")):
        if file.is_dir() and not file.is_symlink():
            continue
        raw = safe_file(file)
        files[file.relative_to(root).as_posix()] = digest(raw)
    return files


def native_model_id(root):
    """Native sim-policy-checkpoint-v1 identity; tested against the shared inspector.

    Hash domain, then sorted UTF8 name length/name, file length and exact file bytes.
    Config, weights, processors/statistics and the optional control contract
    participate. Temporal provenance is inventoried but excluded, matching Isaac.
    """
    from firebird_act.bundle import CORE_FILES

    files = inventory(root)
    config, _, _ = policy_info(root, files)
    if config["use_vae"] is not False:
        raise ValueError("Student inference identity requires a VAE-free policy")
    names = (CORE_FILES | inherited_files(root, config)) - {"temporal-contract.json"}
    value = hashlib.sha256(b"sim-policy-checkpoint-v1\0")
    for name in sorted(names):
        encoded, raw = name.encode("utf-8"), safe_file(root / name)
        value.update(len(encoded).to_bytes(8, "big"))
        value.update(encoded)
        value.update(len(raw).to_bytes(8, "big"))
        value.update(raw)
    return "sha256:" + value.hexdigest()


def implementation_identity():
    import firebird_act.bundle
    import firebird_act.control_schema
    import firebird_act.probe
    import firebird_vla.control_contract
    import firebird_vla.control_schema

    names = ("__init__.py", "application.py", "contracts.py", "prepare.py", "runtime.py", "provenance.py")
    files = {
        "distillation/" + name: digest(safe_file(Path(__file__).parent / name, 1024**2))
        for name in names
    }
    for prefix, modules in (
        ("act/", (firebird_act.bundle, firebird_act.probe, firebird_act.control_schema)),
        ("data/", (firebird_vla.control_contract, firebird_vla.control_schema)),
    ):
        for module in modules:
            files[prefix + Path(module.__file__).name] = digest(
                safe_file(Path(module.__file__), 1024**2)
            )
    return files


def run_job(job):
    teacher, data, output = request(job)
    cfg, camera, processors_sha = teacher_info(teacher, job["teacher"]["files"])
    inherited = policy_metadata(teacher, cfg)
    fps_contract = action_fps(teacher, inherited)
    doc = corpus(data, job["dataset"]["manifest_sha256"], cfg, camera, processors_sha,
                 metadata=inherited, expected_fps=fps_contract)
    preserved = {name: job["teacher"]["files"][name] for name in inherited_files(teacher, cfg)}
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    implementation = implementation_identity()
    with (
        Owner(job["timeout_seconds"]) as owner,
        tempfile.TemporaryDirectory(prefix=".distillation-", dir=output) as scratch,
    ):
        scratch = Path(scratch)
        copied_teacher, copied_data = scratch / "teacher", scratch / "corpus"
        copied_teacher.mkdir()
        try:
            for name, expected in job["teacher"]["files"].items():
                raw = safe_file(teacher / name)
                if digest(raw) != expected["sha256"] or len(raw) != expected["bytes"]:
                    raise ValueError("Teacher changed during copy")
                (copied_teacher / name).write_bytes(raw)
            copy_corpus(data, copied_data, doc)
            # Revalidate copied identities before execution; original paths cannot be read by child.
            teacher_info(copied_teacher, job["teacher"]["files"])
            corpus(copied_data, job["dataset"]["manifest_sha256"], cfg, camera, processors_sha,
                   metadata=inherited, expected_fps=fps_contract)
            copied_job = copy.deepcopy(job)
            copied_job["teacher"]["path"], copied_job["dataset"]["path"] = (
                str(copied_teacher),
                str(copied_data),
            )
            job_path = scratch / "request.json"
            job_path.write_bytes(canonical(copied_job))
            stage = scratch / "artifact"
            stage.mkdir()
            train_result, verify_result = scratch / "train.json", scratch / "verify.json"
            owner.run(
                [
                    sys.executable,
                    "-m",
                    "firebird_distill.runtime",
                    "train",
                    str(job_path),
                    str(stage),
                    str(train_result),
                ],
                environment([teacher, data]),
            )
            report = read(train_result)
            expected_files = inventory(stage / "policy")
            student_cfg, _, _ = policy_info(stage / "policy", expected_files)
            if policy_metadata(stage / "policy", student_cfg) != inherited or any(
                expected_files.get(name) != expected for name, expected in preserved.items()
            ):
                raise ValueError("Student changed inherited processors or sidecar bytes")
            # The fresh reloader cannot use either original or copied teacher weights.
            shutil.rmtree(copied_teacher)
            owner.run(
                [
                    sys.executable,
                    "-m",
                    "firebird_distill.runtime",
                    "verify",
                    str(job_path),
                    str(stage),
                    str(verify_result),
                ],
                environment([teacher, data, copied_teacher]),
            )
            verified = read(verify_result)
            if (
                verified["policy_files"] != expected_files
                or report["policy_files"] != expected_files
                or verified["predictions"] != report["predictions"]
                or verified["versions"] != report["versions"]
                or any(k not in report or k not in verified for k in inherited)
                or canonical({k: report.get(k) for k in inherited}) != canonical(inherited)
                or canonical({k: verified.get(k) for k in inherited}) != canonical(inherited)
            ):
                raise ValueError("Fresh-process student reload differs from the frozen artifact")
            report["fresh_reload_verified"] = True
            report["elapsed_seconds"] = time.monotonic() - started
            report["scope"] = (
                "Offline action imitation only; generated corpus is software proof, "
                "recorded corpus is not closed-loop robot quality"
            )
            (stage / "verification.json").write_bytes(canonical(verified))
            lineage = {
                "schema_version": 1,
                "teacher": {k: v for k, v in job["teacher"].items() if k != "path"},
                "corpus_manifest_sha256": job["dataset"]["manifest_sha256"],
                "corpus": doc,
                "recipe": job["recipe"],
                "teacher_training_overlap": "unknown",
                "student_splits_disjoint": True,
                "implementation_sha256": implementation,
            }
            (stage / "lineage.json").write_bytes(canonical(lineage))
            model_id = native_model_id(stage / "policy")
            metadata = {
                **inherited,
                "architecture": "act",
                "recipe": "act-action-distillation-v1",
                "model_id": model_id,
                "policy_inventory_sha256": digest(canonical(expected_files)),
                "precision": "fp32",
                "policy_subdirectory": "policy",
                "inference_only": True,
                "training_resume_supported": False,
                "fresh_reload_verified": True,
                "cpu_reload_verified": True,
                "quality_verified": False,
                "calibration_verified": False,
                "speedup_verified": False,
                "isaac_runtime_verified": False,
                "task_success": None,
                "dataset_kind": doc["source"]["kind"],
                "teacher_artifact_id": job["teacher"]["artifact_id"],
                "teacher_artifact_manifest_sha256": job["teacher"]["artifact_manifest_sha256"],
            }
            manifest = {"schema_version": 1, "metadata": metadata, "files": tree_inventory(stage)}
            (stage / "manifest.json").write_bytes(canonical(manifest))
            check_originals(teacher, data, job, doc)
            owner.check()
            if implementation_identity() != implementation:
                raise ValueError("Worker source changed during execution")
            destination = output / "distilled-policy"
            publish_new_directory(stage, destination)
            return {
                "schema_version": 1,
                "job_id": job["job_id"],
                "operation": "policy.distill",
                "artifact": {
                    "path": str(destination),
                    "format": "native_checkpoint",
                    "label": "Distilled ACT256 (offline imitation only)",
                },
                "report": report,
            }
        finally:
            check_originals(teacher, data, job, doc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    if os.path.lexists(args.result):
        parser.error("Refusing to replace a result")
    job = read(args.request)
    roots = request(job)
    resolved = args.result.resolve()
    if any(resolved.is_relative_to(root) for root in roots[:2]):
        parser.error("Result must not be inside source inputs")
    if resolved.is_relative_to(roots[2] / "distilled-policy"):
        parser.error("Result must not mutate the published inference artifact")
    result = run_job(job)
    with args.result.open("xb") as stream:
        stream.write(canonical(result))


if __name__ == "__main__":
    main()
