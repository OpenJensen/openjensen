"""Explicit generated ACT 8/3 chain stages; the external supervisor owns all runs.

Commands: generate ROOT; verify-distill ROOT; verify-final ROOT.
Generation creates one teacher/corpus. Verification never imports an ML runtime.
This is generated software evidence, not recording, Isaac, calibration or quality proof.
Every stage preserves partial failures and refuses to replace its receipt/requests.
"""

import argparse
import hashlib
import math
import os
import stat
import sys
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[3]
TESTS = REPOSITORY / "workers/policy_distillation/tests"
JSON_LIMIT = 8 * 1024**2
FILE_LIMIT = 192 * 1024**2
BITS = (8, 4)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def fixed_sources() -> None:
    """Use only this checkout's worker packages and reusable test builders."""
    sources = [TESTS] + [
        REPOSITORY / "workers" / suffix
        for suffix in (
            "policy_distillation/src",
            "act_optimizer/src",
            "smolvla_qlora/src",
            "firebird_quant/src",
            "isaac_sim",
        )
    ]
    sys.path[:0] = [str(source) for source in sources]


def no_network(event: str, _args: tuple[Any, ...]) -> None:
    if event in {"socket.connect", "socket.bind", "socket.getaddrinfo", "socket.gethostbyaddr"}:
        raise RuntimeError("Fixture stages cannot access the network")


class NoML:
    def find_spec(self, fullname: str, *_args: Any) -> None:
        if fullname.split(".")[0] in {"torch", "torchvision", "lerobot", "transformers"}:
            raise ImportError("ML imports are allowed only in the explicit generation stage")


def identity(path: Path) -> dict[str, Any]:
    """Stream bounded regular bytes; never load a complete weight file into memory."""
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Symlink input refused")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(before.st_mode) and 0 < before.st_size <= FILE_LIMIT,
            "Invalid fixture file or size",
        )
        digest = hashlib.sha256()
        count = 0
        while block := stream.read(1024**2):
            count += len(block)
            require(count <= FILE_LIMIT, "Fixture file grew beyond its bound")
            digest.update(block)
        after = os.fstat(stream.fileno())
    require(
        (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        and count == before.st_size,
        "Fixture input changed while hashing",
    )
    return {"sha256": digest.hexdigest(), "bytes": count}


def inventory(root: Path) -> dict[str, Any]:
    require(root.is_dir() and not root.is_symlink(), "Missing immutable fixture directory")
    files = {}
    for item in sorted(root.rglob("*")):
        require(not item.is_symlink(), "Fixture links refused")
        if item.is_dir():
            continue
        require(len(files) < 128, "Too many fixture files")
        files[item.relative_to(root).as_posix()] = identity(item)
    require(bool(files), "Empty fixture inventory")
    return files


def read(path: Path) -> dict[str, Any]:
    from firebird_act.bundle import decode, safe_file

    return decode(safe_file(path, JSON_LIMIT))


def write(path: Path, value: Any) -> None:
    from firebird_act.bundle import canonical

    raw = canonical(value)
    require(len(raw) <= 1024**2, "Stage receipt exceeds the one MiB bound")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def new_outputs(*paths: Path) -> None:
    require(not any(os.path.lexists(path) for path in paths), "Refusing to replace stage output")


def source_identity() -> dict[str, str]:
    from firebird_distill.application import implementation_identity

    result = implementation_identity()
    for relative in (
        "workers/policy_distillation/scripts/control_chain_fixture.py",
        "workers/policy_distillation/tests/native_fixture.py",
        "workers/policy_distillation/tests/semantics_fixture.py",
        "workers/act_optimizer/scripts/temporal_http_fixture.py",
    ):
        result[relative] = identity(REPOSITORY / relative)["sha256"]
    return result


def generate(root: Path) -> None:
    """Generate once and independently seal normalized targets before optimization."""
    new_outputs(root)
    import torch
    from firebird_act.probe import runtime_versions
    from firebird_distill.contracts import load_sample
    from firebird_distill.provenance import inherited_files, policy_metadata
    from firebird_distill.runtime import batch_for, chunk, load_policy, processed, tensor_sha
    from native_fixture import make_job, make_teacher

    root.mkdir()  # Parent exists; no reuse of an earlier or partially generated root.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    versions = runtime_versions()
    teacher = make_teacher(root / "teacher", 8, 3, sidecars=True)
    job = make_job(teacher, root)
    write(
        root / "teacher-manifest.json",
        {"files": job["teacher"]["files"], "scope": "Generated teacher; no registry claim"},
    )
    job["teacher"]["artifact_manifest_sha256"] = identity(root / "teacher-manifest.json")["sha256"]
    config = read(teacher / "config.json")
    metadata = policy_metadata(teacher, config)
    preserved = {name: job["teacher"]["files"][name] for name in inherited_files(teacher, config)}
    doc = read(root / "corpus/manifest.json")
    require(len(doc["samples"]) == 6, "Expected one six-observation corpus")
    require(
        {s["split"] for s in doc["samples"]} == {"train", "validation", "final"}
        and len({s["lineage_group"] for s in doc["samples"]}) == 3,
        "Expected three disjoint generated lineage groups",
    )
    model, pre, post = load_policy(teacher)
    targets, physical = {}, {}
    for sample in doc["samples"]:
        data = load_sample(root / "corpus", sample, doc["image_shape"], 8)
        with torch.no_grad():
            raw = chunk(model, batch_for(data, pre, doc["camera"]))
            actions = processed(raw, post)
        require(
            tuple(raw.shape) == (1, 8, 6) and not raw.requires_grad,
            "Expected detached full normalized teacher chunk",
        )
        require(not torch.equal(raw[0], actions), "Nonidentity target normalization not exercised")
        targets[sample["file"]] = tensor_sha(raw)
        physical[sample["file"]] = tensor_sha(actions)
    require(inventory(teacher) == job["teacher"]["files"], "Generation changed teacher bytes")
    write(root / "distill-request.json", job)
    write(
        root / "fixture.json",
        {
            "schema_version": 1,
            "scope": "Generated ACT512 and corpus; no actual snapshot reader",
            "versions": versions,
            "implementation": source_identity(),
            "metadata": metadata,
            "teacher_files": inventory(teacher),
            "corpus_files": inventory(root / "corpus"),
            "request": identity(root / "distill-request.json"),
            "preserved_files": preserved,
            "teacher_manifest": identity(root / "teacher-manifest.json"),
            "normalized_target_sha256": targets,
            "postprocessed_target_sha256": physical,
            "task_success": None,
            "quality_verified": False,
            "calibration_verified": False,
        },
    )


def originals(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    fixture, job = read(root / "fixture.json"), read(root / "distill-request.json")
    require(fixture["implementation"] == source_identity(), "Fixture worker sources changed")
    require(identity(root / "distill-request.json") == fixture["request"], "Request changed")
    require(
        identity(root / "teacher-manifest.json") == fixture["teacher_manifest"],
        "Generated teacher manifest changed",
    )
    require(
        inventory(root / "teacher") == fixture["teacher_files"] == job["teacher"]["files"],
        "Original teacher changed",
    )
    require(inventory(root / "corpus") == fixture["corpus_files"], "Original corpus changed")
    require(
        job["teacher"]["path"] == str(root / "teacher")
        and job["dataset"]["path"] == str(root / "corpus")
        and job["output_dir"] == str(root / "operation"),
        "Unexpected fixture paths",
    )
    return fixture, job


def artifact(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    files, manifest = inventory(root), read(root / "manifest.json")
    actual = {name: item["sha256"] for name, item in files.items() if name != "manifest.json"}
    require(
        manifest["schema_version"] == 1 and manifest["files"] == actual,
        "Published artifact differs from its complete manifest",
    )
    return files, manifest


def claims(value: dict[str, Any], expected: dict[str, Any], *, temporal: bool = True) -> None:
    for key, item in expected.items():
        if temporal or key != "temporal_contract_sha256":
            require(key in value and value[key] == item, f"Inherited claim differs: {key}")


def checkpoint(root: Path, expected: dict[str, Any], model_id: str) -> None:
    from sim_worker.rollout.checkpoint import inspect_checkpoint

    info = inspect_checkpoint(root)
    record = expected["control_contract"]
    require(
        info.model_id == model_id
        and info.policy_type == "act"
        and (info.chunk_size, info.action_steps, info.state_dim, info.action_dim) == (8, 3, 6, 6)
        and info.camera_key == record["camera"]["key"]
        and (info.width, info.height) == (32, 32)
        and info.control_contract == record
        and info.control_contract_sha256 == expected["control_contract_sha256"],
        "Saved checkpoint inspection differs from claimed identity/semantics",
    )


def verify_distill(root: Path) -> None:
    """Check real saved optimizer/reload evidence, then prepare both exact input requests."""
    from firebird_distill.application import implementation_identity, native_model_id

    outputs = [
        root / "distill-verification.json",
        *(root / f"quant-{b}-request.json" for b in BITS),
    ]
    new_outputs(*outputs)
    fixture, job = originals(root)
    result = read(root / "distill-result.json")
    bundle = root / "operation/distilled-policy"
    require(
        result["job_id"] == job["job_id"]
        and result["operation"] == "policy.distill"
        and result["artifact"]["path"] == str(bundle),
        "Wrong distillation result identity",
    )
    files, manifest = artifact(bundle)
    report, verified = result["report"], read(bundle / "verification.json")
    trained = read(bundle / "training.json")
    require(
        all(report.get(key) == value for key, value in trained.items()), "Training report changed"
    )
    policy_files = inventory(bundle / "policy")
    for record in (report, verified, manifest["metadata"]):
        claims(record, fixture["metadata"])
    for record in (report, manifest["metadata"]):
        require(
            record["task_success"] is None
            and record["quality_verified"] is False
            and record["calibration_verified"] is False,
            "Unjustified student quality claim",
        )
    require(
        report["policy_files"] == verified["policy_files"] == policy_files,
        "Saved student or fresh-reload inventory differs",
    )
    require(report["versions"] == verified["versions"] == fixture["versions"], "Runtime mismatch")
    require(
        report["teacher_target_sha256"] == fixture["normalized_target_sha256"],
        "Training did not use the independently sealed normalized teacher targets",
    )
    require(
        report["steps"] == 8
        and len(report["training_losses"]) == 8
        and len(report["gradient_norms"]) == 8,
        "Wrong optimizer update count",
    )
    require(
        all(
            type(v) in (int, float) and math.isfinite(v) and v >= 0
            for v in report["training_losses"]
        ),
        "Invalid native losses",
    )
    require(
        all(
            type(v) in (int, float) and math.isfinite(v) and v > 0 for v in report["gradient_norms"]
        ),
        "Missing finite nonzero optimizer gradients",
    )
    for key in (
        "fresh_reload_verified",
        "action_head_changed",
        "teacher_gradients_absent",
        "copied_backbone_unchanged",
    ):
        require(report[key] is True, f"Missing optimizer/reload evidence: {key}")
    require(
        report["student_weights_bytes"]
        == policy_files["model.safetensors"]["bytes"]
        < report["teacher_inference_tensor_bytes"],
        "Student compression not established",
    )
    require(
        set(report["untuned_student"]) == {"train", "validation"}
        and set(report["trained_student"]) == {"train", "validation", "final"},
        "Held-out assessment boundaries changed",
    )
    require(
        report["trained_student"]["train"]["teacher_normalized_l1"]
        < report["untuned_student"]["train"]["teacher_normalized_l1"],
        "Generated training imitation objective did not improve",
    )
    require(
        report["predictions"] == verified["predictions"] and len(verified["predictions"]) == 6,
        "Fresh saved student predictions differ",
    )
    for prediction in verified["predictions"]:
        require(
            prediction["prediction_horizon"] == 8
            and prediction["execution_horizon"] == 3
            and prediction["queue_and_reset_exact"] is True
            and prediction["queue_refill_exact"] is True,
            "Student queue/timing evidence missing",
        )
    require(
        all(policy_files[name] == item for name, item in fixture["preserved_files"].items()),
        "Saved student changed processors or sidecars",
    )
    config = read(bundle / "policy/config.json")
    require(
        (config["dim_model"], config["n_encoder_layers"], config["use_vae"]) == (256, 2, False),
        "Saved student architecture differs",
    )
    model_id = native_model_id(bundle / "policy")
    require(manifest["metadata"]["model_id"] == model_id, "FP32 model identity differs")
    checkpoint(bundle / "policy", fixture["metadata"], model_id)
    lineage = read(bundle / "lineage.json")
    require(
        lineage["teacher"] == {k: v for k, v in job["teacher"].items() if k != "path"}
        and lineage["recipe"] == job["recipe"]
        and lineage["corpus_manifest_sha256"] == job["dataset"]["manifest_sha256"]
        and lineage["corpus"] == read(root / "corpus/manifest.json")
        and lineage["implementation_sha256"] == implementation_identity()
        and lineage["student_splits_disjoint"] is True,
        "Distillation lineage differs",
    )
    for bits in BITS:
        write(
            root / f"quant-{bits}-request.json",
            {
                "schema_version": 1,
                "job_id": f"generated-chain-int{bits}",
                "operation": "policy.quantize",
                "source": {
                    "path": str(bundle / "policy"),
                    "files": policy_files,
                    "manifest_sha256": None,
                    "artifact_id": "generated-proof:distilled-policy",
                    "artifact_manifest_sha256": files["manifest.json"]["sha256"],
                },
                "output_dir": str(root / f"quant-{bits}"),
                "native_quantization": {"format": "firebird_quant", "bits": bits, "group_size": 64},
                "timeout_seconds": 120,
            },
        )
    originals(root)
    write(
        outputs[0],
        {
            "schema_version": 1,
            "status": "passed",
            "model_id": model_id,
            "artifact_files": files,
            "policy_files": policy_files,
            "result": identity(root / "distill-result.json"),
            "requests": {str(b): identity(root / f"quant-{b}-request.json") for b in BITS},
            "normalized_targets_exact": True,
            "scope": "Generated native software proof",
        },
    )


def verify_final(root: Path) -> None:
    """Bind both fresh packed HTTP receipts to the same immutable saved student."""
    from firebird_act.bundle import canonical
    from firebird_quant.native_package import inspect_policy

    new_outputs(root / "chain-verification.json")
    fixture, _ = originals(root)
    student = read(root / "distill-verification.json")
    require(
        student["status"] == "passed" and student["normalized_targets_exact"] is True,
        "Missing completed distillation verification stage",
    )
    bundle = root / "operation/distilled-policy"
    require(
        inventory(bundle) == student["artifact_files"]
        and identity(root / "distill-result.json") == student["result"],
        "Saved student changed",
    )
    results = {}
    for bits in BITS:
        require(
            identity(root / f"quant-{bits}-request.json") == student["requests"][str(bits)],
            "Prepared quantization request changed",
        )
        request = read(root / f"quant-{bits}-request.json")
        result = read(root / f"quant-{bits}-result.json")
        packed = root / f"quant-{bits}/native-quantized"
        require(
            result["job_id"] == request["job_id"]
            and result["operation"] == "policy.quantize"
            and result["artifact"]["path"] == str(packed),
            "Wrong packed output identity",
        )
        files, manifest = artifact(packed)
        info = inspect_policy(packed / "policy")
        verified, http = read(packed / "verification.json"), read(root / f"http-{bits}.json")
        for record in (result["report"], manifest["metadata"]):
            claims(record, fixture["metadata"])
            require(
                record["model_id"] == info["model_id"] and record["precision"] == f"int{bits}",
                "Packed model/precision differs",
            )
        for record in (result["report"], manifest["metadata"], verified, http):
            require(
                record["task_success"] is None
                and record["quality_verified"] is False
                and record["calibration_verified"] is False,
                "Unjustified packed quality claim",
            )
        claims(verified, fixture["metadata"], temporal=False)
        require(info["encoding"]["recipe"]["bits"] == bits, "Wrong packed encoding")
        unchanged = set(student["policy_files"]) - {"model.safetensors"}
        require(
            set(info["files"]) == unchanged | {"encoding.json", "model.fbq"}
            and all(info["files"][name] == student["policy_files"][name] for name in unchanged),
            "Packing changed config, processors or inherited contracts",
        )
        lineage = read(packed / "lineage.json")
        require(
            all(
                lineage[key] == request["source"][source_key]
                for key, source_key in (
                    ("source_files", "files"),
                    ("source_manifest_sha256", "manifest_sha256"),
                    ("source_artifact_id", "artifact_id"),
                    ("source_artifact_manifest_sha256", "artifact_manifest_sha256"),
                )
            ),
            "Packed lineage does not bind the saved student",
        )
        checkpoint(packed / "policy", fixture["metadata"], info["model_id"])
        require(
            http["model_id"] == verified["model_id"] == info["model_id"]
            and http["policy_files"] == info["files"]
            and (http["prediction_horizon"], http["execution_horizon"]) == (8, 3),
            "HTTP receipt differs from its packed package",
        )
        for key in (
            "fresh_packed_reload_exact",
            "full_chunk_queue_reset_verified",
            "floating_master_reads_blocked",
            "network_disabled",
        ):
            require(verified[key] is True, f"Missing packed verification: {key}")
        for key in (
            "queue_reset_forces_new_call",
            "saved_processors_exact",
            "source_unchanged",
            "server_closed",
            "external_network_disabled",
            "floating_master_reads_blocked",
        ):
            require(http[key] is True, f"Missing HTTP evidence: {key}")
        require(http["queue_refill_calls"] == 3, "Incomplete native queue refill proof")
        require(
            http["versions"] == verified["versions"] == fixture["versions"],
            "Fresh packed/HTTP runtime differs",
        )
        require([row["seed"] for row in verified["packed"]] == [171, 902], "Wrong probe seeds")
        require([row["seed"] for row in verified["floating"]] == [171, 902], "Wrong FP32 seeds")
        drift = []
        for floating, packed_row in zip(verified["floating"], verified["packed"], strict=True):
            require(
                all(
                    floating[key] == packed_row[key]
                    for key in ("seed", "input_sha256", "image_shape")
                ),
                "Drift inputs differ",
            )
            delta = {"seed": packed_row["seed"], "input_sha256": packed_row["input_sha256"]}
            for key in ("raw", "postprocessed"):
                for row in (floating, packed_row):
                    values = row[key]
                    require(
                        len(values) == 8
                        and all(len(action) == 6 for action in values)
                        and all(
                            type(v) in (int, float) and math.isfinite(v)
                            for action in values
                            for v in action
                        ),
                        "Invalid full drift chunk",
                    )
                errors = [
                    abs(a - b)
                    for first, second in zip(floating[key], packed_row[key], strict=True)
                    for a, b in zip(first, second, strict=True)
                ]
                delta[key] = {
                    "coordinates": 48,
                    "maximum_absolute_difference": max(errors),
                    "rmse": math.sqrt(math.fsum(v * v for v in errors) / len(errors)),
                }
            drift.append(delta)
        require(verified["drift_from_fp32"] == drift, "Recorded drift does not match full chunks")
        expected_rows = []
        for count in (8, 3):
            for prediction in verified["packed"]:
                actions = prediction["postprocessed"]
                require(
                    len(actions) == 8
                    and all(len(row) == 6 for row in actions)
                    and all(
                        type(v) in (int, float) and math.isfinite(v) for row in actions for v in row
                    ),
                    "Invalid full packed action chunk",
                )
                expected_rows.append(
                    {
                        "seed": prediction["seed"],
                        "returned_steps": count,
                        "actions_sha256": hashlib.sha256(canonical(actions[:count])).hexdigest(),
                        "reset_exact": True,
                    }
                )
        require(http["requests"] == expected_rows, "HTTP full/prefix/reset predictions differ")
        require(
            result["report"]["drift_from_fp32"] == verified["drift_from_fp32"],
            "Recorded FP32 drift differs",
        )
        results[str(bits)] = {
            "model_id": info["model_id"],
            "artifact_files": files,
            "http": identity(root / f"http-{bits}.json"),
            "result": identity(root / f"quant-{bits}-result.json"),
            "drift_from_fp32": verified["drift_from_fp32"],
        }
    originals(root)
    require(inventory(bundle) == student["artifact_files"], "Student changed during final checks")
    for bits in BITS:
        require(
            inventory(root / f"quant-{bits}/native-quantized")
            == results[str(bits)]["artifact_files"],
            "Packed output changed during final checks",
        )
    write(
        root / "chain-verification.json",
        {
            "schema_version": 1,
            "status": "passed",
            "fixture": identity(root / "fixture.json"),
            "distillation": identity(root / "distill-verification.json"),
            "packed": results,
            "prediction_horizon": 8,
            "execution_horizon": 3,
            "originals_unchanged": True,
            "scope": "One generated teacher/corpus, saved student and two packed HTTP consumers",
            "task_success": None,
            "quality_verified": False,
            "calibration_verified": False,
            "isaac_runtime_verified": False,
            "recorded_dataset_verified": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generate", "verify-distill", "verify-final"))
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    require(
        args.root.is_absolute() and args.root == args.root.resolve(),
        "Use a canonical absolute root",
    )
    require(not any(p.is_symlink() for p in (args.root, *args.root.parents)), "Root links refused")
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    sys.dont_write_bytecode = True
    sys.addaudithook(no_network)
    if args.stage != "generate":
        sys.meta_path.insert(0, NoML())
    fixed_sources()
    {"generate": generate, "verify-distill": verify_distill, "verify-final": verify_final}[
        args.stage
    ](args.root)
    print(f"{args.stage}: passed; root={args.root}")


if __name__ == "__main__":
    main()
