"""Paired Spatial orchestration. Native quality is a reference, never a CPP score."""

import hashlib
import json
import math
import re

SHA256 = re.compile(r"^[a-f0-9]{64}$")
EXPORT_EVIDENCE = {
    "reload-verification.json",
    "runtime-lock.json",
    "tested-payload.json",
    "workflow-evidence.json",
    "lineage.json",
}


def identity_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def artifact_identity(lifecycle, artifact):
    directory = lifecycle.settings.data_dir / artifact.path
    manifest = json.loads((directory / "manifest.json").read_text())
    model = manifest["files"].get("model.gguf")
    native = artifact.metadata.get("native_model_sha256")
    if (
        not isinstance(model, str)
        or not SHA256.fullmatch(model)
        or not isinstance(native, str)
        or not SHA256.fullmatch(native)
        or artifact.metadata.get("model_sha256") != model
    ):
        raise ValueError("Spatial artifact lacks a consistent GGUF/native reference identity")
    return model, native


def validate_package_receipt(lifecycle, package, chosen, report):
    root = lifecycle.settings.data_dir
    chosen_manifest = json.loads((root / chosen.path / "manifest.json").read_text())
    package_manifest = json.loads((root / package.path / "manifest.json").read_text())
    receipt = json.loads((root / package.path / "tested-payload.json").read_text())
    expected_files = dict(chosen_manifest["files"])
    expected_manifest_sha = chosen.manifest_sha256
    native_weights = "policy/model.safetensors"
    if native_weights in expected_files:
        if chosen.metadata.get("precision") != "float":
            raise ValueError("Packed deployables cannot contain native reference weights")
        assets = json.loads((root / chosen.path / "spatial-assets.json").read_text())
        if assets["files"].get(native_weights) != chosen.metadata["native_model_sha256"]:
            raise ValueError("Floating source native weight identity is inconsistent")
        expected_files.pop(native_weights)
        assets["files"].pop(native_weights)
        assets["reference_weights"] = False
        encoded = json.dumps(assets, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        expected_files["spatial-assets.json"] = hashlib.sha256(encoded.encode()).hexdigest()
        expected_manifest = {
            "schema_version": 1,
            "metadata": chosen_manifest["metadata"],
            "files": expected_files,
        }
        encoded = (
            json.dumps(expected_manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        )
        expected_manifest_sha = hashlib.sha256(encoded.encode()).hexdigest()
    if (
        receipt.get("manifest_sha256") != expected_manifest_sha
        or receipt.get("files") != expected_files
        or report.get("artifact_manifest_sha256") != expected_manifest_sha
        or package.metadata.get("deployment_verified") is not True
        or native_weights in package_manifest["files"]
        or set(package_manifest["files"]) - set(expected_files) - EXPORT_EVIDENCE
        or EXPORT_EVIDENCE.intersection(expected_files)
        or "tested-payload.json" not in package_manifest["files"]
        or any(
            package_manifest["files"].get(name) != digest for name, digest in expected_files.items()
        )
    ):
        raise ValueError("Package receipt does not bind the exact tested candidate inventory")


def pinned_fixture_hashes(lifecycle, artifact, protocol):
    """Bind reports to the registered asset inventory, not matching worker claims alone."""
    directory = lifecycle.settings.data_dir / artifact.path
    manifest_bytes = (directory / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    asset_bytes = (directory / "spatial-assets.json").read_bytes()
    assets = json.loads(asset_bytes)
    if (
        hashlib.sha256(manifest_bytes).hexdigest() != artifact.manifest_sha256
        or manifest["files"].get("spatial-assets.json") != hashlib.sha256(asset_bytes).hexdigest()
    ):
        raise ValueError("Spatial inference asset inventory changed after registration")
    files, fixtures = assets.get("files"), assets.get("fixtures")
    if (
        not isinstance(files, dict)
        or not files
        or any(
            not isinstance(name, str)
            or not isinstance(digest, str)
            or not SHA256.fullmatch(digest)
            or manifest["files"].get(name) != digest
            for name, digest in files.items()
        )
        or not isinstance(fixtures, list)
        or not 1 <= len(fixtures) <= 64
        or any(not isinstance(name, str) or name not in files for name in fixtures)
        or len(set(fixtures)) != len(fixtures)
    ):
        raise ValueError("Spatial parity fixtures lack a pinned artifact inventory")
    expected = {files[name] for name in fixtures}
    if len(expected) != len(fixtures):
        raise ValueError("Spatial parity fixtures require distinct input hashes")
    inference = {
        name: digest for name, digest in files.items() if name != "policy/model.safetensors"
    }
    if protocol.get("inference_assets") != inference:
        raise ValueError("Spatial protocol inference assets differ from the registered policy")
    return expected


def episode_ids(evaluation, final):
    states = evaluation.final_states if final else evaluation.initial_states
    return {
        (task, state, evaluation.seed + state, evaluation.seed + 1000 * state)
        for task in evaluation.task_ids
        for state in states
    }


def validate_report(
    lifecycle,
    job,
    artifact,
    report,
    *,
    backend,
    final=False,
    anchor=None,
    runtime=None,
    target=None,
):
    from .service import complete_measurement

    evaluation = job.request.evaluation
    model, native = artifact_identity(lifecycle, artifact)
    if report.get("backend") != backend or report.get("model_sha256") != model:
        raise ValueError("Spatial measurement is not bound to the requested backend/model")
    native_key = "native_model_sha256" if backend == "native-bf16" else "source_native_model_sha256"
    if report.get(native_key) != native:
        raise ValueError("Spatial native weight lineage does not match the executed policy")
    expected_configuration = (
        "native-bf16"
        if backend == "native-bf16"
        else (
            "cpp-bf16"
            if artifact.metadata.get("precision") == "float"
            else "cpp-"
            + artifact.metadata["precision"]["language"]
            + ("-vision" if artifact.metadata["precision"].get("vision") else "")
        )
    )
    if report.get("backend_configuration") != expected_configuration:
        raise ValueError("Spatial backend precision configuration differs from the artifact")
    if not isinstance(report.get("runtime"), dict) or not report["runtime"]:
        raise ValueError("Spatial runtime identity is missing")
    if runtime is not None and report["runtime"] != runtime:
        raise ValueError("Spatial backend runtime changed between measurements")
    hardware = report.get("target_identity")
    if not isinstance(hardware, dict) or any(
        not isinstance(hardware.get(key), str) or not hardware[key]
        for key in ("gpu_uuid", "name", "driver_version")
    ):
        raise ValueError("Spatial GPU identity is missing")
    if target is not None and hardware != target:
        raise ValueError("Spatial measurements used different hardware")
    protocol = report.get("protocol")
    expected = {
        "suite": "libero_spatial",
        "task_ids": sorted(evaluation.task_ids),
        "state_ids": sorted(evaluation.final_states if final else evaluation.initial_states),
        "seed": evaluation.seed,
        "steps": evaluation.steps,
        "action_steps": 50,
        "warmups": evaluation.warmups,
        "repetitions": evaluation.repetitions,
        "parity_limits": evaluation.parity_limits.model_dump()
        if evaluation.parity_limits
        else None,
    }
    if not isinstance(protocol, dict) or any(protocol.get(k) != v for k, v in expected.items()):
        raise ValueError("Spatial protocol does not match the requested episode plan")
    if report.get("protocol_sha256") != identity_hash(protocol):
        raise ValueError("Spatial protocol digest is invalid")
    if anchor is not None and (
        protocol != anchor.get("protocol")
        or report["protocol_sha256"] != anchor.get("protocol_sha256")
        or hardware != anchor.get("target_identity")
    ):
        raise ValueError("Spatial comparisons require identical protocol and hardware")
    if set(fixture_actions(report)) != pinned_fixture_hashes(lifecycle, artifact, protocol):
        raise ValueError("Spatial parity evidence must include every pinned fixture exactly once")
    if evaluation.mode == "libero":
        expected_ids = episode_ids(evaluation, final)
        episodes = report.get("episodes")
        if (
            not complete_measurement(report, len(expected_ids))
            or type(report.get("requested_episodes")) is not int
            or report["requested_episodes"] != len(expected_ids)
            or not isinstance(episodes, list)
            or len(episodes) != len(expected_ids)
        ):
            raise ValueError(
                "Spatial task quality requires every requested episode and resource sample"
            )
        actual_ids = []
        for episode in episodes:
            keys = ("task_id", "init_state_id", "seed", "noise_seed")
            if not isinstance(episode, dict) or any(type(episode.get(k)) is not int for k in keys):
                raise ValueError("Spatial episode identity is malformed")
            actual_ids.append(tuple(episode[k] for k in keys))
            if (
                episode.get("status") != "episode_complete"
                or type(episode.get("task_success")) is not bool
                or episode.get("benchmark_horizon") != 280
                or type(episode.get("steps")) is not int
                or not 1 <= episode["steps"] <= evaluation.steps
                or (not episode["task_success"] and episode["steps"] != 280)
            ):
                raise ValueError("Spatial episode is truncated or lacks authoritative success")
        if set(actual_ids) != expected_ids or len(set(actual_ids)) != len(actual_ids):
            raise ValueError("Spatial episode IDs do not match the requested paired set")
        success = sum(episode["task_success"] for episode in episodes) / len(episodes)
        if not math.isclose(report["success_rate"], success, abs_tol=1e-12, rel_tol=0):
            raise ValueError("Spatial success rate disagrees with individual episode outcomes")
    return report


def fixture_actions(report):
    if report.get("fixture_actions_deterministic") is not True:
        raise ValueError("Spatial fixed-action repeatability was not established")
    fixtures = report.get("fixture_actions")
    if not isinstance(fixtures, list) or not fixtures:
        raise ValueError("Spatial fixed-action parity evidence is missing")
    parsed = {}
    for fixture in fixtures:
        if not isinstance(fixture, dict):
            raise ValueError("Malformed parity fixture")
        key, actions = fixture.get("fixture_sha256"), fixture.get("actions")
        if not isinstance(key, str) or not SHA256.fullmatch(key) or key in parsed:
            raise ValueError("Parity fixtures need unique input hashes")
        if not isinstance(actions, list) or len(actions) != 50:
            raise ValueError("Parity fixture requires the complete 50-action chunk")
        values = []
        for action in actions:
            if (
                not isinstance(action, list)
                or len(action) != 7
                or any(
                    type(value) not in (int, float) or not math.isfinite(value) for value in action
                )
            ):
                raise ValueError("Parity fixture requires finite seven-channel actions")
            values.extend(action)
        parsed[key] = values
    return parsed


def parity(reference, control, limits):
    native, cpp = fixture_actions(reference), fixture_actions(control)
    if native.keys() != cpp.keys():
        raise ValueError("Native and CPP parity fixtures have different input hashes")
    errors = [
        abs(a - b) for key in sorted(native) for a, b in zip(native[key], cpp[key], strict=True)
    ]
    maximum = max(errors)
    # Scaling avoids overflow while keeping the measurement finite and explicit.
    rmse = (
        maximum * math.sqrt(sum((error / maximum) ** 2 for error in errors) / len(errors))
        if maximum
        else 0.0
    )
    if not math.isfinite(rmse) or not math.isfinite(maximum):
        raise ValueError("Nonfinite native/CPP parity measurement")
    return {
        "scope": "fixed_observation_action_parity",
        "profile": limits.profile,
        "limits": limits.model_dump(),
        "rmse": rmse,
        "max_abs_error": maximum,
        "fixture_sha256s": sorted(native),
        "passed": rmse <= limits.max_rmse and maximum <= limits.max_abs_error,
    }


def feasible(report, references, limits):
    return (
        report["success_rate"] >= limits.min_success_rate
        and all(
            report["success_rate"] >= ref["success_rate"] - limits.max_success_drop
            for ref in references
        )
        and report["p95_ms"] <= limits.max_p95_ms
        and report["peak_device_mib"] <= limits.max_peak_device_mib
    )


async def run_spatial(lifecycle, job, result, baseline):
    request = job.request
    evaluation = request.evaluation
    promotion = evaluation.mode == "libero" and request.limits is not None

    async def stop(stage, message):
        result.decision = "no_feasible_candidate" if promotion else "diagnostics_only"
        await lifecycle.event(job, stage, message)
        await lifecycle.publish(job, result)
        return result

    async def evaluate(artifact, stage, backend="cpp", final=False):
        _, report = await lifecycle.native(
            job,
            "policy.evaluate",
            artifact,
            stage,
            result,
            final=final,
            evaluation_backend=backend,
        )
        return report

    native = await evaluate(baseline, "native-reference", "native-bf16")
    control = await evaluate(baseline, "baseline")
    try:
        validate_report(lifecycle, job, baseline, native, backend="native-bf16")
        validate_report(lifecycle, job, baseline, control, backend="cpp", anchor=native)
        comparison = {"stage": "float-parity", **parity(native, control, evaluation.parity_limits)}
        result.reports.append(comparison)
        if not comparison["passed"]:
            return await stop(
                "float-parity", "Floating CPP parity failed; compression was not started"
            )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return await stop("float-parity", str(exc))
    await lifecycle.publish(job, result)
    measured = [(baseline, control)]
    for index, precision in enumerate(request.candidates):
        try:
            artifact, _ = await lifecycle.native(
                job,
                "policy.quantize",
                baseline,
                f"quantize-{index}",
                result,
                precision=precision,
            )
            measurement = await evaluate(artifact, f"evaluate-{index}")
            validate_report(
                lifecycle,
                job,
                artifact,
                measurement,
                backend="cpp",
                anchor=control,
                runtime=control["runtime"],
            )
            fixture_actions(measurement)
            measured.append((artifact, measurement))
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            result.reports.append(
                {"stage": f"candidate-{index}", "status": "failed", "error": str(exc)[:2000]}
            )
            await lifecycle.event(job, f"candidate-{index}", f"Candidate failed: {exc}")
            await lifecycle.publish(job, result)
    if not promotion or len(measured) < 2:
        return await stop(
            "selection",
            "Diagnostic evidence only; a complete paired comparison and limits are required",
        )
    eligible = [
        (artifact, report)
        for artifact, report in measured
        if type(artifact.metadata.get("weight_bytes")) is int
        and artifact.metadata["weight_bytes"] > 0
        and feasible(report, [native, control], request.limits)
    ]
    if not eligible:
        return await stop(
            "selection", "No candidate meets limits against both native and CPP floating controls"
        )
    chosen, chosen_search = min(eligible, key=lambda item: item[0].metadata["weight_bytes"])
    await lifecycle.event(
        job, "selection", f"Frozen candidate: {chosen.label}; checking unused paired states"
    )
    final_native = await evaluate(baseline, "final-reference", "native-bf16", final=True)
    final_control = await evaluate(baseline, "final-control", final=True)
    final = await evaluate(chosen, "final-evaluation", final=True)
    try:
        validate_report(
            lifecycle,
            job,
            baseline,
            final_native,
            backend="native-bf16",
            final=True,
            runtime=native["runtime"],
            target=native["target_identity"],
        )
        validate_report(
            lifecycle,
            job,
            baseline,
            final_control,
            backend="cpp",
            final=True,
            anchor=final_native,
            runtime=control["runtime"],
        )
        validate_report(
            lifecycle,
            job,
            chosen,
            final,
            backend="cpp",
            final=True,
            anchor=final_control,
            runtime=chosen_search["runtime"],
        )
        repeated_parity = parity(final_native, final_control, evaluation.parity_limits)
        result.reports.append({"stage": "final-float-parity", **repeated_parity})
        if (
            not repeated_parity["passed"]
            or repeated_parity["fixture_sha256s"] != comparison["fixture_sha256s"]
        ):
            raise ValueError(
                "Final floating controls no longer satisfy the fixed-fixture parity gate"
            )
        if fixture_actions(final) != fixture_actions(chosen_search):
            raise ValueError("Frozen candidate actions changed on identical parity inputs")
        if not feasible(final, [final_native, final_control], request.limits):
            raise ValueError("Frozen candidate failed final acceptance; no replacement selected")
        package, reloaded = await lifecycle.native(
            job,
            "policy.run",
            chosen,
            "package-and-reload",
            result,
            final=True,
            evaluation_backend="cpp",
            register_artifact=False,
        )
        validate_report(
            lifecycle,
            job,
            chosen,
            reloaded,
            backend="cpp",
            final=True,
            anchor=final,
            runtime=final["runtime"],
        )
        if (
            not package
            or reloaded.get("fresh_reload_verified") is not True
            or fixture_actions(reloaded) != fixture_actions(final)
            or not feasible(reloaded, [final_native, final_control], request.limits)
        ):
            raise ValueError("Exact package rerun failed final acceptance")
        if artifact_identity(lifecycle, package) != artifact_identity(lifecycle, chosen):
            raise ValueError("Published package differs from the selected model payload")
        validate_package_receipt(lifecycle, package, chosen, reloaded)
        result.artifacts.append(package)
        result.selected_artifact_id = package.id
        result.decision = "validated"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return await stop("final-evaluation", str(exc))
    await lifecycle.publish(job, result)
    return result
