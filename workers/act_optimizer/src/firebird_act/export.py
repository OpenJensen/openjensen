"""Publish an ACT inference-only export only after fresh offline CPU parity passes."""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .bundle import (
    CORE_FILES,
    canonical,
    decode,
    inventory,
    publish_new_directory,
    read_json,
    safe_file,
    strip_vae,
    validate_config,
    validate_processors,
)
from .probe import FIXTURE_SEEDS, VERSIONS

RECIPE = "act-vae-removal-fp32-v1"


def run_probe(
    checkpoint: Path, report: Path, timeout: float, *, forbidden_sources: tuple[Path, ...] = ()
) -> dict[str, Any]:
    env = os.environ | {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "FIREBIRD_ACT_FORBIDDEN_SOURCES": json.dumps([str(p.resolve()) for p in forbidden_sources]),
    }
    # The worker can also be run from its source checkout; never import checkpoint code.
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    with tempfile.TemporaryFile() as log:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "firebird_act.probe", str(checkpoint), str(report)],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise ValueError(
                f"ACT CPU verification exceeded {timeout:g}s; no export published"
            ) from error
        if result.returncode:
            length = log.tell()
            log.seek(max(0, length - 6000))
            raise ValueError("ACT CPU verification failed: " + log.read().decode(errors="replace"))
    return decode(safe_file(report, 2 * 1024 * 1024))


def validate_probe(report: dict[str, Any]) -> None:
    if (
        type(report.get("schema_version")) is not int
        or report["schema_version"] != 1
        or report.get("device") != "cpu"
        or report.get("dtype") != "float32"
        or type(report.get("cpu_threads")) is not int
        or report["cpu_threads"] != 1
        or report.get("network_disabled") is not True
    ):
        raise ValueError("Invalid ACT parity runtime schema")
    versions = report.get("versions")
    if not isinstance(versions, dict) or set(versions) != set(VERSIONS):
        raise ValueError("Incomplete ACT parity runtime versions")
    if any(
        not isinstance(versions[k], str) or versions[k].split("+")[0] != v
        for k, v in VERSIONS.items()
    ):
        raise ValueError("Unexpected ACT parity runtime versions")
    fixtures = report.get("fixtures")
    if not isinstance(fixtures, list) or len(fixtures) != len(FIXTURE_SEEDS):
        raise ValueError("Incomplete ACT parity fixtures")
    for fixture, seed in zip(fixtures, FIXTURE_SEEDS, strict=True):
        if (
            not isinstance(fixture, dict)
            or type(fixture.get("seed")) is not int
            or fixture["seed"] != seed
            or fixture.get("queue_and_reset_exact") is not True
            or not isinstance(fixture.get("input_sha256"), str)
            or not re.fullmatch(r"[a-f0-9]{64}", fixture["input_sha256"])
        ):
            raise ValueError("Invalid ACT parity fixture identity or queue proof")
        for key in ("chunk", "postprocessed"):
            matrix = fixture.get(key)
            if (
                not isinstance(matrix, list)
                or len(matrix) != 100
                or any(
                    not isinstance(row, list)
                    or len(row) != 6
                    or any(type(x) not in (int, float) or not math.isfinite(x) for x in row)
                    for row in matrix
                )
            ):
                raise ValueError("ACT parity requires finite complete 100x6 actions")


def compare_probes(original: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    validate_probe(original)
    validate_probe(candidate)
    if original["versions"] != candidate["versions"] or original["python"] != candidate["python"]:
        raise ValueError("Parity runtimes differ")
    if original["fixtures"] != candidate["fixtures"]:
        raise ValueError("Fresh-process ACT chunk/postprocessor parity failed")
    return {
        "schema_version": 1,
        "status": "passed",
        "scope": "synthetic CPU inference parity",
        "device": "cpu",
        "dtype": "float32",
        "network_disabled": True,
        "versions": original["versions"],
        "python": original["python"],
        "fixture_count": len(original["fixtures"]),
        "action_steps": 100,
        "action_dim": 6,
        "exact_equal": True,
        "maximum_absolute_error": 0.0,
        "source_checkpoint_files": original["checkpoint_files"],
        "export_checkpoint_files": candidate["checkpoint_files"],
        "fixtures": [
            {
                "seed": f["seed"],
                "input_sha256": f["input_sha256"],
                "chunk_sha256": hashlib.sha256(canonical(f["chunk"])).hexdigest(),
                "postprocessed_sha256": hashlib.sha256(canonical(f["postprocessed"])).hexdigest(),
                "queue_and_reset_exact": f["queue_and_reset_exact"],
            }
            for f in original["fixtures"]
        ],
        "calibration_verified": False,
        "task_success": None,
        "gpu_memory_bytes": None,
        "inference_speedup": None,
    }


def export_policy(source: Path, output: Path, *, timeout: float = 120) -> dict[str, Any]:
    """Validate, snapshot, transform and prove parity before atomic no-replace publication."""
    source, output = Path(source), Path(output)
    if os.path.lexists(output):
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    if isinstance(timeout, bool) or not 1 <= timeout <= 600:
        raise ValueError("Each CPU verification deadline must be 1..600 seconds")
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError("Output must be outside the original checkpoint")
    source_files = inventory(source)
    config = read_json(source / "config.json")
    validate_config(config, source=True)
    names = CORE_FILES | validate_processors(source, config)
    if "train_config.json" in source_files:
        read_json(source / "train_config.json")
        names.add("train_config.json")
    if set(source_files) != names:
        raise ValueError("Unexpected or missing checkpoint files")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".act-export-", dir=output.parent) as temporary:
        root = Path(temporary)
        original, candidate = root / "original", root / "candidate"
        original.mkdir()
        candidate.mkdir()
        for name in sorted(names):
            data = safe_file(source / name)
            (original / name).write_bytes(data)
            if name not in {"model.safetensors", "config.json"}:
                (candidate / name).write_bytes(data)
        if inventory(original) != source_files or inventory(source) != source_files:
            raise ValueError("Source changed during snapshot")
        recipe = strip_vae(original / "model.safetensors", candidate / "model.safetensors", config)
        (candidate / "config.json").write_bytes(canonical(config | {"use_vae": False}))
        before = inventory(candidate)
        original_probe = run_probe(original, root / "original-result.json", timeout)
        candidate_probe = run_probe(candidate, root / "candidate-result.json", timeout)
        if (
            original_probe["checkpoint_files"] != source_files
            or candidate_probe["checkpoint_files"] != before
        ):
            raise ValueError("Verified checkpoint inventory differs from export inputs")
        parity = compare_probes(original_probe, candidate_probe)
        if inventory(candidate) != before or inventory(source) != source_files:
            raise ValueError("Source or export changed during verification")
        (candidate / "parity.json").write_bytes(canonical(parity))
        (candidate / "recipe.json").write_bytes(
            canonical(
                recipe
                | {
                    "schema_version": 1,
                    "recipe": RECIPE,
                    "family": "act",
                    "inference_only": True,
                    "config_changes": {"use_vae": {"from": True, "to": False}},
                    "loader_overrides": {
                        "device": "cpu",
                        "pretrained_backbone_weights": None,
                        "processor_device": "cpu",
                    },
                    "source_files": source_files,
                }
            )
        )
        files = inventory(candidate)
        manifest = {
            "schema_version": 1,
            "recipe": RECIPE,
            "family": "act",
            "inference_only": True,
            "files": files,
            "source_model_bytes": source_files["model.safetensors"]["bytes"],
            "export_model_bytes": before["model.safetensors"]["bytes"],
            "calibration_verified": False,
            "task_success": None,
        }
        (candidate / "manifest.json").write_bytes(canonical(manifest))
        # Independently load the assembled package, without the source snapshot available.
        shutil.rmtree(original)
        complete_files = inventory(candidate)
        final_probe = run_probe(
            candidate, root / "final-result.json", timeout, forbidden_sources=(source, original)
        )
        compare_probes(original_probe, final_probe)
        if (
            final_probe["checkpoint_files"] != complete_files
            or inventory(candidate) != complete_files
            or inventory(source) != source_files
        ):
            raise ValueError("Package or source changed during final reload")
        # Flush complete package contents before the one visible directory transition.
        for file in candidate.iterdir():
            with file.open("r+b") as stream:
                os.fsync(stream.fileno())
        publish_new_directory(candidate, output)
    return manifest | {
        "package_bytes": sum(p.stat().st_size for p in output.iterdir()),
        "final_package_reload": {
            "status": "passed",
            "exact_equal": True,
            "source_reads_blocked": True,
            "manifest_sha256": complete_files["manifest.json"]["sha256"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--timeout",
        type=float,
        default=120,
        help="Deadline per fresh CPU process, 1..600 seconds (default120)",
    )
    args = parser.parse_args()
    try:
        result = export_policy(args.source, args.output, timeout=args.timeout)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"ACT export failed: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
