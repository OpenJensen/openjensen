"""Dependency-light contract and offline asset inventory for SmolVLA Spatial.

A protocol identity describes comparable inputs, never interchangeable runtimes.
Native reference weights are excluded from deployable C++ candidate inventories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path, PurePosixPath

from .worker import atomic_json, canonical, sha256

MODEL = "lerobot/smolvla_libero"
REVISION = "31d453f7edd78c839a8bbc39744a292686daf0de"
WEIGHTS_SHA256 = "9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8"
BACKBONE = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
BACKBONE_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
ASSETS_REVISION = "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"
POLICY_FILES = (
    "config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_5_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)
BACKBONE_FILES = (
    "config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "chat_template.json",
    "generation_config.json",
)
REFERENCE_FILE = "policy/model.safetensors"


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def relative_file(root, name):
    if not isinstance(name, str) or not name or "\\" in name or "\0" in name:
        raise ValueError("Invalid bundle asset path")
    parts = PurePosixPath(name)
    if parts.is_absolute() or any(x in {".", "..", ""} for x in name.split("/")):
        raise ValueError("Unsafe bundle asset path")
    path = Path(root)
    for part in parts.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Bundle symlinks are not permitted")
    if not path.is_file():
        raise ValueError("Missing bundle asset: " + name)
    return path


def verify_assets(root, *, require_reference=False):
    root = Path(root)
    record = json.loads(relative_file(root, "spatial-assets.json").read_text())
    if (
        record.get("schema_version") != 1
        or record.get("checkpoint")
        != {
            "repo_id": MODEL,
            "revision": REVISION,
            "weights_sha256": WEIGHTS_SHA256,
        }
        or record.get("backbone") != {"repo_id": BACKBONE, "revision": BACKBONE_REVISION}
    ):
        raise ValueError("Spatial assets do not identify the pinned reference")
    files = record.get("files", {})
    if (
        not isinstance(files, dict)
        or type(record.get("reference_weights")) is not bool
        or not isinstance(record.get("floating_gguf_sha256"), str)
        or not re.fullmatch("[0-9a-f]{64}", record["floating_gguf_sha256"])
        or any(
            not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value)
            for value in files.values()
        )
    ):
        raise ValueError("Invalid Spatial asset identity fields")
    required = {"policy/" + x for x in POLICY_FILES} | {"backbone/" + x for x in BACKBONE_FILES}
    fixtures = record.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures or len(fixtures) > 64:
        raise ValueError("A bounded, fixed fixture inventory is required")
    if (
        any(not isinstance(name, str) for name in fixtures)
        or len(fixtures) != len(set(fixtures))
        or any(not name.startswith("fixtures/") or not name.endswith(".npz") for name in fixtures)
    ):
        raise ValueError("Invalid fixture inventory")
    expected = required | set(fixtures)
    if record.get("reference_weights"):
        expected.add(REFERENCE_FILE)
    elif require_reference:
        raise ValueError("Native BF16 reference weights are absent")
    if set(files) != expected:
        raise ValueError("Spatial asset inventory is incomplete or contains unexpected files")
    if any(
        p.is_symlink()
        for prefix in ("policy", "backbone", "fixtures")
        for p in [root / prefix, *(root / prefix).rglob("*")]
    ):
        raise ValueError("Bundle symlinks are not permitted")
    actual = {
        p.relative_to(root).as_posix()
        for prefix in ("policy", "backbone", "fixtures")
        for p in (root / prefix).rglob("*")
        if p.is_file()
    }
    if actual != expected:
        raise ValueError("Unlisted or missing Spatial inference asset")
    if sum(relative_file(root, name).stat().st_size for name in fixtures) > 256 * 1024 * 1024:
        raise ValueError("Fixture set exceeds the 256 MiB limit")
    for name, expected_hash in files.items():
        if sha256(relative_file(root, name)) != expected_hash:
            raise ValueError("Spatial asset changed: " + name)
    if REFERENCE_FILE in files and files[REFERENCE_FILE] != WEIGHTS_SHA256:
        raise ValueError("Native reference weights differ from the pinned checkpoint")
    if len({files[name] for name in fixtures}) != len(fixtures):
        raise ValueError("Duplicate fixture content does not provide distinct evidence")
    config = json.loads(relative_file(root, "policy/config.json").read_text())
    if any(
        config.get(key) != value
        for key, value in {
            "type": "smolvla",
            "chunk_size": 50,
            "n_action_steps": 50,
            "max_action_dim": 32,
            "max_state_dim": 32,
            "num_steps": 10,
        }.items()
    ) or (
        config.get("output_features", {}).get("action", {}).get("shape") != [7]
        or config.get("input_features", {}).get("observation.state", {}).get("shape") != [6]
    ):
        raise ValueError("Unsupported Spatial policy dimensions or denoising contract")
    for filename in ("policy_preprocessor.json", "policy_postprocessor.json"):
        pipeline = json.loads(relative_file(root, "policy/" + filename).read_text())
        states = {step["state_file"] for step in pipeline["steps"] if "state_file" in step}
        expected_states = {x for x in POLICY_FILES if x.startswith(filename[:-5] + "_step_")}
        if states != expected_states:
            raise ValueError("Processor normalization state inventory differs")
    return record


def copy_assets(source, destination, *, reference=False):
    record = verify_assets(source, require_reference=reference)
    files = {k: v for k, v in record["files"].items() if reference or k != REFERENCE_FILE}
    for name in files:
        target = Path(destination) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(relative_file(source, name), target)
    result = {**record, "files": files, "reference_weights": reference}
    atomic_json(Path(destination) / "spatial-assets.json", result)
    verify_assets(destination, require_reference=reference)
    return result


def episode_ids(evaluation, final=False):
    if evaluation.get("suite") != "libero_spatial":
        raise ValueError("Spatial adapter requires an explicit libero_spatial suite")
    tasks = evaluation.get("task_ids")
    if tasks is None:
        tasks = list(range(10))
    states = evaluation["final_states" if final else "initial_states"]
    for values, ceiling in (
        (tasks, 9),
        (evaluation["initial_states"], 999),
        (evaluation["final_states"], 999),
    ):
        if (
            not values
            or len(values) != len(set(values))
            or any(type(x) is not int or not 0 <= x <= ceiling for x in values)
        ):
            raise ValueError("Task and state IDs must be bounded unique integers")
    if set(evaluation["initial_states"]) & set(evaluation["final_states"]):
        raise ValueError("Search and final states must be disjoint")
    seed = evaluation["seed"]
    if type(seed) is not int or not 0 <= seed <= 2147483647:
        raise ValueError("Invalid Spatial seed")
    return [
        {
            "task_id": task,
            "init_state_id": state,
            "seed": seed + state,
            "noise_seed": seed + 1000 * state,
        }
        for task in sorted(tasks)
        for state in sorted(states)
    ]


def protocol(evaluation, assets, simulator, final=False):
    value = {
        "schema_version": 1,
        "suite": "libero_spatial",
        "checkpoint": assets["checkpoint"],
        "backbone": assets["backbone"],
        "inference_assets": {k: v for k, v in assets["files"].items() if k != REFERENCE_FILE},
        "simulator": simulator,
        "episodes": episode_ids(evaluation, final),
        "observation": "LeRobot 0.4.4 LIBERO raw CPU image/image2/state/task processors",
        "input_cameras": ["observation.images.image", "observation.images.image2"],
        "input_resolution": [360, 360],
        "state_layout": {
            "checkpoint_declared": 6,
            "raw_and_normalization": 8,
            "padded": 32,
            "conversion_rule": "Preserve all 8 saved normalization coordinates in GGUF",
        },
        "model_resolution": [512, 512],
        "noise": "numpy.default_rng(noise_seed + chunk_index).standard_normal FP32 [1,50,32]",
        "output": "unnormalized finite CPU FP32 [1,50,7]",
        "action_steps": 50,
        "task_ids": sorted(evaluation.get("task_ids") or range(10)),
        "state_ids": sorted(evaluation["final_states" if final else "initial_states"]),
        "seed": evaluation["seed"],
        "parity_limits": evaluation.get("parity_limits"),
        "denoising_steps": 10,
        "steps": evaluation["steps"],
        "warmups": evaluation["warmups"],
        "repetitions": evaluation["repetitions"],
        "timing_scope": "raw CPU observation to unnormalized CPU chunk; simulator excluded",
    }
    return value, digest(value)


def prepare(policy, backbone, fixtures, gguf, output):
    """Copy only already-cached local files; never downloads or executes a model."""
    policy, backbone, fixtures, gguf, output = map(Path, (policy, backbone, fixtures, gguf, output))
    if policy.name != REVISION or backbone.name != BACKBONE_REVISION:
        raise ValueError("Use the exact pinned local snapshot directories")
    if sha256(policy / "model.safetensors") != WEIGHTS_SHA256:
        raise ValueError("Native checkpoint hash differs from the pinned reference")
    names = sorted(path.name for path in fixtures.glob("*.npz"))
    if not names or len(names) > 64:
        raise ValueError("Provide 1..64 pre-captured observation/noise fixtures")
    output.mkdir(parents=True, exist_ok=False)
    for directory, files, prefix in (
        (policy, (*POLICY_FILES, "model.safetensors"), "policy"),
        (backbone, BACKBONE_FILES, "backbone"),
        (fixtures, names, "fixtures"),
    ):
        (output / prefix).mkdir()
        for name in files:
            shutil.copyfile(directory / name, output / prefix / name)
    record = {
        "schema_version": 1,
        "checkpoint": {"repo_id": MODEL, "revision": REVISION, "weights_sha256": WEIGHTS_SHA256},
        "backbone": {"repo_id": BACKBONE, "revision": BACKBONE_REVISION},
        "floating_gguf_sha256": sha256(gguf),
        "reference_weights": True,
        "fixtures": ["fixtures/" + x for x in names],
        "files": {
            p.relative_to(output).as_posix(): sha256(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    }
    atomic_json(output / "spatial-assets.json", record)
    verify_assets(output, require_reference=True)
    return record


def main():
    parser = argparse.ArgumentParser(
        description="Prepare pinned local Spatial assets; no downloads"
    )
    for name in ("policy", "backbone", "fixtures", "gguf", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.policy, args.backbone, args.fixtures, args.gguf, args.output)
    print(sha256(args.output / "spatial-assets.json"))


if __name__ == "__main__":
    main()
