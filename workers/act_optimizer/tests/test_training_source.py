"""Source-admission contracts; synthetic serialization is not native0.6.2 reload evidence."""

import hashlib
import json
import math
import os
import shutil
import struct
from pathlib import Path

import pytest

from firebird_act import training_source
from firebird_act.bundle import NORM_MAP, canonical, removal_keys
from firebird_act.training_source import TRAINING_REVISION, admit_training_source


def tensor_file(tensors: dict[str, tuple[list[int], float]]) -> bytes:
    header, payload = {}, b""
    for name, (shape, number) in sorted(tensors.items()):
        block = struct.pack("<f", number) * math.prod(shape)
        header[name] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [len(payload), len(payload) + len(block)],
        }
        payload += block
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + payload


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
    path.write_bytes(canonical(value))


def resign(root: Path) -> str:
    """Rebuild fixture manifests to test semantics independently of integrity errors."""
    for directory in (root / "checkpoint", root):
        path = directory / "manifest.json"
        document = json.loads(path.read_text())
        document["files"] = {
            item.relative_to(directory).as_posix(): digest(item)
            for item in sorted(directory.rglob("*"))
            if item.is_file() and item != path
        }
        write(path, document)
    return digest(root / "manifest.json")


def edit(path: Path, key: str, value: object) -> None:
    document = json.loads(path.read_text())
    document[key] = value
    write(path, document)


@pytest.fixture
def training_bundle(tmp_path: Path) -> Path:
    root = tmp_path / "training-bundle"
    checkpoint = root / "checkpoint"
    policy = checkpoint / "pretrained_model"
    state = checkpoint / "training_state"
    policy.mkdir(parents=True)
    state.mkdir()
    features = {
        "observation.state": {"type": "STATE", "shape": [6]},
        "observation.images.front": {"type": "VISUAL", "shape": [3, 32, 32]},
    }
    actions = {"action": {"type": "ACTION", "shape": [6]}}
    config = {
        "type": "act",
        "n_obs_steps": 1,
        "n_action_steps": 100,
        "chunk_size": 100,
        "use_vae": True,
        "use_peft": False,
        "use_amp": False,
        "temporal_ensemble_coeff": None,
        "vision_backbone": "resnet18",
        "pre_norm": False,
        "replace_final_stride_with_dilation": False,
        "feedforward_activation": "relu",
        "normalization_mapping": NORM_MAP,
        "input_features": features,
        "output_features": actions,
        "dim_model": 32,
        "n_heads": 4,
        "dim_feedforward": 64,
        "n_encoder_layers": 1,
        "n_decoder_layers": 1,
        "n_vae_encoder_layers": 1,
        "latent_dim": 8,
    }
    write(policy / "config.json", config)
    # Deliberately no Torch/model instantiation: these bytes test bounded parsing, not inference.
    model = {name: ([1], 0.0) for name in removal_keys(config)}
    model["model.fixture.weight"] = ([1], 0.0)
    (policy / "model.safetensors").write_bytes(tensor_file(model))
    write(policy / "train_config.json", {"provenance_only_fixture": True})
    for kind in ("pre", "post"):
        chosen = features | actions if kind == "pre" else actions
        registry = "normalizer_processor" if kind == "pre" else "unnormalizer_processor"
        state_file = kind + "-stats.safetensors"
        normalization = {
            "registry_name": registry,
            "config": {"eps": 1e-8, "features": chosen, "norm_map": NORM_MAP},
            "state_file": state_file,
        }
        device = {
            "registry_name": "device_processor",
            "config": {"device": "cuda", "float_dtype": None},
        }
        steps = (
            [
                {
                    "registry_name": "rename_observations_processor",
                    "config": {"rename_map": {}},
                },
                {"registry_name": "to_batch_processor", "config": {}},
                device,
                normalization,
            ]
            if kind == "pre"
            else [normalization, device]
        )
        write(
            policy / f"policy_{kind}processor.json",
            {"name": f"policy_{kind}processor", "steps": steps},
        )
        tensors = {}
        for name, feature in chosen.items():
            shape = [3, 1, 1] if feature["type"] == "VISUAL" else feature["shape"]
            tensors[name + ".mean"] = shape, 0.0
            tensors[name + ".std"] = shape, 1.0
        (policy / state_file).write_bytes(tensor_file(tensors))
    recipe = {
        "policy_type": "act",
        "training_backend": "lerobot",
        "method": "full",
        "upstream_revision": TRAINING_REVISION,
        "model_revision": TRAINING_REVISION,
        "model_id": "code://lerobot/act",
        "steps": 2,
        "chunk_size": 100,
        "camera_keys": ["observation.images.front"],
        "dataset_id": "fixture/robot",
        "dataset_revision": "a" * 40,
    }
    write(checkpoint / "recipe.json", recipe)
    for name in ("probe-batch.pt", "probe-action.pt"):
        (checkpoint / name).write_bytes(b"opaque fixture, never deserialized")
    for name in ("optimizer_state.safetensors", "rng_state.safetensors"):
        (state / name).write_bytes(b"opaque resume fixture, never deserialized")
    write(state / "optimizer_param_groups.json", {})
    write(
        state / "training_step.json",
        {
            "step": 2,
            "dp_world_size": 1,
            "batch_size": 4,
            "grad_accum_steps": 1,
            "mixed_precision": "no",
            "parallelism": {
                "dp_replicate": 1,
                "dp_shard": 1,
                "ring_degree": 1,
                "ulysses_degree": 1,
            },
        },
    )
    write(
        checkpoint / "manifest.json",
        {
            "schema_version": 1,
            "step": 2,
            "representation": "lerobot-native-full",
            "training_backend": "lerobot",
            "policy_type": "act",
            "upstream_revision": TRAINING_REVISION,
            "reload_verified": False,
            "task_success": None,
        },
    )
    write(
        root / "verification.json",
        {"reload_verified": True, "max_abs_action_difference": 0.0},
    )
    write(
        root / "manifest.json",
        {
            "schema_version": 1,
            "metadata": {
                "architecture": "act",
                "training_backend": "lerobot",
                "method": "full",
                "base_model": {
                    "repository": "code://lerobot/act",
                    "revision": TRAINING_REVISION,
                },
                "dataset": {
                    "source": "huggingface",
                    "repo_id": "fixture/robot",
                    "revision": "a" * 40,
                },
                "camera_keys": recipe["camera_keys"],
                "action_dim": 6,
                "reload_verified": True,
            },
        },
    )
    resign(root)
    return root


def admit(root: Path):
    return admit_training_source(
        root,
        artifact_id="job:checkpoint",
        manifest_sha256=digest(root / "manifest.json"),
    )


def test_admission_preserves_every_byte_and_records_unverified_lineage(
    training_bundle: Path,
) -> None:
    before = {str(path): path.read_bytes() for path in training_bundle.rglob("*") if path.is_file()}
    source = admit(training_bundle)
    assert source.source == training_bundle / "checkpoint/pretrained_model"
    receipt = source.receipt()
    assert receipt["source_manifest_sha256"] == digest(training_bundle / "manifest.json")
    assert receipt["checkpoint_manifest_sha256"] == digest(
        training_bundle / "checkpoint/manifest.json"
    )
    assert receipt["checkpoint_step"] == 2
    assert receipt["dataset"] == {"repository": "fixture/robot", "revision": "a" * 40}
    assert receipt["runtime_compatibility_verified"] is False
    assert receipt["export_verified"] is False and receipt["task_success"] is None
    assert set(receipt["source_files"]) == {p.name for p in source.source.iterdir()}
    assert before == {
        str(path): path.read_bytes() for path in training_bundle.rglob("*") if path.is_file()
    }


@pytest.mark.parametrize(
    "where,key,value",
    [
        ("checkpoint/manifest.json", "upstream_revision", "b" * 40),
        ("checkpoint/manifest.json", "policy_type", "smolvla"),
        ("checkpoint/manifest.json", "representation", "inference-only"),
        ("checkpoint/manifest.json", "step", True),
        ("checkpoint/manifest.json", "step", 3),
        ("checkpoint/recipe.json", "model_revision", "b" * 40),
        ("checkpoint/recipe.json", "upstream_revision", "b" * 40),
        ("checkpoint/recipe.json", "dataset_revision", "b" * 40),
        ("checkpoint/recipe.json", "camera_keys", ["observation.images.other"]),
        ("checkpoint/recipe.json", "chunk_size", 20),
        ("checkpoint/pretrained_model/config.json", "chunk_size", 20),
        ("checkpoint/pretrained_model/config.json", "type", "smolvla"),
        ("checkpoint/pretrained_model/config.json", "use_vae", False),
        ("checkpoint/pretrained_model/config.json", "use_amp", True),
        ("verification.json", "reload_verified", False),
        ("verification.json", "max_abs_action_difference", -1),
    ],
)
def test_incompatible_saved_contracts_fail_after_rehash(
    training_bundle: Path, where, key, value
) -> None:
    edit(training_bundle / where, key, value)
    resign(training_bundle)
    with pytest.raises(ValueError):
        admit(training_bundle)


@pytest.mark.parametrize(
    "key,value",
    [
        ("architecture", "smolvla"),
        ("method", "lora"),
        ("reload_verified", False),
        ("action_dim", 2),
        ("base_model", {"repository": "code://lerobot/act", "revision": "b" * 40}),
        (
            "dataset",
            {"source": "local", "repo_id": "fixture/robot", "revision": "a" * 40},
        ),
    ],
)
def test_outer_metadata_must_agree_with_checkpoint(training_bundle: Path, key, value) -> None:
    path = training_bundle / "manifest.json"
    document = json.loads(path.read_text())
    document["metadata"][key] = value
    write(path, document)
    with pytest.raises(ValueError):
        admit(training_bundle)


@pytest.mark.parametrize(
    "name",
    [
        "checkpoint/training_state/optimizer_state.safetensors",
        "checkpoint/probe-action.pt",
        "checkpoint/pretrained_model/pre-stats.safetensors",
    ],
)
def test_missing_resume_or_processor_files_fail_even_with_new_manifest(
    training_bundle: Path, name
) -> None:
    (training_bundle / name).unlink()
    resign(training_bundle)
    with pytest.raises((ValueError, FileNotFoundError)):
        admit(training_bundle)


@pytest.mark.parametrize(
    "name",
    ["checkpoint/probe-batch.pt", "checkpoint/pretrained_model/model.safetensors"],
)
def test_content_tampering_rejected(training_bundle: Path, name: str) -> None:
    (training_bundle / name).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="inventory or hash"):
        admit(training_bundle)


def test_manifest_pinning_and_inner_integrity_are_independent(
    training_bundle: Path,
) -> None:
    registered = digest(training_bundle / "manifest.json")
    edit(training_bundle / "manifest.json", "unexpected", True)
    with pytest.raises(ValueError, match="Registered"):
        admit_training_source(
            training_bundle, artifact_id="job:checkpoint", manifest_sha256=registered
        )
    path = training_bundle / "checkpoint/manifest.json"
    document = json.loads(path.read_text())
    document["files"]["recipe.json"] = "0" * 64
    write(path, document)
    outer = training_bundle / "manifest.json"
    document = json.loads(outer.read_text())
    document["files"]["checkpoint/manifest.json"] = digest(path)
    write(outer, document)
    with pytest.raises(ValueError, match="inventory or hash"):
        admit(training_bundle)


@pytest.mark.parametrize(
    "name",
    [
        "../outside",
        "/absolute",
        "checkpoint/../escape",
        "C:\\outside",
        "checkpoint//recipe.json",
    ],
)
def test_manifest_paths_fail_closed(training_bundle: Path, name: str) -> None:
    path = training_bundle / "manifest.json"
    document = json.loads(path.read_text())
    document["files"][name] = "a" * 64
    write(path, document)
    with pytest.raises(ValueError, match="unsafe paths"):
        admit(training_bundle)


def test_unlisted_and_unfinalized_files_rejected(training_bundle: Path) -> None:
    extra = training_bundle / "unknown.bin"
    extra.write_bytes(b"extra")
    with pytest.raises(ValueError, match="inventory"):
        admit(training_bundle)
    extra.unlink()
    (training_bundle / ".partial").mkdir()
    with pytest.raises(ValueError, match="unsafe paths"):
        admit(training_bundle)


def test_remote_descriptors_and_inference_packages_are_not_training_sources(
    tmp_path: Path,
) -> None:
    (tmp_path / "remote.json").write_text("{}")
    with pytest.raises(ValueError, match="materialized"):
        admit_training_source(tmp_path, artifact_id="cloud:checkpoint", manifest_sha256="a" * 64)


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink and FIFO boundaries")
@pytest.mark.parametrize("kind", ["root", "directory", "file", "fifo"])
def test_symlinks_and_special_files_rejected(training_bundle: Path, tmp_path: Path, kind) -> None:
    if kind == "root":
        link = tmp_path / "link"
        link.symlink_to(training_bundle, target_is_directory=True)
        with pytest.raises(ValueError, match="nonsymlink"):
            admit(link)
        return
    path = training_bundle / (
        "checkpoint/pretrained_model" if kind == "directory" else "verification.json"
    )
    if kind == "fifo":
        path.unlink()
        os.mkfifo(path)
    else:
        target = tmp_path / "outside"
        shutil.move(path, target)
        path.symlink_to(target, target_is_directory=kind == "directory")
    with pytest.raises(ValueError, match="symlinks|regular"):
        admit(training_bundle)


@pytest.mark.parametrize("limit", ["MAX_ENTRIES", "MAX_FILE_BYTES", "MAX_BUNDLE_BYTES"])
def test_verification_has_finite_inventory_and_size_limits(
    training_bundle: Path, monkeypatch, limit
) -> None:
    monkeypatch.setattr(training_source, limit, 1)
    with pytest.raises(ValueError):
        admit(training_bundle)


def test_mid_verification_mutation_is_not_admitted(training_bundle: Path, monkeypatch) -> None:
    original = training_source.validate_processors

    def mutate(root, config):
        result = original(root, config)
        (training_bundle / "checkpoint/probe-batch.pt").write_bytes(
            b"changed after initial inventory"
        )
        return result

    monkeypatch.setattr(training_source, "validate_processors", mutate)
    with pytest.raises(ValueError, match="changed during admission"):
        admit(training_bundle)


@pytest.mark.runtime
def test_real_exporter_fixture_can_be_admitted_without_loading_it(
    training_bundle: Path, act_source: Path
) -> None:
    """A0.6.1 fixture in synthetic training packaging is not a0.6.2 producer test."""
    destination = training_bundle / "checkpoint/pretrained_model"
    shutil.rmtree(destination)
    shutil.copytree(act_source, destination)
    resign(training_bundle)
    assert admit(training_bundle).receipt()["runtime_compatibility_verified"] is False


@pytest.mark.parametrize(
    "mutation",
    ["stats_nan", "custom_processor", "traversal", "missing_vae", "non_fp32", "extra_policy_file"],
)
def test_exporter_semantic_guards_apply_after_valid_manifests(
    training_bundle: Path, mutation: str
) -> None:
    policy = training_bundle / "checkpoint/pretrained_model"
    if mutation == "stats_nan":
        state = policy / "pre-stats.safetensors"
        data = bytearray(state.read_bytes())
        data[-4:] = struct.pack("<f", float("nan"))
        state.write_bytes(data)
    elif mutation in {"custom_processor", "traversal"}:
        path = policy / "policy_preprocessor.json"
        document = json.loads(path.read_text())
        if mutation == "custom_processor":
            document["steps"][0]["registry_name"] = "untrusted.Custom"
        else:
            document["steps"][-1]["state_file"] = "../stats.safetensors"
        write(path, document)
    elif mutation in {"missing_vae", "non_fp32"}:
        path = policy / "model.safetensors"
        data = path.read_bytes()
        size = struct.unpack("<Q", data[:8])[0]
        header = json.loads(data[8 : 8 + size])
        if mutation == "non_fp32":
            next(iter(header.values()))["dtype"] = "F16"
        else:
            key = next(key for key in header if key.startswith("model.vae_encoder"))
            header["wrong_tensor_name"] = header.pop(key)
        encoded = json.dumps(header).encode()
        encoded += b" " * (-len(encoded) % 8)
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data[8 + size :])
    else:
        (policy / "extra.bin").write_bytes(b"not in the narrow source contract")
    resign(training_bundle)
    with pytest.raises(ValueError):
        admit(training_bundle)


def test_rehashed_partial_file_is_not_a_finalized_bundle(training_bundle: Path) -> None:
    (training_bundle / "checkpoint/training_state/optimizer.partial").write_bytes(b"unfinished")
    resign(training_bundle)
    with pytest.raises(ValueError, match="unsafe paths"):
        admit(training_bundle)


@pytest.mark.parametrize("value", [1, True, False, 0, -1, "2", 2.0, None])
def test_saved_native_step_must_match_manifest_without_coercion(
    training_bundle: Path, value
) -> None:
    edit(training_bundle / "checkpoint/training_state/training_step.json", "step", value)
    resign(training_bundle)
    with pytest.raises(ValueError, match="Saved native training step"):
        admit(training_bundle)


@pytest.mark.parametrize(
    "payload",
    [
        b"{",
        b"[]",
        b"{}",
        b'{"step":NaN}',
        b'{"step":2,"step":1}',
    ],
)
def test_malformed_native_step_document_rejected_after_rehash(
    training_bundle: Path, payload: bytes
) -> None:
    (training_bundle / "checkpoint/training_state/training_step.json").write_bytes(payload)
    resign(training_bundle)
    with pytest.raises(ValueError):
        admit(training_bundle)
