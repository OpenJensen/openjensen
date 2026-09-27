"""Small metadata-only fixtures. Weights are inert bytes, never native model evidence."""

import math
import struct
from firebird_act.bundle import NORM_MAP, canonical
from firebird_act.control_schema import SCOPE


def control_record():
    return {
        "schema_version": 1, "kind": "simulator_joint_position",
        "controller": "joint_position_targets", "state_key": "observation.state",
        "action_key": "action", "state_units": "radians", "action_units": "radians",
        "timebase": "simulation_seconds", "joint_order": [f"joint_{i}" for i in range(6)],
        "camera": {"key": "observation.images.front", "width": 32, "height": 32,
                   "prim": "/World/Camera"},
        "action_fps": 20,
        "source": {
            "dataset_snapshot_id": "sha256:" + "a" * 64,
            "dataset_manifest_sha256": "a" * 64, "demonstrations_sha256": "b" * 64,
            "scene_sha256": ["c" * 64], "scene_hash_scope": SCOPE, "origins": ["synthetic"],
        },
        "physical_calibration_verified": False, "task_success_verified": False,
    }


def temporal_record(config, fps=20):
    prediction, execution = config["chunk_size"], config["n_action_steps"]
    return {
        "schema_version": 1, "family": "act", "action_fps": fps,
        "policy_fields": {"chunk_size": prediction, "n_action_steps": execution, "n_obs_steps": 1},
        "action_delta_indices": list(range(prediction)), "observation_delta_indices": None,
        "action_delta_timestamps": [i / fps for i in range(prediction)],
        "observation_delta_timestamps": None, "observation_history": 1, "frame_stride": 1,
        "prediction_horizon": prediction, "execution_horizon": execution,
    }


def statistics(features):
    header, payload = {}, bytearray()
    for name, feature in features.items():
        shape = [3, 1, 1] if feature["type"] == "VISUAL" else feature["shape"]
        for statistic, value in (("mean", 1.25), ("std", 2.5)):
            start = len(payload)
            payload.extend(struct.pack("<" + "f" * math.prod(shape), *([value] * math.prod(shape))))
            header[name + "." + statistic] = {
                "dtype": "F32", "shape": shape, "data_offsets": [start, len(payload)]
            }
    encoded = canonical(header)
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + bytes(payload)


def policy(root, prediction=8, execution=3, sidecars=True):
    root.mkdir()
    config = {
        "type": "act", "chunk_size": prediction, "n_action_steps": execution,
        "n_obs_steps": 1, "use_vae": False, "use_peft": False, "use_amp": False,
        "temporal_ensemble_coeff": None, "replace_final_stride_with_dilation": False,
        "vision_backbone": "resnet18", "pre_norm": False, "feedforward_activation": "relu",
        "normalization_mapping": NORM_MAP, "dim_model": 512, "n_heads": 4,
        "dim_feedforward": 1024, "n_encoder_layers": 2, "n_decoder_layers": 1,
        "n_vae_encoder_layers": 1, "latent_dim": 8,
        "input_features": {"observation.state": {"type": "STATE", "shape": [6]},
                           "observation.images.front": {"type": "VISUAL", "shape": [3, 32, 32]}},
        "output_features": {"action": {"type": "ACTION", "shape": [6]}},
    }
    (root / "config.json").write_bytes(canonical(config))
    (root / "model.safetensors").write_bytes(b"metadata-only generated weight placeholder")
    features = config["input_features"] | config["output_features"]
    def step(name, cfg, file=None):
        return {"registry_name": name, "config": cfg} | ({"state_file": file} if file else {})
    device = step("device_processor", {"device": "cpu", "float_dtype": None})
    pre = [step("rename_observations_processor", {"rename_map": {}}),
           step("to_batch_processor", {}), device,
           step("normalizer_processor", {"eps": 1e-8, "features": features, "norm_map": NORM_MAP},
                "pre-stats.safetensors")]
    post = [step("unnormalizer_processor", {"eps": 1e-8, "features": config["output_features"],
                                           "norm_map": NORM_MAP}, "post-stats.safetensors"), device]
    for name, steps in (("policy_preprocessor", pre), ("policy_postprocessor", post)):
        (root / (name + ".json")).write_bytes(canonical({"name": name, "steps": steps}))
    (root / "pre-stats.safetensors").write_bytes(statistics(features))
    (root / "post-stats.safetensors").write_bytes(statistics(config["output_features"]))
    if sidecars:
        (root / "control-contract.json").write_bytes(canonical(control_record()))
        (root / "temporal-contract.json").write_bytes(canonical(temporal_record(config)))
    return config
