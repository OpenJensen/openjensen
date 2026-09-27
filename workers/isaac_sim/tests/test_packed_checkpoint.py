"""Packed ACT package admission is a byte/shape check, never model-quality proof."""

# ruff: noqa: E402 -- pure validators live in optional sibling workers.
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

WORKERS = Path(__file__).resolve().parents[2]
for source in (WORKERS / "firebird_quant/src", WORKERS / "act_optimizer/src"):
    sys.path.insert(0, str(source))

from firebird_quant.native_package import encoding, inspect_policy
from test_checkpoint_package import export, manifest, tensor_file

from sim_worker.rollout.checkpoint import inspect_checkpoint
from sim_worker.rollout.checkpoint_package import resolve_checkpoint


def packed_fixture(root, bits=8):
    export(root)
    config = json.loads((root / "config.json").read_text())
    config.update(
        n_action_steps=100,
        use_vae=False,
        use_peft=False,
        use_amp=False,
        temporal_ensemble_coeff=None,
        vision_backbone="resnet18",
        pre_norm=False,
        replace_final_stride_with_dilation=False,
        feedforward_activation="relu",
        normalization_mapping={"VISUAL": "MEAN_STD", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        latent_dim=8,
    )
    (root / "config.json").write_text(json.dumps(config))
    device = {"registry_name": "device_processor", "config": {"device": "cpu", "float_dtype": None}}
    features = config["input_features"] | config["output_features"]
    for name, registry, expected in (
        ("preprocessor", "normalizer_processor", features),
        ("postprocessor", "unnormalizer_processor", config["output_features"]),
    ):
        stats = {}
        for key, feature in expected.items():
            shape = [3, 1, 1] if feature["type"] == "VISUAL" else [6]
            count = 3 if feature["type"] == "VISUAL" else 6
            stats[key + ".mean"] = (shape, [0.0] * count)
            stats[key + ".std"] = (shape, [1.0] * count)
        tensor_file(root / (name + ".safetensors"), stats)
        normalizer = {
            "registry_name": registry,
            "config": {
                "features": expected,
                "eps": 1e-8,
                "norm_map": config["normalization_mapping"],
            },
            "state_file": name + ".safetensors",
        }
        steps = (
            [normalizer, device]
            if name == "postprocessor"
            else [
                {"registry_name": "rename_observations_processor", "config": {"rename_map": {}}},
                {"registry_name": "to_batch_processor", "config": {}},
                device,
                normalizer,
            ]
        )
        (root / ("policy_" + name + ".json")).write_text(
            json.dumps({"name": "policy_" + name, "steps": steps})
        )
    (root / "model.safetensors").rename(root / "model.fbq")
    (root / "encoding.json").write_text(json.dumps(encoding(bits)))
    return root


class PackedCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def test_identity_matches_producer_and_manifest_archive(self):
        for bits in (4, 8):
            outer = self.root / str(bits)
            policy = packed_fixture(outer / "policy", bits)
            expected = inspect_policy(policy)["model_id"]
            manifest(outer)
            tar = self.root / f"{bits}.tar"
            with tarfile.open(tar, "w") as stream:
                stream.add(outer, arcname="native-quantized")
            with resolve_checkpoint(tar, archive=True) as admitted:
                self.assertEqual(admitted.checkpoint.model_id, expected)
                self.assertEqual(admitted.checkpoint.policy_type, "act")
                self.assertFalse(admitted.metadata()["runtime_verified"])
                self.assertIn("native-quantized/policy/encoding.json", admitted.files)
                self.assertEqual(len(admitted.manifests), 1)

    def test_inspection_does_not_import_torch(self):
        policy = packed_fixture(self.root / "policy")
        code = (
            "import sys;sys.path[:0]="
            + repr(
                [str(WORKERS / p) for p in ("isaac_sim", "firebird_quant/src", "act_optimizer/src")]
            )
            + ";from sim_worker.rollout.checkpoint import inspect_checkpoint"
            + ";from pathlib import Path;inspect_checkpoint(Path(sys.argv[1]))"
            + ";assert 'torch' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-c", code, str(policy)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_configuration_metadata_is_bound_to_fingerprinted_bytes(self):
        import firebird_quant.native_package as package

        policy = packed_fixture(self.root / "policy")
        original = package.inspect_policy

        def changed(root):
            result = original(root)
            config = json.loads((root / "config.json").read_text())
            config["input_features"]["observation.images.front"]["shape"] = [3, 64, 64]
            (root / "config.json").write_text(json.dumps(config))
            return result

        with patch.object(package, "inspect_policy", changed):
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                inspect_checkpoint(policy)

    def test_extra_float_master_or_missing_encoding_rejected(self):
        policy = packed_fixture(self.root / "policy")
        tensor_file(policy / "model.safetensors")
        with self.assertRaises(ValueError):
            inspect_checkpoint(policy)
        (policy / "model.safetensors").unlink()
        (policy / "encoding.json").unlink()
        with self.assertRaises((ValueError, OSError)):
            inspect_checkpoint(policy)

    def test_packed_smol_and_vae_are_not_silently_admitted(self):
        policy = packed_fixture(self.root / "policy")
        original = json.loads((policy / "config.json").read_text())
        for key, value in (("type", "smolvla"), ("use_vae", True)):
            (policy / "config.json").write_text(json.dumps(original | {key: value}))
            with self.assertRaises(ValueError):
                inspect_checkpoint(policy)

    def test_encoding_types_and_runtime_pin(self):
        policy = packed_fixture(self.root / "policy")
        for modify in (
            lambda d: d.update(recipe=None),
            lambda d: d.update(recipe=[]),
            lambda d: d.update(schema_version=True),
            lambda d: d["recipe"].update(min_ndim=2.0),
            lambda d: d["runtime"].update(lerobot="0.6.2"),
        ):
            value = encoding(8)
            modify(value)
            (policy / "encoding.json").write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                inspect_checkpoint(policy)

    def test_symlink_root_and_processor_are_rejected(self):
        policy = packed_fixture(self.root / "policy")
        link = self.root / "alias"
        link.symlink_to(policy, target_is_directory=True)
        with self.assertRaises(ValueError):
            inspect_checkpoint(link)
        original = policy / "postprocessor.safetensors"
        moved = self.root / "stats"
        original.rename(moved)
        original.symlink_to(moved)
        with self.assertRaises(ValueError):
            inspect_checkpoint(policy)

    def test_corrupt_weight_header_is_rejected_before_runtime(self):
        policy = packed_fixture(self.root / "policy")
        (policy / "model.fbq").write_bytes(b"not a safetensors container")
        with self.assertRaises(ValueError):
            with resolve_checkpoint(policy):
                self.fail("Invalid packed storage admitted")

    def test_mutation_changes_identity_and_outer_manifest_rejects(self):
        policy = packed_fixture(self.root / "outer/policy")
        before = inspect_checkpoint(policy).model_id
        manifest(policy.parent)
        tensor_file(policy / "model.fbq", {"weight": ([1], [2.0])})
        self.assertNotEqual(inspect_checkpoint(policy).model_id, before)
        with self.assertRaisesRegex(ValueError, "manifest"):
            with resolve_checkpoint(policy.parent):
                self.fail("Changed package admitted")


if __name__ == "__main__":
    unittest.main()
