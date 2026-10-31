"""Closed native SmolVLA contracts and data-only assets, without importing ML runtimes."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_contract import (
    CONFIG_KEYS,
    PIPELINES,
    SMOLVLA_PIPELINES,
    SMOLVLM_ASSET_PATH,
    SMOLVLM_MODEL,
    validate_policy_config,
    validate_processors,
    validate_smolvla_assets,
)


class SmolVLAContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / "pretrained_model"
        self.checkpoint.mkdir()
        (self.checkpoint / "model.safetensors").write_bytes(b"fixture-not-native-weights")
        self.config = {
            "type": "smolvla",
            "input_features": {
                "observation.state": {"type": "STATE", "shape": [6]},
                "observation.images.camera1": {"type": "VISUAL", "shape": [3, 480, 640]},
            },
            "output_features": {"action": {"type": "ACTION", "shape": [6]}},
            "chunk_size": 50,
            "n_action_steps": 50,
            "num_vlm_layers": 16,
            "num_expert_layers": -1,
            "tokenizer_max_length": 48,
        }
        self.write_config()
        self.assets = self.checkpoint / SMOLVLM_ASSET_PATH
        self.assets.mkdir(parents=True)
        self.asset_config = {
            "model_type": "smolvlm",
            "scale_factor": 4,
            "text_config": {
                "model_type": "llama",
                "hidden_size": 960,
                "intermediate_size": 2560,
                "num_hidden_layers": 32,
                "num_attention_heads": 15,
                "num_key_value_heads": 5,
                "vocab_size": 49280,
            },
            "vision_config": {
                "model_type": "smolvlm_vision",
                "hidden_size": 768,
                "num_attention_heads": 12,
                "image_size": 512,
                "patch_size": 16,
            },
        }
        for name, value in {
            "config.json": self.asset_config,
            "tokenizer_config.json": {"tokenizer_class": "GPT2TokenizerFast"},
            "tokenizer.json": {"version": "1.0", "model": {"type": "BPE", "vocab": {}}},
            "preprocessor_config.json": {"image_processor_type": "SmolVLMImageProcessor"},
            "processor_config.json": {"processor_class": "SmolVLMProcessor"},
        }.items():
            self.write_asset(name, value)
        self.write_processors("smolvla")

    def tearDown(self):
        self.temporary.cleanup()

    def write_config(self):
        (self.checkpoint / "config.json").write_text(json.dumps(self.config))

    def write_asset(self, name, value):
        (self.assets / name).write_text(json.dumps(value))

    def write_processors(self, policy_type):
        pipelines = SMOLVLA_PIPELINES if policy_type == "smolvla" else PIPELINES
        for filename, names in pipelines.items():
            steps = []
            for name in names:
                config = {}
                if name == "rename_observations_processor":
                    config = {"rename_map": {}}
                if name == "tokenizer_processor":
                    config = {
                        "max_length": 48,
                        "task_key": "task",
                        "padding_side": "right",
                        "padding": "max_length",
                        "truncation": True,
                        "tokenizer_name": SMOLVLM_MODEL,
                    }
                step = {"registry_name": name, "config": config}
                if name in {"normalizer_processor", "unnormalizer_processor"}:
                    state = name + ".safetensors"
                    (self.checkpoint / state).write_bytes(b"fixture")
                    step["state_file"] = state
                steps.append(step)
            (self.checkpoint / filename).write_text(json.dumps({"name": filename, "steps": steps}))

    def test_closed_dispatch_and_exact_native_processors(self):
        self.assertEqual(validate_policy_config(self.checkpoint)["type"], "smolvla")
        self.assertEqual(validate_smolvla_assets(self.checkpoint), self.assets)
        validate_processors(self.checkpoint, policy_type="smolvla")
        with self.assertRaises(ValueError):
            validate_processors(self.checkpoint)
        self.config["type"] = "custom"
        self.write_config()
        with self.assertRaisesRegex(ValueError, "ACT and SmolVLA"):
            validate_policy_config(self.checkpoint)

    def test_act_default_contract_remains_available_without_smol_assets(self):
        self.config.update(type="act", chunk_size=8, n_action_steps=4)
        self.write_config()
        self.write_processors("act")
        self.assertEqual(validate_policy_config(self.checkpoint)["type"], "act")
        validate_processors(self.checkpoint)

    def test_up_to_three_visual_features_and_legacy_native_camera_names(self):
        for key in ("observation.image2", "observation.images.camera3"):
            self.config["input_features"][key] = {"type": "VISUAL", "shape": [3, 256, 256]}
        self.write_config()
        validate_policy_config(self.checkpoint)
        self.config["input_features"]["observation.images.camera4"] = {
            "type": "VISUAL",
            "shape": [3, 256, 256],
        }
        self.write_config()
        with self.assertRaisesRegex(ValueError, "three RGB"):
            validate_policy_config(self.checkpoint)

    def test_six_joints_are_required_and_extra_features_are_rejected(self):
        for change in ("state", "action", "extra"):
            with self.subTest(change=change):
                original = copy.deepcopy(self.config)
                if change == "state":
                    self.config["input_features"]["observation.state"]["shape"] = [7]
                elif change == "action":
                    self.config["output_features"]["action"]["shape"] = [32]
                else:
                    self.config["input_features"]["observation.environment_state"] = {
                        "type": "ENV",
                        "shape": [6],
                    }
                self.write_config()
                with self.assertRaises(ValueError):
                    validate_policy_config(self.checkpoint)
                self.config = original

    def test_allocation_and_execution_bounds_are_strict(self):
        for field, value in {
            "chunk_size": 51,
            "n_action_steps": 51,
            "num_steps": 11,
            "num_vlm_layers": 17,
            "num_expert_layers": 3,
            "expert_width_multiplier": 0.9,
            "max_state_dim": 33,
            "max_action_dim": 33,
            "tokenizer_max_length": 129,
            "rtc_config": {},
            "compile_model": True,
            "adapt_to_pi_aloha": True,
            "use_delta_joint_actions_aloha": True,
            "resize_imgs_with_padding": [1024, 512],
            "n_obs_steps": 2,
            "empty_cameras": 1,
            "self_attn_every_n_layers": True,
            "vlm_model_name": "/external/model",
        }.items():
            with self.subTest(field=field, value=value):
                original = copy.deepcopy(self.config)
                self.config[field] = value
                self.write_config()
                with self.assertRaises(ValueError):
                    validate_policy_config(self.checkpoint)
                self.config = original

    def test_native_tokenizer_is_fixed_bounded_and_cannot_import_classes(self):
        path = self.checkpoint / "policy_preprocessor.json"
        original = json.loads(path.read_text())
        for field, value in {
            "max_length": 100000,
            "tokenizer_name": "https://untrusted/tokenizer",
            "padding": "longest",
            "padding_side": "left",
            "truncation": False,
            "task_key": "other",
            "trust_remote_code": True,
        }.items():
            with self.subTest(field=field):
                pipeline = copy.deepcopy(original)
                pipeline["steps"][3]["config"][field] = value
                path.write_text(json.dumps(pipeline))
                with self.assertRaises(ValueError):
                    validate_processors(self.checkpoint, "smolvla")
        for extra in (
            {"class": "subprocess.Popen"},
            {"artifacts": {"tokenizer_name": "../../outside"}},
        ):
            pipeline = copy.deepcopy(original)
            pipeline["steps"][3].update(extra)
            path.write_text(json.dumps(pipeline))
            with self.assertRaises(ValueError):
                validate_processors(self.checkpoint, "smolvla")
        original["steps"][3]["config"]["tokenizer_name"] = SMOLVLM_ASSET_PATH
        path.write_text(json.dumps(original))
        validate_processors(self.checkpoint, "smolvla")
        self.assertEqual(set(original["steps"][3]["config"]), CONFIG_KEYS["tokenizer_processor"])

    def test_dynamic_code_metadata_is_rejected_in_all_asset_configs(self):
        for name in ("config.json", "tokenizer_config.json", "processor_config.json"):
            with self.subTest(name=name):
                original = json.loads((self.assets / name).read_text())
                self.write_asset(name, dict(original, auto_map={"AutoConfig": "custom.Config"}))
                with self.assertRaisesRegex(ValueError, "custom code"):
                    validate_smolvla_assets(self.checkpoint)
                self.write_asset(name, original)

    def test_inert_native_name_metadata_cannot_resolve_an_external_path(self):
        for inert in (None, "", "None"):
            self.write_asset("config.json", {**self.asset_config, "_name_or_path": inert})
            validate_smolvla_assets(self.checkpoint)
        self.write_asset("config.json", {**self.asset_config, "_name_or_path": "../../outside"})
        with self.assertRaisesRegex(ValueError, "fixed local data files"):
            validate_smolvla_assets(self.checkpoint)

    def test_asset_metadata_cannot_override_tokenizer_or_processor_files(self):
        for name in ("tokenizer_config.json", "processor_config.json", "preprocessor_config.json"):
            original = json.loads((self.assets / name).read_text())
            for key in (
                "tokenizer_file",
                "vocab_file",
                "merges_file",
                "added_tokens_file",
                "special_tokens_map_file",
                "tokenizer_config_file",
                "chat_template_file",
                "vocab_files",
                "cache_path",
                "additional_paths",
            ):
                for target in ("/private/outside.json", "../../outside.json"):
                    with self.subTest(name=name, key=key, target=target):
                        self.write_asset(name, {**original, key: target})
                        with self.assertRaisesRegex(ValueError, "fixed local data files"):
                            validate_smolvla_assets(self.checkpoint)
            self.write_asset(name, original)
        self.write_asset(
            "tokenizer_config.json",
            {
                "tokenizer_class": "GPT2TokenizerFast",
                "nested": {"tokenizer_file": "../../outside.json"},
            },
        )
        with self.assertRaisesRegex(ValueError, "fixed local data files"):
            validate_smolvla_assets(self.checkpoint)

    def test_native_backbone_dimensions_are_bounded_before_constructor(self):
        for section, key, value in (
            ("text_config", "num_hidden_layers", 10000),
            ("text_config", "hidden_size", 100000),
            ("text_config", "vocab_size", 10000000),
            ("vision_config", "image_size", 100000),
            ("vision_config", "patch_size", 1),
            ("vision_config", "model_type", "custom"),
        ):
            with self.subTest(section=section, key=key):
                changed = copy.deepcopy(self.asset_config)
                changed[section][key] = value
                self.write_asset("config.json", changed)
                with self.assertRaises(ValueError):
                    validate_smolvla_assets(self.checkpoint)
        self.write_asset("config.json", self.asset_config)

    def test_unexpected_assets_symlinks_and_missing_closure_are_rejected(self):
        bad = self.assets / "model.py"
        bad.write_text("raise RuntimeError('must never import')")
        with self.assertRaisesRegex(ValueError, "known data files"):
            validate_smolvla_assets(self.checkpoint)
        bad.unlink()
        tokenizer = self.assets / "tokenizer.json"
        tokenizer.rename(self.root / "outside.json")
        tokenizer.symlink_to(self.root / "outside.json")
        with self.assertRaisesRegex(ValueError, "symbolic links"):
            validate_smolvla_assets(self.checkpoint)
        tokenizer.unlink()
        with self.assertRaisesRegex(ValueError, "complete local"):
            validate_smolvla_assets(self.checkpoint)

    def test_asset_parent_rejects_unreviewed_sibling_files_and_directories(self):
        for name, directory in (("evil.py", False), ("other-tokenizer", True)):
            with self.subTest(name=name):
                sibling = self.assets.parent / name
                if directory:
                    sibling.mkdir()
                else:
                    sibling.write_text("unreviewed payload")
                with self.assertRaisesRegex(ValueError, "only the fixed smolvlm"):
                    validate_smolvla_assets(self.checkpoint)
                if directory:
                    sibling.rmdir()
                else:
                    sibling.unlink()

    def test_assets_have_a_total_byte_budget_and_cannot_hide_subdirectories(self):
        with (self.assets / "added_tokens.json").open("wb") as stream:
            stream.truncate(32 * 1024**2)
        with self.assertRaisesRegex(ValueError, "32 MiB"):
            validate_smolvla_assets(self.checkpoint)
        (self.assets / "added_tokens.json").unlink()
        (self.assets / "nested").mkdir()
        with self.assertRaisesRegex(ValueError, "known data files"):
            validate_smolvla_assets(self.checkpoint)


if __name__ == "__main__":
    unittest.main()
