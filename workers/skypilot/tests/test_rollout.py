"""Fail before GPU submission and verify private discovery/artifact durability."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_IMAGE = "us-east4-docker.pkg.dev/project/simulation/worker@sha256:" + "a" * 64
_RUN_ID = "b" * 32
_NETWORK = "https://www.googleapis.com/compute/v1/projects/test/global/networks/sim-network"

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str((_ROOT / "../isaac_sim").resolve()))
from sim_worker.rollout.checkpoint import inspect_checkpoint  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = _load("rollout_launcher", _ROOT / "rollout_launch.py")
cloud = _load("rollout_cloud", _ROOT / "remote/cloud.py")
runner = _load("rollout_runner", _ROOT / "remote/rollout_runner.py")


class RolloutLaunchTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        root = Path(self.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.receipt_dir = root / "submission"
        self.receipt_dir.mkdir()
        self.stack.enter_context(
            patch.object(launcher.tempfile, "mkdtemp", return_value=str(self.receipt_dir))
        )
        self.worker = root / "isaac_sim"
        self.worker.mkdir()
        self.checkpoint = root / "checkpoint"
        self.checkpoint.mkdir()
        features = {
            "observation.state": {"type": "STATE", "shape": [1]},
            "observation.images.front": {"type": "VISUAL", "shape": [3, 360, 640]},
        }
        actions = {"action": {"type": "ACTION", "shape": [1]}}
        checkpoint_config = {
            "type": "act",
            "n_obs_steps": 1,
            "chunk_size": 100,
            "input_features": features,
            "output_features": actions,
            "normalization_mapping": {
                "STATE": "MEAN_STD",
                "ACTION": "MEAN_STD",
                "VISUAL": "MEAN_STD",
            },
        }
        (self.checkpoint / "config.json").write_text(json.dumps(checkpoint_config))
        (self.checkpoint / "model.safetensors").write_bytes(b"weights")
        for name, registry, feature_set in (
            ("preprocessor", "normalizer_processor", features),
            ("postprocessor", "unnormalizer_processor", actions),
        ):
            stats_name = f"{name}.safetensors"
            document = {
                "name": f"policy_{name}",
                "steps": [
                    {
                        "registry_name": registry,
                        "config": {
                            "features": feature_set,
                            "norm_map": checkpoint_config["normalization_mapping"],
                        },
                        "state_file": stats_name,
                    }
                ],
            }
            (self.checkpoint / f"policy_{name}.json").write_text(json.dumps(document))
            (self.checkpoint / stats_name).write_bytes(b"statistics")
        self.model_id = inspect_checkpoint(self.checkpoint).model_id
        config = {
            "gcp": {
                "vpc_name": "sim-network",
                "subnet_names": ["sim-us-central1"],
                "use_internal_ips": True,
                "ssh_proxy_command": {"us-central1": "iap"},
            },
            "jobs": {
                "force_disable_cloud_bucket": True,
                "controller": {"resources": {"infra": "gcp/us-central1"}},
            },
        }
        (root / "config.yaml").write_text(yaml.safe_dump(config))
        self.manifest = {
            "calibration": "calibration.yaml",
            "scene": {"uri": "scene.usda", "joints": ["joint"]},
            "policy": {"model_id": self.model_id},
            "control": {"steps": 300, "execute_steps": 1},
            "capture": {"width": 640, "height": 360},
        }
        (self.worker / "manifest.yaml").write_text(yaml.safe_dump(self.manifest))
        for name in ("scene.usda", "calibration.yaml"):
            (self.worker / name).write_text("contents")
        evidence = self.worker / "evidence"
        evidence.mkdir()
        for name in ("dataset.json", "front-frame-000.jpg"):
            (evidence / name).write_bytes(b"readiness fixture")
        self.documents = list(yaml.safe_load_all((_ROOT / "rollout.example.yaml").read_text()))
        for doc in self.documents[1:]:
            doc["workdir"] = str(self.worker)
        isaac, vla = self.documents[1:]
        isaac["envs"].update(
            SIM_IMAGE=_IMAGE, SIM_RESULTS_URI="gs://results/rollouts", SIM_MANIFEST="manifest.yaml"
        )
        vla["envs"].update(MODEL_ID=self.model_id, POLICY_STATE_DIM="1")
        vla["file_mounts"]["~/vla-checkpoint"] = str(self.checkpoint)
        self.task = root / "rollout.yaml"
        self.stack.enter_context(patch.object(launcher, "_ROOT", root))
        self.stack.enter_context(patch.object(launcher, "_WORKER", self.worker))
        self.stack.enter_context(
            patch.dict(os.environ, {"SIM_PROJECT_ID": "project", "ACCEPT_EULA": "Y"})
        )
        self.dispatch = self.stack.enter_context(patch.object(launcher.subprocess, "run"))
        self.stack.enter_context(patch.object(launcher, "_run_sdk", self.dispatch))

    def _prepare(self, mode=launcher._Mode.ROLLOUT):
        self.task.write_text(yaml.safe_dump_all(self.documents))
        return launcher._prepare(self.task, mode)

    def test_private_labels(self):
        result = self._prepare()
        for task in result[1:]:
            labels = task["resources"]["labels"]
            self.assertEqual(labels["rollout-id"], task["envs"]["ROLLOUT_ID"])
            self.assertEqual(labels["rollout-role"], task["name"])
            self.assertNotIn("ports", task["resources"])
        self.assertEqual(result[1]["envs"]["ROLLOUT_ID"], result[2]["envs"]["ROLLOUT_ID"])
        self.assertIn("--validate-only", self.dispatch.call_args.args[0])

    def _select(self, steps=None, mode=launcher._Mode.EXPERIMENTAL):
        self.task.write_text(yaml.safe_dump_all(self.documents))
        return launcher._select_checkpoint(self.task, self.checkpoint, mode, steps)

    def test_select_policy_export(self):
        manifest_path = self.worker / "manifest.yaml"
        self.manifest["control"]["execute_steps"] = 100
        manifest_path.write_text(yaml.safe_dump(self.manifest))
        original = manifest_path.read_bytes()
        config_path = self.checkpoint / "config.json"
        config = json.loads(config_path.read_text())
        for kind, chunk, action_steps in (("act", 100, 100), ("smolvla", 50, 25)):
            config.update(type=kind, chunk_size=chunk, n_action_steps=action_steps)
            config_path.write_text(json.dumps(config))
            info = inspect_checkpoint(self.checkpoint)
            with self.subTest(kind=kind), self._select() as selected:
                result = launcher._prepare(selected, launcher._Mode.EXPERIMENTAL)
                isaac, policy = result[1:]
                snapshot = self.worker / isaac["envs"]["SIM_MANIFEST"]
                data = yaml.safe_load(snapshot.read_text())
                self.assertEqual(data["policy"]["model_id"], info.model_id)
                self.assertEqual(data["control"]["execute_steps"], action_steps)
                self.assertEqual(policy["envs"]["POLICY_ACTION_STEPS"], str(action_steps))
                self.assertEqual(policy["envs"]["MODEL_ID"], info.model_id)
                self.assertEqual(data["calibration"], self.manifest["calibration"])
                self.assertEqual(manifest_path.read_bytes(), original)
                self.assertTrue(result[0]["name"].startswith(f"isaac-{kind}-test-"))
                remote = f"~/sky_workdir/{snapshot.relative_to(self.worker)}"
                self.assertEqual(isaac["file_mounts"][remote], str(snapshot))
            self.assertFalse(snapshot.exists())
            self.assertFalse(selected.exists())

    def test_select_capture_and_camera(self):
        path = self.checkpoint / "config.json"
        config = json.loads(path.read_text())
        camera = config["input_features"].pop("observation.images.front")
        camera["shape"] = [3, 480, 640]
        config["input_features"]["observation.images.overhead"] = camera
        path.write_text(json.dumps(config))
        with self._select(10) as selected:
            result = launcher._prepare(selected, launcher._Mode.EXPERIMENTAL)
            data = yaml.safe_load((self.worker / result[1]["envs"]["SIM_MANIFEST"]).read_text())
            self.assertEqual(data["capture"], {"width": 640, "height": 480})
            self.assertEqual(data["control"]["execute_steps"], 10)
            self.assertEqual(result[2]["envs"]["POLICY_CAMERA_KEY"], "observation.images.overhead")

    def test_select_horizon_limits(self):
        for steps in (0, True, 101):
            with self.subTest(steps=steps), self.assertRaisesRegex(ValueError, "execute.steps"):
                with self._select(steps):
                    self.fail("Invalid horizon accepted")
        self.dispatch.assert_not_called()

    def test_select_joint_mismatch(self):
        self.manifest["scene"]["joints"].append("unexpected")
        (self.worker / "manifest.yaml").write_text(yaml.safe_dump(self.manifest))
        with self.assertRaisesRegex(ValueError, "joint"):
            with self._select():
                self.fail("Incompatible robot accepted")
        self.dispatch.assert_not_called()

    def test_select_failure_cleanup(self):
        original = (self.worker / "manifest.yaml").read_bytes()
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            with self._select() as selected:
                task = list(yaml.safe_load_all(selected.read_text()))
                snapshot = self.worker / task[1]["envs"]["SIM_MANIFEST"]
                raise RuntimeError("cancelled")
        self.assertFalse(selected.exists())
        self.assertFalse(snapshot.exists())
        self.assertEqual((self.worker / "manifest.yaml").read_bytes(), original)

    def test_isolated_snapshots(self):
        with self._select(1) as first, self._select(10) as second:
            files = []
            for selected, steps in ((first, 1), (second, 10)):
                docs = list(yaml.safe_load_all(selected.read_text()))
                snapshot = self.worker / docs[1]["envs"]["SIM_MANIFEST"]
                data = yaml.safe_load(snapshot.read_text())
                self.assertEqual(data["control"]["execute_steps"], steps)
                files.append(snapshot)
            self.assertNotEqual(first, second)
            self.assertNotEqual(*files)

    def test_model_placeholders(self):
        self.documents[2]["envs"]["MODEL_ID"] = "CHANGE_ME_MODEL_ID"
        self.documents[2]["file_mounts"]["~/vla-checkpoint"] = "CHANGE_ME_CHECKPOINT"
        with self._select() as selected:
            launcher._prepare(selected, launcher._Mode.EXPERIMENTAL)

    def test_select_readiness(self):
        with self._select(mode=launcher._Mode.READINESS) as selected:
            result = launcher._prepare(selected, launcher._Mode.READINESS)
            self.assertIn("check_ready.sh", result[1]["run"])

    def test_horizon_needs_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            with launcher._select_checkpoint(self.task, None, launcher._Mode.ROLLOUT, 10):
                self.fail("Missing checkpoint accepted")

    def test_readiness_no_robot(self):
        result = self._prepare(launcher._Mode.READINESS)
        self.assertTrue(result[0]["name"].startswith("isaac-ready-"))
        self.assertIn("check_ready.sh", result[1]["run"])
        self.assertNotIn("rollout_runner.py", result[1]["run"])
        self.assertIn("--kill-after=60", result[1]["run"])
        self.assertNotIn("--validate-only", self.dispatch.call_args.args[0])
        self.assertIn("config import load", self.dispatch.call_args.args[0][2])
        self.assertIn("sim_worker.rollout.server", result[2]["run"])

    def test_experimental_motion(self):
        result = self._prepare(launcher._Mode.EXPERIMENTAL)
        self.assertTrue(result[0]["name"].startswith("isaac-act-test-"))
        self.assertEqual(result[1]["envs"]["SIM_EXPERIMENTAL"], "1")
        self.assertIn("rollout_runner.py", result[1]["run"])
        self.assertIn("--experimental", self.dispatch.call_args.args[0])
        for task in result[1:]:
            self.assertEqual(task["resources"]["job_recovery"]["max_restarts_on_errors"], 0)

    def test_experimental_l4_fallbacks(self):
        resources = self.documents[1]["resources"]
        resources.pop("instance_type")
        resources["ordered"] = [
            {"instance_type": "g2-standard-32"},
            {"instance_type": "g2-standard-12"},
        ]
        result = self._prepare(launcher._Mode.EXPERIMENTAL)
        self.assertEqual(result[1]["resources"]["ordered"], resources["ordered"])
        self.assertIn("rollout-id", result[1]["resources"]["labels"])

    def test_experimental_l4_shapes(self):
        for machine in ("g2-standard-12", "g2-standard-16", "g2-standard-32"):
            with self.subTest(machine=machine):
                self.documents[1]["resources"]["instance_type"] = machine
                self._prepare(launcher._Mode.EXPERIMENTAL)

    def test_experimental_l4_zones(self):
        for zone in ("a", "b", "c", "f"):
            with self.subTest(zone=zone):
                self.documents[1]["resources"]["infra"] = f"gcp/us-central1/us-central1-{zone}"
                self._prepare(launcher._Mode.EXPERIMENTAL)

    def test_experimental_zone_scope(self):
        for infra in (
            "gcp/us-east4/us-east4-a",
            "gcp/us-central1/us-central1-z",
            "gcp/us-central1/us-east4-a",
            "aws/us-central1/us-central1-c",
            "gcp/us-central1/us-central1-c/extra",
        ):
            with self.subTest(infra=infra):
                self.documents[1]["resources"]["infra"] = infra
                with self.assertRaisesRegex(ValueError, "GPU resource"):
                    self._prepare(launcher._Mode.EXPERIMENTAL)
        self.dispatch.assert_not_called()

    def test_production_zone_refused(self):
        self.documents[1]["resources"]["infra"] = "gcp/us-central1/us-central1-c"
        with self.assertRaisesRegex(ValueError, "GPU resource"):
            self._prepare()
        self.dispatch.assert_not_called()

    def test_policy_zone_refused(self):
        self.documents[2]["resources"]["infra"] = "gcp/us-central1/us-central1-a"
        with self.assertRaisesRegex(ValueError, "GPU resource"):
            self._prepare(launcher._Mode.EXPERIMENTAL)
        self.dispatch.assert_not_called()

    def test_separate_zone_refused(self):
        self.documents[1]["resources"]["zone"] = "us-central1-c"
        with self.assertRaisesRegex(ValueError, "infra"):
            self._prepare(launcher._Mode.EXPERIMENTAL)
        self.dispatch.assert_not_called()

    def test_strict_shape_retained(self):
        self.documents[1]["resources"]["instance_type"] = "g2-standard-32"
        with self.assertRaisesRegex(ValueError, "GPU resource"):
            self._prepare()
        self.dispatch.assert_not_called()

    def test_fallback_override_refused(self):
        resources = self.documents[1]["resources"]
        resources.pop("instance_type")
        for candidate in (
            {"instance_type": "g2-standard-32", "accelerators": "H100:1"},
            {"instance_type": "g2-standard-32", "ports": [8080]},
            {"instance_type": "g2-standard-32", "labels": {"rollout-id": "other"}},
            {"instance_type": "g2-standard-32", "job_recovery": {"max_restarts_on_errors": 10}},
        ):
            with self.subTest(candidate=candidate):
                resources["ordered"] = [candidate]
                with self.assertRaisesRegex(ValueError, "instance_type"):
                    self._prepare(launcher._Mode.EXPERIMENTAL)
        self.dispatch.assert_not_called()

    def test_fallback_gpu_refused(self):
        resources = self.documents[1]["resources"]
        resources.pop("instance_type")
        resources["ordered"] = [{"instance_type": "g2-standard-32"}]
        resources["accelerators"] = "H100:1"
        with self.assertRaisesRegex(ValueError, "GPU resource"):
            self._prepare(launcher._Mode.EXPERIMENTAL)
        self.dispatch.assert_not_called()

    def test_fallback_shape_refused(self):
        resources = self.documents[1]["resources"]
        resources.pop("instance_type")
        for machine in ("a3-highgpu-1g", "g2-standard-24"):
            with self.subTest(machine=machine):
                resources["ordered"] = [{"instance_type": machine}]
                with self.assertRaisesRegex(ValueError, "GPU resource"):
                    self._prepare(launcher._Mode.EXPERIMENTAL)
        self.dispatch.assert_not_called()

    def test_fallback_needs_experiment(self):
        resources = self.documents[1]["resources"]
        resources.pop("instance_type")
        resources["ordered"] = [{"instance_type": "g2-standard-32"}]
        with self.assertRaisesRegex(ValueError, "experimental"):
            self._prepare()
        self.dispatch.assert_not_called()

    def test_experimental_step_bound(self):
        for steps in (301, 0, -1, True, 1.5):
            with self.subTest(steps=steps):
                self.manifest["control"]["steps"] = steps
                (self.worker / "manifest.yaml").write_text(yaml.safe_dump(self.manifest))
                with self.assertRaisesRegex(ValueError, "300"):
                    self._prepare(launcher._Mode.EXPERIMENTAL)
        self.dispatch.assert_not_called()

    def test_experimental_needs_flag(self):
        self.documents[1]["envs"]["SIM_EXPERIMENTAL"] = "1"
        with self.assertRaisesRegex(ValueError, "--experimental"):
            self._prepare()
        self.dispatch.assert_not_called()

    def test_exclusive_test_modes(self):
        with (
            patch.object(sys, "argv", ["launch", "--check-ready", "--experimental"]),
            patch.object(launcher, "_prepare") as prepare,
            self.assertRaises(SystemExit) as error,
        ):
            launcher._main()
        self.assertEqual(error.exception.code, 2)
        prepare.assert_not_called()

    def test_experimental_wall_timeout(self):
        self.dispatch.side_effect = [subprocess.TimeoutExpired("launch", 7200), None]
        documents = [{"name": "isaac-act-test-unique"}]
        with self.assertRaises(subprocess.TimeoutExpired):
            launcher._launch(documents, self.task, launcher._Mode.EXPERIMENTAL)
        self.assertEqual(self.dispatch.call_args_list[0].kwargs["timeout"], 7200)
        cancel = self.dispatch.call_args_list[1].args[0]
        self.assertEqual(cancel[-4:], ["cancel", "--name", "isaac-act-test-unique", "--yes"])

    def test_missing_ready_evidence(self):
        (self.worker / "evidence/front-frame-000.jpg").unlink()
        with self.assertRaisesRegex(ValueError, "readiness evidence"):
            self._prepare(launcher._Mode.READINESS)
        self.dispatch.assert_not_called()

    def test_ready_wall_timeout(self):
        self.dispatch.side_effect = [subprocess.TimeoutExpired("launch", 7200), None]
        documents = [{"name": "isaac-ready-unique"}]
        with self.assertRaises(subprocess.TimeoutExpired):
            launcher._launch(documents, self.task, launcher._Mode.READINESS)
        self.assertEqual(self.dispatch.call_args_list[0].kwargs["timeout"], 7200)
        cancel = self.dispatch.call_args_list[1].args[0]
        self.assertEqual(cancel[-4:], ["cancel", "--name", "isaac-ready-unique", "--yes"])

    def test_timeout_cancels_exact_known_job_and_retains_uncertain_cleanup(self):
        submitted = {
            "schema_version": 1,
            "skypilot": "0.13.0",
            "job_id": 7,
            "group_name": "isaac-ready-unique",
            "request_id": "request-1",
            "tasks": [
                {"task_id": 0, "task_name": "isaac", "is_primary_in_job_group": True},
                {"task_id": 1, "task_name": "vla", "is_primary_in_job_group": False},
            ],
        }
        (self.receipt_dir / "submission.json").write_text(json.dumps(submitted))
        self.dispatch.side_effect = [subprocess.TimeoutExpired("launch", 7200), None]
        with self.assertRaises(subprocess.TimeoutExpired):
            launcher._launch([{"name": "isaac-ready-unique"}], self.task, launcher._Mode.READINESS)
        self.assertEqual(self.dispatch.call_args_list[1].args[0][-3:], ["cancel", "7", "--yes"])
        result = json.loads((self.receipt_dir / "cancellation.json").read_text())
        self.assertEqual(result["job_id"], 7)
        self.assertEqual(result["status"], "requested")
        self.assertEqual(result["resource_deletion"], "unverified")

    def test_failed_cancellation_and_interruption_never_become_success(self):
        self.dispatch.side_effect = [KeyboardInterrupt(), subprocess.TimeoutExpired("cancel", 120)]
        with self.assertRaises(subprocess.TimeoutExpired):
            launcher._launch([{"name": "isaac-ready-unique"}], self.task, launcher._Mode.READINESS)
        result = json.loads((self.receipt_dir / "cancellation.json").read_text())
        self.assertEqual(result["status"], "failed_or_unknown")
        self.assertEqual(result["reason"], "KeyboardInterrupt")
        self.assertEqual(self.dispatch.call_args_list[1].kwargs["timeout"], 120)

    def test_ordinary_interruption_keeps_existing_no_auto_cancellation(self):
        self.dispatch.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            launcher._launch([{"name": "ordinary"}], self.task, launcher._Mode.ROLLOUT)
        self.assertEqual(self.dispatch.call_count, 1)
        self.assertIsNone(self.dispatch.call_args.kwargs["timeout"])

    def test_launch_options(self):
        launcher._launch(
            [{"name": "isaac-ready-unique"}],
            self.task,
            launcher._Mode.READINESS,
            ("--yes", "--detach-run"),
        )
        self.assertEqual(self.dispatch.call_args.args[0][-2:], ["--yes", "--detach-run"])

    def test_missing_checkpoint(self):
        (self.checkpoint / "model.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "model.safetensors"):
            self._prepare()
        self.assertEqual(self.dispatch.call_count, 1)
        self.assertNotIn("jobs", self.dispatch.call_args.args[0])

    def test_invalid_calibration(self):
        self.dispatch.side_effect = subprocess.CalledProcessError(1, "validate-only")
        with self.assertRaises(subprocess.CalledProcessError):
            self._prepare()
        self.assertEqual(self.dispatch.call_count, 1)

    def test_outside_calibration(self):
        self.manifest["calibration"] = "/tmp/calibration.yaml"
        (self.worker / "manifest.yaml").write_text(yaml.safe_dump(self.manifest))
        with self.assertRaisesRegex(ValueError, "relative"):
            self._prepare()
        self.dispatch.assert_not_called()

    def test_wrong_model_refused(self):
        self.documents[2]["envs"]["MODEL_ID"] = "different-model"
        with self.assertRaisesRegex(ValueError, "MODEL_ID"):
            self._prepare()

    def test_changed_artifact(self):
        (self.checkpoint / "model.safetensors").write_bytes(b"different weights")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self._prepare()

    def test_missing_stats(self):
        (self.checkpoint / "preprocessor.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "preprocessor.safetensors"):
            self._prepare()

    def test_checkpoint_type(self):
        path = self.checkpoint / "config.json"
        config = json.loads(path.read_text())
        config["type"] = "diffusion"
        path.write_text(json.dumps(config))
        with self.assertRaises(ValueError):
            self._prepare()

    def test_capture_dimensions(self):
        self.manifest["capture"]["width"] = 320
        (self.worker / "manifest.yaml").write_text(yaml.safe_dump(self.manifest))
        with self.assertRaisesRegex(ValueError, "dimensions"):
            self._prepare()

    def test_state_dimensions(self):
        self.documents[2]["envs"]["POLICY_STATE_DIM"] = "2"
        self.manifest["scene"]["joints"].append("second_joint")
        (self.worker / "manifest.yaml").write_text(yaml.safe_dump(self.manifest))
        with self.assertRaisesRegex(ValueError, "state/actions"):
            self._prepare()

    def test_checkpoint_chunk(self):
        self.documents[2]["envs"]["POLICY_ACTION_STEPS"] = "101"
        with self.assertRaisesRegex(ValueError, "chunk size"):
            self._prepare()

    def test_missing_network(self):
        (launcher._ROOT / "config.yaml").write_text("gcp: {}\n")
        with self.assertRaisesRegex(ValueError, "us-central1"):
            self._prepare()
        self.dispatch.assert_not_called()

    def test_public_port_refused(self):
        self.documents[2]["resources"]["ports"] = [8080]
        with self.assertRaisesRegex(ValueError, "public"):
            self._prepare()
        self.dispatch.assert_not_called()

    def test_unpinned_controller(self):
        path = launcher._ROOT / "config.yaml"
        config = yaml.safe_load(path.read_text())
        config.pop("jobs")
        path.write_text(yaml.safe_dump(config))
        with self.assertRaisesRegex(ValueError, "jobs.controller"):
            self._prepare()
        self.dispatch.assert_not_called()


class PolicyDiscoveryTests(unittest.TestCase):
    def test_scoped_pagination(self):
        responses = [
            {
                "items": {
                    "zones/a": {
                        "instances": [
                            {
                                "networkInterfaces": [
                                    {"network": "other-network", "networkIP": "10.0.0.2"}
                                ]
                            }
                        ]
                    }
                },
                "nextPageToken": "next",
            },
            {
                "items": {
                    "zones/b": {
                        "instances": [
                            {"networkInterfaces": [{"network": _NETWORK, "networkIP": "10.43.0.3"}]}
                        ]
                    }
                }
            },
        ]
        with (
            patch.object(cloud, "_token", return_value="token"),
            patch.object(
                cloud, "_send", side_effect=[json.dumps(value).encode() for value in responses]
            ) as send,
        ):
            self.assertEqual(cloud._policy_addresses("project", _RUN_ID, _NETWORK), ["10.43.0.3"])
        query = parse_qs(urlsplit(send.call_args_list[0].args[0].full_url).query)
        self.assertIn(_RUN_ID, query["filter"][0])
        self.assertIn("rollout-role = vla", query["filter"][0])
        self.assertIn("pageToken=next", send.call_args_list[1].args[0].full_url)

    def test_unsafe_target(self):
        for addresses, message in (
            (["10.43.0.2", "10.43.0.3"], "Multiple"),
            (["8.8.8.8"], "private"),
        ):
            with (
                self.subTest(addresses=addresses),
                patch.object(cloud, "_metadata", return_value="project"),
                patch.object(
                    cloud, "_compute", return_value={"networkInterfaces": [{"network": _NETWORK}]}
                ),
                patch.object(cloud, "_policy_addresses", return_value=addresses),
                self.assertRaisesRegex(ValueError, message),
            ):
                cloud._discover_policy(_RUN_ID, 900)

    def test_discovery_expires(self):
        with (
            patch.object(cloud, "_metadata", return_value="project"),
            patch.object(
                cloud, "_compute", return_value={"networkInterfaces": [{"network": _NETWORK}]}
            ),
            patch.object(cloud, "_policy_addresses", return_value=[]),
            patch.object(cloud.time, "monotonic", side_effect=[0, 901]),
            self.assertRaises(TimeoutError),
        ):
            cloud._discover_policy(_RUN_ID, 900)


class RolloutArtifactTests(unittest.TestCase):
    def test_publish_tree_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "outputs").mkdir()
            (root / "outputs/result.json").write_text("{}")
            (root / "outputs/trajectory.jsonl").write_text("{}\n")
            (root / "outputs/final.ppm").write_bytes(b"P6\n1 1\n255\n\0\0\0")
            (root / "outputs/result.pending.json").write_text("{}")
            (root / "job-result.json").write_text("{}")
            with patch.object(cloud, "_upload") as upload:
                cloud._publish_tree(root, "gs://results/episode")
            self.assertEqual(upload.call_args_list[-1].args[0].name, "job-result.json")
            self.assertEqual(upload.call_args_list[0].args[1], "gs://results/episode/outputs")

    def test_publish_refuses_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "job-result.json").write_text("{}")
            (root / "worker.log").symlink_to("/etc/passwd")
            with (
                patch.object(cloud, "_upload") as upload,
                self.assertRaisesRegex(ValueError, "symlinks"),
            ):
                cloud._publish_tree(root, "gs://results/episode")
            upload.assert_not_called()

    def test_zero_without_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "result.json").write_text('{"status":"succeeded"}')
            with self.assertRaisesRegex(ValueError, "trajectory"):
                runner._check_result(root)

    def test_container_options(self):
        command = runner._command(
            _IMAGE,
            Path("/source"),
            Path("/out"),
            "manifest.yaml",
            "http://10.43.0.3:8080",
            "test-container",
            3600,
        )
        self.assertIn("POLICY_ENDPOINT=http://10.43.0.3:8080", command)
        self.assertIn("/usr/bin/timeout", command)
        self.assertIn("sim_worker.rollout", command)
        self.assertIn("type=bind,src=/source,dst=/opt/sim-worker,readonly", command)
        self.assertNotIn("--experimental", command)

    def test_experimental_container(self):
        command = runner._command(
            _IMAGE,
            Path("/source"),
            Path("/out"),
            "manifest.yaml",
            "http://10.43.0.3:8080",
            "test-container",
            3600,
            runner._Mode.EXPERIMENTAL,
        )
        self.assertEqual(command[-1], "--experimental")

    def test_runner_mode(self):
        for value in ("1", "yes"):
            with self.subTest(value=value), patch.dict(os.environ, {"SIM_EXPERIMENTAL": value}):
                self.assertEqual(runner._mode(), runner._Mode.EXPERIMENTAL)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(runner._mode(), runner._Mode.ROLLOUT)
        with (
            patch.dict(os.environ, {"SIM_EXPERIMENTAL": "maybe"}),
            self.assertRaisesRegex(ValueError, "SIM_EXPERIMENTAL"),
        ):
            runner._mode()


if __name__ == "__main__":
    unittest.main()
