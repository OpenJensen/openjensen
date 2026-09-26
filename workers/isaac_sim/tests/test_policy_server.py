import base64
import copy
import json
import threading
import unittest
from contextlib import nullcontext
from http import HTTPStatus
from http.client import HTTPConnection
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sim_worker.rollout.backend import LeRobotPolicy, MockPolicy, _check_features
from sim_worker.rollout.checkpoint import inspect_checkpoint
from sim_worker.rollout.server import create_server

_MODEL = "checkpoint-sha256-test"
_EPISODE = "episode-one"
_CAMERA = "observation.images.front"
_STATE = [0.0, 1.0, -1.0, 2.0, -2.0, 0.5]
_RGB = bytes([255, 0, 0, 0, 255, 0])
_BODY = {
    "episode_id": _EPISODE,
    "model_id": _MODEL,
    "step": 0,
    "sim_time": 0.0,
    "task": "Pick up the object",
    "state": _STATE,
    "image": {
        "width": 2,
        "height": 1,
        "encoding": "rgb8",
        "data": base64.b64encode(_RGB).decode(),
    },
}


def _write_export(path, policy_type="act"):
    config = {
        "type": policy_type,
        "n_obs_steps": 1,
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [6]},
            _CAMERA: {"type": "VISUAL", "shape": [3, 360, 640]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [6]}},
        "chunk_size": 100,
    }
    (path / "config.json").write_text(json.dumps(config))
    (path / "model.safetensors").write_bytes(b"test weights")
    for direction, registry in (
        ("pre", "normalizer_processor"),
        ("post", "unnormalizer_processor"),
    ):
        filename = f"{direction}.safetensors"
        (path / filename).write_bytes(b"test statistics")
        processor = {
            "steps": [
                {
                    "registry_name": registry,
                    "config": {"norm_map": {"STATE": "MEAN_STD"}},
                    "state_file": filename,
                }
            ]
        }
        (path / f"policy_{direction}processor.json").write_text(json.dumps(processor))


def _policy_config(policy_type="smolvla"):
    return SimpleNamespace(
        type=policy_type,
        input_features={"observation.state": None, _CAMERA: None},
        output_features={"action": None},
        image_features={_CAMERA: SimpleNamespace(shape=(3, 360, 640))},
        robot_state_feature=SimpleNamespace(shape=(6,)),
        action_feature=SimpleNamespace(shape=(6,)),
        n_obs_steps=1,
        chunk_size=100,
        temporal_ensemble_coeff=None,
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
    )


class _SpyPolicy(MockPolicy):
    def __init__(self):
        super().__init__()
        self.calls = []

    def predict(self, state, rgb, width, height, task):
        self.calls.append((state, rgb, width, height, task))
        return super().predict(state, rgb, width, height, task)


class PolicyServerTests(unittest.TestCase):
    def setUp(self):
        self.policy = _SpyPolicy()
        self.server = create_server(("127.0.0.1", 0), self.policy, _MODEL)
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, path, data=None):
        connection = HTTPConnection(*self.server.server_address, timeout=2)
        try:
            if data is None:
                connection.request("GET", path)
            else:
                raw = data if isinstance(data, bytes) else json.dumps(data).encode()
                connection.request(
                    "POST", path, body=raw, headers={"Content-Type": "application/json"}
                )
            response = connection.getresponse()
            self.assertEqual(response.getheader("Content-Type"), "application/json")
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def _reset(self, episode=_EPISODE):
        return self._request("/reset", {"episode_id": episode, "model_id": _MODEL})

    def test_hold_round_trip(self):
        self.assertEqual(
            self._request("/health"), (HTTPStatus.OK, {"status": "ready", "model_id": _MODEL})
        )
        self.assertEqual(
            self._reset(), (HTTPStatus.OK, {"episode_id": _EPISODE, "model_id": _MODEL})
        )
        status, response = self._request("/predict", _BODY)
        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(
            response, {"episode_id": _EPISODE, "step": 0, "model_id": _MODEL, "actions": [_STATE]}
        )
        self.assertEqual(self.policy.calls, [(_STATE, _RGB, 2, 1, _BODY["task"])])

    def test_reset_is_required(self):
        self.assertEqual(self._request("/predict", _BODY)[0], HTTPStatus.CONFLICT)
        self.assertEqual(self.policy.calls, [])

    def test_rejects_model_mismatch(self):
        self._reset()
        body = dict(_BODY, model_id="another-checkpoint")
        self.assertEqual(self._request("/predict", body)[0], HTTPStatus.CONFLICT)
        self.assertEqual(self.policy.calls, [])
        body = {"episode_id": "new", "model_id": "another-checkpoint"}
        self.assertEqual(self._request("/reset", body)[0], HTTPStatus.CONFLICT)

    def test_stale_step_and_time(self):
        self._reset()
        self.assertEqual(self._request("/predict", dict(_BODY, step=1))[0], HTTPStatus.CONFLICT)
        self.assertEqual(self._request("/predict", _BODY)[0], HTTPStatus.OK)
        self.assertEqual(self._request("/predict", _BODY)[0], HTTPStatus.CONFLICT)
        self.assertEqual(self._request("/predict", dict(_BODY, step=1))[0], HTTPStatus.CONFLICT)
        body = dict(_BODY, step=1, sim_time=1 / 30)
        self.assertEqual(self._request("/predict", body)[0], HTTPStatus.OK)
        self.assertEqual(len(self.policy.calls), 2)

    def test_retires_old_episode(self):
        self._reset()
        self.assertEqual(self._reset()[0], HTTPStatus.CONFLICT)
        self.assertEqual(self._reset("episode-two")[0], HTTPStatus.OK)
        self.assertEqual(self._request("/predict", _BODY)[0], HTTPStatus.CONFLICT)
        self.assertEqual(self._reset()[0], HTTPStatus.CONFLICT)
        body = dict(_BODY, episode_id="episode-two")
        self.assertEqual(self._request("/predict", body)[0], HTTPStatus.OK)

    def test_rejects_invalid_inputs(self):
        self._reset()
        cases = [
            ("state", _STATE[:-1]),
            ("state", [True] * len(_STATE)),
            ("state", [float("inf")] * len(_STATE)),
            ("step", True),
            ("step", -1),
            ("step", 0.1),
            ("sim_time", -0.1),
            ("sim_time", float("nan")),
            ("sim_time", True),
            ("sim_time", 10**400),
            ("episode_id", ""),
            ("task", ""),
            ("task", "a" * 4097),
            ("extra", "field"),
        ]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                self.assertEqual(
                    self._request("/predict", dict(_BODY, **{key: value}))[0],
                    HTTPStatus.BAD_REQUEST,
                )
        self.assertEqual(self.policy.calls, [])

    def test_rejects_invalid_images(self):
        self._reset()
        for key, value in [
            ("width", True),
            ("width", 0),
            ("height", -1),
            ("width", 10**10),
            ("encoding", "jpeg"),
            ("data", "???"),
            ("data", ""),
            ("data", 10),
            ("extra", "field"),
        ]:
            with self.subTest(key=key, value=value):
                body = copy.deepcopy(_BODY)
                body["image"][key] = value
                self.assertEqual(self._request("/predict", body)[0], HTTPStatus.BAD_REQUEST)
        self.assertEqual(self.policy.calls, [])

    def test_rejects_invalid_json(self):
        for raw in (b"{", b"[]", b"null", b'{"episode_id":"one","episode_id":"two"}', b"\xff"):
            with self.subTest(raw=raw):
                self.assertEqual(self._request("/reset", raw)[0], HTTPStatus.BAD_REQUEST)

    def test_rejects_large_body_early(self):
        connection = HTTPConnection(*self.server.server_address, timeout=2)
        try:
            connection.request(
                "POST",
                "/predict",
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(17 * 1024 * 1024),
                },
            )
            response = connection.getresponse()
            self.assertEqual(response.status, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            self.assertIn("error", json.loads(response.read()))
        finally:
            connection.close()

    def test_rejects_backend_output(self):
        for actions in ([[0.0]], [[float("nan")] * len(_STATE)], [], [_STATE, _STATE]):
            with self.subTest(actions=actions):
                episode = f"failure-{len(str(actions))}"
                self._reset(episode)
                body = dict(_BODY, episode_id=episode)
                with (
                    patch.object(self.policy, "predict", return_value=actions),
                    self.assertLogs(level="ERROR"),
                ):
                    status, response = self._request("/predict", body)
                self.assertEqual(status, HTTPStatus.INTERNAL_SERVER_ERROR)
                self.assertEqual(response, {"error": "Policy inference failed; reset required"})
                self.assertEqual(self._request("/predict", body)[0], HTTPStatus.CONFLICT)

    def test_hides_backend_errors(self):
        self._reset()
        with patch.object(self.policy, "predict", side_effect=RuntimeError("sensitive detail")):
            with self.assertLogs(level="ERROR"):
                status, body = self._request("/predict", _BODY)
        self.assertEqual(status, HTTPStatus.INTERNAL_SERVER_ERROR)
        self.assertNotIn("sensitive detail", json.dumps(body))

    def test_unknown_routes(self):
        self.assertEqual(self._request("/unknown")[0], HTTPStatus.NOT_FOUND)
        self.assertEqual(self._request("/unknown", {})[0], HTTPStatus.NOT_FOUND)


class CheckpointTests(unittest.TestCase):
    def test_saved_action_horizon(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            _write_export(path, "smolvla")
            config = json.loads((path / "config.json").read_text())
            config.update(chunk_size=50, n_action_steps=25)
            (path / "config.json").write_text(json.dumps(config))
            info = inspect_checkpoint(path)
            self.assertEqual((info.chunk_size, info.action_steps), (50, 25))

            for value in (0, True, 51):
                config["n_action_steps"] = value
                (path / "config.json").write_text(json.dumps(config))
                with self.subTest(value=value), self.assertRaises(ValueError):
                    inspect_checkpoint(path)

    def test_postprocess_each_action(self):
        policy = LeRobotPolicy.__new__(LeRobotPolicy)
        policy._torch = MagicMock()
        policy._torch.inference_mode.side_effect = nullcontext
        policy._camera_key = _CAMERA
        policy._image_size = (2, 1)
        policy._action_steps = 2
        policy._state_dim = len(_STATE)
        chunk = MagicMock()
        chunk.shape = (1, 2, len(_STATE))
        calls = []

        def action_at(index):
            if not isinstance(index[1], int):
                return chunk
            action = MagicMock()
            action.shape = (1, len(_STATE))
            output = action.__getitem__.return_value.detach.return_value.cpu.return_value
            output.tolist.return_value = [value * 10 + index[1] for value in _STATE]
            return action

        def postprocess(action):
            self.assertEqual(action.shape, (1, len(_STATE)))
            calls.append(action)
            return action

        chunk.__getitem__.side_effect = action_at
        policy._policy = SimpleNamespace(predict_action_chunk=lambda observation: chunk)
        policy._pre = lambda observation: observation
        policy._post = postprocess
        actions = policy.predict(_STATE, _RGB, 2, 1, _BODY["task"])
        self.assertEqual(actions, [[value * 10 + step for value in _STATE] for step in range(2)])
        self.assertEqual(len(calls), 2)

    def test_requires_complete_export(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "missing config.json"):
                LeRobotPolicy(Path(directory), "cpu", 1, 6, _CAMERA)

    def test_requires_exact_runtime(self):
        with TemporaryDirectory() as directory:
            _write_export(Path(directory))
            with patch("sim_worker.rollout.backend.version", return_value="0.6.0"):
                with self.assertRaisesRegex(ValueError, "0.6.1"):
                    LeRobotPolicy(Path(directory), "cpu", 1, 6, _CAMERA)

    def test_checkpoint_contract(self):
        config = _policy_config()
        _check_features(config, 6, _CAMERA, 1)
        _check_features(_policy_config("act"), 6, _CAMERA, 1)
        for key, value in [
            ("type", "diffusion"),
            ("n_obs_steps", 2),
            ("chunk_size", 0),
            ("robot_state_feature", SimpleNamespace(shape=(7,))),
            ("action_feature", SimpleNamespace(shape=(5,))),
            ("image_features", {_CAMERA: SimpleNamespace(shape=(1, 480, 640))}),
            ("input_features", {"observation.state": None, _CAMERA: None, "extra": None}),
        ]:
            with self.subTest(key=key):
                changed = copy.copy(config)
                setattr(changed, key, value)
                with self.assertRaises(ValueError):
                    _check_features(changed, 6, _CAMERA, 1)

    def test_act_temporal_ensemble(self):
        config = _policy_config("act")
        config.temporal_ensemble_coeff = 0.01
        with self.assertRaisesRegex(ValueError, "temporal ensembling"):
            _check_features(config, 6, _CAMERA, 1)

    def test_loads_full_policy(self):
        for policy_type in ("act", "smolvla"):
            with self.subTest(policy_type=policy_type), TemporaryDirectory() as directory:
                path = Path(directory)
                _write_export(path, policy_type)
                config = _policy_config(policy_type)
                policy_class = MagicMock()
                factory = SimpleNamespace(
                    get_policy_class=MagicMock(return_value=policy_class),
                    make_pre_post_processors=MagicMock(return_value=(MagicMock(), MagicMock())),
                )
                configs = SimpleNamespace(
                    PreTrainedConfig=SimpleNamespace(from_pretrained=MagicMock(return_value=config))
                )
                modules = {
                    "torch": MagicMock(),
                    "lerobot.configs": configs,
                    "lerobot.policies.factory": factory,
                }
                with (
                    patch.dict("sys.modules", modules),
                    patch("sim_worker.rollout.backend.version", return_value="0.6.1"),
                ):
                    policy = LeRobotPolicy(path, "cpu", 1, 6, _CAMERA)
                factory.get_policy_class.assert_called_once_with(policy_type)
                policy_class.from_pretrained.assert_called_once_with(
                    path, config=config, local_files_only=True, strict=True
                )
                if policy_type == "act":
                    self.assertIsNone(config.pretrained_backbone_weights)
                self.assertEqual(policy.model_id(), inspect_checkpoint(path).model_id)

    def test_requires_image_shape(self):
        policy = LeRobotPolicy.__new__(LeRobotPolicy)
        policy._image_size = (640, 360)
        policy._torch = MagicMock()
        with self.assertRaisesRegex(ValueError, "Camera dimensions"):
            policy.predict(_STATE, _RGB, 2, 1, _BODY["task"])
        policy._torch.frombuffer.assert_not_called()

    def test_fingerprints_runtime(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            _write_export(path)
            original = inspect_checkpoint(path)
            self.assertEqual(
                (original.policy_type, original.width, original.height), ("act", 640, 360)
            )
            self.assertEqual(
                (original.state_dim, original.action_dim, original.chunk_size), (6, 6, 100)
            )
            self.assertEqual(original.action_steps, 100)
            self.assertEqual(original, inspect_checkpoint(path))
            (path / "train_config.json").write_text('{"unrelated": true}')
            self.assertEqual(original, inspect_checkpoint(path))
            for filename in ("model.safetensors", "pre.safetensors", "post.safetensors"):
                target = path / filename
                previous = target.read_bytes()
                target.write_bytes(previous + b"modified")
                self.assertNotEqual(original.model_id, inspect_checkpoint(path).model_id)
                target.write_bytes(previous)

    def test_requires_statistics(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            _write_export(path)
            (path / "pre.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "missing pre.safetensors"):
                inspect_checkpoint(path)

    def test_rejects_external_stats(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            _write_export(path)
            processor = path / "policy_preprocessor.json"
            content = json.loads(processor.read_text())
            content["steps"][0]["state_file"] = "../outside.safetensors"
            processor.write_text(json.dumps(content))
            with self.assertRaisesRegex(ValueError, "inside the export"):
                inspect_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
