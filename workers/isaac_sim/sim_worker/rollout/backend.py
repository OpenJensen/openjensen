"""Policy implementations; heavy dependencies stay on the inference VM."""

from importlib.metadata import version
from pathlib import Path
from typing import Protocol

from .checkpoint import inspect_checkpoint

_LEROBOT_VERSION = "0.6.1"
_RGB_CHANNELS = 3
_UINT8_MAX = 255
_STATE_KEY = "observation.state"
_ACTION_KEY = "action"
_ACT = "act"
_SMOLVLA = "smolvla"


class Backend(Protocol):
    def reset(self) -> None: ...

    def predict(
        self, state: list[float], rgb: bytes, width: int, height: int, task: str
    ) -> list[list[float]]: ...


class MockPolicy:
    """Hold observed positions to exercise transport without an ML runtime."""

    def __init__(self, action_steps: int = 1):
        self._action_steps = action_steps

    def reset(self) -> None:
        pass

    def predict(
        self, state: list[float], rgb: bytes, width: int, height: int, task: str
    ) -> list[list[float]]:
        return [list(state) for _ in range(self._action_steps)]


class LeRobotPolicy:
    """Load a complete, local ACT or SmolVLA export and its saved processors."""

    def __init__(
        self,
        checkpoint: Path,
        device: str,
        action_steps: int,
        state_dim: int,
        camera_key: str,
    ):
        packed = (checkpoint / "model.fbq").exists() or (checkpoint / "encoding.json").exists()
        info = inspect_checkpoint(checkpoint)
        if packed != (
            (checkpoint / "model.fbq").exists() or (checkpoint / "encoding.json").exists()
        ):
            raise ValueError("Checkpoint format changed during inspection")
        if packed and device != "cpu":
            raise ValueError("Native packed ACT serving is verified for CPU only")
        if packed and (
            state_dim != info.state_dim
            or camera_key != info.camera_key
            or not 1 <= action_steps <= info.chunk_size
        ):
            raise ValueError("Packed checkpoint features differ from the serving request")
        if version("lerobot") != _LEROBOT_VERSION:
            raise ValueError(f"Install lerobot[smolvla]=={_LEROBOT_VERSION}")

        # Isaac and CPU mock tests never import the policy runtime.
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")

        if packed:
            from firebird_quant.native_consumer import load_packed_act

            self._policy, config, self._pre, self._post, loaded_id = load_packed_act(
                checkpoint.absolute(), device=device, expected_model_id=info.model_id
            )
            if loaded_id != info.model_id:
                raise ValueError("Packed model identity changed during loading")
        else:
            config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
        _check_features(config, state_dim, camera_key, action_steps)
        config.device = device
        if config.type == _ACT:
            # Full checkpoint weights include the trained vision backbone.
            config.pretrained_backbone_weights = None
        self._torch = torch
        self._model_id = info.model_id
        self._image_size = (info.width, info.height)
        self._camera_key = camera_key
        self._action_steps = action_steps
        self._state_dim = state_dim
        if not packed:
            self._policy = (
                get_policy_class(config.type)
                .from_pretrained(checkpoint, config=config, local_files_only=True, strict=True)
                .to(device)
                .eval()
            )
            self._pre, self._post = make_pre_post_processors(
                config,
                pretrained_path=str(checkpoint),
                preprocessor_overrides={"device_processor": {"device": device}},
            )
        # Bind loaded config/processors/contract to the pre-load identity.
        if inspect_checkpoint(checkpoint) != info:
            raise ValueError("Checkpoint changed during native policy loading")
        self.reset()

    def model_id(self) -> str:
        return self._model_id

    def reset(self) -> None:
        self._policy.reset()
        self._pre.reset()
        self._post.reset()

    def predict(
        self, state: list[float], rgb: bytes, width: int, height: int, task: str
    ) -> list[list[float]]:
        if (width, height) != self._image_size:
            raise ValueError(f"Camera dimensions must match checkpoint: {self._image_size}")
        torch = self._torch
        pixels = torch.frombuffer(bytearray(rgb), dtype=torch.uint8)
        pixels = pixels.reshape(height, width, _RGB_CHANNELS).permute(2, 0, 1)
        observation = {
            _STATE_KEY: torch.tensor(state, dtype=torch.float32),
            self._camera_key: pixels.to(torch.float32) / _UINT8_MAX,
            "task": task,
        }

        # Saved processors own normalization and policy-specific transforms.
        with torch.inference_mode():
            batch = self._pre(observation)
            chunk = self._policy.predict_action_chunk(batch)
            shape = tuple(chunk.shape)
            if len(shape) != 3 or shape[0] != 1 or shape[2] != self._state_dim:
                raise ValueError("Policy chunk shape must be (1, steps, state-dim)")
            if shape[1] < self._action_steps:
                raise ValueError("Policy returned too few action steps")

            # LeRobot postprocessors accept one batched action at a time.
            actions = []
            for index in range(self._action_steps):
                action = self._post(chunk[:, index, :])
                if tuple(action.shape) != (1, self._state_dim):
                    raise ValueError("Policy action shape must be (1, state-dim)")
                actions.append(action[0].detach().cpu().tolist())
        return actions


def _check_features(config, state_dim, camera_key, action_steps):
    if config.type not in {_ACT, _SMOLVLA}:
        raise ValueError("Only ACT and SmolVLA checkpoint exports are supported")
    if config.type == _ACT and getattr(config, "temporal_ensemble_coeff", None) is not None:
        raise ValueError("ACT temporal ensembling is not supported by chunk inference")
    if set(config.input_features) != {_STATE_KEY, camera_key}:
        raise ValueError("Checkpoint must expect one camera and observation.state")
    if set(config.output_features) != {_ACTION_KEY}:
        raise ValueError("Checkpoint must output only joint actions")
    if set(config.image_features) != {camera_key}:
        raise ValueError("Configured camera must match the checkpoint camera")
    state = config.robot_state_feature
    action = config.action_feature
    if state is None or action is None:
        raise ValueError("Checkpoint must declare state and action features")
    if tuple(state.shape) != (state_dim,) or tuple(action.shape) != (state_dim,):
        raise ValueError("Checkpoint state/action dimensions must match state-dim")
    image_shape = config.image_features[camera_key].shape
    if len(image_shape) != 3 or image_shape[0] != _RGB_CHANNELS:
        raise ValueError("Checkpoint camera must contain RGB channels")
    if config.n_obs_steps != 1:
        raise ValueError("Only single-observation checkpoints are supported")
    if not 1 <= action_steps <= config.chunk_size:
        raise ValueError("action-steps exceeds the checkpoint chunk size")
