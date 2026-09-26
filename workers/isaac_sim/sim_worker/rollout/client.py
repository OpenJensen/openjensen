import base64
import json
import math
import time
from http import HTTPStatus
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from sim_worker.rollout.config import check_endpoint
from sim_worker.rollout.contracts import RGB_CHANNELS, ActionChunk, Observation

_MAX_RESPONSE_BYTES = 1024 * 1024
_MAX_ACTIONS = 1000
_READY_POLL_SECONDS = 2.0


class RemotePolicy:
    def __init__(self, endpoint: str, model_id: str, task: str, timeout_seconds: float):
        self._endpoint = check_endpoint(endpoint)
        self._model_id = model_id
        self._task = task
        self._timeout = timeout_seconds
        self._episode = None
        # Private VPC traffic should not inherit a workstation's HTTP proxy.
        self._http = build_opener(ProxyHandler({}))

    def _request(self, path, payload=None, timeout=None):
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        request = Request(
            self._endpoint + path, data=data, headers={"Content-Type": "application/json"}
        )
        with self._http.open(request, timeout=timeout or self._timeout) as response:
            if response.status != HTTPStatus.OK:
                raise RuntimeError(f"Policy server returned HTTP {response.status}")
            contents = response.read(_MAX_RESPONSE_BYTES + 1)
        if len(contents) > _MAX_RESPONSE_BYTES:
            raise ValueError("Policy response exceeds limit")
        result = json.loads(contents)
        if not isinstance(result, dict):
            raise ValueError("Policy response must be an object")
        if result.get("model_id") != self._model_id:
            raise ValueError("Policy model identity differs from manifest")
        return result

    def wait_ready(self, timeout_seconds: float) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Readiness timeout must be positive and finite")
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Policy server did not become ready")
            try:
                result = self._request("/health", timeout=min(self._timeout, remaining))
                if result.get("status") != "ready":
                    raise ValueError("Policy server is not ready")
                return
            except HTTPError:
                raise
            except (URLError, TimeoutError, ConnectionError):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Policy server did not become ready") from None
                time.sleep(min(_READY_POLL_SECONDS, remaining))

    def reset(self, episode_id: str) -> None:
        self._episode = None
        result = self._request("/reset", {"episode_id": episode_id, "model_id": self._model_id})
        if result.get("episode_id") != episode_id:
            raise ValueError("Policy reset returned a different episode")
        self._episode = episode_id

    def predict(self, observation: Observation) -> ActionChunk:
        if observation.episode_id != self._episode:
            raise ValueError("Reset policy before sending a new episode")
        frame = observation.frame
        if len(frame.rgb) != frame.width * frame.height * RGB_CHANNELS:
            raise ValueError("Invalid RGB frame size")
        result = self._request(
            "/predict",
            {
                "episode_id": observation.episode_id,
                "step": observation.step,
                "sim_time": observation.sim_time,
                "model_id": self._model_id,
                "task": self._task,
                "state": observation.state,
                "image": {
                    "width": frame.width,
                    "height": frame.height,
                    "encoding": "rgb8",
                    "data": base64.b64encode(frame.rgb).decode("ascii"),
                },
            },
        )
        if (
            result.get("episode_id") != observation.episode_id
            or type(result.get("step")) is not int
            or result["step"] != observation.step
        ):
            raise ValueError("Stale or mismatched policy response")
        actions = result.get("actions")
        if not isinstance(actions, list) or not 1 <= len(actions) <= _MAX_ACTIONS:
            raise ValueError("Policy returned an invalid action chunk")
        for action in actions:
            if (
                not isinstance(action, list)
                or len(action) != len(observation.state)
                or any(
                    type(value) not in (int, float) or not math.isfinite(value) for value in action
                )
            ):
                raise ValueError("Policy action has invalid dimensions or nonfinite values")
        return ActionChunk(
            observation.episode_id,
            observation.step,
            self._model_id,
            tuple(tuple(float(value) for value in action) for action in actions),
        )
