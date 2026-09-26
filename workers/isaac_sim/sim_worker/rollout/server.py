"""Bounded JSON policy transport for one simulation client on a private network."""

import argparse
import base64
import binascii
import json
import logging
import math
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from .backend import Backend, LeRobotPolicy, MockPolicy

_PORT = 8080
_MAX_BODY = 16 * 1024 * 1024
_MAX_PIXELS = 4096 * 2160
_MAX_TEXT = 4096
_MAX_ID = 256
_MAX_EPISODES = 4096
_SOCKET_TIMEOUT = 30
_RGB_CHANNELS = 3
_JSON_TYPE = "application/json"
_RESET_KEYS = {"episode_id", "model_id"}
_PREDICT_KEYS = _RESET_KEYS | {"step", "sim_time", "task", "state", "image"}
_IMAGE_KEYS = {"width", "height", "encoding", "data"}


class _RequestError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self._status = status

    def status(self):
        return self._status


class _Session:
    def __init__(self, backend, model_id, state_dim, action_steps):
        self._backend = backend
        self._model_id = model_id
        self._state_dim = state_dim
        self._action_steps = action_steps
        self._episode = None
        self._seen = set()
        self._step = -1
        self._sim_time = -1.0

    def health(self):
        return {"status": "ready", "model_id": self._model_id}

    def dispatch(self, path, data):
        if path not in {"/reset", "/predict"}:
            raise _RequestError(HTTPStatus.NOT_FOUND, "Unknown endpoint")
        _keys(data, _RESET_KEYS if path == "/reset" else _PREDICT_KEYS)
        episode = _text(data["episode_id"], _MAX_ID)
        if _text(data["model_id"], _MAX_ID) != self._model_id:
            raise _RequestError(HTTPStatus.CONFLICT, "Model identity mismatch")
        if path == "/reset":
            return self._reset(episode)
        return self._predict(episode, data)

    def _reset(self, episode):
        if episode in self._seen:
            raise _RequestError(HTTPStatus.CONFLICT, "Use a new episode ID for reset")
        if len(self._seen) >= _MAX_EPISODES:
            raise _RequestError(HTTPStatus.SERVICE_UNAVAILABLE, "Restart the policy server")

        self._episode = None
        self._backend.reset()
        self._episode = episode
        self._seen.add(episode)
        self._step = -1
        self._sim_time = -1.0
        return {"episode_id": episode, "model_id": self._model_id}

    def _predict(self, episode, data):
        if episode != self._episode:
            raise _RequestError(HTTPStatus.CONFLICT, "Reset this episode before prediction")
        step = _integer(data["step"], 0)
        sim_time = _number(data["sim_time"])
        if sim_time < 0:
            raise ValueError("Simulation time cannot be negative")
        if self._step == -1 and step != 0:
            raise _RequestError(HTTPStatus.CONFLICT, "The first step must be zero")
        if step <= self._step or sim_time <= self._sim_time:
            raise _RequestError(HTTPStatus.CONFLICT, "Stale observation")
        state = _vector(data["state"], self._state_dim)
        task = _text(data["task"], _MAX_TEXT)
        image = data["image"]
        _keys(image, _IMAGE_KEYS)
        width = _integer(image["width"], 1)
        height = _integer(image["height"], 1)
        if width * height > _MAX_PIXELS or image["encoding"] != "rgb8":
            raise ValueError("Image must be bounded RGB8")
        if not isinstance(image["data"], str):
            raise ValueError("Image data must be base64 RGB bytes")
        try:
            rgb = base64.b64decode(image["data"], validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError("Image data must be base64 RGB bytes") from error
        if len(rgb) != width * height * _RGB_CHANNELS:
            raise ValueError("Image byte count does not match dimensions")

        # HTTPServer serializes reset and prediction, including backend state.
        try:
            actions = self._backend.predict(state, rgb, width, height, task)
            if not isinstance(actions, list) or len(actions) != self._action_steps:
                raise ValueError("Unexpected action chunk length")
            actions = [_vector(action, self._state_dim) for action in actions]
        except Exception:
            self._episode = None
            logging.exception("Policy inference failed; reset required")
            raise _RequestError(
                HTTPStatus.INTERNAL_SERVER_ERROR, "Policy inference failed; reset required"
            ) from None
        self._step = step
        self._sim_time = sim_time
        return {
            "episode_id": episode,
            "step": step,
            "model_id": self._model_id,
            "actions": actions,
        }


def create_server(
    address: tuple[str, int],
    backend: Backend,
    model_id: str,
    state_dim: int = 6,
    action_steps: int = 1,
) -> HTTPServer:
    """Create a serialized server; callers own its lifetime."""
    session = _Session(
        backend, _text(model_id, _MAX_ID), _integer(state_dim, 1), _integer(action_steps, 1)
    )

    class _Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(_SOCKET_TIMEOUT)

        def do_GET(self):
            if self.path != "/health":
                self._reply(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint"})
                return
            self._reply(HTTPStatus.OK, session.health())

        def do_POST(self):
            try:
                data = self._read_json()
                result = session.dispatch(self.path, data)
                self._reply(HTTPStatus.OK, result)
            except _RequestError as error:
                self._reply(error.status(), {"error": str(error)})
            except (ValueError, UnicodeError, RecursionError):
                self._reply(HTTPStatus.BAD_REQUEST, {"error": "Invalid request body"})
            except TimeoutError:
                self._reply(HTTPStatus.REQUEST_TIMEOUT, {"error": "Request timed out"})
            except Exception:
                logging.exception("Policy request failed")
                self._reply(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Policy request failed"})

        def _read_json(self):
            if self.headers.get_content_type() != _JSON_TYPE:
                raise _RequestError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Use application/json")
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1 or self.headers.get("Transfer-Encoding"):
                raise ValueError("Exactly one Content-Length is required")
            length = int(lengths[0])
            if length > _MAX_BODY:
                raise _RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Request is too large")
            if length <= 0:
                raise ValueError("Request body is empty")
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("Request body is incomplete")
            return json.loads(raw, object_pairs_hook=_unique, parse_constant=_nonfinite)

        def _reply(self, status, value):
            body = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", _JSON_TYPE)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

        def log_message(self, format, *args):
            logging.info("%s", format % args)

    return HTTPServer(address, _Handler)


def _keys(value, expected):
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("Unexpected request fields")


def _text(value, limit):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError("Invalid text field")
    return value


def _integer(value, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError("Invalid integer field")
    return value


def _number(value):
    if type(value) not in (int, float):
        raise ValueError("Expected a finite number")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError("Expected a finite number") from None
    if not math.isfinite(number):
        raise ValueError("Expected a finite number")
    return number


def _vector(value, dimension):
    if not isinstance(value, list) or len(value) != dimension:
        raise ValueError("Unexpected joint vector shape")
    return [_number(item) for item in value]


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError("Nonfinite JSON number")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=_PORT)
    parser.add_argument("--backend", choices=("mock", "lerobot"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--action-steps", type=int, default=1)
    parser.add_argument("--state-dim", type=int, default=6)
    parser.add_argument("--camera-key", default="observation.images.front")
    args = parser.parse_args()
    if args.backend == "lerobot" and args.checkpoint is None:
        parser.error("--checkpoint is required for the lerobot backend")
    if args.state_dim < 1 or args.action_steps < 1:
        parser.error("--state-dim and --action-steps must be positive")

    logging.basicConfig(level=logging.INFO)
    backend = MockPolicy(args.action_steps)
    if args.backend == "lerobot":
        backend = LeRobotPolicy(
            args.checkpoint, args.device, args.action_steps, args.state_dim, args.camera_key
        )
        if args.model_id != backend.model_id():
            parser.error("--model-id must match the checkpoint fingerprint")
    with create_server(
        (args.host, args.port), backend, args.model_id, args.state_dim, args.action_steps
    ) as server:
        logging.info("Policy ready: %s on %s:%s", args.model_id, args.host, args.port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
