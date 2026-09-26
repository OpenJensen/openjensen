import re
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from sim_worker.contracts import API_VERSION, BUILTIN_SCENE, RunSpec


_MAX_MANIFEST_BYTES = 64 * 1024
_MAX_WIDTH = 1920
_MAX_HEIGHT = 1080
_MAX_FPS = 60
_MAX_SECONDS = 600
_CHROMA_ALIGNMENT = 2
_USD_SUFFIXES = {".usd", ".usda", ".usdc"}


class _Loader(yaml.SafeLoader):
    """Reject aliases and duplicate keys rather than silently changing a run."""

    def compose_node(self, parent, index):
        if self.check_event(yaml.AliasEvent):
            raise ValueError("YAML aliases are unsupported")
        return super().compose_node(parent, index)

    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in result:
                raise ValueError("Manifest keys must be unique strings")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def _fields(value: object, keys: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} requires exactly: {', '.join(sorted(keys))}")
    return value


def _integer(value: object, maximum: int, label: str) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{label} must be an integer from 1 to {maximum}")
    return value


def _scene_path(uri: object, manifest: Path) -> str:
    if not isinstance(uri, str):
        raise ValueError("scene.uri must be a string")
    if uri == BUILTIN_SCENE:
        return uri

    target = urlsplit(uri)
    location = Path(uri)
    if (target.scheme or target.netloc or target.query or target.fragment
            or location.suffix.lower() not in _USD_SUFFIXES):
        raise ValueError("scene.uri must be builtin:falling-cube or a USD file path")
    if location.is_absolute():
        return uri

    # Resolve after transfer so a bundle works on both client and container.
    location = (manifest.parent / location).resolve()
    if not location.is_file():
        raise ValueError(f"Scene file not found: {location}")
    return str(location)


def load(path: Path) -> RunSpec:
    with path.open("rb") as source:
        contents = source.read(_MAX_MANIFEST_BYTES + 1)
    if len(contents) > _MAX_MANIFEST_BYTES:
        raise ValueError("Manifest exceeds 64 KiB")

    data = _fields(yaml.load(contents, Loader=_Loader),
                   {"api_version", "scene", "capture", "outputs"}, "manifest")
    if data["api_version"] != API_VERSION:
        raise ValueError(f"api_version must be {API_VERSION}")

    scene = _fields(data["scene"], {"uri", "camera"}, "scene")
    capture = _fields(data["capture"], {"width", "height", "fps", "frames"}, "capture")
    outputs = _fields(data["outputs"], {"uri"}, "outputs")
    uri = _scene_path(scene["uri"], path)

    camera = scene["camera"]
    if not isinstance(camera, str) or not re.fullmatch(r"(/[A-Za-z_][A-Za-z_0-9]*)+", camera):
        raise ValueError("scene.camera must be an absolute USD prim path")

    width = _integer(capture["width"], _MAX_WIDTH, "width")
    height = _integer(capture["height"], _MAX_HEIGHT, "height")
    if width % _CHROMA_ALIGNMENT or height % _CHROMA_ALIGNMENT:
        raise ValueError("H.264 dimensions must be even")
    fps = _integer(capture["fps"], _MAX_FPS, "fps")
    frames = _integer(capture["frames"], _MAX_SECONDS * fps, "frames")

    output = outputs["uri"]
    if not isinstance(output, str):
        raise ValueError("outputs.uri must be a GCS prefix")
    target = urlsplit(output)
    if (target.scheme != "gs" or not re.fullmatch(r"[a-z0-9][a-z0-9._-]+[a-z0-9]", target.netloc)
            or target.query or target.fragment):
        raise ValueError("outputs.uri must be gs://bucket/optional-prefix")

    return RunSpec(uri, camera, width, height, fps, frames, output.rstrip("/"))
