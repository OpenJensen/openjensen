"""Adopt an existing pinned ACT interpreter; HTTP callers never supply executable paths."""

import json
import os
import stat
import tempfile
from pathlib import Path

REVISION = "e595b7902714ba51f91e47523f66f89c5181b649"
BINDING = "macos-act-runtime.json"


def installed_python():
    return Path.home() / ".local/share/jensen/policy-runtimes/act-e595b790/bin/python"


def binding_python(data_dir):
    path = Path(data_dir) / "local-environments" / BINDING
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
            or not 0 < info.st_size <= 8192
        ):
            raise ValueError("The ACT interpreter binding must be a private owned file")
        value = json.loads(stream.read(8193))
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "python", "revision"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["revision"] != REVISION
        or not isinstance(value["python"], str)
    ):
        raise ValueError("The ACT interpreter binding is invalid")
    python = Path(value["python"])
    if (
        not python.is_absolute()
        or ".." in python.parts
        or not python.is_file()
        or not os.access(python, os.X_OK)
    ):
        raise ValueError("The installed ACT interpreter is unavailable")
    return python


def configure_existing(data_dir):
    """Persist only the fixed known installation, preserving an existing operator binding."""
    directory = Path(data_dir) / "local-environments"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if directory.is_symlink():
        raise ValueError("The local environment configuration cannot use a symbolic link")
    path = directory / BINDING
    if path.exists() or path.is_symlink():
        return binding_python(data_dir)
    python = installed_python()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("The existing ACT runtime is not installed; no packages were downloaded")
    fd, temporary = tempfile.mkstemp(prefix=".macos-act-", dir=directory)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump({"schema_version": 1, "python": str(python), "revision": REVISION}, stream)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return python


def validate_training_recipe(recipe, features):
    """Initial Apple GPU support stays inside the measured single-camera ACT envelope."""
    cameras = recipe.get("camera_keys") or [recipe.get("camera_key")]
    prediction = recipe.get("prediction_horizon", recipe.get("chunk_size", 100))
    execution = recipe.get("execution_horizon", recipe.get("chunk_size", 100))
    if (
        type(recipe.get("batch_size", 1)) is not int
        or recipe.get("batch_size", 1) != 1
        or type(recipe.get("gradient_accumulation_steps", 1)) is not int
        or recipe.get("gradient_accumulation_steps", 1) != 1
        or not isinstance(cameras, list)
        or len(cameras) != 1
        or not isinstance(cameras[0], str)
        or type(prediction) is not int
        or not 1 <= prediction <= 32
        or type(execution) is not int
        or not 1 <= execution <= 8
        or execution > prediction
        or not isinstance(features, dict)
    ):
        raise ValueError(
            "Initial Apple GPU ACT training uses batch 1, one camera, "
            "prediction horizon at most 32 and execution horizon at most 8."
        )
    feature = features.get(cameras[0], {})
    shape = feature.get("shape") if isinstance(feature, dict) else None
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or shape[2] != 3
        or any(type(n) is not int or n < 1 for n in shape)
        or shape[0] > 480
        or shape[1] > 640
    ):
        raise ValueError("Initial Apple GPU ACT training supports one RGB camera up to 640 by 480.")
