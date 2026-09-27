"""Private, explicitly saved OpenRouter settings; no provider verification or discovery."""

import json
import os
import re
import stat
import tempfile
import threading
import uuid
from pathlib import Path

from pydantic import BaseModel, ConfigDict, SecretStr, model_validator

_LOCK = threading.RLock()
FILENAME = "teaching-intelligence.json"
MODEL = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.:/-]{1,160}\Z")


class SettingsError(ValueError):
    """Fixed safe message only; no credential values or filesystem paths."""


class IntelligenceSettingsInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    openrouter_api_key: SecretStr
    voice_model: str

    @model_validator(mode="after")
    def valid(self):
        key = self.openrouter_api_key.get_secret_value()
        if (
            not 16 <= len(key) <= 4096
            or not key.isascii()
            or any(not 33 <= ord(c) <= 126 for c in key)
        ):
            raise ValueError("Invalid OpenRouter key")
        if (
            not MODEL.fullmatch(self.voice_model)
            or self.voice_model.startswith("typesafe/")
            or key in self.voice_model
        ):
            raise ValueError("Use an explicit chat model, not Jev")
        return self


def read(path: Path):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("Duplicate configuration field")
            value[key] = item
        return value

    if any(item.is_symlink() for item in (path, *path.parents)):
        raise SettingsError("Saved intelligence settings cannot use symbolic links.")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_size > 8192
            or (os.name == "posix" and (info.st_mode & 0o077 or info.st_uid != os.getuid()))
        ):
            raise SettingsError("Saved intelligence settings are not a private regular file.")
        raw = stream.read(8193)
    if len(raw) > 8192:
        raise SettingsError("Saved intelligence settings exceed limits.")
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "revision", "openrouter_api_key", "voice_model"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or not isinstance(value["revision"], str)
        or not re.fullmatch(r"[a-f0-9]{32}", value["revision"])
    ):
        raise SettingsError("Saved intelligence settings are invalid.")
    IntelligenceSettingsInput.model_validate(
        {key: value[key] for key in ("openrouter_api_key", "voice_model")}
    )
    return value


def status(data_dir: Path):
    path = data_dir / FILENAME
    with _LOCK:
        try:
            value = read(path)
        except FileNotFoundError:
            return {
                "saved": False,
                "revision": None,
                "voice_model": None,
                "message": "No app-managed OpenRouter settings saved.",
            }
        except OSError, ValueError, UnicodeError, RecursionError:
            return {
                "saved": False,
                "revision": None,
                "voice_model": None,
                "message": "Saved settings could not be read securely. Replace them explicitly.",
            }
    return {
        "saved": True,
        "revision": value["revision"],
        "voice_model": value["voice_model"],
        "message": (
            "Saved locally; provider access is unverified. Voice services must "
            "explicitly load this file."
        ),
    }


def save(data_dir: Path, value: IntelligenceSettingsInput):
    path = data_dir / FILENAME
    with _LOCK:
        if any(item.is_symlink() for item in (path, *path.parents)):
            raise SettingsError("Saved intelligence settings cannot use symbolic links.")
        if path.exists() and not path.is_file():
            raise SettingsError("Saved intelligence settings must be a regular file.")
        document = {
            "schema_version": 1,
            "revision": uuid.uuid4().hex,
            "openrouter_api_key": value.openrouter_api_key.get_secret_value(),
            "voice_model": value.voice_model,
        }
        fd, temporary = tempfile.mkstemp(prefix=".teaching-intelligence-", dir=data_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                os.chmod(temporary, 0o600)
                json.dump(document, stream, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            if os.name == "posix":
                directory = os.open(data_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return status(data_dir)


def remove(data_dir: Path):
    path = data_dir / FILENAME
    with _LOCK:
        if any(item.is_symlink() for item in (path, *path.parents)):
            raise SettingsError("Saved intelligence settings cannot use symbolic links.")
        if path.exists() and not path.is_file():
            raise SettingsError("Saved intelligence settings must be a regular file.")
        path.unlink(missing_ok=True)
    return status(data_dir)
