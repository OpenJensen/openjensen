"""Optional app-owned OpenRouter file. Existing operator environment wins."""

from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

CONFIG_ENV = "FIREBIRD_TEACHING_INTELLIGENCE_FILE"
MODEL = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.:/-]{1,160}\Z")


@dataclass(frozen=True)
class IntelligenceConfig:
    key: str = field(repr=False)
    voice_model: str
    key_source: str
    model_source: str
    revision: str | None


def valid_key(value: object) -> str:
    if not isinstance(value, str) or not 16 <= len(value) <= 4096 or not value.isascii():
        raise ValueError("OpenRouter key is missing or invalid")
    if any(not 33 <= ord(c) <= 126 for c in value):
        raise ValueError("OpenRouter key is missing or invalid")
    return value


def valid_model(value: object) -> str:
    if not isinstance(value, str) or not MODEL.fullmatch(value):
        raise ValueError("Configure an explicit provider/model voice model")
    if value.startswith(("typesafe/", "~typesafe/")):
        raise ValueError("Jev is a typed decision provider, not the voice chat model")
    return value


def read_saved(path: Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate configuration field")
            result[key] = value
        return result

    if not path.is_absolute() or any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("Intelligence configuration must use a real absolute path")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_size > 8192
            or (os.name == "posix" and (info.st_mode & 0o077 or info.st_uid != os.getuid()))
        ):
            raise ValueError("Intelligence configuration is not a private regular file")
        raw = stream.read(8193)
        if len(raw) > 8192:
            raise ValueError("Intelligence configuration exceeds limits")
    document = json.loads(raw.decode("utf-8"), object_pairs_hook=unique)
    if (
        not isinstance(document, dict)
        or set(document) != {"schema_version", "revision", "openrouter_api_key", "voice_model"}
        or type(document["schema_version"]) is not int
        or document["schema_version"] != 1
        or not isinstance(document["revision"], str)
        or not re.fullmatch(r"[a-f0-9]{32}", document["revision"])
    ):
        raise ValueError("Invalid intelligence configuration")
    valid_key(document["openrouter_api_key"])
    valid_model(document["voice_model"])
    if document["openrouter_api_key"] in document["voice_model"]:
        raise ValueError("Voice model cannot contain credentials")
    return document


def load_config(environ: Mapping[str, str] | None = None) -> IntelligenceConfig:
    source = os.environ if environ is None else environ
    key, model = source.get("OPENROUTER_API_KEY", ""), source.get("FIREBIRD_VOICE_MODEL", "")
    saved: dict = {}
    # Never discover dotenv files or overwrite an operator's configured environment.
    if source.get(CONFIG_ENV) and (not key or not model):
        try:
            saved = read_saved(Path(source[CONFIG_ENV]))
        except (OSError, ValueError, UnicodeError, RecursionError):
            raise ValueError(
                "Managed intelligence configuration could not be read securely"
            ) from None
    effective_key = valid_key(key or saved.get("openrouter_api_key", ""))
    effective_model = model or saved.get("voice_model", "")
    if effective_model and effective_key in effective_model:
        raise ValueError("Voice model cannot contain credentials")
    return IntelligenceConfig(
        effective_key,
        valid_model(model or saved.get("voice_model", ""))
        if model or saved.get("voice_model")
        else "",
        "environment" if key else "managed_file",
        "environment" if model else "managed_file" if saved.get("voice_model") else "missing",
        saved.get("revision"),
    )


def voice_environment() -> dict[str, str]:
    source = dict(os.environ)
    # Without explicit opt-in, preserve existing missing-variable diagnostics.
    if source.get(CONFIG_ENV):
        configuration = load_config(source)
        if not source.get("OPENROUTER_API_KEY"):
            source["OPENROUTER_API_KEY"] = configuration.key
        if not source.get("FIREBIRD_VOICE_MODEL"):
            source["FIREBIRD_VOICE_MODEL"] = configuration.voice_model
    return source
