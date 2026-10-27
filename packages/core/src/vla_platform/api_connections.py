"""Private provider keys, shared by installed integrations and never returned to HTTP clients."""

import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Literal

from filelock import FileLock
from pydantic import BaseModel, ConfigDict, SecretStr, field_validator

ApiProvider = Literal["openai", "gemini"]
NAMES = {"openai": "OpenAI", "gemini": "Gemini Omni"}
ENVIRONMENT = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}


class ApiKeyError(ValueError):
    """Only fixed, non-secret messages may cross the API boundary."""


def valid_key(value: str) -> str:
    if not isinstance(value, str):
        raise ApiKeyError("Enter a valid API key.")
    value = value.strip()
    if not 16 <= len(value) <= 4096 or any(not 33 <= ord(c) <= 126 for c in value):
        raise ApiKeyError("Enter a valid API key with no spaces or control characters.")
    return value


class ApiKeyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_key: SecretStr

    @field_validator("api_key")
    @classmethod
    def key_format(cls, value: SecretStr) -> SecretStr:
        return SecretStr(valid_key(value.get_secret_value()))


class ApiConnectionStatus(BaseModel):
    id: ApiProvider
    name: str
    configured: bool = False
    source: Literal["saved", "environment", "profile"] | None = None
    message: str | None = None


class ApiConnectionsStatus(BaseModel):
    providers: list[ApiConnectionStatus]


class ApiConnections:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.directory = self.data_dir / "api-credentials"

    def _path(self, ident: ApiProvider) -> Path:
        if ident not in NAMES:
            raise ApiKeyError("Unknown API provider.")
        if self.directory.is_symlink():
            raise ApiKeyError("The saved API keys need administrator attention.")
        return self.directory / (ident + ".json")

    def saved_key(self, ident: ApiProvider) -> str | None:
        path = self._path(ident)
        try:
            descriptor = os.open(
                path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
            )
        except FileNotFoundError:
            return None
        try:
            with os.fdopen(descriptor, "r") as stream:
                info = os.fstat(stream.fileno())
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_size > 8192
                    or (os.name == "posix" and (info.st_mode & 0o077 or info.st_uid != os.getuid()))
                ):
                    raise ValueError
                value = json.loads(stream.read(8193))
            if (
                not isinstance(value, dict)
                or set(value) != {"version", "api_key"}
                or value["version"] != 1
            ):
                raise ValueError
            return valid_key(value["api_key"])
        except OSError, ValueError, TypeError, KeyError:
            raise ApiKeyError(
                "The saved API key could not be read securely. Save it again."
            ) from None

    def _profile_key(self, ident: ApiProvider) -> str | None:
        if ident != "openai":
            return None
        try:
            from vla_platform.agent_control.settings import Profiles

            profiles = Profiles(self.data_dir).read()
            for key in sorted(profiles, key=lambda item: (item != "openai", item)):
                profile = profiles[key]
                if profile.get("provider") == "openai_responses" and profile.get("api_key"):
                    return valid_key(profile["api_key"])
        except ImportError, OSError, ValueError:
            return None
        return None

    def _resolve(self, ident: ApiProvider) -> tuple[str | None, str | None]:
        saved = self.saved_key(ident)
        if saved:
            return saved, "saved"
        environment = os.environ.get(ENVIRONMENT[ident], "").strip()
        if environment:
            return valid_key(environment), "environment"
        profile = self._profile_key(ident)
        return (profile, "profile") if profile else (None, None)

    def key(self, ident: ApiProvider) -> str | None:
        """Internal use only, resolved afresh so key changes need no server restart."""
        return self._resolve(ident)[0]

    def provider(self, ident: ApiProvider) -> ApiConnectionStatus:
        try:
            key, source = self._resolve(ident)
            return ApiConnectionStatus(
                id=ident, name=NAMES[ident], configured=bool(key), source=source
            )
        except OSError, ApiKeyError:
            return ApiConnectionStatus(
                id=ident,
                name=NAMES[ident],
                message="The saved API key needs administrator attention.",
            )

    def status(self) -> ApiConnectionsStatus:
        return ApiConnectionsStatus(providers=[self.provider(ident) for ident in NAMES])

    def configured(self) -> dict[str, bool]:
        return {item.id: item.configured for item in self.status().providers}

    def save(self, ident: ApiProvider, key: SecretStr) -> ApiConnectionStatus:
        value = valid_key(key.get_secret_value())
        path = self._path(ident)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        with FileLock(self.directory / ".write.lock", timeout=5):
            if path.is_symlink():
                raise ApiKeyError("The saved API key needs administrator attention.")
            descriptor, temporary = tempfile.mkstemp(prefix=".api-key-", dir=self.directory)
            try:
                with os.fdopen(descriptor, "w") as stream:
                    json.dump({"version": 1, "api_key": value}, stream)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                Path(temporary).unlink(missing_ok=True)
        return self.provider(ident)

    def disconnect(self, ident: ApiProvider) -> ApiConnectionStatus:
        path = self._path(ident)
        if self.directory.exists():
            with FileLock(self.directory / ".write.lock", timeout=5):
                if path.is_symlink():
                    raise ApiKeyError("The saved API key needs administrator attention.")
                path.unlink(missing_ok=True)
        return self.provider(ident)
