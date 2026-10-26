"""Private provider profiles. Only fixed transports are executable."""

import json
import os
import re
import stat
import tempfile
from pathlib import Path
from uuid import uuid4

from .contracts import ProviderInput, ProviderStatus


class ProfileError(ValueError):
    pass


class Profiles:
    def __init__(self, directory: Path):
        self.path = directory / "agent-providers.json"

    def read(self):
        if any(p.is_symlink() for p in (self.path, *self.path.parents)):
            raise ProfileError("Provider profiles cannot use symbolic links")
        try:
            fd = os.open(
                self.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
            )
        except FileNotFoundError:
            return {}
        with os.fdopen(fd, "rb") as f:
            info = os.fstat(f.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_size > 65536
                or (os.name == "posix" and (info.st_mode & 0o077 or info.st_uid != os.getuid()))
            ):
                raise ProfileError("Provider profiles are not a private regular file")
            raw = f.read(65537)
        try:
            value = json.loads(raw)
            if not isinstance(value, dict) or len(value) > 16:
                raise ValueError
            for key, item in value.items():
                if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", key) or key == "mock":
                    raise ValueError
                ProviderInput.model_validate(
                    {k: item[k] for k in ("provider", "model", "reasoning_effort", "api_key")}
                )
                if not re.fullmatch(r"[a-f0-9]{32}", item["revision"]):
                    raise ValueError
                if type(item.get("checked")) is not bool:
                    raise ValueError
                capabilities = item.get("capabilities")
                allowed = {
                    "text",
                    "images",
                    "tools",
                    "tool_continuation",
                    "network",
                    "joint_correction",
                }
                if not isinstance(capabilities, list) or any(
                    c not in allowed for c in capabilities
                ):
                    raise ValueError
        except ValueError, KeyError, TypeError:
            raise ProfileError("Provider profiles are invalid") from None
        return value

    def write(self, value):
        self.read()  # Security check also applies before replacement/removal.
        fd, tmp = tempfile.mkstemp(prefix=".agent-providers-", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w") as f:
                os.fchmod(f.fileno(), 0o600)
                json.dump(value, f, allow_nan=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            Path(tmp).unlink(missing_ok=True)

    def get(self, identity):
        if identity == "mock":
            return dict(
                provider="mock",
                model="deterministic-mock-v1",
                reasoning_effort="low",
                revision="mock-v1",
                api_key=None,
                checked=True,
                capabilities=["text", "joint_correction", "tool_continuation"],
            )
        value = self.read().get(identity)
        if not value:
            raise ProfileError("Provider profile not found")
        return value.copy()

    def public(self, identity):
        item = self.get(identity)
        return ProviderStatus(
            id=identity,
            provider=item["provider"],
            model=item["model"],
            reasoning_effort=item["reasoning_effort"],
            revision=item["revision"],
            saved=True,
            checked=item.get("checked", False),
            capabilities=item.get("capabilities", []),
            message="Transport capability check passed; target admission is separate."
            if item.get("checked")
            else "Saved privately. Provider access and capabilities have not been checked.",
        )

    def list(self):
        return [self.public(k) for k in ["mock", *sorted(self.read())]]

    def save(self, identity, payload):
        if not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", identity):
            raise ProfileError("Invalid provider profile identity")
        if identity == "mock":
            raise ProfileError("The built-in mock profile cannot be changed")
        value = self.read()
        old = value.get(identity, {})
        key = (
            payload.api_key.get_secret_value()
            if payload.api_key is not None
            else old.get("api_key")
        )
        if payload.provider == "mock":
            key = None
        if payload.provider == "openai_responses" and not key:
            raise ProfileError("Save an API key for the Responses provider")
        if key and key in payload.model:
            raise ProfileError("Model identity is invalid")
        if len(value) >= 16 and identity not in value:
            raise ProfileError("Provider profile limit reached")
        value[identity] = dict(
            provider=payload.provider,
            model="deterministic-mock-v1" if payload.provider == "mock" else payload.model,
            reasoning_effort=payload.reasoning_effort,
            api_key=key,
            revision=uuid4().hex,
            checked=False,
            capabilities=[],
        )
        self.write(value)
        return self.public(identity)

    def checked(self, identity, revision, capabilities):
        if identity != "mock":
            value = self.read()
            if identity not in value or value[identity]["revision"] != revision:
                raise ProfileError("Provider profile changed during its check")
            value[identity].update(checked=True, capabilities=capabilities)
            self.write(value)
        return self.public(identity)

    def remove(self, identity):
        if identity == "mock":
            raise ProfileError("The built-in mock profile cannot be removed")
        value = self.read()
        value.pop(identity, None)
        self.write(value)
