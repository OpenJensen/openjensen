"""Explicitly entered Hugging Face credentials, separate from jobs and task bundles."""

import asyncio
import json
import os
import re
import stat
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, SecretStr

WHOAMI_URL = "https://huggingface.co/api/whoami-v2"
MAX_IDENTITY_BYTES = 512 * 1024


class HuggingFaceTokenInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token: SecretStr


class HuggingFaceStatus(BaseModel):
    configured: bool = False
    username: str | None = None
    token_hint: str | None = None
    checked_at: str | None = None
    message: str | None = None


class CredentialError(ValueError):
    """Only fixed, non-secret messages may be returned to the client."""


def _validate_token(token: str) -> str:
    token = token.strip()
    if not re.fullmatch(r"hf_[A-Za-z0-9_-]{8,509}", token):
        raise CredentialError("Enter a valid Hugging Face access token beginning with hf_.")
    return token


async def verify_huggingface_token(token: str) -> str:
    """Verify only identity; never echo provider diagnostics or follow a redirect."""
    try:
        async with (
            asyncio.timeout(15),
            httpx.AsyncClient(timeout=12, follow_redirects=False) as client,
        ):
            async with client.stream(
                "GET", WHOAMI_URL, headers={"Authorization": f"Bearer {token}"}
            ) as response:
                if response.status_code in {401, 403}:
                    raise CredentialError(
                        "Hugging Face rejected this token. Check its validity and permissions."
                    )
                if response.status_code != 200:
                    raise CredentialError(
                        "Hugging Face could not verify the token. Try again shortly."
                    )
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(body) + len(chunk) > MAX_IDENTITY_BYTES:
                        raise CredentialError("Hugging Face returned an invalid identity response.")
                    body.extend(chunk)
        identity = json.loads(body)
        username = identity.get("name") if isinstance(identity, dict) else None
        if (
            not isinstance(username, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", username)
            or token in username
        ):
            raise CredentialError("Hugging Face returned an invalid account identity.")
        return username
    except httpx.TimeoutException, TimeoutError:
        raise CredentialError("Hugging Face verification timed out. Try again.") from None
    except httpx.HTTPError:
        raise CredentialError("Could not reach Hugging Face to verify the token.") from None
    except json.JSONDecodeError, UnicodeDecodeError:
        raise CredentialError("Hugging Face returned an invalid identity response.") from None


class HuggingFaceConnection:
    def __init__(self, data_dir: Path):
        self.path = data_dir / "huggingface-credential.json"
        self._lock = asyncio.Lock()
        self._secret: SecretStr | None = None
        self._status = HuggingFaceStatus()
        self._load()

    def _load(self) -> None:
        if not self.path.exists() and not self.path.is_symlink():
            return
        try:
            if self.path.is_symlink():
                raise ValueError("Credential file must not be a symlink")
            # Opening a FIFO must not block before fstat can reject it.
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
            handle = os.open(self.path, flags)
            with os.fdopen(handle, "r") as stream:
                info = os.fstat(stream.fileno())
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_size > 8192
                    or (os.name == "posix" and (info.st_mode & 0o077 or info.st_uid != os.getuid()))
                ):
                    raise ValueError("Credential file is not private")
                document = json.loads(stream.read(8193))
            if not isinstance(document, dict) or document.get("version") != 1:
                raise ValueError("Invalid saved credential")
            token = _validate_token(document["token"])
            username = document["username"]
            checked_at = document["checked_at"]
            if (
                not isinstance(username, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", username)
                or token in username
                or not isinstance(checked_at, str)
            ):
                raise ValueError("Invalid saved account")
            datetime.fromisoformat(checked_at)
            self._secret = SecretStr(token)
            self._status = self._saved_status(username, token, checked_at)
        except OSError, ValueError, KeyError, TypeError:
            self._status = HuggingFaceStatus(
                message="The Hugging Face credential could not be read securely. Save it again."
            )

    @staticmethod
    def _saved_status(username: str, token: str, checked_at: str) -> HuggingFaceStatus:
        return HuggingFaceStatus(
            configured=True,
            username=username,
            token_hint="••••" + token[-4:],
            checked_at=checked_at,
            message="Saved for model downloads. Access still depends on the token's permissions.",
        )

    def status(self) -> HuggingFaceStatus:
        return self._status.model_copy(deep=True)

    def token(self) -> str | None:
        """Internal delivery to worker process secrets only; never an HTTP response."""
        return self._secret.get_secret_value() if self._secret is not None else None

    async def save(self, value: SecretStr) -> HuggingFaceStatus:
        token = _validate_token(value.get_secret_value())
        async with self._lock:
            username = await verify_huggingface_token(token)
            checked_at = datetime.now(UTC).isoformat()
            document = {
                "version": 1,
                "token": token,
                "username": username,
                "checked_at": checked_at,
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle, temporary = tempfile.mkstemp(
                prefix=".huggingface-credential-", dir=self.path.parent
            )
            try:
                with os.fdopen(handle, "w") as stream:
                    json.dump(document, stream)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            self._secret = SecretStr(token)
            self._status = self._saved_status(username, token, checked_at)
            return self.status()

    async def disconnect(self) -> HuggingFaceStatus:
        async with self._lock:
            self.path.unlink(missing_ok=True)
            self._secret = None
            self._status = HuggingFaceStatus()
            return self.status()
