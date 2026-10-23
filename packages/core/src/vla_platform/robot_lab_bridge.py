"""Private loopback connection to the computer holding the SO-101 USB devices."""

import asyncio
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from vla_platform.datasets.library import (
    MAX_BYTES,
    MAX_FILES,
    LibraryError,
    canonical_name,
    name_key,
)

MAX_JSON = 4 * 1024**2
MAX_IMAGE = 8 * 1024**2
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")


class RobotLabError(ValueError):
    def __init__(self, message, status=502):
        self.status = status
        super().__init__(message)


def strict_json(raw):
    def nonfinite(_value):
        raise ValueError("Non-finite JSON number")

    return json.loads(raw, parse_constant=nonfinite)


def safe_id(value):
    if not SAFE_ID.fullmatch(value) or value in {".", ".."}:
        raise RobotLabError("Invalid local recording identifier", 422)
    return value


class RobotLabBridge:
    def __init__(self, data_dir: Path, *, client=None):
        self.data_dir = Path(data_dir)
        self.config_path = Path(
            os.environ.get("JENSEN_ROBOT_LAB_CONFIG", self.data_dir / "robot-lab-connection.json")
        )
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(30, connect=3), follow_redirects=False, trust_env=False
        )
        self.import_lock = asyncio.Lock()

    async def close(self):
        await self.client.aclose()

    def config(self):
        try:
            descriptor = os.open(self.config_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, "rb") as handle:
                info = os.fstat(handle.fileno())
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.getuid()
                    or info.st_mode & 0o077
                    or info.st_size > 16384
                ):
                    raise ValueError
                value = json.load(handle)
            if not isinstance(value, dict) or set(value) - {"url", "token", "label"}:
                raise ValueError
            url = value["url"]
            token = value["token"]
            label = value.get("label", "Local computer")
            parsed = urlsplit(url)
            if (
                parsed.scheme != "http"
                or parsed.hostname not in {"127.0.0.1", "::1"}
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
                or parsed.path not in {"", "/"}
                or any(character.isspace() or not character.isprintable() for character in url)
                or parsed.port is None
                or not 1024 <= parsed.port <= 65535
                or not isinstance(token, str)
                or not 32 <= len(token) <= 512
                or not token.isascii()
                or any(character.isspace() or not character.isprintable() for character in token)
                or not isinstance(label, str)
                or not 1 <= len(label) <= 100
                or any(not character.isprintable() for character in label)
            ):
                raise ValueError
            return {"url": url.rstrip("/"), "token": token, "label": label}
        except FileNotFoundError:
            raise RobotLabError(
                "Connect the local Robot Lab companion to use the arm and cameras.", 503
            ) from None
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            raise RobotLabError(
                "The private Robot Lab connection needs operator setup.", 503
            ) from None

    def url(self, config, path):
        return config["url"] + "/v1/" + path

    def headers(self, config):
        return {"Authorization": "Bearer " + config["token"]}

    def sanitize(self, value, config):
        if isinstance(value, dict):
            return {
                key: self.sanitize(item, config)
                for key, item in value.items()
                if key.lower() not in {"token", "authorization", "password", "secret"}
            }
        if isinstance(value, list):
            return [self.sanitize(item, config) for item in value]
        if isinstance(value, str):
            return value.replace(config["token"], "[private]")
        return value

    async def read(self, response, limit):
        chunks = []
        size = 0
        async for chunk in response.aiter_bytes():
            size += len(chunk)
            if size > limit:
                raise RobotLabError("The local companion response exceeded its size limit.")
            chunks.append(chunk)
        return b"".join(chunks)

    async def request(self, method, path, *, payload=None, query=None, image=False):
        config = self.config()
        # Episode compression and final metadata can exceed an ordinary device read.
        seconds = 80 if path in {"recording/episode/save", "recording/finish"} else 30
        try:
            async with (
                asyncio.timeout(seconds + 5),
                self.client.stream(
                    method,
                    self.url(config, path),
                    headers=self.headers(config),
                    json=payload,
                    params=query,
                    timeout=httpx.Timeout(seconds, connect=3),
                    follow_redirects=False,
                ) as response,
            ):
                if 300 <= response.status_code < 400:
                    raise RobotLabError("The local companion returned an unexpected redirect.")
                raw = await self.read(response, MAX_IMAGE if image else MAX_JSON)
                if image and response.is_success:
                    content_type = response.headers.get("content-type", "").split(";")[0]
                    if content_type not in {"image/jpeg", "image/png"}:
                        raise RobotLabError("The local companion did not return a camera image.")
                    return raw, content_type
                try:
                    body = self.sanitize(strict_json(raw), config)
                except (ValueError, UnicodeDecodeError, RecursionError):
                    raise RobotLabError(
                        "The local companion returned an invalid response."
                    ) from None
                if not response.is_success:
                    detail = (
                        body.get("detail", body.get("error")) if isinstance(body, dict) else None
                    )
                    if not isinstance(detail, str):
                        detail = "The local companion could not complete this operation."
                    status = response.status_code
                    if status not in {400, 404, 409, 413, 422, 423, 503}:
                        status = 502
                    raise RobotLabError(detail[:1000], status)
                return body
        except (httpx.HTTPError, TimeoutError):
            raise RobotLabError(
                "The local Robot Lab companion is unavailable. Check its connection on the Mac."
            ) from None

    async def status(self):
        try:
            config = self.config()
            result = await self.request("GET", "status")
            if not isinstance(result, dict):
                raise RobotLabError("The local companion returned an invalid status.")
            return {**result, "connected": True, "label": self.sanitize(config["label"], config)}
        except RobotLabError as exc:
            return {"connected": False, "message": str(exc), "status": "offline"}

    async def manifest(self, ident):
        value = await self.request("GET", f"datasets/{safe_id(ident)}/manifest")
        if not isinstance(value, dict) or value.get("id") != ident:
            raise RobotLabError("The recording manifest has an invalid identity.")
        files = value.get("files")
        if not isinstance(files, list) or not 1 <= len(files) <= MAX_FILES:
            raise RobotLabError("Recordings support between 1 and 4096 files.", 413)
        names = set()
        total = 0
        inventory = []
        for item in files:
            if not isinstance(item, dict):
                raise RobotLabError("The recording manifest is invalid.")
            path, size, checksum = item.get("path"), item.get("size"), item.get("sha256")
            try:
                canonical_name(path)
                key = name_key(path)
            except (LibraryError, TypeError):
                raise RobotLabError(
                    "The recording manifest contains an unsafe file path."
                ) from None
            if (
                key in names
                or type(size) is not int
                or size < 0
                or not isinstance(checksum, str)
                or not re.fullmatch(r"[a-f0-9]{64}", checksum)
            ):
                raise RobotLabError("The recording manifest has invalid or duplicate files.")
            names.add(key)
            total += size
            inventory.append({"path": path, "size": size, "sha256": checksum})
        if total > MAX_BYTES:
            raise RobotLabError("Local recording imports support at most 2 GiB.", 413)
        if value.get("total_bytes") != total:
            raise RobotLabError("The recording manifest byte count is inconsistent.")
        if "meta/info.json" not in {item["path"] for item in inventory}:
            raise RobotLabError("A recording must contain LeRobot metadata.")
        inventory.sort(key=lambda item: item["path"])
        fingerprint = hashlib.sha256(
            json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return value, inventory, fingerprint

    async def file_stream(self, ident, item):
        config = self.config()
        digest, size = hashlib.sha256(), 0
        try:
            async with (
                asyncio.timeout(300),
                self.client.stream(
                    "GET",
                    self.url(config, f"datasets/{safe_id(ident)}/files"),
                    headers=self.headers(config),
                    params={"path": item["path"]},
                    follow_redirects=False,
                ) as response,
            ):
                if response.status_code != 200:
                    raise RobotLabError("A recording file could not be read from the Mac.")
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > item["size"]:
                        raise RobotLabError(
                            "A recording file changed during transfer. Retry the import."
                        )
                    digest.update(chunk)
                    yield chunk
                if size != item["size"] or digest.hexdigest() != item["sha256"]:
                    raise RobotLabError(
                        "A recording file changed during transfer. Retry the import."
                    )
        except (httpx.HTTPError, TimeoutError):
            raise RobotLabError(
                "The recording transfer was interrupted. Retry to resume it."
            ) from None

    async def import_dataset(self, project_id, ident, library):
        async with self.import_lock:
            manifest, inventory, fingerprint = await self.manifest(ident)
            # Receipts live alongside the upload, so a failed transfer can resume without
            # publishing a half-recording or replacing the Mac's original files.
            existing = None
            for record in library.records():
                receipt = library.directory(record["id"]) / "robot-lab-import.json"
                try:
                    saved = json.loads(receipt.read_bytes())
                except (OSError, ValueError):
                    continue
                if saved == {"project_id": project_id, "fingerprint": fingerprint}:
                    existing = record
                    break
            if existing and existing["status"] == "ready":
                return library.public(existing)
            if not existing or existing["status"] != "uploading":
                name = manifest.get("name", ident)
                if not isinstance(name, str) or not 1 <= len(name) <= 100:
                    name = ident
                existing = library.get(library.begin(project_id, name)["id"])
                receipt = library.directory(existing["id"]) / "robot-lab-import.json"
                receipt.write_text(
                    json.dumps({"project_id": project_id, "fingerprint": fingerprint})
                )
                receipt.chmod(0o600)
            for item in inventory:
                uploaded = library.get(existing["id"])["files"]
                if item["path"] in uploaded:
                    if uploaded[item["path"]] != item["size"]:
                        raise RobotLabError("Saved transfer inventory is inconsistent.")
                    continue
                await library.upload(existing["id"], item["path"], self.file_stream(ident, item))
            accepted = await library.finalize(existing["id"])
            if accepted["status"] != "ready":
                raise RobotLabError(
                    "The recording was preserved, but it is not a finalized LeRobot v3 dataset.",
                    422,
                )
            return accepted
