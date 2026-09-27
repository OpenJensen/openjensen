"""Authenticated /firebird gateway. Run with FIREBIRD_PUBLIC_CONFIG pointing to a 0600 JSON file."""

import asyncio
import base64
import hashlib
import hmac
import json
import os
import re
import stat
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import anyio
import httpx
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response, StreamingResponse
from starlette.routing import Route
from starlette.staticfiles import StaticFiles

PREFIX = "/firebird"
UPSTREAM = httpx.URL("http://127.0.0.1:8096")
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
HOP_HEADERS = {
    b"connection",
    b"keep-alive",
    b"proxy-authenticate",
    b"proxy-authorization",
    b"te",
    b"trailer",
    b"transfer-encoding",
    b"upgrade",
}
SECURITY_HEADERS = {
    "cache-control": "no-store",
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
    "content-security-policy": "frame-ancestors 'none'",
    "x-frame-options": "DENY",
}


@dataclass(frozen=True)
class GatewayConfig:
    public_origin: str
    static_dir: Path
    username: str
    salt: bytes
    password_hash: bytes
    iterations: int
    preview_origins: tuple[str, ...] = ()
    public_readonly: bool = False

    @classmethod
    def from_value(cls, value):
        expected = {
            "public_origin",
            "static_dir",
            "username",
            "password_salt_hex",
            "password_hash_hex",
            "password_iterations",
            "preview_origins",
            "public_readonly",
        }
        if not isinstance(value, dict) or set(value) - expected:
            raise ValueError("Invalid gateway configuration")
        try:
            origin = value["public_origin"].rstrip("/")
            parsed = urlsplit(origin)
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.path
                or parsed.query
                or parsed.fragment
                or parsed.port not in (None, 443)
            ):
                raise ValueError
            directory = Path(value["static_dir"])
            salt = bytes.fromhex(value["password_salt_hex"])
            password_hash = bytes.fromhex(value["password_hash_hex"])
            iterations = value["password_iterations"]
            previews = value.get("preview_origins", [])
            if (
                value["username"] != "firebird"
                or len(salt) != 16
                or len(password_hash) != 32
                or type(iterations) is not int
                or not 210000 <= iterations <= 2000000
                or not directory.is_absolute()
                or not directory.is_dir()
                or not isinstance(previews, list)
                or type(value.get("public_readonly", False)) is not bool
                or any(
                    item
                    not in {
                        "http://127.0.0.1:18097",
                        "http://localhost:18097",
                        "http://127.0.0.1:8097",
                        "http://localhost:8097",
                    }
                    for item in previews
                )
            ):
                raise ValueError
        except KeyError, TypeError, AttributeError, ValueError:
            raise ValueError("Invalid gateway configuration") from None
        return cls(
            origin,
            directory,
            "firebird",
            salt,
            password_hash,
            iterations,
            tuple(previews),
            value.get("public_readonly", False),
        )

    @classmethod
    def load(cls, path):
        # This deployment template relies on POSIX owner/mode and no-follow checks.
        # Fail closed rather than silently replacing those checks with Windows mode bits.
        if os.name != "posix":
            raise ValueError("The Xbox gateway requires POSIX file ownership and permissions")
        # A credential file must be owned by this service user and unreadable to others.
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_mode & 0o077
                or info.st_size > 16384
            ):
                raise ValueError("Gateway configuration must be a private owned regular file")
            return cls.from_value(json.load(stream))


def single_header(scope, name):
    values = [value for key, value in scope["headers"] if key.lower() == name]
    return values[0].decode("latin-1") if len(values) == 1 else None


def filtered_headers(headers, *, request=False):
    connection = set()
    for key, value in headers:
        if key.lower() == b"connection":
            connection.update(name.strip().lower() for name in value.split(b","))
    blocked = HOP_HEADERS | connection
    if request:
        # The gateway has already authenticated and checked Origin. The local backend
        # deliberately accepts only local origins, and must never receive Basic secrets.
        blocked |= {
            b"authorization",
            b"cookie",
            b"origin",
            b"host",
            b"content-length",
            b"forwarded",
        }
    else:
        blocked |= {b"set-cookie"}
    return [
        (key.lower(), value)
        for key, value in headers
        if key.lower() not in blocked and not key.lower().startswith(b"x-forwarded-")
    ]


def public_location(location, origin):
    parsed = urlsplit(location)
    internal = not parsed.netloc and location.startswith("/")
    if parsed.netloc:
        internal = (
            parsed.scheme in {"http", "https"}
            and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
            and parsed.port == 8096
        ) or location.startswith(origin + "/")
    if internal:
        path = parsed.path
        if path != PREFIX and not path.startswith(PREFIX + "/"):
            path = PREFIX + (path if path.startswith("/") else "/" + path)
        return urlunsplit(("", "", path, parsed.query, parsed.fragment))
    return location


def public_read_allowed(path):
    """Public demonstrations expose saved application results, never account administration."""
    if path == PREFIX:
        return True
    if not path.startswith(PREFIX + "/"):
        return False
    path = path[len(PREFIX) :]
    if path == "/openapi.json":
        return True
    if path != "/api" and not path.startswith("/api/"):
        return True  # StaticFiles still enforces filesystem containment.
    if path in {
        "/api/v1/health",
        "/api/v1/projects",
        "/api/v1/capabilities",
        "/api/v1/policy-options",
    }:
        return True
    return (
        re.fullmatch(
            r"/api/v1/(?:projects/[A-Za-z0-9_-]+/(?:jobs|artifacts)|"
            r"jobs/[A-Za-z0-9_-]+(?:/(?:events|training|episodes(?:/[0-9]+)?))?)",
            path,
        )
        is not None
    )


class PublicGateway:
    def __init__(self, config, *, transport=None):
        self.config = config
        self.transport = transport
        self.client = None
        self.static = StaticFiles(directory=config.static_dir, html=True, follow_symlink=False)
        self.auth_lock = asyncio.Lock()
        self.verified_digest = None
        self.origins = {config.public_origin, *config.preview_origins}
        self.hosts = {
            urlsplit(config.public_origin).netloc.lower(),
            f"{urlsplit(config.public_origin).hostname}:443",
            "127.0.0.1:8097",
            "localhost:8097",
            "127.0.0.1:18097",
            "localhost:18097",
        }
        self.application = Starlette(
            routes=[
                Route(
                    "/{path:path}",
                    self.handle,
                    methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
                )
            ],
            lifespan=self.lifespan,
        )

    @asynccontextmanager
    async def lifespan(self, _app):
        async with httpx.AsyncClient(
            transport=self.transport,
            follow_redirects=False,
            trust_env=False,
            timeout=httpx.Timeout(60, connect=5),
            limits=httpx.Limits(max_connections=100),
        ) as client:
            self.client = client
            yield
        self.client = None

    async def authenticated(self, scope):
        authorization = single_header(scope, b"authorization")
        if not authorization or len(authorization) > 2048:
            return False
        try:
            scheme, encoded = authorization.split(" ", 1)
            if scheme.lower() != "basic":
                return False
            credentials = base64.b64decode(encoded, validate=True)
            username, separator, password = credentials.partition(b":")
            if not separator or not password:
                return False
            username.decode("utf-8")
            password.decode("utf-8")
        except ValueError, UnicodeError:
            return False
        candidate = hashlib.sha256(credentials).digest()
        if self.verified_digest is not None and hmac.compare_digest(
            candidate, self.verified_digest
        ):
            return True
        # Config is immutable until restart, so a verified fingerprint remains valid
        # for this process. Established sessions never wait behind expensive failures.
        # At most one uncached verification runs; reject excess attempts, never queue.
        if self.auth_lock.locked():
            return False
        async with self.auth_lock:
            derived = await asyncio.to_thread(
                hashlib.pbkdf2_hmac, "sha256", password, self.config.salt, self.config.iterations
            )
            valid_user = hmac.compare_digest(username, self.config.username.encode())
            valid_password = hmac.compare_digest(derived, self.config.password_hash)
            if valid_user and valid_password:
                # Cache only the successful credential fingerprint, never the password.
                self.verified_digest = candidate
                return True
            return False

    def error(self, scope, status, message, **headers):
        return Response(
            b"" if scope["method"] == "HEAD" else message.encode(),
            status_code=status,
            headers={**SECURITY_HEADERS, **headers},
            media_type="text/plain",
        )

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            return await self.application(scope, receive, send)
        if scope["type"] != "http":
            await send({"type": "websocket.close", "code": 1008})
            return
        host = single_header(scope, b"host")
        if not host or host.lower() not in self.hosts:
            response = self.error(scope, 400, "Invalid host")
        elif not self.config.public_readonly and not await self.authenticated(scope):
            response = self.error(
                scope,
                401,
                "Authentication required",
                **{"www-authenticate": 'Basic realm="Firebird", charset="UTF-8"'},
            )
        elif not (scope["path"] == PREFIX or scope["path"].startswith(PREFIX + "/")):
            response = self.error(scope, 404, "Not found")
        elif (
            self.config.public_readonly
            and not (scope["method"] in {"GET", "HEAD"} and public_read_allowed(scope["path"]))
            and not await self.authenticated(scope)
        ):
            # Do not trigger a browser login dialog while someone explores the demo.
            response = JSONResponse(
                {
                    "detail": (
                        "This public demo is read-only. Cloud jobs, downloads "
                        "and account settings require owner access."
                    )
                },
                status_code=403,
                headers=SECURITY_HEADERS,
            )
        else:
            origin = single_header(scope, b"origin")
            count = sum(key.lower() == b"origin" for key, _ in scope["headers"])
            unsafe = scope["method"] not in SAFE_METHODS
            if (
                count > 1
                or origin is not None
                and origin not in self.origins
                or unsafe
                and (
                    origin not in self.origins
                    or single_header(scope, b"sec-fetch-site") == "cross-site"
                )
            ):
                response = self.error(scope, 403, "Origin is not allowed")
            else:
                return await self.application(scope, receive, send)
        await response(scope, receive, send)

    async def handle(self, request: Request):
        path = request.scope["path"][len(PREFIX) :] or "/"
        if request.scope["path"] == PREFIX:
            response = RedirectResponse(PREFIX + "/", status_code=307)
        elif path == "/api" or path.startswith("/api/") or path == "/openapi.json":
            raw_path = request.scope.get("raw_path", request.scope["path"].encode())
            if not raw_path.startswith(PREFIX.encode() + b"/"):
                return self.error(request.scope, 400, "Invalid path")
            upstream_path = raw_path[len(PREFIX) :]
            if request.scope["query_string"]:
                upstream_path += b"?" + request.scope["query_string"]
            url = UPSTREAM.copy_with(raw_path=upstream_path)
            headers = filtered_headers(request.scope["headers"], request=True)
            try:
                upstream = await self.client.send(
                    self.client.build_request(
                        request.method, url, headers=headers, content=request.stream()
                    ),
                    stream=True,
                )
            except httpx.HTTPError:
                return self.error(request.scope, 502, "Application is unavailable")

            async def body():
                try:
                    async for chunk in upstream.aiter_raw():
                        yield chunk
                finally:
                    with anyio.CancelScope(shield=True):
                        await upstream.aclose()

            response = StreamingResponse(body(), status_code=upstream.status_code)
            response.raw_headers = filtered_headers(upstream.headers.raw)
            if "location" in response.headers:
                response.headers["location"] = public_location(
                    response.headers["location"], self.config.public_origin
                )
        else:
            try:
                response = await self.static.get_response(path.lstrip("/"), request.scope)
                if "location" in response.headers:
                    # TLS terminates at Funnel. A directory redirect must not expose
                    # the loopback listener or downgrade the browser to HTTP.
                    destination = urlsplit(response.headers["location"])
                    response.headers["location"] = urlunsplit(
                        ("", "", destination.path, destination.query, destination.fragment)
                    )
            except HTTPException as error:
                response = self.error(request.scope, error.status_code, "Not found")
        for name, value in SECURITY_HEADERS.items():
            response.headers[name] = value
        return response


if __name__ == "__main__":
    import uvicorn

    config = GatewayConfig.load(os.environ["FIREBIRD_PUBLIC_CONFIG"])
    uvicorn.run(
        PublicGateway(config),
        host="127.0.0.1",
        port=8097,
        access_log=False,
        proxy_headers=False,
        server_header=False,
        log_level="warning",
        limit_concurrency=100,
    )
