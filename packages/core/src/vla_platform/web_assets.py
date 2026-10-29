"""Serve Next static route payloads alongside ordinary exported HTML."""

from pathlib import Path
from urllib.parse import quote

from starlette.datastructures import Headers
from starlette.staticfiles import StaticFiles
from starlette.types import Scope


class WorkspaceStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope: Scope):
        requested = Path(path)
        route = not requested.suffix
        rsc = Headers(scope=scope).get("rsc") == "1"
        target = requested / "index.txt" if rsc and route else requested
        response = await super().get_response(str(target), scope)
        payload = target.name == "index.txt" or (
            target.name.startswith("__next.") and target.suffix == ".txt"
        )
        if route or payload:
            response.headers.add_vary_header("RSC")
        if rsc and payload and response.status_code in {200, 304}:
            response.headers["Content-Type"] = "text/x-component; charset=utf-8"
            # Payload filenames are transport details, not the rendered route.
            rendered = "/" + "/".join(target.parent.parts)
            response.headers["x-nextjs-rewritten-path"] = quote(
                rendered if rendered.endswith("/") else rendered + "/", safe="/"
            )
        return response
