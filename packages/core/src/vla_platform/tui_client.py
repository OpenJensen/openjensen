"""Bounded asynchronous API transport for the optional terminal client."""

import asyncio
import json
import unicodedata
from urllib.parse import quote, urlsplit

import httpx
from pydantic import TypeAdapter, ValidationError

from vla_platform.contracts import Job, Project

MAX_RESPONSE = 8 * 1024 * 1024


class ApiError(Exception):
    """An actionable public error; mutations are never retried."""


def plain(value: object) -> str:
    # Never interpret terminal controls, Rich markup or OSC hyperlinks from data.
    return "".join(
        c for c in str(value) if c in "\n\t" or not unicodedata.category(c).startswith("C")
    )


def endpoint(value: str) -> str:
    error = "API URL must be HTTP(S), without credentials, query or fragment"
    try:
        url = urlsplit(value)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
            or any(ord(c) < 32 for c in value)
        ):
            raise ValueError(error)
        _ = url.port
    except ValueError:
        # urllib's invalid-port message includes user input; never reflect secrets.
        raise ValueError(error) from None
    return value.rstrip("/")


def segment(value: str) -> str:
    return quote(value, safe="")


class ApiClient:
    def __init__(self, base: str, *, transport=None, deadline: float = 15):
        self.base = endpoint(base)
        self.deadline = deadline
        self.http = httpx.AsyncClient(
            timeout=deadline, trust_env=False, follow_redirects=False, transport=transport
        )

    async def close(self):
        await self.http.aclose()

    async def request(self, method: str, path: str, payload=None):
        uncertain = (
            " Outcome unknown; the request was sent once. Refresh jobs before submitting again."
            if method != "GET"
            else " Start 'firebird serve' or check FIREBIRD_API_URL; press Ctrl+R to refresh."
        )
        try:
            async with asyncio.timeout(self.deadline):
                async with self.http.stream(
                    method, self.base + "/api/v1" + path, json=payload
                ) as response:
                    raw = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(raw) + len(chunk) > MAX_RESPONSE:
                            raise ApiError(
                                "Application response exceeds the 8 MiB limit." + uncertain
                            )
                        raw.extend(chunk)
                    if not 200 <= response.status_code < 300:
                        try:
                            detail = json.loads(raw).get("detail", "Request was not accepted")
                        except ValueError, AttributeError:
                            detail = "Request was not accepted"
                        # A server error or redirect may follow an accepted mutation.
                        ambiguous = response.status_code >= 500 or (
                            method != "GET" and 300 <= response.status_code < 400
                        )
                        suffix = uncertain if ambiguous else ""
                        raise ApiError(
                            f"API {response.status_code}: {plain(detail)[:600]}" + suffix
                        )
                    try:
                        return json.loads(raw)
                    except ValueError:
                        raise ApiError("Application returned invalid JSON." + uncertain) from None
        except httpx.HTTPError, TimeoutError:
            raise ApiError("Cannot reach the application." + uncertain) from None

    async def projects(self):
        records = self.validate(list[Project], await self.request("GET", "/projects"))
        self.unique(records)
        return records

    async def jobs(self, project_id: str):
        jobs = self.validate(
            list[Job], await self.request("GET", f"/projects/{segment(project_id)}/jobs")
        )
        self.unique(jobs)
        if any(job["project_id"] != project_id for job in jobs):
            raise ApiError("Application returned jobs for another project.")
        return jobs

    async def job(self, job_id: str):
        job = self.validate(Job, await self.request("GET", f"/jobs/{segment(job_id)}"))
        if job["id"] != job_id:
            raise ApiError("Application returned another job's details.")
        return job

    @staticmethod
    def unique(records):
        if len({record["id"] for record in records}) != len(records):
            raise ApiError("Application returned duplicate record identities.")

    @staticmethod
    def validate(schema, value):
        try:
            parsed = TypeAdapter(schema).validate_python(value)
            return TypeAdapter(schema).dump_python(parsed, mode="json")
        except ValidationError:
            raise ApiError(
                "Application returned an invalid record; refresh or update the client."
            ) from None
