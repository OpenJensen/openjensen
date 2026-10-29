"""Local stdio MCP adapter. Only session-scoped proposal and observation tools."""

import argparse
import base64
import json
import os
import re
from typing import Literal
from urllib.parse import urlsplit

import httpx

from vla_platform.agent_control.contracts import LocalAgentResult, SegmentArguments

TOKEN_ENV = "OPENJENSEN_AGENT_TOKEN"


class LocalAgentClient:
    def __init__(self, api_url, session_id, token, *, transport=None):
        url = urlsplit(api_url)
        if (
            url.scheme not in {"http", "https"}
            or url.hostname not in {"127.0.0.1", "localhost", "::1"}
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.path not in {"", "/"}
        ):
            raise ValueError("Use a loopback application URL or an existing local SSH forward")
        try:
            _ = url.port
        except ValueError:
            raise ValueError("Invalid loopback application port") from None
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", session_id):
            raise ValueError("Invalid local agent session")
        if not re.fullmatch(r"[A-Za-z0-9_-]{43}", token or ""):
            raise ValueError("Set OPENJENSEN_AGENT_TOKEN from the local agent connection panel")
        self.base = api_url.rstrip("/") + "/api/v1/agent-sessions/" + session_id + "/local"
        self.token = token
        self.transport = transport

    async def request(self, method, operation, body=None, params=None, *, image=False):
        try:
            async with httpx.AsyncClient(
                timeout=15, trust_env=False, follow_redirects=False, transport=self.transport
            ) as client:
                async with client.stream(
                    method,
                    self.base + "/" + operation,
                    json=body,
                    params=params,
                    headers={"Authorization": "Bearer " + self.token},
                ) as response:
                    data = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(data) + len(chunk) > 4 * 1024**2:
                            raise ValueError("Local agent response exceeds its bound")
                        data.extend(chunk)
                    if not 200 <= response.status_code < 300:
                        raise ValueError(
                            "OpenJensen refused the local agent request. Check its connection, "
                            "authority and pending context. Mutations are not retried."
                        )
                    if image:
                        mime = response.headers.get("content-type", "").split(";")[0]
                        if mime not in {"image/jpeg", "image/png"}:
                            raise ValueError("The admitted observation has no supported image")
                        return bytes(data), mime
                    return json.loads(data)
        except httpx.HTTPError, json.JSONDecodeError:
            raise ValueError(
                "Request not acknowledged. Inspect the session before any new mutation."
            ) from None


def make_server(client):
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ImageContent, ToolAnnotations

    server = FastMCP(
        "OpenJensen teleoperation",
        instructions=(
            "Read get_context for the pending request, measured state, units and limits. "
            "Propose small joint segments: gripper in percent, arm in degrees. "
            "Read the admitted camera image. Proposals need operator approval "
            "and a fresh admission image. Motion receipts do not verify a grasp. "
            "Tools cannot arm, approve, renew presence, calibrate or write motors. "
            "The external agent owns its own model usage and data-sharing settings."
        ),
    )
    read = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
    write = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)

    @server.tool(annotations=read)
    async def get_context() -> dict:
        """Read the session task, pending request, state and bounded action space."""
        return await client.request("GET", "context")

    @server.tool(annotations=read)
    async def get_camera_image(observation_id: str) -> list[ImageContent]:
        """Read the admitted image for an observation ID from get_context."""
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", observation_id):
            raise ValueError("Invalid observation identity")
        data, mime = await client.request(
            "GET", "image", params={"observation_id": observation_id}, image=True
        )
        return [ImageContent(type="image", mimeType=mime, data=base64.b64encode(data).decode())]

    @server.tool(annotations=write)
    async def propose_joint_segment(
        request_id: str,
        authority_epoch: int,
        observation_id: str,
        observation_sha256: str,
        targets: dict[str, float],
        duration_seconds: float,
        purpose: str,
        agent_name: str = "local-agent",
    ) -> dict:
        """Submit a bounded proposal for human review; never execute it."""
        result = LocalAgentResult(
            request_id=request_id,
            authority_epoch=authority_epoch,
            observation_id=observation_id,
            observation_sha256=observation_sha256,
            agent_name=agent_name,
            purpose=purpose,
            arguments=SegmentArguments(targets=targets, duration_seconds=duration_seconds),
        )
        session = await client.request("POST", "result", result.model_dump())
        return {"received_for_validation": True, "session": session}

    @server.tool(annotations=write)
    async def report_assessment(
        request_id: str,
        authority_epoch: int,
        observation_id: str,
        observation_sha256: str,
        outcome: Literal["achieved", "needs_help", "cannot_proceed"],
        summary: str,
        agent_name: str = "local-agent",
    ) -> dict:
        """End this planning turn with an assessment. Human task verification remains separate."""
        result = LocalAgentResult.model_validate(
            {
                "request_id": request_id,
                "authority_epoch": authority_epoch,
                "observation_id": observation_id,
                "observation_sha256": observation_sha256,
                "agent_name": agent_name,
                "assessment": {"outcome": outcome, "summary": summary},
            }
        )
        session = await client.request("POST", "result", result.model_dump())
        return {"received_for_validation": True, "session": session}

    @server.tool(annotations=write)
    async def stop_robot() -> dict:
        """Stop this session and release its local controller authority."""
        return await client.request("POST", "stop", {})

    return server


def serve(api_url: str, session_id: str):
    client = LocalAgentClient(api_url, session_id, os.environ.get(TOKEN_ENV, ""))
    make_server(client).run(transport="stdio")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()
    try:
        serve(args.api_url, args.session_id)
    except ValueError as exc:
        parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    main()
