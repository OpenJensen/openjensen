"""Separate optional LiveKit credential broker; never imported into the simulator."""

from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import os
import threading
import time
import uuid
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from .contracts import canonical, decode
from .control import Client
from .credentials import control_token


class JoinBroker:
    def __init__(self, control: Client, *, clock=time.monotonic):
        self.control = control
        self.clock = clock
        self.lock = threading.Lock()
        self.current = None

    async def _create(self, state):
        from .livekit_agent import VoiceSettings

        VoiceSettings.from_env()  # Do not issue a room when the voice model is unconfigured.
        from aiohttp import ClientTimeout
        from livekit import api

        url = os.environ.get("LIVEKIT_URL", "")
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"ws", "wss"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("LiveKit URL is missing or invalid")
        key, secret = (
            os.environ.get(name, "") for name in ("LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
        )
        if not key or not secret:
            raise ValueError("LiveKit API credentials are not configured")
        room = "firebird-teaching-" + uuid.uuid4().hex
        identity = "operator-" + uuid.uuid4().hex
        metadata = json.dumps(
            {"operator_identity": identity, "teaching_session_id": state["session_id"]}
        )
        service = api.LiveKitAPI(
            url=url, api_key=key, api_secret=secret, timeout=ClientTimeout(total=10)
        )
        created = False
        try:
            await service.room.create_room(
                api.CreateRoomRequest(
                    name=room, empty_timeout=60, max_participants=2, departure_timeout=20
                )
            )
            created = True
            await service.agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(
                    agent_name="firebird-teaching", room=room, metadata=metadata
                )
            )
            token = (
                api.AccessToken(key, secret)
                .with_identity(identity)
                .with_ttl(timedelta(minutes=5))
                .with_grants(
                    api.VideoGrants(
                        room_join=True,
                        room=room,
                        can_publish=True,
                        can_publish_sources=["microphone"],
                        can_subscribe=True,
                        can_publish_data=False,
                        can_update_own_metadata=False,
                    )
                )
                .to_jwt()
            )
            return {
                "status": "room_created",
                "url": url,
                "token": token,
                "room": room,
                "operator_identity": identity,
                "session_id": state["session_id"],
                "expires_in_seconds": 300,
                "agent_name": "firebird-teaching",
                "microphone_or_executor_connection_verified": False,
            }
        except Exception:
            if created:
                try:
                    await service.room.delete_room(api.DeleteRoomRequest(room=room))
                except Exception:
                    pass
            raise ValueError(
                "LiveKit room setup failed; check server-side service configuration"
            ) from None
        finally:
            await service.aclose()

    def join(self, payload: dict) -> dict:
        if set(payload) != {"session_id"} or not isinstance(payload["session_id"], str):
            raise ValueError("Voice join requires exactly session_id")
        with self.lock:
            state = self.control.request("/state")
            if state.get("session_id") != payload["session_id"] or state.get("mode") not in {
                "idle",
                "running",
                "paused",
            }:
                raise ValueError("Teaching executor is not ready or session changed")
            if self.current:
                if self.current["session_id"] != state["session_id"]:
                    raise ValueError("Broker belongs to an older executor; restart voice services")
                remaining = int(self.current["expires_at"] - self.clock())
                if remaining <= 0:
                    raise ValueError(
                        "Voice room lease expired; restart voice services for a new bounded session"
                    )
                return self.current["result"] | {"expires_in_seconds": remaining}
            result = asyncio.run(self._create(state))
            self.current = {
                "session_id": state["session_id"],
                "result": result,
                "expires_at": self.clock() + 300,
            }
            return result


def serve(broker: JoinBroker, token: str, port: int):
    if len(token) < 32:
        raise ValueError("Control token must contain at least32 characters")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            status = 200
            try:
                if self.path != "/voice/join":
                    raise ValueError("Unknown voice endpoint")
                if self.headers.get("Origin") or not hmac.compare_digest(
                    self.headers.get("Authorization", ""), "Bearer " + token
                ):
                    status = 401
                    raise ValueError("Unauthorized voice request")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024 or self.headers.get("Transfer-Encoding"):
                    raise ValueError("Invalid voice request size")
                self.connection.settimeout(2)
                result = broker.join(decode(self.rfile.read(length), limit=1024))
            except (ValueError, OSError, ImportError) as error:
                status = status if status != 200 else 400
                # Only the typed missing-variable error is safe to return verbatim.
                from .livekit_agent import VoiceConfigurationError

                detail = (
                    str(error)
                    if isinstance(error, VoiceConfigurationError)
                    else "Voice join unavailable; check executor and voice configuration"
                )
                result = {"error": detail}
            raw = canonical(result)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file")
    parser.add_argument("--port", type=int, default=8769)
    args = parser.parse_args()
    if args.env_file:
        from dotenv import load_dotenv

        load_dotenv(args.env_file, override=False)
    if not 1024 <= args.port <= 65535:
        parser.error("Use an unprivileged local port")
    client = Client(os.environ.get("FIREBIRD_TEACHING_CONTROL_URL", ""), control_token())
    http = serve(JoinBroker(client), client.token, args.port)
    try:
        http.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        http.server_close()


if __name__ == "__main__":
    main()
