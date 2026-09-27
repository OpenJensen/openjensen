"""One LiveKit conversational loop driving the authenticated local teaching executor."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from .control import Client
from .credentials import control_token


class VoiceConfigurationError(ValueError):
    """Missing public variable names only; never contains secret values."""


@dataclass(frozen=True)
class VoiceSettings:
    control_url: str
    control_token: str = field(repr=False)
    mode: str = "openrouter"
    model: str = ""
    stt_model: str = "deepgram/flux-general"
    tts_model: str = "fishaudio/s2.1-pro"
    tts_voice: str = "fa4c9eb3dccc4806b382b40d61c6b10a"
    openrouter_key: str = field(default="", repr=False)
    stt_url: str = ""
    stt_key: str = field(default="", repr=False)
    tts_url: str = ""
    tts_key: str = field(default="", repr=False)

    @classmethod
    def from_env(cls):
        mode = os.environ.get("FIREBIRD_VOICE_MODE", "openrouter")
        if mode not in {"openrouter", "openrouter-custom-speech"}:
            raise ValueError("FIREBIRD_VOICE_MODE must be openrouter or openrouter-custom-speech")
        required = [
            "FIREBIRD_TEACHING_CONTROL_URL",
            "FIREBIRD_VOICE_MODEL",
            "OPENROUTER_API_KEY",
            "LIVEKIT_URL",
            "LIVEKIT_API_KEY",
            "LIVEKIT_API_SECRET",
        ]
        if mode == "openrouter-custom-speech":
            required += [
                "FIREBIRD_VOICE_MODEL",
                "OPENROUTER_API_KEY",
                "FIREBIRD_STT_MODEL",
                "FIREBIRD_STT_BASE_URL",
                "FIREBIRD_STT_API_KEY",
                "FIREBIRD_TTS_MODEL",
                "FIREBIRD_TTS_BASE_URL",
                "FIREBIRD_TTS_API_KEY",
                "FIREBIRD_TTS_VOICE",
            ]
        missing = [name for name in required if not os.environ.get(name, "").strip()]
        if missing:
            raise VoiceConfigurationError("Voice configuration missing: " + ", ".join(missing))
        result = cls(
            control_url=os.environ["FIREBIRD_TEACHING_CONTROL_URL"],
            control_token=control_token(),
            mode=mode,
            model=os.environ.get("FIREBIRD_VOICE_MODEL", cls.model),
            stt_model=os.environ.get("FIREBIRD_STT_MODEL", cls.stt_model),
            tts_model=os.environ.get("FIREBIRD_TTS_MODEL", cls.tts_model),
            tts_voice=os.environ.get("FIREBIRD_TTS_VOICE", cls.tts_voice),
            openrouter_key=os.environ.get("OPENROUTER_API_KEY", ""),
            stt_url=os.environ.get("FIREBIRD_STT_BASE_URL", ""),
            stt_key=os.environ.get("FIREBIRD_STT_API_KEY", ""),
            tts_url=os.environ.get("FIREBIRD_TTS_BASE_URL", ""),
            tts_key=os.environ.get("FIREBIRD_TTS_API_KEY", ""),
        )
        Client(result.control_url, result.control_token)
        livekit = urlsplit(os.environ["LIVEKIT_URL"])
        if livekit.username or livekit.password or livekit.query or livekit.fragment:
            raise ValueError("LiveKit URL must not contain embedded credentials/query")
        if mode == "openrouter":
            if livekit.scheme != "wss" or not (livekit.hostname or "").endswith(".livekit.cloud"):
                raise ValueError(
                    "Default speech uses LiveKit Cloud; use openrouter-custom-speech for self "
                    "hosting"
                )
        else:
            for url in (result.stt_url, result.tts_url):
                parsed = urlsplit(url)
                if (
                    parsed.username
                    or parsed.password
                    or parsed.query
                    or parsed.fragment
                    or not parsed.hostname
                    or (
                        parsed.scheme != "https"
                        and not (parsed.scheme == "http" and parsed.hostname == "127.0.0.1")
                    )
                ):
                    raise ValueError(
                        "Speech endpoints must use HTTPS or explicit loopback HTTP, without "
                        "embedded credentials"
                    )
        if result.model.startswith(("typesafe/", "~typesafe/")):
            raise ValueError("Jev is a typed decision provider, not the voice chat model")
        return result


def validate_dispatch(room_name: str, metadata: str) -> dict:
    from .contracts import IDENTITY, decode

    if not room_name.startswith("firebird-teaching-") or not IDENTITY.fullmatch(room_name):
        raise ValueError("Voice teaching requires a distinct firebird-teaching-* room")
    data = decode(metadata.encode())
    if set(data) != {"operator_identity", "teaching_session_id"}:
        raise ValueError("Dispatch requires exact operator identity and teaching session")
    for value in data.values():
        if not isinstance(value, str) or not IDENTITY.fullmatch(value):
            raise ValueError("Invalid dispatch identity")
    if not data["operator_identity"].startswith("operator-"):
        raise ValueError("Teaching operator identity must start with operator-")
    return data


class SessionClient(Client):
    """A room cannot continue commanding an executor that restarted underneath it."""

    def __init__(self, origin, token, session_id):
        super().__init__(origin, token)
        self.session_id = session_id

    def request(self, path, body=None):
        if path != "/state":
            self.request("/state")
        value = super().request(path, body)
        if path == "/state" and value.get("session_id") != self.session_id:
            raise ValueError("Teaching executor session changed; create a new room")
        return value


def build_agent(client: Client):
    # Optional SDK is isolated from simulator and dataset conversion processes.
    from livekit.agents import Agent, function_tool

    class TeachingAgent(Agent):
        def __init__(self):
            super().__init__(
                instructions=(
                    "Help the operator teach a simulated robot using only the declared tools. "
                    "A spoken task labels data; it does not invent a working pickup controller. "
                    "Use named joint corrections only when explicitly requested; never infer "
                    "joint motions from a vague task. "
                    "Report a command as applied only when status is acknowledged. "
                    "Queued/executing means pending, "
                    "rejected means not completed; "
                    "do not claim no motion occurred after a simulator fault. "
                    "Use session_state to check actual state. "
                    "Never claim task success, learning, calibration, or training completion. "
                    "Pause immediately when requested. Explain configuration errors plainly."
                )
            )

        @function_tool()
        async def session_state(self) -> dict:
            """Read mode, episode, task and exact native joint names before commanding."""
            return await asyncio.to_thread(client.request, "/state")

        @function_tool()
        async def set_task(self, instruction: str) -> dict:
            """Label the intended task. This does not move the robot."""
            return await asyncio.to_thread(client.command, "task", {"instruction": instruction})

        @function_tool()
        async def start_demonstration(self) -> dict:
            """Start recording a new episode, or resume a paused episode at its held target."""
            return await asyncio.to_thread(client.command, "start", {})

        @function_tool()
        async def pause_demonstration(self) -> dict:
            """Stop further simulation steps and clear the active correction target."""
            return await asyncio.to_thread(client.command, "pause", {})

        @function_tool()
        async def correct_joint(self, joint: str, delta_rad: float) -> dict:
            """
            Apply an explicit named-joint correction of at most0.15 radians, guarded locally.
            """
            return await asyncio.to_thread(
                client.command, "correct", {"joint": joint, "delta_rad": delta_rad}
            )

        @function_tool()
        async def mark_failure(self, reason: str) -> dict:
            """Record the operator's failure report without inferring objective task success."""
            return await asyncio.to_thread(client.command, "mark_failure", {"reason": reason})

        @function_tool()
        async def finish_demonstration(self) -> dict:
            """Finalize this captured episode with its recorded outcome; no training is launched."""
            return await asyncio.to_thread(client.command, "finish", {})

        @function_tool()
        async def reset_scene(self) -> dict:
            """
            Finalize the episode, invalidate queued commands, and reset the scene.
            """
            return await asyncio.to_thread(client.command, "reset", {})

    return TeachingAgent()


async def publish_video(room, client: Client):
    """Preview transport only. Authoritative training pixels stay in the local journal."""
    from livekit import rtc

    source = None
    last = None
    while True:
        frame = await asyncio.to_thread(client.request, "/frame")
        identity = (frame.get("episode_id"), frame.get("step"))
        if "rgb_base64" in frame and identity != last:
            width, height = frame["width"], frame["height"]
            pixels = base64.b64decode(frame["rgb_base64"], validate=True)
            if len(pixels) != width * height * 3:
                raise ValueError("Preview frame is truncated")
            if source is None:
                source = rtc.VideoSource(width, height)
                track = rtc.LocalVideoTrack.create_video_track("teaching-camera", source)
                await room.local_participant.publish_track(track)
            source.capture_frame(rtc.VideoFrame(width, height, rtc.VideoBufferType.RGB24, pixels))
            last = identity
        await asyncio.sleep(1 / 15)


def main():
    # Explicit file load; no shell evaluation, implicit cwd search or secret output.
    if "--env-file" in sys.argv:
        position = sys.argv.index("--env-file")
        if position + 1 >= len(sys.argv):
            raise SystemExit("--env-file requires a path")
        from dotenv import load_dotenv

        load_dotenv(sys.argv[position + 1], override=False)
        del sys.argv[position : position + 2]
    try:
        settings = VoiceSettings.from_env()
    except ValueError as error:
        raise SystemExit(str(error)) from None
    if "--check-config" in sys.argv:
        print(
            json.dumps(
                {
                    "status": "configuration_present",
                    "mode": settings.mode,
                    "network_or_microphone_tested": False,
                }
            )
        )
        return
    from livekit import agents
    from livekit.agents import inference, room_io
    from livekit.plugins import openai, silero

    server = agents.AgentServer(num_idle_processes=0)

    @server.rtc_session(agent_name="firebird-teaching")
    async def entrypoint(ctx: agents.JobContext):
        dispatch = validate_dispatch(ctx.room.name, ctx.job.metadata)
        client = SessionClient(
            settings.control_url, settings.control_token, dispatch["teaching_session_id"]
        )
        state = await asyncio.to_thread(client.request, "/state")
        if state["mode"] in {"starting", "faulted", "closed"}:
            raise ValueError("Teaching executor is not ready")
        if settings.mode == "openrouter":
            stt = inference.STT(model=settings.stt_model, language="en")
            llm = openai.LLM.with_openrouter(
                model=settings.model,
                api_key=settings.openrouter_key,
                parallel_tool_calls=False,
                provider={"require_parameters": True},
            )
            tts = inference.TTS(model=settings.tts_model, voice=settings.tts_voice)
        else:
            stt = openai.STT(
                model=settings.stt_model, base_url=settings.stt_url, api_key=settings.stt_key
            )
            llm = openai.LLM.with_openrouter(
                model=settings.model,
                api_key=settings.openrouter_key,
                parallel_tool_calls=False,
                provider={"require_parameters": True},
            )
            tts = openai.TTS(
                model=settings.tts_model,
                base_url=settings.tts_url,
                api_key=settings.tts_key,
                voice=settings.tts_voice,
            )
        session = agents.AgentSession(vad=silero.VAD.load(), stt=stt, llm=llm, tts=tts)
        await session.start(
            agent=build_agent(client),
            room=ctx.room,
            room_options=room_io.RoomOptions(
                participant_identity=dispatch["operator_identity"],
                close_on_disconnect=True,
                text_input=False,
                video_input=False,
            ),
        )

        async def guarded_video():
            try:
                await publish_video(ctx.room, client)
            except asyncio.CancelledError:
                raise
            except Exception:
                # Stop conversation on preview/executor disconnect; never expose credentials.
                ctx.shutdown(reason="Teaching executor or preview unavailable")

        video = asyncio.create_task(guarded_video())

        async def bounded_session():
            await asyncio.sleep(300)
            ctx.shutdown(reason="Bounded teaching voice session finished")

        deadline = asyncio.create_task(bounded_session())

        async def stop_teaching():
            video.cancel()
            deadline.cancel()
            await asyncio.gather(video, deadline, return_exceptions=True)
            try:
                state = await asyncio.to_thread(client.request, "/state")
                if state.get("mode") == "running":
                    await asyncio.to_thread(client.command, "pause", {})
            except Exception:
                pass  # Executor TTL/session bound still applies; do not claim a successful pause.

        ctx.add_shutdown_callback(stop_teaching)

    agents.cli.run_app(server)


if __name__ == "__main__":
    main()
