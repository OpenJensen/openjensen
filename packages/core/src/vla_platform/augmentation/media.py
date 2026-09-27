"""Bounded, local-only video trimming. No model execution or GPU worker required."""

import asyncio
import json
from pathlib import Path

MAX_INPUT_BYTES = 14 * 1024 * 1024  # Leave room for base64 and JSON within a 20 MiB request.
MAX_OUTPUT_BYTES = 40 * 1024 * 1024


async def command(*args: str) -> bytes:
    process = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
    )
    try:
        output, _ = await asyncio.wait_for(process.communicate(), 90)
        if process.returncode:
            raise ValueError("Video processing failed; check the camera's video format")
        return output
    finally:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()


async def duration(path: Path) -> float:
    raw = await command(
        "ffprobe",
        "-v",
        "error",
        "-protocol_whitelist",
        "file,pipe",
        "-f",
        "mov",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=duration,width,height",
        "-of",
        "json",
        str(path),
    )
    try:
        stream = json.loads(raw)["streams"][0]
        seconds = float(stream["duration"])
        if not 0 < seconds <= 10 or not 0 < stream["width"] <= 4096:
            raise ValueError()
        if not 0 < stream["height"] <= 4096:
            raise ValueError()
        return seconds
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ValueError("Expected a valid video clip of at most 10 seconds") from exc


async def trim(source: Path, target: Path, start: float, seconds: float) -> None:
    await command(
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-threads",
        "2",
        "-protocol_whitelist",
        "file,pipe",
        "-f",
        "mov",
        "-i",
        str(source),
        "-ss",
        str(start),
        "-t",
        str(seconds),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        "scale=w=1280:h=720:force_original_aspect_ratio=decrease:force_divisible_by=2",
        "-c:v",
        "libx264",
        "-threads",
        "2",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-fs",
        str(MAX_INPUT_BYTES),
        str(target),
    )
    if target.stat().st_size >= MAX_INPUT_BYTES:
        raise ValueError("The selected clip exceeds the 14 MiB upload limit")
    actual = await duration(target)
    if abs(actual - seconds) > 0.15:
        raise ValueError("Camera media does not cover the requested clip interval")


async def verify_output(path: Path, source_seconds: float) -> None:
    if path.stat().st_size > MAX_OUTPUT_BYTES:
        raise ValueError("Generated video exceeds the output limit")
    actual = await duration(path)
    if abs(actual - source_seconds) > 0.15:
        raise ValueError("Gemini changed the clip duration; this result cannot preserve timing")
