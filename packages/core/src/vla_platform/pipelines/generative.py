"""Generative appearance edits for whole episodes, kept only when the robot did not move.

Gemini Omni repaints a camera clip from a text request. It can also redraw the arm in a
different pose, which would pair the recorded joint angles with a picture of another pose.
Every edit therefore passes a gate: the Decisions API compares original and edited frames
at several moments and the edit is kept only if the arm, gripper, object and goal match.
"""

import asyncio
import base64
import os
import shutil
import tempfile
from pathlib import Path

import httpx

from vla_platform.augmentation.contracts import PRESERVATION_PROMPT
from vla_platform.augmentation.provider import AugmentationError, GeminiOmniProvider
from vla_platform.pipelines import vision

PROMPTS = [
    "Change only the white tabletop surface to matte light oak wood.",
    "Change the lighting to warm late-afternoon sunlight with soft shadows.",
    "Change only the white tabletop surface to a matte dark grey.",
]
# Gemini Omni video output, standard tier, per second of output video.
USD_PER_SECOND = 0.10
GATE_MOMENTS = 5
SAME_SCENE = (
    "Ignoring colours, lighting and surface textures: is the robot arm in the same pose in "
    "both images (same joint angles, within a few degrees), with the gripper opened the same "
    "amount, and are the object and the goal in the same places?"
)
ENCODE = [
    "-an",
    "-c:v",
    "libx264",
    "-pix_fmt",
    "yuv420p",
    "-crf",
    "20",
    "-g",
    "2",
    "-keyint_min",
    "2",
    "-sc_threshold",
    "0",
    "-bf",
    "0",
]


async def ffmpeg(*args):
    process = await asyncio.create_subprocess_exec(
        shutil.which("ffmpeg") or "ffmpeg",
        "-v",
        "error",
        "-y",
        *args,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, error = await process.communicate()
    if process.returncode:
        raise AugmentationError("Video processing failed: " + error.decode()[-200:])


async def still(video, seconds, start=0.0):
    with tempfile.NamedTemporaryFile(suffix=".jpg") as handle:
        await ffmpeg(
            "-ss",
            f"{start + seconds:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-vf",
            "scale=512:384",
            "-q:v",
            "4",
            handle.name,
        )
        return "data:image/jpeg;base64," + base64.b64encode(Path(handle.name).read_bytes()).decode()


async def gate(client, key, original, original_start, edited, task, length, fps):
    """Probability, per moment, that the edit shows the same physical scene state."""
    questions = []
    # Frame numbers spread over the episode; seeking half a frame early lands on each one.
    frames = [min(length - 1, int((k + 0.5) * length / GATE_MOMENTS)) for k in range(GATE_MOMENTS)]
    moments = [round(max(0.0, index - 0.5) / fps, 4) for index in frames]
    for number, seconds in enumerate(moments):
        questions.append(
            {
                "type": "predicate",
                "name": f"same_{number}",
                "instructions": [
                    {
                        "type": "input_text",
                        "text": f"First image: the original camera frame at {seconds} s.",
                    },
                    {
                        "type": "input_image",
                        "image_url": await still(original, seconds, original_start),
                    },
                    {
                        "type": "input_text",
                        "text": "Second image: the same moment after an appearance edit.",
                    },
                    {"type": "input_image", "image_url": await still(edited, seconds)},
                    {
                        "type": "input_text",
                        "text": SAME_SCENE,
                    },
                ],
            }
        )
    body = {
        "model": vision.DECISIONS_MODEL,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"Robot demonstration of the task: {task}"}
                ],
            }
        ],
        "questions": questions,
    }
    response = await client.post(
        vision.DECISIONS_API, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=180
    )
    if response.status_code >= 400:
        raise vision.VisionError(f"The pose check was refused: HTTP {response.status_code}")
    payload = response.json()
    by_name = {answer["name"]: answer for answer in payload.get("answers", [])}
    probabilities = [
        round(by_name.get(f"same_{n}", {}).get("probability", 0.0), 3) for n in range(GATE_MOMENTS)
    ]
    return probabilities, int((payload.get("usage") or {}).get("input_tokens", 0))


async def edit_entries(entries, params, folder, *, provider, decisions_key, progress=None):
    """Append gated Gemini copies of selected kept recordings; return a summary."""
    camera = params.get("camera", "observation.images.front")
    prompts = params.get("prompts") or PROMPTS
    threshold = float(params.get("threshold", 0.6))
    candidates = [
        e for e in entries if e["keep"] and e["origin"] == "recorded" and camera in e["videos"]
    ]
    count = int(params.get("count", 10))
    step = max(1, len(candidates) // max(1, count))
    chosen = candidates[::step][:count]
    limit = asyncio.Semaphore(int(params.get("concurrency", 3)))
    results, seconds_billed, gate_tokens = [], 0.0, 0
    folder = Path(folder)

    async with httpx.AsyncClient() as client:

        async def one(position, entry):
            nonlocal seconds_billed, gate_tokens
            prompt = prompts[position % len(prompts)]
            video = entry["videos"][camera]
            target = folder / f"{entry['id']}-g{position}"
            target.mkdir(parents=True, exist_ok=True)
            source = target / "source.mp4"
            if not source.exists():
                await ffmpeg(
                    "-ss",
                    f"{video['from']:.6f}",
                    "-i",
                    video["path"],
                    "-frames:v",
                    str(entry["length"]),
                    *ENCODE,
                    str(source),
                )
            raw = target / "gemini.mp4"
            async with limit:
                if not raw.exists():
                    edited, _ = await provider.edit(
                        source.read_bytes(), PRESERVATION_PROMPT + prompt
                    )
                    raw.write_bytes(edited)
                    seconds_billed += entry["length"] / entry["fps"]
            # Gemini returns 16:9 at 24 fps by stretching the frame; squash it back and
            # take the frame nearest each recorded time, so frame i still shows time i/fps.
            final = target / f"{camera}.mp4"
            width, height = params.get("size", [640, 480])
            await ffmpeg(
                "-i",
                str(raw),
                "-vf",
                f"scale={width}:{height},fps={entry['fps']},tpad=stop_mode=clone:stop=30",
                "-frames:v",
                str(entry["length"]),
                *ENCODE,
                str(final),
            )
            probabilities, tokens = await gate(
                client,
                decisions_key,
                source,
                0.0,
                final,
                entry["task"],
                entry["length"],
                entry["fps"],
            )
            gate_tokens += tokens
            accepted = min(probabilities) >= threshold
            results.append(
                {"id": entry["id"], "prompt": prompt, "gate": probabilities, "accepted": accepted}
            )
            if progress:
                await progress(len(results), len(chosen))
            if not accepted:
                return None
            return {
                **entry,
                "id": target.name,
                "origin": "augmented",
                "derived_from": entry["id"],
                "augmentation": {"method": "gemini_omni", "prompt": prompt, "gate": probabilities},
                "videos": {
                    **entry["videos"],
                    camera: {
                        "path": str(final),
                        "from": 0.0,
                        "to": entry["length"] / entry["fps"],
                        "shared": False,
                    },
                },
                "labels": {k: v for k, v in entry["labels"].items() if k != "frames"},
            }

        made = await asyncio.gather(
            *(one(n, e) for n, e in enumerate(chosen)), return_exceptions=True
        )
    errors = [str(item) for item in made if isinstance(item, BaseException)]
    added = [item for item in made if isinstance(item, dict)]
    entries.extend(added)
    return {
        "edited": len(chosen) - len(errors),
        "accepted": len(added),
        "rejected": [r for r in results if not r["accepted"]],
        "errors": errors[:5],
        "results": results,
        "gemini_seconds": round(seconds_billed, 1),
        "cost_usd": round(
            seconds_billed * USD_PER_SECOND + gate_tokens / 1e6 * vision.DECISIONS_PRICE, 2
        ),
    }


def provider_from(params, api_keys):
    project = params.get("google_cloud_project") or os.environ.get("JENSEN_GEMINI_CLOUD_PROJECT")
    if project:
        return GeminiOmniProvider(google_cloud_project=project)
    key = api_keys.key("gemini") if api_keys is not None else None
    return GeminiOmniProvider(key or os.environ.get("GEMINI_API_KEY"))
