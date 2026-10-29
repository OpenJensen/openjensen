"""Complete pinned dataset catalog and agent-selected, bounded video investigation."""

import json
import math

from vla_platform.augmentation.source import download_video
from vla_platform.datasets.explore import (
    MAX_PARQUET_BYTES,
    Budget,
    feature_names,
    hub_url,
    source_profile,
    summary,
)
from vla_platform.datasets.hub_parquet import read_parquet
from vla_platform.datasets.local_preview import reader_python
from vla_platform.scenes.agent import SceneError, command, run_role
from vla_platform.scenes.contracts import Investigation


def investigation_rounds(video_entries):
    if type(video_entries) is not int or not 1 <= video_entries <= 96:
        raise SceneError("Dataset investigation supports 1 to 96 episode-camera videos.")
    return max(3, math.ceil(video_entries / 4) + 1)


JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


async def catalog(scenes, run, source, explorer):
    from .service import digest, write

    profile = source_profile(source)
    budget = Budget()
    info = await explorer.metadata(profile, budget)
    rows = await explorer.episodes(profile, budget, 0, profile.total_episodes)
    if len(rows) != profile.total_episodes or sorted(row["episode_index"] for row in rows) != list(
        range(profile.total_episodes)
    ):
        raise SceneError(
            "The complete dataset episode index is unavailable; no partial catalog was substituted."
        )
    episodes = []
    for row in rows:
        item = summary(row, profile.fps)
        warnings = []
        cameras = await explorer.cameras(profile, info, row, item, budget, warnings, verify=False)
        expected = {key for key, value in profile.features.items() if value.get("dtype") == "video"}
        if warnings or {camera.key for camera in cameras} != expected:
            raise SceneError("The dataset camera catalog is incomplete or unsupported.")
        episodes.append({**item.model_dump(), "cameras": [c.model_dump() for c in cameras]})
    result = {
        "scope": "entire_dataset",
        "source": run.source,
        "total_episodes": profile.total_episodes,
        "fps": profile.fps,
        "state_names": feature_names(profile.features, "observation.state"),
        "episodes": episodes,
    }
    if len(json.dumps(result).encode()) > 2 * 1024**2:
        raise SceneError("The complete dataset catalog exceeds the investigation size limit.")
    path = scenes.folder(run.id) / "dataset-catalog.json"
    write(path, result)
    run.artifacts.append(path.name)
    run.source.update(
        total_episodes=profile.total_episodes, catalog_sha256=digest(path), scope="entire_dataset"
    )
    scenes.save(run)
    return result, info, {row["episode_index"]: row for row in rows}


async def investigate(scenes, run, source, explorer):
    from .service import GUARD, digest, write

    folder = scenes.folder(run.id)
    inventory, info, rows = await catalog(scenes, run, source, explorer)
    frames, clips, seen = [], [], set()
    rounds = investigation_rounds(sum(len(episode["cameras"]) for episode in inventory["episodes"]))
    videos = {}
    selected = None
    try:
        for round_index in range(rounds):
            work = folder / f"investigation-{round_index + 1}"
            work.mkdir(mode=0o700)
            scenes.event(
                run,
                "investigating",
                f"Codex is selecting dataset evidence ({round_index + 1}/{rounds}).",
            )
            decision, tokens = await run_role(
                work,
                "investigation",
                Investigation,
                GUARD + "\nYou have the complete dataset episode/video catalog. Select informative "
                "episode/camera time intervals to inspect. The host will decode five timestamped "
                "frames from every requested interval, and provide those frames on the next round. "
                "Request additional intervals when evidence is missing; you need not inspect every "
                "video. ready may be true only after viewing evidence. Choose replay_episode from "
                "an inspected episode. Select diverse views of the workspace and motion. "
                f"At most {rounds} investigation rounds and four clips per round are available. "
                "The visual input is decoded temporal frames, rather than video bytes. "
                "Catalog: "
                + json.dumps(inventory)
                + "\nAlready inspected clips: "
                + json.dumps(clips),
                frames,
                scenes.data_dir,
                model=run.agent.model,
                reasoning_effort=run.agent.reasoning_effort,
            )
            for key, value in tokens.items():
                run.usage[key] = run.usage.get(key, 0) + value
            scenes.save(run)
            run.artifacts.append(str((work / "investigation.json").relative_to(folder)))
            if decision.ready and not decision.clips and frames:
                selected = decision.replay_episode
                break
            if not decision.clips:
                raise SceneError("Agent requested no video evidence before reconstruction.")
            if round_index == rounds - 1:
                raise SceneError(
                    "Agent investigation needs more evidence; its configured budget was reached."
                )
            for requested in decision.clips:
                episode = next(
                    (
                        ep
                        for ep in inventory["episodes"]
                        if ep["episode_index"] == requested.episode_index
                    ),
                    None,
                )
                camera = (
                    next((c for c in episode["cameras"] if c["key"] == requested.camera_key), None)
                    if episode
                    else None
                )
                if camera is None:
                    raise SceneError("Agent selected a video outside the dataset catalog.")
                identity = (
                    requested.episode_index,
                    requested.camera_key,
                    requested.start_fraction,
                    requested.end_fraction,
                )
                if identity in seen:
                    raise SceneError("Agent repeated an already inspected video segment.")
                seen.add(identity)
                if camera["url"] not in videos:
                    target = folder / f".source-{len(videos)}.mp4"
                    await download_video(explorer.client, camera["url"], target)
                    videos[camera["url"]] = target
                    if sum(path.stat().st_size for path in videos.values()) > 1024**3:
                        raise SceneError(
                            "Dataset video investigation exceeds its 1 GiB download budget."
                        )
                video = videos[camera["url"]]
                duration = camera["end_seconds"] - camera["start_seconds"]
                start = camera["start_seconds"] + duration * requested.start_fraction
                span = duration * (requested.end_fraction - requested.start_fraction)
                if not 0 < span <= 600:
                    raise SceneError("Selected video interval exceeds the investigation limit.")
                clip_index = len(clips)
                evidence_frames = []
                for index, fraction in enumerate((0, 0.25, 0.5, 0.75, 0.98)):
                    target = folder / f"evidence-{clip_index:02}-{index}.jpg"
                    await command(
                        [
                            "ffmpeg",
                            "-nostdin",
                            "-v",
                            "error",
                            "-threads",
                            "2",
                            "-protocol_whitelist",
                            "file,pipe",
                            "-ss",
                            str(start + span * fraction),
                            "-i",
                            video,
                            "-frames:v",
                            "1",
                            "-vf",
                            "scale=960:540:force_original_aspect_ratio=decrease",
                            "-threads",
                            "2",
                            target,
                        ],
                        cwd=folder,
                        timeout=45,
                    )
                    if not target.is_file() or target.stat().st_size < 1024:
                        raise SceneError(
                            "Selected video segment did not yield the required frames."
                        )
                    frames.append(target)
                    run.artifacts.append(target.name)
                    evidence_frames.append(
                        {
                            "path": target.name,
                            "seconds": start + span * fraction,
                            "sha256": digest(target),
                        }
                    )
                clips.append(
                    {
                        **requested.model_dump(),
                        "video_sha256": digest(video),
                        "frames": evidence_frames,
                    }
                )
            selected = decision.replay_episode
        inspected = sorted({clip["episode_index"] for clip in clips})
        if selected not in inspected:
            raise SceneError("Replay episode was not among the inspected dataset videos.")
        profile = source_profile(source)
        row = rows[selected]
        item = summary(row, profile.fps)
        raw = await explorer.reader.read(
            hub_url(profile, explorer.data_path(profile, info, row)), MAX_PARQUET_BYTES, Budget()
        )
        trajectory_rows = await read_parquet(
            raw,
            "trajectory",
            episode_index=selected,
            python=explorer.reader_python or reader_python(scenes.data_dir),
        )
        names = feature_names(profile.features, "observation.state")
        if [name.removesuffix(".pos") for name in names] != JOINTS:
            raise SceneError("Recorded state columns do not match the registered robot joints.")
        if len(trajectory_rows) != item.frame_count or not 2 <= len(trajectory_rows) <= 18000:
            raise SceneError(
                "A complete recorded episode is required; preview samples cannot be replayed."
            )
        if [row["frame_index"] for row in trajectory_rows] != list(range(item.frame_count)):
            raise SceneError("Recorded episode frames are not complete and contiguous.")
        samples = []
        previous = -1
        for row in trajectory_rows:
            stamp, positions = row["timestamp"], row["state"]
            if (
                isinstance(stamp, bool)
                or not isinstance(stamp, (int, float))
                or not math.isfinite(stamp)
                or stamp <= previous
            ):
                raise SceneError("Recorded episode timestamps are invalid.")
            if abs(stamp - row["frame_index"] / profile.fps) > 0.5 / profile.fps:
                raise SceneError("Recorded timestamps differ from the dataset frame rate.")
            if (
                not isinstance(positions, list)
                or len(positions) != 6
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or abs(value) > 200
                    for value in positions
                )
            ):
                raise SceneError("Recorded joint states are invalid or outside the replay bound.")
            samples.append({"timestamp": stamp, "positions": positions})
            previous = stamp
        if not float(profile.fps).is_integer() or not 1 <= profile.fps <= 60:
            raise SceneError("Recorded replay requires an integer dataset frame rate from 1 to 60.")
        trajectory = {
            "episode_index": selected,
            "fps": int(profile.fps),
            "joint_names": names,
            "frames": samples,
            "mapping_status": "uncalibrated_dataset_normalized",
        }
        write(folder / "trajectory.json", trajectory)
        run.artifacts.append("trajectory.json")
        run.source.update(
            inspected_episodes=inspected,
            replay_episode=selected,
            replay_frames=len(samples),
            trajectory_sha256=digest(folder / "trajectory.json"),
        )
        evidence = {
            "source": run.source,
            "dataset_catalog": inventory,
            "inspected_clips": clips,
            "replay": {
                "episode_index": selected,
                "frame_count": len(samples),
                "fps": profile.fps,
                "trajectory_sha256": run.source["trajectory_sha256"],
                "mapping_status": trajectory["mapping_status"],
            },
            "template": json.loads((scenes.template / "scene_config.json").read_text()),
            "template_source_sha256": digest(scenes.template / "scene_config.json"),
        }
        write(folder / "evidence.json", evidence)
        run.artifacts.append("evidence.json")
        run.gates["source_evidence"] = True
        scenes.save(run)
        return frames, evidence
    finally:
        for video in videos.values():
            video.unlink(missing_ok=True)
