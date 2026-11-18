"""Data-heavy dataset pipeline steps, run in the CPU dataset reader's interpreter.

Every step reads one episode manifest (JSON lines, one episode per line) and writes a
new one, so each stage of a run can be inspected, reviewed and resumed on its own.
The reader starts this file with ``-I``; it imports nothing from the application.
"""

import hashlib
import importlib.util
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path


def _shared():
    """The LeRobot helpers the dataset combiner already trusts, loaded by path."""
    path = Path(__file__).resolve().parents[1] / "datasets" / "import_worker.py"
    spec = importlib.util.spec_from_file_location("jensen_import_worker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shared = _shared()


def read_manifest(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_manifest(path, entries):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        for entry in entries:
            stream.write(json.dumps(entry, allow_nan=False, sort_keys=True) + "\n")
    temporary.replace(path)
    return len(entries)


def ffmpeg():
    return shared.ffmpeg()


def run(command):
    completed = subprocess.run(command, capture_output=True, timeout=600)
    if completed.returncode:
        tail = completed.stderr.decode("utf-8", "replace").strip().splitlines()[-1:]
        raise ValueError("Video processing failed: " + (tail[0] if tail else "unknown error"))


def episode_rows(entry):
    """Frames of one episode, in order, as a column dictionary."""
    import pyarrow.parquet as pq

    table = pq.read_table(entry["data"], filters=[("episode_index", "=", entry["episode_index"])])
    rows = table.to_pydict()
    order = sorted(range(len(rows["frame_index"])), key=rows["frame_index"].__getitem__)
    return {key: [values[i] for i in order] for key, values in rows.items()}


# Steps -----------------------------------------------------------------------------


def read(payload):
    """One manifest entry per recorded episode of a finalized LeRobot v3 dataset."""
    root = Path(payload["root"]).resolve()
    info, tasks, rows = shared._dataset(root)
    videos = [key for key, value in info["features"].items() if value.get("dtype") == "video"]
    usage = {}
    for row in rows:
        for key in videos:
            origin = (key, row[f"videos/{key}/chunk_index"], row[f"videos/{key}/file_index"])
            usage[origin] = usage.get(origin, 0) + 1
    entries = []
    for row in rows:
        data = shared.safe(
            root,
            info["data_path"].format(
                chunk_index=row["data/chunk_index"], file_index=row["data/file_index"]
            ),
        )
        entry = {
            "id": f"e{row['episode_index']:04d}",
            "episode_index": row["episode_index"],
            "root": str(root),
            "data": str(data),
            "length": row["length"],
            "fps": info["fps"],
            "task": (row.get("tasks") or [""])[0],
            "videos": {},
            "keep": True,
            "origin": "recorded",
            "labels": {},
        }
        for key in videos:
            chunk, number = row[f"videos/{key}/chunk_index"], row[f"videos/{key}/file_index"]
            entry["videos"][key] = {
                "path": str(
                    shared.safe(
                        root,
                        info["video_path"].format(
                            video_key=key, chunk_index=chunk, file_index=number
                        ),
                    )
                ),
                "from": row[f"videos/{key}/from_timestamp"],
                "to": row[f"videos/{key}/to_timestamp"],
                "shared": usage[(key, chunk, number)] > 1,
            }
        entries.append(entry)
    write_manifest(payload["output"], entries)
    return {
        "entries": len(entries),
        "cameras": videos,
        "fps": info["fps"],
        "action_names": info["features"]["action"].get("names") or [],
    }


def grasp_events(commanded, measured, open_above=20.0, closed_below=10.0, held_gap=2.5):
    """Grasp attempts read from the commanded and measured gripper positions.

    A gripper that closes on an object stalls short of its command; one that closes on
    nothing reaches it. Counting empty closes before the first stalled close finds missed
    grasps and re-grasps without looking at a single image.
    """
    events, state = [], "closed"
    for frame, value in enumerate(commanded):
        if state == "closed" and value > open_above:
            state = "open"
        elif state == "open" and value < closed_below:
            state = "closed"
            window = range(frame + 3, min(frame + 18, len(commanded)))
            gaps = sorted(measured[i] - commanded[i] for i in window)
            gap = gaps[len(gaps) // 2] if gaps else 0.0
            events.append({"frame": frame, "gap": round(gap, 2), "held": gap >= held_gap})
    for position, event in enumerate(events):
        if not event["held"]:
            continue
        # The object leaves the gripper when the operator opens it, even partly, or
        # when the gripper closes further on its own because the object slipped out.
        end = events[position + 1]["frame"] if position + 1 < len(events) else len(commanded)
        for frame in range(event["frame"] + 18, end):
            opened = commanded[frame] > closed_below + 3
            emptied = all(
                measured[i] - commanded[i] < held_gap * 0.6
                for i in range(frame, min(frame + 3, len(commanded)))
            )
            if opened or emptied:
                event["released_at"] = frame
                break
        if position + 1 < len(events) and "released_at" not in event:
            event["released_at"] = events[position + 1]["frame"]
    first_hold = next((i for i, event in enumerate(events) if event["held"]), None)
    if first_hold is None:
        verdict = "no_grasp"
    elif "released_at" not in events[first_hold]:
        verdict = "not_released"
    elif first_hold > 0:
        verdict = "regrasp"
    else:
        verdict = "clean"
    return {
        "verdict": verdict,
        "missed_grasps": first_hold if first_hold is not None else len(events),
        "closes": len(events),
        "events": events,
    }


def gripper(payload):
    """Label every episode with its grasp attempts; see ``grasp_events``."""
    entries = read_manifest(payload["input"])
    names = payload.get("action_names") or []
    index = next((i for i, name in enumerate(names) if "gripper" in name.lower()), None)
    if index is None:
        write_manifest(payload["output"], entries)
        return {"skipped": "This dataset names no gripper value, so grasps were not counted."}
    thresholds = {
        key: float(payload[key])
        for key in ("open_above", "closed_below", "held_gap")
        if key in payload
    }
    counts = {}
    for entry in entries:
        rows = episode_rows(entry)
        label = grasp_events(
            [vector[index] for vector in rows["action"]],
            [vector[index] for vector in rows["observation.state"]],
            **thresholds,
        )
        counts[label["verdict"]] = counts.get(label["verdict"], 0) + 1
        entry["labels"]["gripper"] = label
    write_manifest(payload["output"], entries)
    return {"verdicts": counts}


def frames(payload):
    """Evenly spaced still frames of each camera, plus one review strip per episode."""
    from PIL import Image, ImageDraw

    entries = read_manifest(payload["input"])
    folder = Path(payload["folder"])
    count = int(payload.get("frames", 8))
    width = int(payload.get("width", 512))
    cameras = payload.get("cameras") or None
    executable = ffmpeg()
    for entry in entries:
        selected = [key for key in entry["videos"] if cameras is None or key in cameras]
        target = folder / entry["id"]
        target.mkdir(parents=True, exist_ok=True)
        stills = {}
        length, fps = entry["length"], entry["fps"]
        # Frame numbers, not times: seeking half a frame early always lands on the frame,
        # including the very last one, whatever the container's timestamp rounding.
        picks = sorted({min(length - 1, int((k + 0.5) * length / count)) for k in range(count)})
        times = [(index, None) for index in picks]
        if payload.get("grasp_frames", True):
            # A still shortly after every gripper close shows whether that grasp held.
            events = (entry["labels"].get("gripper") or {}).get("events", [])[:6]
            for number, event in enumerate(events, start=1):
                times.append(
                    (min(event["frame"] + 15, length - 1), f"after gripper close {number}")
                )
            times.sort(key=lambda item: item[0])
        for key in selected:
            video = entry["videos"][key]
            paths = []
            for k, (index, event) in enumerate(times):
                offset = round(index / fps, 3)
                path = target / f"{key.rsplit('.', 1)[-1]}-{k:02d}.jpg"
                if not path.exists():
                    run(
                        [
                            executable,
                            "-v",
                            "error",
                            "-ss",
                            f"{video['from'] + max(0.0, index - 0.5) / fps:.4f}",
                            "-i",
                            video["path"],
                            "-frames:v",
                            "1",
                            "-vf",
                            f"scale={width}:-2",
                            "-q:v",
                            "3",
                            "-y",
                            str(path),
                        ]
                    )
                paths.append({"path": str(path), "seconds": offset, "event": event})
            stills[key] = paths
        strip = target / "strip.jpg"
        if not strip.exists() and stills:
            thumbs = [[Image.open(item["path"]) for item in stills[key]] for key in selected]
            tile_w = 192
            tile_h = round(thumbs[0][0].height * tile_w / thumbs[0][0].width)
            sheet = Image.new("RGB", (tile_w * len(times), tile_h * len(selected)), "black")
            draw = ImageDraw.Draw(sheet)
            for row, images in enumerate(thumbs):
                for column, image in enumerate(images):
                    sheet.paste(image.resize((tile_w, tile_h)), (column * tile_w, row * tile_h))
                    still = stills[selected[row]][column]
                    label = f"{still['seconds']:.1f}s" + (" grab" if still["event"] else "")
                    colour = (255, 80, 80) if still["event"] else (255, 255, 255)
                    draw.text((column * tile_w + 4, row * tile_h + 2), label, fill=colour)
            sheet.save(strip, quality=82)
        entry["labels"]["frames"] = {"strip": str(strip), "stills": stills}
    write_manifest(payload["output"], entries)
    return {"entries": len(entries)}


def _jitter_filter(params, width, height):
    parts = [
        "eq=brightness={brightness:.3f}:contrast={contrast:.3f}:saturation={saturation:.3f}".format(
            **params
        ),
        "hue=h={hue:.1f}".format(**params),
    ]
    zoom = params.get("zoom", 1.0)
    if zoom < 0.999:
        crop_w, crop_h = int(width * zoom) // 2 * 2, int(height * zoom) // 2 * 2
        x = int((width - crop_w) * params.get("x", 0.5))
        y = int((height - crop_h) * params.get("y", 0.5))
        parts += [f"crop={crop_w}:{crop_h}:{x}:{y}", f"scale={width}:{height}"]
    if params.get("blur", 0) > 0:
        parts.append(f"gblur=sigma={params['blur']:.2f}")
    if params.get("noise", 0) > 0:
        parts.append(f"noise=alls={int(params['noise'])}:allf=t")
    return ",".join(parts)


RANGES = {
    "brightness": (-0.15, 0.15),
    "contrast": (0.8, 1.25),
    "saturation": (0.7, 1.4),
    "hue": (-12.0, 12.0),
    "zoom": (0.85, 1.0),
    "x": (0.0, 1.0),
    "y": (0.0, 1.0),
    "blur": (0.0, 0.0),
    "noise": (0.0, 0.0),
}


def jitter(payload):
    """Copies of kept episodes with one fixed colour, crop and blur change per copy.

    The same change holds for the whole episode, so the policy sees a consistent scene.
    Joint positions, actions and timing are copied unchanged.
    """
    entries = read_manifest(payload["input"])
    folder = Path(payload["folder"])
    copies = int(payload.get("copies", 1))
    cameras = payload.get("cameras") or None
    ranges = {**RANGES, **{k: tuple(v) for k, v in (payload.get("ranges") or {}).items()}}
    executable = ffmpeg()
    output = list(entries)
    made = 0
    for entry in entries:
        if not entry["keep"] or entry["origin"] != "recorded":
            continue
        for copy in range(copies):
            seed = int(
                hashlib.sha256(
                    f"{payload.get('seed', 0)}:{entry['id']}:{copy}".encode()
                ).hexdigest()[:12],
                16,
            )
            rng = random.Random(seed)
            params = {name: rng.uniform(*span) for name, span in ranges.items()}
            ident = f"{entry['id']}-j{copy}"
            videos = {}
            for key, video in entry["videos"].items():
                if cameras is not None and key not in cameras:
                    videos[key] = video
                    continue
                path = folder / ident / f"{key}.mp4"
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    width, height = video_size(video["path"])
                    run(
                        [
                            executable,
                            "-v",
                            "error",
                            "-ss",
                            f"{video['from']:.6f}",
                            "-i",
                            video["path"],
                            "-frames:v",
                            str(entry["length"]),
                            "-vf",
                            _jitter_filter(params, width, height),
                            *ENCODE,
                            "-y",
                            str(path),
                        ]
                    )
                videos[key] = {
                    "path": str(path),
                    "from": 0.0,
                    "to": entry["length"] / entry["fps"],
                    "shared": False,
                }
            output.append(
                {
                    **entry,
                    "id": ident,
                    "origin": "augmented",
                    "derived_from": entry["id"],
                    "augmentation": {"method": "jitter", "params": params},
                    "videos": videos,
                    "labels": {k: v for k, v in entry["labels"].items() if k != "frames"},
                }
            )
            made += 1
    write_manifest(payload["output"], output)
    return {"augmented": made, "entries": len(output)}


def trim(payload):
    """Cut the still time before the arm starts moving and after it stops.

    A fixed-length recording pads every demonstration with seconds of the arm resting;
    a policy trained on them learns to rest. The cut is decided from the actions, so an
    augmented copy is cut exactly like its source. Run it after augmentation.
    """
    entries = read_manifest(payload["input"])
    threshold = float(payload.get("motion_threshold", 0.5))
    before = int(payload.get("pad_before", 5))
    after = int(payload.get("pad_after", 12))
    trimmed = removed = 0
    for entry in entries:
        if not entry["keep"] or "trim" in entry:
            continue
        actions = episode_rows(entry)["action"]
        moving = [
            i
            for i in range(1, len(actions))
            if sum(abs(a - b) for a, b in zip(actions[i], actions[i - 1])) > threshold
        ]
        if not moving:
            continue
        start = max(0, moving[0] - before)
        end = min(len(actions), moving[-1] + 1 + after)
        if start + len(actions) - end < 10:
            continue
        entry["trim"] = [start, end]
        entry["source_length"] = entry["length"]
        entry["length"] = end - start
        trimmed += 1
        removed += entry["source_length"] - entry["length"]
    write_manifest(payload["output"], entries)
    return {"trimmed": trimmed, "frames_removed": removed}


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


def video_size(path):
    probe = shutil.which("ffprobe") or str(Path(ffmpeg()).with_name("ffprobe"))
    completed = subprocess.run(
        [
            probe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        timeout=60,
        check=True,
    )
    width, height = completed.stdout.decode().strip().split(",")[:2]
    return int(width), int(height)


def frame_count(path):
    probe = shutil.which("ffprobe") or str(Path(ffmpeg()).with_name("ffprobe"))
    completed = subprocess.run(
        [
            probe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        timeout=120,
        check=True,
    )
    return int(completed.stdout.decode().strip().split(",")[0])


def image_stats(path, length, samples=24):
    """Per-channel image statistics in LeRobot's [0, 1] layout from sampled frames."""
    import numpy as np

    completed = subprocess.run(
        [
            ffmpeg(),
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            f"select='not(mod(n\\,{max(1, length // samples)}))',scale=160:120",
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
        timeout=300,
        check=True,
    )
    pixels = np.frombuffer(completed.stdout, dtype=np.uint8).reshape(-1, 3) / 255.0

    def shape(values):
        return [[[float(v)]] for v in values]

    stats = {
        "min": shape(pixels.min(0)),
        "max": shape(pixels.max(0)),
        "mean": shape(pixels.mean(0)),
        "std": shape(pixels.std(0)),
        "count": [int(length)],
    }
    for name, q in (("q01", 1), ("q10", 10), ("q50", 50), ("q90", 90), ("q99", 99)):
        stats[name] = shape(np.percentile(pixels, q, axis=0))
    return stats


def numeric_stats(values):
    import numpy as np

    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    stats = {
        "min": array.min(0).tolist(),
        "max": array.max(0).tolist(),
        "mean": array.mean(0).tolist(),
        "std": array.std(0).tolist(),
        "count": [int(array.shape[0])],
    }
    for name, q in (("q01", 1), ("q10", 10), ("q50", 50), ("q90", 90), ("q99", 99)):
        stats[name] = np.percentile(array, q, axis=0).tolist()
    return stats


def export(payload):
    """Write the kept entries as a new LeRobot v3 dataset, one file per episode."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    entries = [entry for entry in read_manifest(payload["input"]) if entry["keep"]]
    if not entries:
        raise ValueError("No episodes are kept, so there is nothing to save")
    output = Path(payload["output"])
    if output.exists() and any(output.iterdir()):
        raise ValueError("The new dataset folder must be empty")
    roots = sorted({entry["root"] for entry in entries})
    sources = {root: shared._dataset(Path(root)) for root in roots}
    first = sources[roots[0]][0]
    for root in roots[1:]:
        if (
            sources[root][0]["features"] != first["features"]
            or sources[root][0]["fps"] != first["fps"]
        ):
            raise ValueError("Every source dataset must share cameras, values and frame rate")
    source_rows = {
        (root, row["episode_index"]): row for root, (_, _, rows) in sources.items() for row in rows
    }
    videos = [key for key, value in first["features"].items() if value.get("dtype") == "video"]
    task_ids, episode_rows_out, lineage = {}, [], []
    frame_total = 0
    executable = ffmpeg()
    for number, entry in enumerate(entries):
        rows = episode_rows(entry)
        start, end = entry.get("trim") or (0, len(rows["frame_index"]))
        rows = {name: values[start:end] for name, values in rows.items()}
        length = len(rows["frame_index"])
        if length != entry["length"]:
            raise ValueError(f"Episode {entry['id']} has {length} frames, not {entry['length']}")
        task = payload.get("task") or entry["task"]
        task_ids.setdefault(task, len(task_ids))
        rows["episode_index"] = [number] * length
        rows["index"] = list(range(frame_total, frame_total + length))
        rows["task_index"] = [task_ids[task]] * length
        rows["frame_index"] = list(range(length))
        rows["timestamp"] = [i / entry["fps"] for i in range(length)]
        schema = pq.ParquetFile(entry["data"], arrow_extensions_enabled=False).schema_arrow
        target = output / f"data/chunk-{number // 1000:03d}/file-{number % 1000:03d}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pydict({name: rows[name] for name in schema.names}, schema=schema),
            target,
        )
        source_row = source_rows[(entry["root"], entry["episode_index"])]
        row = {
            "episode_index": number,
            "tasks": [task],
            "length": length,
            "data/chunk_index": number // 1000,
            "data/file_index": number % 1000,
            "dataset_from_index": frame_total,
            "dataset_to_index": frame_total + length,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        for name in (
            "observation.state",
            "action",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ):
            if name in rows:
                for stat, value in numeric_stats(rows[name]).items():
                    row[f"stats/{name}/{stat}"] = value
        for key in videos:
            video = entry["videos"][key]
            copied = (
                output / f"videos/{key}/chunk-{number // 1000:03d}/file-{number % 1000:03d}.mp4"
            )
            copied.parent.mkdir(parents=True, exist_ok=True)
            original = "trim" not in entry and (
                entry["origin"] == "recorded" or video["path"] == source_videos(sources, entry, key)
            )
            # Seek half a frame early so the cut lands exactly on its first frame.
            offset = (
                video["from"] + max(0.0, start - 0.5) / entry["fps"] if start else video["from"]
            )
            if video["from"] == 0 and not video.get("shared") and "trim" not in entry:
                shutil.copyfile(video["path"], copied)
            else:
                run(
                    [
                        executable,
                        "-v",
                        "error",
                        "-ss",
                        f"{offset:.6f}",
                        "-i",
                        video["path"],
                        "-frames:v",
                        str(length),
                        # A cut keeps its source timestamps; LeRobot expects frame 0 at 0 s.
                        "-vf",
                        "setpts=PTS-STARTPTS",
                        *ENCODE,
                        "-y",
                        str(copied),
                    ]
                )
            if frame_count(copied) < length:
                raise ValueError(f"Episode {entry['id']} camera {key} has too few frames")
            row[f"videos/{key}/chunk_index"] = number // 1000
            row[f"videos/{key}/file_index"] = number % 1000
            row[f"videos/{key}/from_timestamp"] = 0.0
            row[f"videos/{key}/to_timestamp"] = length / entry["fps"]
            if original and f"stats/{key}/mean" in source_row:
                for stat in (
                    "min",
                    "max",
                    "mean",
                    "std",
                    "count",
                    "q01",
                    "q10",
                    "q50",
                    "q90",
                    "q99",
                ):
                    if f"stats/{key}/{stat}" in source_row:
                        row[f"stats/{key}/{stat}"] = source_row[f"stats/{key}/{stat}"]
            else:
                for stat, value in image_stats(copied, length).items():
                    row[f"stats/{key}/{stat}"] = value
        episode_rows_out.append(row)
        # Copies share their source episode's group, so a held-out split never validates
        # on a copy of an episode it trained on.
        source = hashlib.sha256(entry["root"].encode()).hexdigest()[:12]
        lineage.append(
            {
                "episode_index": number,
                "origin": "augmented" if entry["origin"] == "augmented" else "imported",
                "lineage_group": f"{source}:{entry['episode_index']}",
            }
        )
        frame_total += length
    meta = output / "meta"
    (meta / "episodes/chunk-000").mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(episode_rows_out), meta / "episodes/chunk-000/file-000.parquet"
    )
    pq.write_table(
        pa.Table.from_pydict(
            {"task_index": list(task_ids.values()), "task": list(task_ids)},
            schema=pa.schema([("task_index", pa.int64()), ("task", pa.string())]),
        ),
        meta / "tasks.parquet",
    )
    info = {
        **first,
        "total_episodes": len(entries),
        "total_frames": frame_total,
        "total_tasks": len(task_ids),
        "splits": {"train": f"0:{len(entries)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    (meta / "info.json").write_text(json.dumps(info, allow_nan=False, indent=2))
    per_episode = []
    for row in episode_rows_out:
        stats = {}
        for column, value in row.items():
            if column.startswith("stats/"):
                _, feature, stat = column.split("/")
                stats.setdefault(feature, {})[stat] = value
        per_episode.append(stats)
    (meta / "stats.json").write_text(
        json.dumps(shared._combined_statistics(per_episode), allow_nan=False)
    )
    statement = shared._recorder_statement([Path(root) for root in roots])
    if statement is not None:
        (output / "openjensen.json").write_text(
            json.dumps(
                {
                    **statement,
                    "episodes": len(entries),
                    "frames": frame_total,
                    "fps": first["fps"],
                    "task": next(iter(task_ids), ""),
                    "finalized": True,
                    "abandoned": False,
                    "pending_frames": 0,
                    "camera_features": videos,
                },
                allow_nan=False,
            )
        )
    (meta / "firebird-lineage.json").write_text(
        json.dumps({"schema_version": 1, "episodes": lineage}, allow_nan=False, indent=2)
    )
    return {
        "episodes": len(entries),
        "frames": frame_total,
        "augmented": sum(1 for entry in entries if entry["origin"] == "augmented"),
    }


def source_videos(sources, entry, key):
    info = sources[entry["root"]][0]
    for row in sources[entry["root"]][2]:
        if row["episode_index"] == entry["episode_index"]:
            return str(
                shared.safe(
                    Path(entry["root"]),
                    info["video_path"].format(
                        video_key=key,
                        chunk_index=row[f"videos/{key}/chunk_index"],
                        file_index=row[f"videos/{key}/file_index"],
                    ),
                )
            )
    return None


OPERATIONS = {
    "read": read,
    "gripper": gripper,
    "frames": frames,
    "jitter": jitter,
    "trim": trim,
    "export": export,
}


def main():
    payload = json.loads(sys.stdin.read())
    try:
        result = OPERATIONS[payload["operation"]](payload)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(json.dumps({"error": str(error) or type(error).__name__}))
        return 1
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
