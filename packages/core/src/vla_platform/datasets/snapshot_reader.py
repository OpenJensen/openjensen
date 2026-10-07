"""Fixed isolated CPU validator for complete bounded LeRobot v3 video datasets."""

import json
import math
import re
import subprocess
import sys
import threading
from pathlib import Path

DATA = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
VIDEO_INPUT = [
    "-f",
    "mov",
    "-enable_drefs",
    "0",
    "-use_absolute_path",
    "0",
    "-protocol_whitelist",
    "file,pipe",
]


def bounded_output(command, maximum, timeout=30):
    """Bound native diagnostic allocation while retaining the outer worker deadline."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    output = []

    def collect():
        output.append(process.stdout.read(maximum + 1))
        if len(output[0]) > maximum:
            process.kill()

    collector = threading.Thread(target=collect, daemon=True)
    collector.start()
    try:
        process.wait(timeout=timeout)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        collector.join()
        process.stdout.close()
    require(output and len(output[0]) <= maximum, "Video probe output exceeds byte budget")
    require(process.returncode == 0, "Camera video probe failed")
    return output[0]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def read_json(root, name):
    path = root / name
    require(path.stat().st_size <= 8 * 1024**2, f"Metadata too large: {name}")
    return json.loads(path.read_bytes(), object_pairs_hook=unique)


def finite(value):
    if isinstance(value, list):
        return bool(value) and all(finite(item) for item in value)
    return type(value) in (int, float) and math.isfinite(value)


def rows(path, limits, features=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    with path.open("rb") as handle:
        table = pq.ParquetFile(handle)
        if features is not None:
            for key, feature in features.items():
                require(key in table.schema_arrow.names, f"Missing declared frame column: {key}")
                actual = table.schema_arrow.field(key).type
                while pa.types.is_list(actual) or pa.types.is_fixed_size_list(actual):
                    actual = actual.value_type
                expected = {
                    "float32": pa.float32(),
                    "float64": pa.float64(),
                    "int64": pa.int64(),
                    "bool": pa.bool_(),
                }.get(feature.get("dtype"))
                require(
                    expected is not None and actual == expected,
                    f"Frame storage dtype differs from declared feature: {key}",
                )
        metadata = table.metadata
        require(metadata.num_rows <= limits["max_rows"], "Parquet row budget exceeded")
        decoded = sum(metadata.row_group(i).total_byte_size for i in range(metadata.num_row_groups))
        require(decoded <= limits["max_decoded_bytes"], "Parquet decoded byte budget exceeded")
        for batch in table.iter_batches(batch_size=512, use_threads=False):
            require(
                batch.nbytes <= limits["max_decoded_bytes"], "Parquet batch byte budget exceeded"
            )
            yield from batch.to_pylist()


def validate(root, limits):
    import pyarrow

    require(
        pyarrow.__version__ == "25.0.1",
        "Complete validation requires the pinned PyArrow25.0.1 reader",
    )
    info = read_json(root, "meta/info.json")
    require(
        info.get("codebase_version") == "v3.0",
        "Local training currently supports finalized LeRobot v3.0",
    )
    require(
        info.get("data_path") == DATA and info.get("video_path") == VIDEO,
        "Local training requires the standard v3 Parquet/video layout",
    )
    episodes_count, frames_count = info.get("total_episodes"), info.get("total_frames")
    require(
        integer(episodes_count, 2) and episodes_count <= limits["max_episodes"],
        "Need2+ finalized episodes within budget",
    )
    require(
        integer(frames_count, episodes_count) and frames_count <= limits["max_rows"],
        "Invalid or oversized frame count",
    )
    fps = info.get("fps")
    require(integer(fps, 1) and fps <= 240, "Supported integer camera/control fps is1..240")
    features = info.get("features", {})
    require(isinstance(features, dict), "Invalid feature metadata")
    cameras = {key: value for key, value in features.items() if value.get("dtype") == "video"}
    require(1 <= len(cameras) <= 8, "Need1..8 finalized video cameras")
    for key, value in cameras.items():
        shape = value.get("shape")
        require(
            re.fullmatch(r"observation\.images\.[a-zA-Z0-9_.-]+", key) is not None,
            "Unsupported camera name",
        )
        require(
            isinstance(shape, list)
            and len(shape) == 3
            and all(integer(n, 1) for n in shape)
            and shape[0] <= 2160
            and shape[1] <= 3840
            and shape[2] == 3,
            "Video camera shape must be height,width,3 within2160x3840",
        )
    for key in ("action", "observation.state"):
        feature = features.get(key, {})
        require(
            feature.get("dtype") == "float32"
            and isinstance(feature.get("shape"), list)
            and len(feature["shape"]) == 1
            and integer(feature["shape"][0], 1)
            and feature["shape"][0] <= 256,
            f"Unsupported training feature: {key}",
        )
    stats = read_json(root, "meta/stats.json")
    for key in ("action", "observation.state", *cameras):
        value = stats.get(key, {})
        for kind in ("mean", "std", "min", "max", "count"):
            require(
                finite(value.get(kind)),
                f"Missing or nonfinite normalization statistics: {key}.{kind}",
            )

        def flat(v):
            return [x for item in v for x in flat(item)] if isinstance(v, list) else [v]

        def has_shape(value, shape):
            if not shape:
                return type(value) in (int, float)
            return (
                isinstance(value, list)
                and len(value) == shape[0]
                and all(has_shape(item, shape[1:]) for item in value)
            )

        shapes = ((3,), (3, 1, 1)) if key in cameras else (tuple(features[key]["shape"]),)
        require(
            all(
                any(has_shape(value[kind], shape) for shape in shapes)
                for kind in ("mean", "std", "min", "max")
            ),
            f"Normalization statistics broadcast shape mismatch: {key}",
        )
        require(has_shape(value["count"], (1,)), f"Statistics count shape mismatch: {key}")
        required = 3 if key in cameras else features[key]["shape"][0]
        require(
            all(len(flat(value[kind])) == required for kind in ("mean", "std", "min", "max")),
            f"Normalization statistics shape mismatch: {key}",
        )
        require(
            all(v >= 0 for v in flat(value["std"])) and all(v > 0 for v in flat(value["count"])),
            f"Invalid normalization scale/count: {key}",
        )
    tasks = list(rows(root / "meta/tasks.parquet", limits))
    require(
        len(tasks) == info.get("total_tasks") and len(tasks) > 0,
        "Task count differs from finalized metadata",
    )
    task_ids = set()
    task_names = {}
    for row in tasks:
        task = row.get("task", row.get("__index_level_0__"))
        require(
            integer(row.get("task_index")) and isinstance(task, str) and bool(task.strip()),
            "Invalid task table",
        )
        task_ids.add(row["task_index"])
        task_names[row["task_index"]] = task
    require(task_ids == set(range(len(tasks))), "Task indices must be unique and contiguous")
    require(len(set(task_names.values())) == len(tasks), "Duplicate task descriptions")
    episodes = []
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        episodes.extend(rows(path, limits))
        require(len(episodes) <= episodes_count, "Episode metadata exceeds finalized count")
    require(len(episodes) == episodes_count, "Missing or unfinished episode metadata")
    episodes.sort(key=lambda row: row["episode_index"])
    position = 0
    media = {}
    expected_data = set()
    for index, episode in enumerate(episodes):
        require(
            episode.get("episode_index") == index and integer(episode.get("episode_index")),
            "Episode IDs must be contiguous",
        )
        length = episode.get("length")
        require(
            integer(length, 1)
            and episode.get("dataset_from_index") == position
            and episode.get("dataset_to_index") == position + length,
            "Episode row ranges are incomplete or overlap",
        )
        require(
            isinstance(episode.get("tasks"), list)
            and episode["tasks"]
            and all(isinstance(task, str) and task.strip() for task in episode["tasks"]),
            "Episode tasks are missing",
        )
        position += length
        for key in ("data/chunk_index", "data/file_index"):
            require(integer(episode.get(key)), "Episode data file identity missing")
        data_path = DATA.format(
            chunk_index=episode["data/chunk_index"], file_index=episode["data/file_index"]
        )
        expected_data.add(data_path)
        for camera in cameras:
            prefix = "videos/" + camera
            chunk, file = episode.get(prefix + "/chunk_index"), episode.get(prefix + "/file_index")
            start, end = (
                episode.get(prefix + "/from_timestamp"),
                episode.get(prefix + "/to_timestamp"),
            )
            require(
                integer(chunk)
                and integer(file)
                and finite(start)
                and finite(end)
                and start >= 0
                and abs(end - start - length / fps) < 1e-4,
                f"Missing/incomplete video range for episode{index}: {camera}",
            )
            name = VIDEO.format(video_key=camera, chunk_index=chunk, file_index=file)
            ranges = media.setdefault(name, {"camera": camera, "ranges": []})["ranges"]
            require(
                abs(start - (ranges[-1][1] if ranges else 0)) < 1e-4,
                "Video ranges contain gaps or overlap",
            )
            ranges.append((start, end))
    require(position == frames_count, "Finalized episode lengths differ from total_frames")
    actual_data = {path.relative_to(root).as_posix() for path in (root / "data").rglob("*.parquet")}
    require(actual_data == expected_data, "Missing or unreferenced data table")
    position = 0
    observed_tasks = {index: set() for index in range(episodes_count)}
    for name in sorted(actual_data):
        numeric_features = {key: value for key, value in features.items() if key not in cameras}
        for row in rows(root / name, limits, numeric_features):
            require(
                set(row) == set(features) - set(cameras),
                "Frame columns differ from declared features",
            )
            require(
                position < frames_count and integer(row.get("index")) and row["index"] == position,
                "Frame indices are missing, duplicate or unfinalized",
            )
            ep = row.get("episode_index")
            require(integer(ep) and ep < episodes_count, "Frame episode index invalid")
            episode = episodes[ep]
            frame = position - episode["dataset_from_index"]
            require(
                0 <= frame < episode["length"]
                and row.get("frame_index") == frame
                and integer(row.get("frame_index")),
                "Frame is outside its episode range",
            )
            require(
                name
                == DATA.format(
                    chunk_index=episode["data/chunk_index"], file_index=episode["data/file_index"]
                ),
                "Episode points to a different data file",
            )
            require(
                finite(row.get("timestamp")) and abs(row["timestamp"] - frame / fps) <= 1e-4,
                "Frame timestamps do not match control cadence",
            )
            require(
                integer(row.get("task_index")) and row["task_index"] in task_ids,
                "Unknown frame task",
            )
            observed_tasks[ep].add(task_names[row["task_index"]])
            for key in ("action", "observation.state"):
                value = row.get(key)
                require(
                    isinstance(value, list)
                    and len(value) == features[key]["shape"][0]
                    and all(type(v) in (float, int) and math.isfinite(v) for v in value),
                    f"Invalid {key} vector",
                )
            for key, feature in numeric_features.items():
                shape = feature.get("shape")
                require(
                    isinstance(shape, list)
                    and 1 <= len(shape) <= 4
                    and all(integer(n, 1) for n in shape)
                    and math.prod(shape) <= 65536,
                    f"Unsupported numeric feature shape: {key}",
                )
                values = row[key]

                def leaves(value):
                    return (
                        [leaf for part in value for leaf in leaves(part)]
                        if isinstance(value, list)
                        else [value]
                    )

                values = leaves(values)
                require(len(values) == math.prod(shape), f"Frame feature shape mismatch: {key}")
                kind = feature["dtype"]
                require(
                    all(
                        (
                            type(value) is bool
                            if kind == "bool"
                            else type(value) is int
                            if kind == "int64"
                            else type(value) in (int, float) and math.isfinite(value)
                        )
                        for value in values
                    ),
                    f"Invalid numeric feature: {key}",
                )
            position += 1
    require(position == frames_count, "Missing finalized frame rows")
    require(
        all(
            observed_tasks[index] == set(episode["tasks"]) for index, episode in enumerate(episodes)
        ),
        "Episode task labels differ from actual frame task IDs",
    )
    actual_media = {path.relative_to(root).as_posix() for path in (root / "videos").rglob("*.mp4")}
    require(actual_media == set(media), "Missing or unreferenced camera media")
    for name, entry in media.items():
        path = root / name
        result = bounded_output(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_read_frames,r_frame_rate",
                "-of",
                "json",
                *VIDEO_INPUT,
                str(path),
            ],
            maximum=16384,
        )
        streams = json.loads(result)["streams"]
        require(len(streams) == 1, "Camera video has no decodable stream")
        stream = streams[0]
        height, width, _ = cameras[entry["camera"]]["shape"]
        rate = stream["r_frame_rate"].split("/")
        require(
            stream["height"] == height
            and stream["width"] == width
            and abs(int(rate[0]) / int(rate[1]) - fps) < 1e-6
            and int(stream["nb_read_frames"]) == round(entry["ranges"][-1][1] * fps),
            "Camera dimensions, cadence or decoded frame count mismatch",
        )
        count = int(stream["nb_read_frames"])
        timing = bounded_output(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-select_streams",
                "v:0",
                "-show_entries",
                # The common section name also works with FFmpeg 4.4; its
                # frame-specific alias is unavailable on older installations.
                "frame=best_effort_timestamp_time:side_data=",
                "-of",
                "json=compact=1",
                *VIDEO_INPUT,
                str(path),
            ],
            maximum=min(limits["max_decoded_bytes"], count * 128 + 16384),
        )
        decoded_frames = json.loads(timing)["frames"]
        require(len(decoded_frames) == count, "Decoded video timestamps are incomplete")
        for index, frame in enumerate(decoded_frames):
            timestamp = float(frame["best_effort_timestamp_time"])
            require(
                math.isfinite(timestamp) and abs(timestamp - index / fps) < 1e-4,
                "Camera frame timestamps differ from the declared control cadence",
            )
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "quiet",
                "-xerror",
                *VIDEO_INPUT,
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-f",
                "null",
                "-",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=True,
        )
    sidecar = root / "meta/firebird-lineage.json"
    declared = sidecar.exists()
    if (root / "meta/firebird-demonstrations.json").exists():
        require(declared, "Recorded demonstration provenance requires complete episode lineage")
        provenance = read_json(root, "meta/firebird-demonstrations.json")
        require(
            isinstance(provenance, dict)
            and type(provenance.get("schema_version")) is int
            and provenance["schema_version"] == 1,
            "Unsupported demonstration provenance",
        )
    if declared:
        lineage = read_json(root, "meta/firebird-lineage.json")
        require(
            type(lineage.get("schema_version")) is int
            and lineage["schema_version"] == 1
            and set(lineage) == {"schema_version", "episodes"},
            "Unsupported lineage declaration",
        )
        lineage = lineage["episodes"]
        require(
            isinstance(lineage, list) and len(lineage) == episodes_count,
            "Lineage must cover every episode",
        )
        lineage = sorted(lineage, key=lambda row: row["episode_index"])
        for index, row in enumerate(lineage):
            require(
                set(row) == {"episode_index", "origin", "lineage_group"}
                and integer(row["episode_index"])
                and row["episode_index"] == index
                and row["origin"] in ("recorded", "imported", "augmented", "synthetic")
                and isinstance(row["lineage_group"], str)
                and re.fullmatch(r"[a-zA-Z0-9_.:-]{1,160}", row["lineage_group"]),
                "Invalid or incomplete episode lineage",
            )
    else:
        lineage = [
            {"episode_index": index, "origin": "imported", "lineage_group": f"unknown:{index}"}
            for index in range(episodes_count)
        ]
    warnings = [
        "Action units, robot calibration and task success "
        "are not established by dataset validation."
    ]
    if declared:
        warnings.append(
            "Declared lineage coverage was checked; "
            "collection ancestry is not independently verified."
        )
    if not declared:
        warnings.append(
            "Imported dataset ancestry is unknown; held-out episodes do not "
            "establish independent generalization."
        )
    return {
        "total_episodes": episodes_count,
        "total_frames": frames_count,
        "fps": fps,
        "features": features,
        "lineage": lineage,
        "lineage_validated": declared,
        "warnings": warnings,
    }


if __name__ == "__main__":
    try:
        limits = json.loads(sys.stdin.buffer.read(16384))
        print(json.dumps(validate(Path(sys.argv[1]), limits), allow_nan=False))
    except Exception as exc:
        print(json.dumps({"error": f"Dataset validation failed: {exc}"}))
        sys.exit(1)
