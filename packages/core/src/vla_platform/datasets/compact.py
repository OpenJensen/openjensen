"""Create a video-backed LeRobot copy without changing robot samples or the source."""

import io
import json
import re
import shutil
import subprocess
from pathlib import Path


def compact(root, output, *, progress=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    def inside(name):
        path = Path(name)
        target = root / path
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError("Unsafe dataset path")
        if not target.resolve().is_relative_to(root.resolve()) or target.is_symlink():
            raise ValueError("Dataset path escapes its source")
        return target

    def check_space():
        if shutil.disk_usage(output.parent).free < 1024**3:
            raise ValueError("Not enough free storage to compact this dataset")

    info = json.loads(inside("meta/info.json").read_bytes())
    if not str(info.get("codebase_version", "")).startswith("v3."):
        raise ValueError("Compaction supports LeRobot v3 recordings")
    cameras = {
        key: feature for key, feature in info["features"].items() if feature.get("dtype") == "image"
    }
    if not cameras:
        raise ValueError("This dataset already stores its camera images as videos")
    if any(
        not re.fullmatch(r"observation\.images\.[a-zA-Z0-9_.-]+", key)
        or len(feature.get("shape", [])) != 3
        or feature["shape"][2] != 3
        for key, feature in cameras.items()
    ):
        raise ValueError("Compaction needs RGB camera images with fixed dimensions")
    fps = info["fps"]
    if type(fps) not in (int, float) or not 0 < fps <= 240:
        raise ValueError("Invalid recording frame rate")
    binary = shutil.which("ffmpeg")
    probe = shutil.which("ffprobe")
    if not binary or not probe:
        raise ValueError("Install FFmpeg and FFprobe on the app host to compact datasets")
    files = sorted(inside("meta/episodes").rglob("*.parquet"))
    episodes = []
    for file in files:
        episodes.extend(pq.read_table(file).to_pylist())
    if len(episodes) != info["total_episodes"]:
        raise ValueError("Episode metadata does not match this dataset")
    groups = {}
    for episode in episodes:
        key = (episode["data/chunk_index"], episode["data/file_index"])
        groups.setdefault(key, []).append(episode)
    if not groups:
        raise ValueError("This recording has no saved episodes")
    if output.exists() and any(output.iterdir()):
        raise ValueError("The compacted dataset folder must be empty")
    output.mkdir(parents=True, exist_ok=True)
    check_space()
    source_bytes = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    data_names = {
        info["data_path"].format(chunk_index=chunk, file_index=file) for chunk, file in groups
    }
    # Retain tasks, statistics, annotations, calibration and recording metadata.
    # Source files are private workspace data, never executed.
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ValueError("Dataset symlinks are unsupported")
        if not path.is_file() or relative.as_posix() in data_names:
            continue
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(output.parent).free < path.stat().st_size + 1024**3:
            raise ValueError("Not enough free storage to preserve dataset metadata")
        shutil.copyfile(path, target)
    video_path = info.get("video_path") or (
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    )

    def video_file(key, chunk, file):
        name = video_path.format(video_key=key, chunk_index=chunk, file_index=file)
        path = output / name
        if not path.resolve().is_relative_to(output.resolve()):
            raise ValueError("Video path escapes its output folder")
        return path

    frames = 0
    for completed, ((chunk, file), group) in enumerate(sorted(groups.items())):
        check_space()
        name = info["data_path"].format(chunk_index=chunk, file_index=file)
        source = pq.ParquetFile(inside(name), arrow_extensions_enabled=False)
        if source.metadata.num_rows != sum(ep["length"] for ep in group):
            raise ValueError("Compaction requires complete episodes within each data file")
        columns = [name for name in source.schema_arrow.names if name not in cameras]
        schema = pa.schema(
            [source.schema_arrow.field(name) for name in columns],
            metadata=source.schema_arrow.metadata,
        )
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        encoders = {}
        try:
            for key, feature in cameras.items():
                height, width, _ = feature["shape"]
                path = video_file(key, chunk, file)
                path.parent.mkdir(parents=True, exist_ok=True)
                pixel_format = "yuv420p" if height % 2 == width % 2 == 0 else "yuv444p"
                encoders[key] = subprocess.Popen(
                    [
                        binary,
                        "-y",
                        "-v",
                        "error",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb24",
                        "-s",
                        f"{width}x{height}",
                        "-r",
                        str(fps),
                        "-i",
                        "pipe:0",
                        "-an",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "18",
                        "-g",
                        "2",
                        "-keyint_min",
                        "2",
                        "-sc_threshold",
                        "0",
                        "-bf",
                        "0",
                        "-threads",
                        "1",
                        "-pix_fmt",
                        pixel_format,
                        "-movflags",
                        "+faststart",
                        str(path),
                    ],
                    stdin=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
            count = 0
            with pq.ParquetWriter(target, schema) as writer:
                for batch in source.iter_batches(batch_size=4, use_threads=False):
                    check_space()
                    writer.write_batch(batch.select(columns))
                    for key, feature in cameras.items():
                        for image in batch.column(batch.schema.get_field_index(key)).to_pylist():
                            if not isinstance(image, dict) or not image.get("bytes"):
                                raise ValueError("Compaction needs embedded image bytes")
                            with Image.open(io.BytesIO(image["bytes"])) as opened:
                                if opened.size != (feature["shape"][1], feature["shape"][0]):
                                    raise ValueError(
                                        "Camera dimensions changed within the recording"
                                    )
                                encoders[key].stdin.write(opened.convert("RGB").tobytes())
                    count += batch.num_rows
            for encoder in encoders.values():
                encoder.stdin.close()
                if encoder.wait(timeout=120):
                    raise ValueError("FFmpeg could not encode this recording")
            # Compare in bounded batches, including all extension columns, timestamps,
            # task IDs, indices, actions and states. Never round/rebuild robot samples.
            converted = pq.ParquetFile(target, arrow_extensions_enabled=False)
            original_batches = source.iter_batches(
                batch_size=64, columns=columns, use_threads=False
            )
            result_batches = converted.iter_batches(batch_size=64, use_threads=False)
            for before, after in zip(original_batches, result_batches, strict=True):
                if not before.equals(after):
                    raise ValueError("Robot samples changed during compaction")
            for key in cameras:
                path = video_file(key, chunk, file)
                result = subprocess.run(
                    [
                        probe,
                        "-v",
                        "error",
                        "-count_frames",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=nb_read_frames,width,height",
                        "-of",
                        "json",
                        str(path),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=120,
                )
                stream = json.loads(result.stdout)["streams"][0]
                feature = cameras[key]
                if (
                    int(stream["nb_read_frames"]) != count
                    or stream["width"] != feature["shape"][1]
                    or stream["height"] != feature["shape"][0]
                ):
                    raise ValueError("Compacted camera video does not match its robot samples")
            offset = 0
            for episode in sorted(group, key=lambda ep: ep["dataset_from_index"]):
                for key in cameras:
                    prefix = f"videos/{key}/"
                    episode.update(
                        {
                            prefix + "chunk_index": chunk,
                            prefix + "file_index": file,
                            prefix + "from_timestamp": offset / fps,
                            prefix + "to_timestamp": (offset + episode["length"]) / fps,
                        }
                    )
                offset += episode["length"]
            frames += count
            if progress:
                temporary = progress.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps({"completed": completed + 1, "total": len(groups), "frames": frames})
                )
                temporary.replace(progress)
        finally:
            for encoder in encoders.values():
                if encoder.poll() is None:
                    encoder.kill()
                encoder.wait()
    if frames != info["total_frames"]:
        raise ValueError("Compacted frame count differs from this dataset")
    for source_meta in files:
        rows = pq.read_table(source_meta).to_pylist()
        by_id = {ep["episode_index"]: ep for ep in episodes}
        pq.write_table(
            pa.Table.from_pylist([by_id[row["episode_index"]] for row in rows]),
            output / source_meta.relative_to(root),
        )
    for key, feature in cameras.items():
        height, width, _ = feature["shape"]
        feature.update(
            dtype="video",
            info={
                "video.fps": fps,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p" if height % 2 == width % 2 == 0 else "yuv444p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        )
    info["video_path"] = video_path
    (output / "meta/info.json").write_text(json.dumps(info, allow_nan=False, indent=2) + "\n")
    receipt = {
        "source_bytes": source_bytes,
        "frames": frames,
        "episodes": len(episodes),
        "codec": "h264",
        "crf": 18,
        "original_preserved": True,
    }
    (output / "openjensen").mkdir(exist_ok=True)
    (output / "openjensen/compaction.json").write_text(json.dumps(receipt, indent=2) + "\n")
    receipt["output_bytes"] = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    return receipt
