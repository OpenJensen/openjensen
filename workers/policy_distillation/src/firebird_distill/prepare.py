"""Offline LeRobot0.6.2 snapshot reader; writes the small immutable ACT training corpus.

Run in the existing dataset environment, separately from the pinned0.6.1 model runtime.
No dataset download, augmentation, coordinate conversion, or random split is performed.
"""

import argparse
import importlib.metadata
import os
import sys
import tempfile
from pathlib import Path

from firebird_act.bundle import canonical, inventory, publish_new_directory
from firebird_act.probe import offline_audit

from .contracts import (
    MAX_CORPUS_BYTES,
    MAX_SAMPLES,
    corpus,
    digest,
    exact_keys,
    identity,
    integer,
    path,
    read,
    teacher_info,
)


def prepare(value):
    exact_keys(
        value,
        {
            "schema_version",
            "teacher",
            "dataset_snapshot",
            "splits",
            "frame_stride",
            "semantics",
            "output_dir",
        },
    )
    integer(value["schema_version"], 1, 1)
    integer(value["frame_stride"], 1, 10000)
    exact_keys(value["splits"], {"train", "validation", "final"})
    output, teacher = path(value["output_dir"]), path(value["teacher"])
    if os.path.lexists(output):
        raise FileExistsError("Corpus output already exists")
    if importlib.metadata.version("lerobot") != "0.6.2":
        raise ValueError("Preparation requires the existing LeRobot0.6.2 dataset environment")
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
    sys.addaudithook(offline_audit)
    import torch
    from firebird_vla.local_dataset import offline_dataset_loading, verify_local_snapshot
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from safetensors.torch import save

    source, manifest = verify_local_snapshot(value["dataset_snapshot"])
    for root in (teacher, source):
        if output == root or output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("Corpus output must be separate from the source/teacher")
    before = inventory(teacher)
    cfg, camera, processors_sha = teacher_info(teacher, before)
    if manifest.get("lineage_validated") is not True:
        raise ValueError("Explicit episode lineage is required before distillation")
    selected = {}
    for split, episodes in value["splits"].items():
        if not isinstance(episodes, list) or not episodes:
            raise ValueError("Three nonempty explicit episode splits are required")
        for episode in episodes:
            integer(episode, 0, manifest["total_episodes"] - 1)
            if episode in selected:
                raise ValueError("Episode appears in multiple splits")
            selected[episode] = split
    lineage = {row["episode_index"]: row for row in manifest["lineage"]}
    groups = {}
    for episode, split in selected.items():
        group = identity(lineage[episode]["lineage_group"])
        if groups.setdefault(group, split) != split:
            raise ValueError("Lineage group leaks between splits")
    origins = {lineage[e]["origin"] for e in selected}
    generated = "synthetic" in origins
    if generated and origins != {"synthetic"}:
        raise ValueError("Mixed generated/recorded corpus is not supported")
    feature = manifest["features"].get(camera, {})
    shape = cfg["input_features"][camera]["shape"]
    if feature.get("shape") != [shape[1], shape[2], 3]:
        raise ValueError("Dataset camera resolution differs from teacher; no implicit resize")
    semantics = value["semantics"]
    if not isinstance(semantics, dict):
        raise ValueError("Explicit teacher-coordinate semantics are required")
    for key, field in (("state_names", "observation.state"), ("action_names", "action")):
        if semantics.get(key) != manifest["features"].get(field, {}).get("names"):
            raise ValueError("Coordinate names/order differ from the native dataset")
    semantics = dict(semantics, teacher_processors_sha256=processors_sha)
    source_doc = {
        "kind": "generated_fixture" if generated else "lerobot",
        "identity": value["dataset_snapshot"]["id"],
        "revision": value["dataset_snapshot"]["manifest_sha256"],
        "inventory_sha256": digest(canonical(manifest["files"])),
    }
    doc = {
        "schema_version": 1,
        "format": "act-observation-corpus-v1",
        "source": source_doc,
        "semantics": semantics,
        "camera": camera,
        "image_shape": shape,
        "chunk_size": 100,
        "samples": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".corpus-", dir=output.parent) as temporary:
        stage = Path(temporary) / "corpus"
        stage.mkdir()
        fps = manifest["fps"]
        with offline_dataset_loading():
            dataset = LeRobotDataset(
                repo_id="firebird/local-" + value["dataset_snapshot"]["manifest_sha256"][:16],
                root=source,
                episodes=sorted(selected),
                download_videos=False,
                video_backend="pyav",
                delta_timestamps={"action": [i / fps for i in range(100)]},
                return_uint8=True,
            )
        total = 0
        seen_episodes = set()
        planned, cursor = [], 0
        for episode in sorted(selected):
            length = int(dataset.meta.episodes[episode]["length"])
            for frame in range(0, length, value["frame_stride"]):
                planned.append((cursor + frame, episode, frame, length))
                if len(planned) > MAX_SAMPLES:
                    raise ValueError("Selected corpus exceeds256 samples; increase frame_stride")
            cursor += length
        if len(dataset) != cursor:
            raise ValueError("Native episode lengths differ from the selected row inventory")
        for index, expected_episode, expected_frame, length in planned:
            row = dataset[index]
            episode, frame = int(row["episode_index"]), int(row["frame_index"])
            if (episode, frame) != (expected_episode, expected_frame):
                raise ValueError("Native reader returned the wrong episode/frame identity")
            data = {
                "image": row[camera],
                "state": row["observation.state"],
                "actions": row["action"],
                "padding": row["action_is_pad"],
            }
            if data["image"].dtype != torch.uint8:
                raise ValueError("Native reader must return original uint8 pixels")
            raw = save({k: v.contiguous() for k, v in data.items()})
            total += len(raw)
            if total > MAX_CORPUS_BYTES:
                raise ValueError("Corpus exceeds 2GiB")
            name = f"sample-{len(doc['samples']):06d}.safetensors"
            (stage / name).write_bytes(raw)
            doc["samples"].append(
                {
                    "file": name,
                    "sha256": digest(raw),
                    "bytes": len(raw),
                    "episode_id": episode,
                    "lineage_group": lineage[episode]["lineage_group"],
                    "frame_index": frame,
                    "episode_length": length,
                    "split": selected[episode],
                }
            )
            seen_episodes.add(episode)
        if seen_episodes != set(selected):
            raise ValueError("Native reader omitted selected episodes")
        (stage / "manifest.json").write_bytes(canonical(doc))
        manifest_sha = digest(canonical(doc))
        corpus(stage, manifest_sha, cfg, camera, processors_sha)
        from .contracts import load_sample

        for sample in doc["samples"]:
            load_sample(stage, sample, shape)
        verify_local_snapshot(value["dataset_snapshot"])
        if inventory(teacher) != before:
            raise ValueError("Teacher changed during preparation")
        publish_new_directory(stage, output)
    return {
        "path": str(output),
        "manifest_sha256": manifest_sha,
        "samples": len(doc["samples"]),
        "source": source_doc,
        "scope": "Decoded observation corpus; coordinate semantics are operator-attested",
    }


def verify_lengths(value):
    """Read only bounded episode-index/length columns from the immutable native metadata.

    This is separate from corpus production, not an untrusted-worker security boundary.
    The app compares the independently parsed source bounds with every prepared sample.
    """
    exact_keys(value, {"schema_version", "dataset_snapshot", "episodes"})
    integer(value["schema_version"], 1, 1)
    selected = value["episodes"]
    if not isinstance(selected, list) or not 3 <= len(selected) <= MAX_SAMPLES:
        raise ValueError("Select3..256 distinct episode IDs")
    for episode in selected:
        integer(episode, 0, 19999)
    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate metadata episode selection")
    if importlib.metadata.version("lerobot") != "0.6.2":
        raise ValueError("Metadata reader requires pinned LeRobot0.6.2")
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
    sys.addaudithook(offline_audit)
    import pyarrow.parquet as pq
    from firebird_vla.local_dataset import _open, verify_local_snapshot

    root, manifest = verify_local_snapshot(value["dataset_snapshot"])
    total = integer(manifest["total_episodes"], 3, 20000)
    if max(selected) >= total:
        raise ValueError("Selected metadata episode is absent")
    found, frames = {}, 0
    for item in manifest["files"]:
        if not item["path"].startswith("meta/episodes/"):
            continue
        with _open(root, item["path"]) as handle:
            table = pq.ParquetFile(
                handle, thrift_string_size_limit=1024**2, thrift_container_size_limit=100000
            )
            if table.metadata.num_rows > total:
                raise ValueError("Episode metadata row count exceeds snapshot")
            for batch in table.iter_batches(batch_size=1024, columns=["episode_index", "length"]):
                if batch.schema.names != ["episode_index", "length"]:
                    raise ValueError("Missing native episode identity/length columns")
                for row in batch.to_pylist():
                    episode = integer(row["episode_index"], 0, total - 1)
                    length = integer(row["length"], 1, manifest["total_frames"])
                    if episode in found:
                        raise ValueError("Duplicate native episode metadata")
                    found[episode] = length
                    frames += length
                    if len(found) > total or frames > manifest["total_frames"]:
                        raise ValueError("Episode metadata exceeds snapshot bounds")
    if len(found) != total or frames != manifest["total_frames"]:
        raise ValueError("Incomplete native episode metadata")
    verify_local_snapshot(value["dataset_snapshot"])
    return {
        "schema_version": 1,
        "snapshot_id": value["dataset_snapshot"]["id"],
        "snapshot_manifest_sha256": value["dataset_snapshot"]["manifest_sha256"],
        "episode_lengths": [{"episode_id": e, "length": found[e]} for e in sorted(selected)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    if os.path.lexists(args.result):
        parser.error("Result already exists")
    value = read(args.request)
    destination = args.result.resolve()
    for source in (
        value.get("teacher"),
        value.get("dataset_snapshot", {}).get("path"),
        value.get("output_dir"),
    ):
        if source and destination.is_relative_to(Path(source).resolve()):
            parser.error("Result must remain outside all source/corpus directories")
    result = (verify_lengths if args.metadata_only else prepare)(value)
    with args.result.open("xb") as stream:
        stream.write(canonical(result))


if __name__ == "__main__":
    main()
