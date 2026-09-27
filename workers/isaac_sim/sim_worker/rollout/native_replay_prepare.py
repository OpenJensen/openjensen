"""Prepare exact recorded observations in the separate pinned LeRobot0.6.2 reader."""

import argparse
import importlib.metadata
import os
import sys
import tempfile
from pathlib import Path

from firebird_quant.native_package import canonical, checked_path, read_json, sha, write_new

from .native_replay_contracts import (
    MAX_SAMPLES,
    MAX_TOTAL_BYTES,
    integer,
    keys,
    observations,
    policy,
    publish,
)


def prepare(value):
    keys(
        value,
        {"schema_version", "source", "dataset_snapshot", "selection", "semantics", "output_dir"},
    )
    integer(value["schema_version"], 1, 1)
    source = value["source"]
    keys(source, {"path", "files", "model_id", "artifact_id", "artifact_manifest_sha256"})
    model = checked_path(source["path"])
    output = checked_path(value["output_dir"])
    if os.path.lexists(output):
        raise FileExistsError("Prepared observations already exist")
    selection = value["selection"]
    if not isinstance(selection, list) or not 1 <= len(selection) <= MAX_SAMPLES:
        raise ValueError("Select1..32 explicit episode/frame observations")
    pairs = []
    for row in selection:
        keys(row, {"episode_index", "frame_index"})
        pair = (integer(row["episode_index"], 0, 2**31 - 1), integer(row["frame_index"], 0, 10**7))
        if pair in pairs:
            raise ValueError("Duplicate selected observation")
        pairs.append(pair)
    info = policy(model, source)
    camera = next(
        k for k in info["config"]["input_features"] if k.startswith("observation.images.")
    )
    shape = info["config"]["input_features"][camera]["shape"]
    if len(pairs) * shape[1] * shape[2] * 3 > MAX_TOTAL_BYTES:
        raise ValueError("Selected RGB observations exceed128MiB before decoding")
    if importlib.metadata.version("lerobot") != "0.6.2":
        raise ValueError("Preparation requires the isolated LeRobot0.6.2 dataset runtime")
    from firebird_act.probe import offline_audit

    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
    sys.addaudithook(offline_audit)
    import torch
    from firebird_vla.local_dataset import offline_dataset_loading, verify_local_snapshot
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    data, manifest = verify_local_snapshot(value["dataset_snapshot"])
    for original in (model, data):
        a, b = original.resolve(), output.resolve()
        if a == b or a.is_relative_to(b) or b.is_relative_to(a):
            raise ValueError("Prepared output must be separate from immutable inputs")
    semantics = value["semantics"]
    keys(semantics, {"state_names", "action_names", "units", "compatibility"})
    for field, feature in (("state_names", "observation.state"), ("action_names", "action")):
        if semantics[field] != manifest["features"].get(feature, {}).get("names"):
            raise ValueError("Explicit coordinate names/order differ from recorded dataset")
    if manifest["features"].get(camera, {}).get("shape") != [shape[1], shape[2], 3]:
        raise ValueError("Recorded camera must match policy exactly; no resizing")
    if any(episode >= manifest["total_episodes"] for episode, _ in pairs):
        raise ValueError("Unknown selected episode")
    lineage = {row["episode_index"]: row for row in manifest.get("lineage", [])}
    origins = {lineage.get(episode, {}).get("origin", "imported") for episode, _ in pairs}
    generated = "synthetic" in origins
    if generated and origins != {"synthetic"}:
        raise ValueError("Do not mix generated and recorded observations")
    kind = "generated_fixture" if generated else "lerobot_snapshot"
    expected_attestation = (
        "generated_fixture" if generated else "operator_attested_policy_recorded_coordinates"
    )
    if semantics["compatibility"] != expected_attestation:
        raise ValueError("Explicit compatible coordinate attestation is required")
    doc = {
        "schema_version": 1,
        "format": "native-policy-observations-v1",
        "source": {
            "kind": kind,
            "identity": value["dataset_snapshot"]["id"],
            "manifest_sha256": value["dataset_snapshot"]["manifest_sha256"],
        },
        "camera_key": camera,
        "semantics": semantics,
        "samples": [],
    }
    episodes = sorted({e for e, _ in pairs})
    with offline_dataset_loading():
        dataset = LeRobotDataset(
            repo_id="firebird/local-" + value["dataset_snapshot"]["manifest_sha256"][:16],
            root=data,
            episodes=episodes,
            download_videos=False,
            video_backend="pyav",
            return_uint8=True,
        )
    offsets, cursor = {}, 0
    for episode in episodes:
        length = int(dataset.meta.episodes[episode]["length"])
        offsets[episode] = (cursor, length)
        cursor += length
    if len(dataset) != cursor:
        raise ValueError("Native selected episode inventory differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".replay-observations-", dir=output.parent
    ) as temporary:
        stage = Path(temporary).resolve() / "observations"
        stage.mkdir()
        for episode, frame in pairs:
            offset, length = offsets[episode]
            if frame >= length:
                raise ValueError("Selected frame exceeds its episode")
            row = dataset[offset + frame]
            if (int(row["episode_index"]), int(row["frame_index"])) != (episode, frame):
                raise ValueError("Native reader returned wrong observation identity")
            pixels, state = row[camera], row["observation.state"]
            if (
                pixels.dtype != torch.uint8
                or list(pixels.shape) != shape
                or state.dtype != torch.float32
                or list(state.shape) != [6]
                or not torch.isfinite(state).all()
            ):
                raise ValueError("Expected original uint8 RGB and finite raw float32 state")
            raw = pixels.permute(1, 2, 0).contiguous().numpy().tobytes()
            name = f"frame-{len(doc['samples']):06d}.rgb"
            write_new(stage / name, raw)
            ancestry = lineage.get(episode, {})
            doc["samples"].append(
                {
                    "episode_index": episode,
                    "frame_index": frame,
                    "timestamp_seconds": float(row["timestamp"]),
                    "task": row["task"],
                    "state": state.tolist(),
                    "origin": ancestry.get("origin", "imported"),
                    "lineage_group": ancestry.get("lineage_group"),
                    "image": {
                        "file": name,
                        "width": shape[2],
                        "height": shape[1],
                        "bytes": len(raw),
                        "sha256": sha(raw),
                    },
                }
            )
        manifest_raw = canonical(doc)
        write_new(stage / "manifest.json", manifest_raw)
        _, files = observations(stage, sha(manifest_raw), info["config"])
        verify_local_snapshot(value["dataset_snapshot"])
        if policy(model, source) != info:
            raise ValueError("Original packed policy changed during preparation")
        publish(stage, output)
    return {
        "schema_version": 1,
        "path": str(output),
        "manifest_sha256": sha(manifest_raw),
        "files": files,
        "observations": len(pairs),
        "source": doc["source"],
        "scope": (
            "Original immutable inputs; policy-coordinate compatibility "
            "is operator-attested for recorded sources"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    checked_path(str(args.result))
    if os.path.lexists(args.result):
        parser.error("Refusing to overwrite result")
    value = read_json(args.request)
    destination = args.result.resolve()
    output = checked_path(value["output_dir"])
    if destination.is_relative_to(output.resolve()):
        parser.error("Result must not be inside the published observations")
    for name in (
        value.get("source", {}).get("path"),
        value.get("dataset_snapshot", {}).get("path"),
    ):
        if name and destination.is_relative_to(Path(name).resolve()):
            parser.error("Result must not be inside source inputs")
    write_new(args.result, canonical(prepare(value)))


if __name__ == "__main__":
    main()
