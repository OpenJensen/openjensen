"""Exercise the public run service, fake bus, mock provider, private import and reload."""

import argparse
import asyncio
import hashlib
import io
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace


async def capture(destination):
    from vla_platform.collection_api import CollectionInput, CollectionRuns, Experiment
    from vla_platform.datasets.library import DatasetLibrary

    destination.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        library = DatasetLibrary(root)
        app = SimpleNamespace(state=SimpleNamespace(robot_lab=None, dataset_library=library))
        service = CollectionRuns(app, root)
        config = CollectionInput(
            experiment=Experiment(
                configuration_id="synthetic-tray-1",
                success_criterion="Fixture evidence only; no task outcome assigned",
                mock_timing_fixture=True,
            )
        )
        run = await service.start("fixture", config)
        await asyncio.wait_for(service.tasks[run["id"]], 60)
        reloaded = CollectionRuns(app, root).get(run["id"])
        if reloaded["status"] != "saved" or not reloaded["dataset"]:
            raise ValueError(reloaded["error"])
        archive = library.export(reloaded["dataset"]["id"])
        shutil.copyfile(archive, destination / "episode.zip")
        archive.unlink()
        replay = await service.start(
            "fixture", CollectionInput(experiment=Experiment(parent_run_id=run["id"]))
        )
        await asyncio.wait_for(service.tasks[replay["id"]], 60)
        replay = service.get(replay["id"])
        assert (
            replay["status"] == "saved"
            and replay["source_family_id"] == reloaded["source_family_id"]
        )
        assert (
            replay["episode_id"] != reloaded["episode_id"]
            and replay["dataset"]["id"] != reloaded["dataset"]["id"]
        )
        (destination / "replay.json").write_text(
            json.dumps(
                {
                    "source_family_id": replay["source_family_id"],
                    "episode_id": replay["episode_id"],
                    "dataset_id": replay["dataset"]["id"],
                    "frames": replay["frames"],
                    "status": replay["status"],
                    "outcome": "unknown",
                }
            )
            + "\n"
        )
        (destination / "run.json").write_text(json.dumps(reloaded) + "\n")


def verify(folder, upstream):
    from collection_isr import build, join
    from PIL import Image

    archive = folder / "episode.zip"
    view = build(archive, upstream, folder / "views.json", budget=24)
    again = build(archive, upstream, folder / "reload.json", budget=24)
    assert again == view
    categories = {}
    by_index = {f["raw_index"]: f for f in view["samples"]}
    with zipfile.ZipFile(archive) as z:
        control = {
            r["sample_id"]: r
            for r in map(json.loads, z.read("evidence/controller.jsonl").splitlines())
        }
        frames = list(map(json.loads, z.read("evidence/frames.jsonl").splitlines()))
        meta = json.loads(z.read("evidence/recording.json"))
        streams = {
            r["frame_id"]: r
            for r in map(json.loads, z.read("evidence/camera-streams.jsonl").splitlines())
        }
        with zipfile.ZipFile(io.BytesIO(z.read("evidence/camera-streams.zip"))) as images:
            assert len(images.namelist()) == len(streams) > 2 * len(frames)
            for stream in streams.values():
                with images.open(stream["member"]) as f:
                    Image.open(f).load()
        for row in by_index.values():
            raw = frames[row["raw_index"]]
            sample = control[row["action_sample_id"]]
            assert (
                row["original_elapsed_ns"] + meta["origin_monotonic_ns"]
                == raw["pair_available_robot_ns"]
            )
            assert row["sent"] == sample["sent"] and row["sent_raw"] == sample["sent_raw"]
            assert row["command_id"] == sample["command_id"]
            for role, image in row["images"].items():
                assert image["id"] in streams
                assert (
                    image["id"] == raw["cameras"][role]["frame_id"]
                    and image["delta_to_dispatch_ns"] >= 0
                )
                with z.open(image["path"]) as f:
                    Image.open(f).load()
            for label, span in row["masks"]:
                categories.setdefault(label, []).append(row["raw_index"])
        assert {"api_motion", "stable_wait", "protected_dwell"} <= categories.keys(), categories
        protected = categories["protected_dwell"]
        assert (
            max(by_index[i]["gripper_fraction"] for i in protected)
            - min(by_index[i]["gripper_fraction"] for i in protected)
            < 0.002
        )
        assert any(by_index[i]["command_id"] for i in categories["api_motion"])
        # Independent camera read rates exceed the paired sampling rate. These are
        # intentional skips, and their original PNGs still exist.
        assert any(
            c["unselected_reader_frames"] and c["unselected_reader_frames"] > 0
            for f in frames
            for c in f["cameras"].values()
        )
        assert all(span["start_uncertainty_ns"] >= 0 for span in view["intervals"])
        for command in view["commands"]:
            assert (
                command["accepted_ns"]
                <= command["first_dispatch_ns"]
                <= command["last_write_returned_ns"]
            )
            assert command["encoder_target_reached_ns"] and command["model_observation_id"]
        assert any(c["gripper_direction"] == "closing" for c in view["commands"])
        assert set(protected) <= {x["raw_index"] for x in view["views"]["assisted"]}
        assert len(view["views"]["assisted"]) == len(view["views"]["blind"])
        # Reset annotations split chunks even when the retained frames skip the reset.
        reset = protected[len(protected) // 2]
        review = {
            "episode_id": meta["episode_id"],
            "label": "reset",
            "first_frame": reset,
            "last_frame": reset,
        }
        revised = build(archive, upstream, folder / "review-fixture.json", reviews=[review])
        assert reset not in {s["raw_index"] for s in revised["samples"]}
        assert len({s["segment"] for s in revised["samples"]}) >= 2
        # A missing frame and a stale sensor bound each independently cut chunks.
        for bad_frames in (
            frames[:reset] + frames[reset + 1 :],
            [dict(f, pair_skew_ns=151_000_000) if f["frame_index"] == reset else f for f in frames],
        ):
            joined = join(meta, bad_frames, list(control.values()), view["intervals"], [])
            before = next(r for r in joined if r["raw_index"] == reset - 1)
            after = next(r for r in joined if r["raw_index"] == reset + 1)
            assert before["segment"] != after["segment"]
        # Uncertainty wider than the request cannot attest a stable wait.
        ambiguous = dict(
            next(s for s in view["intervals"] if s["kind"] == "api" and s["injected_ns"] > 0),
            start_uncertainty_ns=10**10,
        )
        joined = join(meta, frames, list(control.values()), [ambiguous], [])
        assert any(["ambiguous", ambiguous["id"]] in r["masks"] for r in joined)
        assert all(["stable_wait", ambiguous["id"]] not in r["masks"] for r in joined)
    with archive.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    receipt = {
        "episode_id": view["episode_id"],
        "archive_sha256": sha,
        "raw_frames": len(frames),
        "camera_frames": len(streams),
        "causal_joins": len(by_index),
        "retained_per_arm": view["retained_per_arm"],
        "mask_frame_indices": categories,
        "intervals": view["intervals"],
        "clocks": view["clocks"],
        "physical": False,
        "api_calls": 0,
        "reload_equal": True,
    }
    (folder / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in receipt.items()
                if k not in {"clocks", "intervals", "mask_frame_indices"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["capture", "verify"])
    p.add_argument("folder", type=Path)
    p.add_argument("--upstream", type=Path)
    args = p.parse_args()
    if args.mode == "capture":
        asyncio.run(capture(args.folder))
    else:
        verify(args.folder, args.upstream)
