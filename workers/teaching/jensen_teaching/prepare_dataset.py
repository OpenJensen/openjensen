"""Fixed offline application adapter around the existing native LeRobot writer.

Private request paths are supplied by the trusted app supervisor, never the web
client. This process never controls a simulator, downloads data or trains a model.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import signal
import sys
from pathlib import Path

from . import dataset
from .contracts import decode
from .journal import atomic_new

ID = re.compile(r"[a-f0-9]{32}\Z")
HASH = re.compile(r"[a-f0-9]{64}\Z")


def exact(value, fields):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError("Unexpected recording preparation fields")


def absolute(value):
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise ValueError("Invalid private preparation path")
    result = Path(value)
    if not result.is_absolute() or ".." in result.parts or "\\" in value or "\0" in value:
        raise ValueError("Private preparation paths must be absolute without traversal")
    return result


def directory(value):
    item = absolute(str(value))
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open(item.anchor, flags)
    try:
        for name in item.parts[1:]:
            child = os.open(name, flags, dir_fd=fd)
            os.close(fd)
            fd = child
    finally:
        os.close(fd)


def equal_file(path, expected):
    if not isinstance(expected, str) or not HASH.fullmatch(expected):
        raise ValueError("Invalid expected recording hash")
    if hashlib.sha256(dataset._read(path, 2 * 1024**2)).hexdigest() != expected:
        raise ValueError("Recording metadata changed after selection")


def admit(request):
    exact(
        request,
        {
            "schema_version",
            "job_id",
            "configuration_sha256",
            "selection_sha256",
            "captures",
            "output",
            "ffmpeg",
            "ffprobe",
        },
    )
    if type(request["schema_version"]) is not int or request["schema_version"] != 1:
        raise ValueError("Preparation schema must be integer 1")
    if not isinstance(request["job_id"], str) or not re.fullmatch(
        r"[a-zA-Z0-9_-]{1,100}", request["job_id"]
    ):
        raise ValueError("Invalid preparation job identity")
    for key in ("configuration_sha256", "selection_sha256"):
        if not isinstance(request[key], str) or not HASH.fullmatch(request[key]):
            raise ValueError("Invalid preparation identity")
    values = request["captures"]
    if not isinstance(values, list) or not 1 <= len(values) <= 100:
        raise ValueError("Select bounded capture identities")
    selections, sessions, episodes = [], set(), set()
    for item in values:
        exact(item, {"path", "session_id", "session_sha256", "episodes"})
        root = absolute(item["path"])
        ident = item["session_id"]
        if not isinstance(ident, str) or not ID.fullmatch(ident) or ident in sessions:
            raise ValueError("Capture identities must be unique")
        sessions.add(ident)
        equal_file(root / "session.json", item["session_sha256"])
        meta = decode(dataset._read(root / "session.json", 2 * 1024**2), limit=2 * 1024**2)
        if meta.get("session_id") != ident:
            raise ValueError("Selected session identity differs")
        if not isinstance(item["episodes"], list) or not 1 <= len(item["episodes"]) <= 100:
            raise ValueError("Select finalized episodes explicitly")
        names = []
        for row in item["episodes"]:
            exact(row, {"episode_id", "receipt_sha256"})
            name = row["episode_id"]
            if not isinstance(name, str) or not ID.fullmatch(name) or name in episodes:
                raise ValueError("Episode identities must be unique")
            episodes.add(name)
            if len(episodes) > 100:
                raise ValueError("Select at most 100 total episodes")
            equal_file(root / name / "episode.json", row["receipt_sha256"])
            names.append(name)
        selections.append(dataset.CaptureSelection(root, tuple(names)))
    output = absolute(request["output"])
    directory(output.parent)
    if os.path.lexists(output):
        raise ValueError("Preparation output must be new")
    for selection in selections:
        # Keep lexical paths; resolving them here would hide symlinked ancestors.
        if output.is_relative_to(selection.root) or selection.root.is_relative_to(output):
            raise ValueError("Preparation output overlaps capture input")
    return selections, output


def preflight(request):
    if sys.version_info[:2] != (3, 12):
        raise ValueError("Recording preparation requires isolated Python3.12")
    dataset.verify_writer()
    import av
    import pyarrow
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if av.codec.Codec("h264", "r").type != "video" or pyarrow.table({"frame": [0]}).num_rows != 1:
        raise ValueError("Native recording reader is unavailable")
    RGBEncoderConfig(vcodec="h264", crf=18)
    if not callable(LeRobotDataset.create):
        raise ValueError("Native recording writer is unavailable")
    for name in ("ffmpeg", "ffprobe"):
        executable = absolute(request[name])
        if (
            executable.name != name
            or not executable.is_file()
            or not os.access(executable, os.X_OK)
        ):
            raise ValueError("Configure the exact FFmpeg/ffprobe executables")
        selected = shutil.which(name)
        if selected is None or Path(selected).resolve() != executable.resolve():
            raise ValueError("Configured encoder differs from the executable on the fixed PATH")
        # Popen inherits the supervisor-owned process group. The outer deadline
        # and cleanup cover this bounded native preflight and all encoder children.
        import subprocess

        with subprocess.Popen(
            [str(executable), "-version"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ) as child:
            try:
                code = child.wait(timeout=5)
            except BaseException:
                child.kill()
                child.wait(timeout=5)
                raise
        if code != 0:
            raise ValueError("Configured FFmpeg/ffprobe preflight failed")


def prepare(request):
    selections, output = admit(request)
    preflight(request)
    result = dataset.convert_captures(selections, output, repo_id="local/teaching")
    # The converter embeds each complete source inventory and rechecks it before
    # publishing. Bind the reviewed metadata again to prevent a stale selection.
    for item in request["captures"]:
        root = Path(item["path"])
        equal_file(root / "session.json", item["session_sha256"])
        for episode in item["episodes"]:
            equal_file(root / episode["episode_id"] / "episode.json", episode["receipt_sha256"])
    files = dataset._inventory(output)
    return {
        "schema_version": 1,
        "job_id": request["job_id"],
        "configuration_sha256": request["configuration_sha256"],
        "selection_sha256": request["selection_sha256"],
        "conversion": result,
        "files": files,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    request = decode(dataset._read(args.request, 128 * 1024), limit=128 * 1024)
    selections, output = admit(request)
    result = absolute(str(args.result))
    directory(result.parent)
    if os.path.lexists(result) or result == args.request or result.is_relative_to(output):
        raise ValueError("Result must be new and outside the published dataset")
    if any(result.is_relative_to(s.root) for s in selections):
        raise ValueError("Result cannot modify source captures")
    signal.signal(
        signal.SIGTERM, lambda number, frame: (_ for _ in ()).throw(SystemExit(128 + number))
    )
    atomic_new(result, prepare(request))


if __name__ == "__main__":
    main()
