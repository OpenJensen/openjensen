"""Fetch the pinned, Apache-2.0 SO101 CAD source; scene loading is offline."""

import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

_REPOSITORY = "TheRobotStudio/SO-ARM100"
_REVISION = "5f6d2b876a53a4872e405b991dd925556c9e38a4"
_RAW_URL = f"https://raw.githubusercontent.com/{_REPOSITORY}/{_REVISION}"
_REMOTE_FOLDER = "Simulation/SO101"
_URDF_NAME = "so101_new_calib.urdf"
_SOURCE = Path(__file__).resolve().parent / "source"


def _fetch(remote, local):
    local.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                    f"{_RAW_URL}/{remote}", "--output", str(local)], check=True)
    data = local.read_bytes()
    return {
        "path": str(local.relative_to(_SOURCE)),
        "url": f"{_RAW_URL}/{remote}",
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def _main():
    urdf = _SOURCE / _URDF_NAME
    files = [_fetch(f"{_REMOTE_FOLDER}/{_URDF_NAME}", urdf)]
    files.append(_fetch("LICENSE", _SOURCE / "LICENSE"))

    # Keep exactly the meshes referenced by this follower, including repeated motors.
    robot = ET.parse(urdf).getroot()
    meshes = sorted({mesh.attrib["filename"] for mesh in robot.iter("mesh")})
    for mesh in meshes:
        files.append(_fetch(f"{_REMOTE_FOLDER}/{mesh}", _SOURCE / mesh))

    manifest = {"repository": _REPOSITORY, "revision": _REVISION, "files": files}
    (_SOURCE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Fetched {len(files)} source files at {_REVISION}.")


if __name__ == "__main__":
    _main()
