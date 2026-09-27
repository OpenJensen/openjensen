"""Selective checkpoint download via the same REST transport as sky.download_logs.

Run only in the installed SkyPilot Python environment. /download_logs performs
incremental remote-to-server rsync; /download retrieves only unseen immutable
checkpoint subdirectories, so earlier model weights are not downloaded again.
"""

import json
import re
import shutil
import sys
from pathlib import Path


def main():
    import sky
    from sky.client import common as client_common
    from sky.server import common
    from sky.server.requests import payloads

    cluster, receipt_path, mode = sys.argv[1:]
    receipts = json.loads(Path(receipt_path).read_text())
    if not isinstance(receipts, dict):
        raise ValueError("Invalid checkpoint receipts")
    body = payloads.ClusterJobsDownloadLogsBody(cluster_name=cluster, job_ids=["1"])
    response = common.make_authenticated_request(
        "POST", "/download_logs", json=json.loads(body.model_dump_json())
    )
    paths = sky.get(common.get_request_id(response))
    if not isinstance(paths, dict) or set(paths) != {"1"}:
        raise ValueError("The training log directory could not be identified")
    remote = paths["1"].rstrip("/")
    if not isinstance(remote, str) or f"/{cluster}/" not in remote:
        raise ValueError("Unexpected SkyPilot training log directory")
    remote_index = remote + "/checkpoint-index"
    local_root = str(Path.home() / "sky_logs" / cluster / "1-firebird-training")

    def download(folder):
        return client_common.download_logs_from_api_server(
            [folder],
            remote_machine_prefix=remote,
            local_machine_prefix=local_root,
        )[folder]

    local_index = download(remote_index)
    index_path = Path(local_index).expanduser() / "index.json"
    if index_path.stat().st_size > 4 * 1024**2:
        raise ValueError("Checkpoint index exceeds its size limit")
    index = json.loads(index_path.read_text())
    entries = index.get("checkpoints")
    if not isinstance(entries, list) or len(entries) > 10000:
        raise ValueError("Invalid checkpoint index")
    seen = set()
    for entry in entries:
        name = entry.get("name", "") if isinstance(entry, dict) else ""
        if (
            not re.fullmatch(r"checkpoint-\d{6,}", name)
            or name in seen
            or type(entry.get("step")) is not int
            or entry["step"] != int(name[11:])
            or not re.fullmatch(r"[a-f0-9]{64}", entry.get("manifest_sha256", ""))
        ):
            raise ValueError("Invalid checkpoint index entry")
        seen.add(name)
        if receipts.get(name) == entry["manifest_sha256"]:
            continue
        if entry.get("remote_pruned"):
            raise ValueError("A previously acknowledged checkpoint is missing from local storage")
        if type(entry.get("file_bytes")) is not int or not 0 < entry["file_bytes"] <= 20 * 1024**3:
            raise ValueError("Invalid checkpoint size")
        if shutil.disk_usage(Path.home()).free < entry["file_bytes"] * 3 + 1024**3:
            raise ValueError("Not enough local disk space to download the next checkpoint safely")
        remote_snapshot = remote + "/checkpoint-snapshots/" + name
        local_snapshot = download(remote_snapshot)
        print(
            "FIREBIRD_CHECKPOINT="
            + json.dumps(
                {
                    **entry,
                    "descriptor": str(Path(local_snapshot).expanduser() / "firebird-output.json"),
                }
            ),
            flush=True,
        )
    if mode == "final":
        # Download only the small final artifact folder. Historical checkpoints
        # are already on the application host and are never re-zipped here.
        remote_final = remote + "/final-output"
        local_final = download(remote_final)
        print("FIREBIRD_FINAL=" + str(Path(local_final).expanduser()), flush=True)


if __name__ == "__main__":
    main()
