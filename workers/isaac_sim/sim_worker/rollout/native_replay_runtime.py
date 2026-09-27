"""Fresh CPU child: bounded recorded observations through the existing HTTP protocol."""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

from firebird_quant.native_package import canonical, read_json, sha, write_new

from .native_replay_contracts import observations, policy


def run(job):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
    permitted = [None]
    forbidden = tuple(Path(p) for p in json.loads(os.environ["FIREBIRD_REPLAY_FORBIDDEN"]))

    def audit(event, args):
        if event == "socket.connect":
            address = args[1]
            if address != ("127.0.0.1", permitted[0]) or permitted[0] is None:
                raise RuntimeError("Only this owned loopback server may be contacted")
        if event in {"socket.getaddrinfo", "socket.gethostbyaddr"} and args[0] != "127.0.0.1":
            raise RuntimeError("External name resolution disabled")
        if event == "socket.bind" and args[1] != ("127.0.0.1", 0):
            raise RuntimeError("Only ephemeral loopback binding is permitted")
        if event == "open" and args and isinstance(args[0], (str, bytes)):
            path = Path(os.fsdecode(args[0])).resolve()
            if path.name == "model.safetensors" or any(path.is_relative_to(p) for p in forbidden):
                raise RuntimeError("Original source and floating master reads are disabled")

    sys.addaudithook(audit)
    from firebird_act.probe import runtime_versions

    versions = runtime_versions()
    import torch

    from .backend import LeRobotPolicy
    from .client import RemotePolicy
    from .contracts import Frame, Observation
    from .server import create_server

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    source, inputs = Path(job["source"]["path"]), Path(job["observations"]["path"])
    info = policy(source, job["source"])
    doc, input_files = observations(inputs, job["observations"]["manifest_sha256"], info["config"])
    camera = doc["camera_key"]
    prediction = info["config"]["chunk_size"]
    backend = LeRobotPolicy(source, "cpu", prediction, 6, camera)
    if backend.model_id() != info["model_id"]:
        raise ValueError("Loaded packed model identity changed")
    if any(
        p.device.type != "cpu"
        for p in list(backend._policy.parameters()) + list(backend._policy.buffers())
    ):
        raise ValueError("Native replay must remain CPU only")
    server = create_server(("127.0.0.1", 0), backend, info["model_id"], 6, prediction)
    permitted[0] = server.server_address[1]
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    thread.start()
    records = []
    try:
        for index, sample in enumerate(doc["samples"]):
            from firebird_quant.native_package import read

            from .native_replay_contracts import MAX_RGB_BYTES

            image = sample["image"]
            rgb = read(inputs / image["file"], MAX_RGB_BYTES)
            if sha(rgb) != image["sha256"]:
                raise ValueError("Observation changed before inference")
            client = RemotePolicy(
                f"http://127.0.0.1:{permitted[0]}", info["model_id"], sample["task"], 30
            )
            client.wait_ready(5)
            runs, actions = [], None
            for repeat in range(2):
                episode = f"replay-{index}-{repeat}"
                client.reset(episode)
                observation = Observation(
                    episode,
                    0,
                    0.0,
                    tuple(sample["state"]),
                    Frame(image["width"], image["height"], rgb),
                )
                started = time.monotonic()
                chunk = client.predict(observation)
                duration = time.monotonic() - started
                current = [list(action) for action in chunk.actions]
                if len(current) != prediction:
                    raise ValueError(
                        "Native replay requires the complete prediction-horizon x 6 action chunk"
                    )
                if actions is not None and current != actions:
                    raise ValueError(
                        "Fresh reset changed packed predictions on identical observation"
                    )
                actions = current
                runs.append({"seconds": duration, "actions_sha256": sha(canonical(current))})
            records.append(
                {
                    "sample_index": index,
                    "episode_index": sample["episode_index"],
                    "frame_index": sample["frame_index"],
                    "timestamp_seconds": sample["timestamp_seconds"],
                    "input_rgb_sha256": image["sha256"],
                    "input_state_sha256": sha(canonical(sample["state"])),
                    "actions": actions,
                    "reset_repeat_exact": True,
                    "requests": runs,
                }
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        if thread.is_alive():
            raise RuntimeError("Owned policy server did not stop")
    if (
        policy(source, job["source"]) != info
        or observations(inputs, job["observations"]["manifest_sha256"], info["config"])[1]
        != input_files
    ):
        raise ValueError("Private inputs changed during replay")
    return {
        "schema_version": 1,
        "model_id": info["model_id"],
        "versions": versions,
        "device": "cpu",
        "mode": "independent_observation_replay",
        "records": records,
        "server_closed": True,
        "external_network_disabled": True,
        "floating_master_reads_blocked": True,
        "task_success": None,
        "quality_verified": False,
        "calibration_verified": False,
        "speedup_verified": False,
        "isaac_runtime_verified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    result = run(read_json(args.request))
    write_new(args.result, canonical(result))


if __name__ == "__main__":
    main()
