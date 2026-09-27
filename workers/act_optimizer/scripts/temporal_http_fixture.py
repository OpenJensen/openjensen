"""Generated ACT temporal HTTP acceptance; not a simulator or task-quality test.

Run in the pinned ACT0.6.1 CPU environment with act_optimizer/src,
firebird_quant/src and isaac_sim on PYTHONPATH. The verification argument is the
packing operation's verification.json: floating and packed outputs remain distinct.
Every run is a fresh interpreter; only its ephemeral loopback listener is allowed.
"""

import argparse
import hashlib
import json
import os
import sys
import threading
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path)
    parser.add_argument("verification", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--forbid", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.result.exists() or args.result.resolve().is_relative_to(args.policy.resolve()):
        parser.error("Result must be new and outside the immutable package")
    permitted = set()
    forbidden = [p.resolve() for p in args.forbid]

    def audit(event, values):
        if event == "socket.connect" and values[1] not in permitted:
            raise RuntimeError("Only this fixture's own loopback listener is allowed")
        if event == "socket.bind" and values[1] != ("127.0.0.1", 0):
            raise RuntimeError("Only ephemeral loopback binding is allowed")
        if event in {"socket.getaddrinfo", "socket.gethostbyaddr"} and values[0] != "127.0.0.1":
            raise RuntimeError("External name resolution is forbidden")
        if event == "open" and values and isinstance(values[0], (str, bytes)):
            path = Path(os.fsdecode(values[0])).resolve()
            if any(path.is_relative_to(root) for root in forbidden) or (
                args.packed and path.name == "model.safetensors"
            ):
                raise RuntimeError("Original or floating-master reads are forbidden")

    sys.addaudithook(audit)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
    import torch
    from sim_worker.rollout.backend import LeRobotPolicy
    from sim_worker.rollout.checkpoint import inspect_checkpoint
    from sim_worker.rollout.client import RemotePolicy
    from sim_worker.rollout.contracts import Frame, Observation
    from sim_worker.rollout.server import create_server

    from firebird_act.bundle import canonical, inventory
    from firebird_act.probe import check_queue, runtime_versions

    versions = runtime_versions()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    before = inventory(args.policy)
    info = inspect_checkpoint(args.policy)
    proof = json.loads(args.verification.read_bytes())
    fixtures = proof["packed" if args.packed else "floating"]
    rows = []
    for count in dict.fromkeys((info.chunk_size, info.action_steps)):
        backend = LeRobotPolicy(args.policy, "cpu", count, info.state_dim, info.camera_key)
        server = create_server(("127.0.0.1", 0), backend, info.model_id, info.state_dim, count)
        endpoint = server.server_address
        permitted.add(endpoint)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        try:
            client = RemotePolicy(
                f"http://127.0.0.1:{endpoint[1]}", info.model_id, "Generated fixture", 10
            )
            client.wait_ready(5)
            for fixture in fixtures:
                generator = torch.Generator().manual_seed(fixture["seed"])
                pixels = torch.randint(
                    0, 256, fixture["image_shape"], generator=generator, dtype=torch.uint8
                )
                state = torch.randn(6, generator=generator)
                rgb = pixels.permute(1, 2, 0).contiguous().numpy().tobytes()
                with torch.inference_mode():
                    backend.reset()
                    batch = backend._pre(
                        {info.camera_key: pixels.float() / 255, "observation.state": state}
                    )
                    raw = backend._policy.predict_action_chunk(batch)
                    assert tuple(raw.shape) == (1, info.chunk_size, 6)
                    assert raw[0].tolist() == fixture["raw"]
                    check_queue(backend._policy, batch, raw, info.action_steps, torch)
                responses = []
                for repeat in range(2):
                    episode = f"seed-{fixture['seed']}-reset-{repeat}"
                    client.reset(episode)
                    observation = Observation(
                        episode, 0, 0.0, tuple(state.tolist()), Frame(info.width, info.height, rgb)
                    )
                    actual = [list(action) for action in client.predict(observation).actions]
                    assert actual == fixture["postprocessed"][:count]
                    responses.append(hashlib.sha256(canonical(actual)).hexdigest())
                assert responses[0] == responses[1]
                rows.append(
                    {
                        "seed": fixture["seed"],
                        "returned_steps": count,
                        "actions_sha256": responses[0],
                        "reset_exact": True,
                    }
                )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            if thread.is_alive():
                raise RuntimeError("Owned HTTP thread did not stop")
            permitted.remove(endpoint)
    assert inventory(args.policy) == before
    result = {
        "schema_version": 1,
        "model_id": info.model_id,
        "versions": versions,
        "prediction_horizon": info.chunk_size,
        "execution_horizon": info.action_steps,
        "policy_files": before,
        "requests": rows,
        "queue_refill_calls": 3,
        "queue_reset_forces_new_call": True,
        "saved_processors_exact": True,
        "source_unchanged": True,
        "server_closed": True,
        "external_network_disabled": True,
        "floating_master_reads_blocked": args.packed,
        "scope": (
            "Generated observations, actual HTTP prediction and native queue; no actions applied"
        ),
        "task_success": None,
        "calibration_verified": False,
        "quality_verified": False,
    }
    with args.result.open("xb") as stream:
        stream.write(canonical(result))
    print(json.dumps({"status": "passed", "model_id": info.model_id, "packed": args.packed}))


if __name__ == "__main__":
    main()
