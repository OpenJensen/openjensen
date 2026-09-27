"""Packed ACT consumer boundaries and optional genuine generated-model HTTP proof."""

# ruff: noqa: E402 -- optional sibling workers are isolated source environments.
import base64
import json
import os
import sys
import threading
import urllib.request
from pathlib import Path
from unittest.mock import patch

import pytest

WORKERS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKERS / "isaac_sim"))
sys.path.insert(0, str(WORKERS / "act_optimizer/src"))
sys.path.insert(0, str(WORKERS / "isaac_sim/tests"))
from test_native_runtime import real_act as _real_act
from test_packed_checkpoint import packed_fixture

from firebird_quant.native_consumer import load_packed_act
from firebird_quant.native_package import inspect_policy

native_act_source = _real_act


def test_consumer_rejects_unverified_device_without_model_import(tmp_path):
    with pytest.raises(ValueError, match="CPU only"):
        load_packed_act(tmp_path / "missing", device="cuda", expected_model_id="x")


def test_consumer_rejects_wrong_identity_before_runtime(tmp_path):
    policy = packed_fixture(tmp_path / "policy")
    with patch("firebird_act.probe.runtime_versions") as runtime:
        with pytest.raises(ValueError, match="identity"):
            load_packed_act(policy, device="cpu", expected_model_id="sha256:" + "0" * 64)
        runtime.assert_not_called()


def test_consumer_rejects_runtime_version_before_architecture(tmp_path):
    policy = packed_fixture(tmp_path / "policy")
    identity = inspect_policy(policy)["model_id"]
    with patch("firebird_act.probe.runtime_versions", side_effect=ValueError("wrong pin")):
        with pytest.raises(ValueError, match="wrong pin"):
            load_packed_act(policy, device="cpu", expected_model_id=identity)


def test_consumer_source_change_during_copy_rejected(tmp_path):
    import firebird_quant.native_consumer as consumer

    policy = packed_fixture(tmp_path / "policy")
    identity = inspect_policy(policy)["model_id"]
    original = consumer.write_new

    def changed(path, raw):
        original(path, raw)
        if path.name == "model.fbq":
            (policy / "model.fbq").write_bytes(b"changed source")

    with (
        patch("firebird_act.probe.runtime_versions", return_value={}),
        patch.object(consumer, "write_new", changed),
    ):
        with pytest.raises(ValueError, match="changed"):
            load_packed_act(policy, device="cpu", expected_model_id=identity)


def test_backend_rejects_packed_cuda_or_feature_mismatch_before_runtime(tmp_path):
    from sim_worker.rollout.backend import LeRobotPolicy

    policy = packed_fixture(tmp_path / "policy")
    with patch("sim_worker.rollout.backend.version") as version:
        for device, steps, dim, camera in (
            ("cuda", 100, 6, "observation.images.front"),
            ("cpu", 100, 5, "observation.images.front"),
            ("cpu", 101, 6, "observation.images.front"),
            ("cpu", 100, 6, "observation.images.other"),
        ):
            with pytest.raises(ValueError):
                LeRobotPolicy(policy, device, steps, dim, camera)
        version.assert_not_called()


def test_backend_cannot_relabel_removed_packed_format_as_float(tmp_path):
    from sim_worker.rollout import backend
    from sim_worker.rollout.checkpoint import inspect_checkpoint

    policy = packed_fixture(tmp_path / "policy")
    info = inspect_checkpoint(policy)

    def changed(_):
        (policy / "encoding.json").unlink()
        (policy / "model.fbq").unlink()
        return info

    with (
        patch.object(backend, "inspect_checkpoint", changed),
        patch.object(backend, "version") as version,
    ):
        with pytest.raises(ValueError, match="format changed"):
            backend.LeRobotPolicy(policy, "cpu", 100, 6, info.camera_key)
        version.assert_not_called()


@pytest.mark.native_act
@pytest.mark.skipif(
    os.environ.get("FIREBIRD_TEST_NATIVE_ACT") != "1",
    reason="Requires unchanged pinned ACT0.6.1 environment",
)
@pytest.mark.parametrize("bits", [8, 4])
def test_generated_act_http_full_chunk_and_reset(native_act_source, tmp_path, bits):
    import torch
    from sim_worker.rollout.backend import LeRobotPolicy
    from sim_worker.rollout.checkpoint import inspect_checkpoint
    from sim_worker.rollout.server import create_server

    from firebird_quant.native_application import run_job
    from firebird_quant.native_package import inventory

    result = run_job(
        {
            "schema_version": 1,
            "job_id": "consumer-fixture",
            "operation": "policy.quantize",
            "source": {
                "path": str(native_act_source),
                "files": inventory(native_act_source),
                "manifest_sha256": None,
                "artifact_id": "generated",
                "artifact_manifest_sha256": "a" * 64,
            },
            "output_dir": str(tmp_path / "output"),
            "native_quantization": {"format": "firebird_quant", "bits": bits, "group_size": 64},
            "timeout_seconds": 120,
        }
    )
    package = Path(result["artifact"]["path"])
    policy = package / "policy"
    before = inventory(policy)
    expected = json.loads((package / "verification.json").read_text())["packed"][0]
    info = inspect_checkpoint(policy)
    backend = LeRobotPolicy(policy, "cpu", 100, 6, info.camera_key)
    server = create_server(("127.0.0.1", 0), backend, info.model_id, 6, 100)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def post(endpoint, data):
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_address[1]}" + endpoint,
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        with opener.open(request, timeout=10) as response:
            assert response.status == 200
            return json.loads(response.read(256 * 1024))

    generator = torch.Generator().manual_seed(expected["seed"])
    pixels = torch.randint(0, 256, expected["image_shape"], generator=generator, dtype=torch.uint8)
    state = torch.randn(6, generator=generator).tolist()
    image = {
        "width": info.width,
        "height": info.height,
        "encoding": "rgb8",
        "data": base64.b64encode(pixels.permute(1, 2, 0).contiguous().numpy().tobytes()).decode(),
    }
    try:
        for episode in ("first", "after-reset"):
            post("/reset", {"episode_id": episode, "model_id": info.model_id})
            response = post(
                "/predict",
                {
                    "episode_id": episode,
                    "model_id": info.model_id,
                    "step": 0,
                    "sim_time": 0,
                    "task": "generated fixture",
                    "state": state,
                    "image": image,
                },
            )
            assert response["actions"] == expected["postprocessed"]
            assert response["model_id"] == result["report"]["model_id"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert inventory(policy) == before
    # Encoding cannot relabel packed INT8 storage as INT4 (or the reverse).
    from firebird_quant.native_package import encoding

    (policy / "encoding.json").write_text(json.dumps(encoding(4 if bits == 8 else 8)))
    with pytest.raises(ValueError, match="recipe|precision"):
        load_packed_act(policy, device="cpu", expected_model_id=inspect_policy(policy)["model_id"])
