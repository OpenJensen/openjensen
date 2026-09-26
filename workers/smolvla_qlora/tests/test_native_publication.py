"""Native/Psi publication fault tests use fixture bytes, never model execution."""

import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from firebird_vla import checkpoint, lerobot_train, psi_train


@pytest.fixture(params=["native", "psi"])
def publisher(request, tmp_path, monkeypatch):
    monkeypatch.delenv("FIREBIRD_CHECKPOINT_EXPORT_ROOT", raising=False)
    output = tmp_path / "training"
    output.mkdir()
    if request.param == "native":
        source = tmp_path / "upstream"
        (source / "pretrained_model").mkdir(parents=True)
        (source / "training_state").mkdir()
        (source / "pretrained_model/model.safetensors").write_bytes(b"weights")
        (source / "training_state/optimizer.safetensors").write_bytes(b"optimizer")

        def publish():
            return lerobot_train.commit_checkpoint(
                source, output / "checkpoint-000002", {"policy_type": "act"}, step=2
            )

        weight_name = "model.safetensors"
    else:

        class Tensor:
            def detach(self):
                return self

            def cpu(self):
                return self

            def contiguous(self):
                return self

        torch = ModuleType("torch")
        torch.save = lambda value, path: Path(path).write_bytes(b"optimizer and RNG fixture")
        torch.get_rng_state = lambda: b"cpu RNG"
        torch.cuda = SimpleNamespace(get_rng_state=lambda device: b"CUDA RNG fixture")
        safe = ModuleType("safetensors.torch")
        safe.save_file = lambda values, path: Path(path).write_bytes(b"tensor fixture")
        monkeypatch.setitem(sys.modules, "torch", torch)
        monkeypatch.setitem(sys.modules, "safetensors.torch", safe)
        monkeypatch.setattr(psi_train, "predict", lambda *args: Tensor())
        monkeypatch.setattr(psi_train, "report", lambda *args, **kwargs: None)
        state = SimpleNamespace(state_dict=lambda: {"weight": Tensor()})
        trainer = SimpleNamespace(model=SimpleNamespace(action_header=state))

        def publish():
            return psi_train.save_checkpoint(
                output, trainer, {"steps": 2}, {}, {}, state, state, 2, 2, {"input": Tensor()}
            )

        weight_name = "action_header.safetensors"
    return output, publish, weight_name


def test_native_model_flush_failure_never_publishes_or_advances_pointer(publisher, monkeypatch):
    output, publish, weight_name = publisher
    previous = {"checkpoint": "checkpoint-000001", "step": 1}
    checkpoint.write_json(output / "latest.json", previous)
    original_fsync = os.fsync

    def fail_weights(descriptor):
        for path in output.rglob(weight_name):
            if os.path.samestat(os.fstat(descriptor), path.stat()):
                raise OSError("injected model data flush failure")
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_weights)
    with pytest.raises(OSError, match="model data flush"):
        publish()
    assert not (output / "checkpoint-000002").exists()
    assert json.loads((output / "latest.json").read_text()) == previous
    assert not list(output.glob(".*checkpoint-*"))


def test_native_publication_flushes_payload_before_final_visibility(publisher, monkeypatch):
    output, publish, weight_name = publisher
    original_fsync = os.fsync
    model_flushed = False
    original_rename = os.rename

    def observe_flush(descriptor):
        nonlocal model_flushed
        for path in output.rglob(weight_name):
            if os.path.samestat(os.fstat(descriptor), path.stat()):
                model_flushed = True
        original_fsync(descriptor)

    def check_commit(source, destination, *args, **kwargs):
        if Path(destination).name == "checkpoint-000002":
            assert model_flushed, "model payload must reach fsync before directory publication"
        return original_rename(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "fsync", observe_flush)
    monkeypatch.setattr(os, "rename", check_commit)
    publish()
    assert checkpoint.verify_bundle(output / "checkpoint-000002")["step"] == 2
    assert json.loads((output / "latest.json").read_text())["step"] == 2


def test_native_publication_never_replaces_existing_empty_destination(publisher):
    output, publish, _ = publisher
    destination = output / "checkpoint-000002"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        publish()
    assert destination.is_dir()
    assert list(destination.iterdir()) == []
    assert not (output / "latest.json").exists()
