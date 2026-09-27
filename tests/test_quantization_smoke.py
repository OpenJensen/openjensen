"""Native inference verification validates variable checkpoint dimensions and evidence."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def smoke(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "workers/vla_cpp"))
    from policykit import inference_smoke

    return inference_smoke


def contract(chunk=4):
    return {
        "chunk_size": chunk,
        "max_action_dim": 32,
        "real_action_dim": 6,
        "image_size": 512,
        "packed_language_tensors": 224,
        "packed_vision_tensors": 72,
    }


def output(values=128, *, language=224, vision=72):
    return (
        "vla: backend = CPU (2 threads)\n"
        f"vla: packed resident matrices: lm={language} vision={vision}\n"
        f"action_len={values}\n" + "0.125\n" * values
    )


@pytest.mark.parametrize("chunk", [4, 10, 50])
def test_smoke_uses_checkpoint_action_shape_not_fixed_libero_dimensions(smoke, chunk):
    assert len(smoke.parse_prediction(output(chunk * 32), contract(chunk))) == chunk * 32


@pytest.mark.parametrize(
    "invalid",
    [
        output(127),
        output().replace("0.125", "nan", 1),
        output().replace("0.125", "inf", 1),
        output(language=0),
        output(vision=0),
        output().replace("CPU", "CUDA"),
    ],
)
def test_failed_native_inference_never_reports_success(smoke, invalid):
    with pytest.raises(ValueError):
        smoke.parse_prediction(invalid, contract())


def test_smoke_keeps_actual_actions_and_binary_model_identity(smoke, tmp_path, monkeypatch):
    model, binary = tmp_path / "model.gguf", tmp_path / "vla_predict_check"
    model.write_bytes(b"fixture model")
    binary.write_bytes(b"fixture executable")
    monkeypatch.setattr(smoke, "model_contract", lambda _: contract())

    def run(argv, **kwargs):
        assert argv == [str(binary), str(model), "", "1"]
        assert kwargs["timeout"] == 300
        assert kwargs["env"]["VLA_N_THREADS"] == kwargs["env"]["OMP_NUM_THREADS"] == "2"
        assert kwargs["env"]["VLA_BENCH_ITERS"] == "0"
        kwargs["stdout"].write(output())
        kwargs["stdout"].flush()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(smoke.subprocess, "run", run)
    report = smoke.verify_cpu_inference(model, binary, tmp_path, cameras=1)
    assert report["finite_action_values"] == 128
    assert report["model_sha256"] == smoke.sha256(model)
    assert report["executable_sha256"] == smoke.sha256(binary)
    assert report["task_success"] is None
    assert report["deployment_verified"] is False
    assert report["scope"] == "synthetic_input_native_inference"
    assert (
        len(json.loads((tmp_path / "cpu-inference-smoke-actions.json").read_text())["values"])
        == 128
    )


def test_smoke_reads_dimensions_and_packing_from_gguf_metadata(smoke, monkeypatch):
    fields = {
        name: SimpleNamespace(contents=lambda value=value: value)
        for name, value in {
            "general.architecture": "smolvla",
            "smolvla.chunk_size": 10,
            "smolvla.max_action_dim": 32,
            "smolvla.real_action_dim": 6,
            "smolvla.image_size": 512,
        }.items()
    }
    tensors = [
        SimpleNamespace(name="vlm.blk.0.attn_q.weight", tensor_type=SimpleNamespace(name="Q4_0")),
        SimpleNamespace(name="vit.blk.0.attn_q.weight", tensor_type=SimpleNamespace(name="Q8_0")),
        SimpleNamespace(name="aex.blk.0.attn_q.weight", tensor_type=SimpleNamespace(name="F32")),
    ]
    monkeypatch.setitem(
        sys.modules,
        "gguf",
        SimpleNamespace(GGUFReader=lambda _: SimpleNamespace(fields=fields, tensors=tensors)),
    )
    value = smoke.model_contract(Path("fixture.gguf"))
    assert value["chunk_size"] == 10
    assert value["packed_language_tensors"] == value["packed_vision_tensors"] == 1
