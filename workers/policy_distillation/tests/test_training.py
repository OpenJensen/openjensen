from pathlib import Path

import pytest
from firebird_act.bundle import inventory
from firebird_distill.application import run_job
from firebird_distill.contracts import digest, read


def test_actual_native_act_distillation_and_fresh_reload(job):
    source = Path(job["teacher"]["path"])
    before = inventory(source)
    result = run_job(job)
    assert result["operation"] == "policy.distill"
    report = result["report"]
    assert report["fresh_reload_verified"] is True
    assert report["steps"] == 8
    assert report["action_head_changed"] and report["teacher_gradients_absent"]
    assert report["copied_backbone_unchanged"]
    assert report["student_weights_bytes"] < report["teacher_inference_tensor_bytes"]
    assert all(n > 0 for n in report["gradient_norms"])
    assert (
        report["trained_student"]["train"]["teacher_normalized_l1"]
        < (report["untuned_student"]["train"]["teacher_normalized_l1"])
    )
    assert set(report["untuned_student"]) == {"train", "validation"}
    assert set(report["trained_student"]) == {"train", "validation", "final"}
    assert report["dataset_kind"] == "generated_fixture"
    assert report["task_success"] is None and report["quality_verified"] is False
    root = Path(result["artifact"]["path"])
    cfg = read(root / "policy/config.json")
    assert (cfg["dim_model"], cfg["n_encoder_layers"], cfg["use_vae"]) == (256, 2, False)
    assert cfg["replace_final_stride_with_dilation"] is False
    manifest = read(root / "manifest.json")
    assert manifest["metadata"]["training_resume_supported"] is False
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "isaac_sim"))
    from sim_worker.rollout.checkpoint import inspect_checkpoint

    assert manifest["metadata"]["model_id"] == inspect_checkpoint(root / "policy").model_id
    for relative, sha in manifest["files"].items():
        assert digest((root / relative).read_bytes()) == sha
    assert inventory(source) == before
    assert not list(root.parent.glob(".distillation-*"))
    with pytest.raises(FileExistsError):
        run_job(job)


def test_native_forward_uses_valid_coordinate_mean_and_real_gradients():
    from types import SimpleNamespace

    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy

    prediction = torch.nn.Parameter(torch.ones(1, 100, 6))
    target = torch.zeros(1, 100, 6)
    target[:, 4:] = 9999
    batch = {"action": target, "action_is_pad": (torch.arange(100) >= 4).unsqueeze(0)}
    # Exercise the installed native loss, without another expensive vision model.
    holder = SimpleNamespace(
        config=SimpleNamespace(image_features={}, use_vae=False),
        model=lambda _: (prediction, (None, None)),
    )
    loss, _ = ACTPolicy.forward(holder, batch)
    assert float(loss.detach()) == 1.0  # Pinned0.6.1 divides by 4*6, not 100*6.
    loss.backward()
    assert prediction.grad[:, :4].abs().sum() > 0
    assert prediction.grad[:, 4:].abs().sum() == 0
    target[:, 4:] = -12345
    repeated, _ = ACTPolicy.forward(holder, batch)
    assert torch.equal(loss, repeated)


def test_existing_result_and_source_never_replaced(job):
    out = Path(job["output_dir"]) / "distilled-policy"
    out.mkdir(parents=True)
    (out / "sentinel").write_bytes(b"retained")
    with pytest.raises(FileExistsError):
        run_job(job)
    assert (out / "sentinel").read_bytes() == b"retained"


def test_missing_observation_does_not_publish(job):
    root = Path(job["dataset"]["path"])
    (root / "sample-000000.safetensors").unlink()
    with pytest.raises(ValueError):
        run_job(job)
    assert not (Path(job["output_dir"]) / "distilled-policy").exists()


def test_worker_failure_does_not_publish_and_removes_private_snapshots(job, monkeypatch):
    from firebird_distill.application import Owner

    def failed(*_):
        raise TimeoutError("injected bounded training deadline")

    monkeypatch.setattr(Owner, "run", failed)
    with pytest.raises(TimeoutError):
        run_job(job)
    root = Path(job["output_dir"])
    assert not (root / "distilled-policy").exists()
    assert not list(root.glob(".distillation-*"))


def test_original_corpus_mutation_during_worker_refuses_publication(job, monkeypatch):
    from firebird_distill.application import Owner

    def mutation(*_):
        path = Path(job["dataset"]["path"]) / "sample-000000.safetensors"
        path.write_bytes(path.read_bytes() + b"changed")
        raise InterruptedError("cancelled")

    monkeypatch.setattr(Owner, "run", mutation)
    with pytest.raises(ValueError, match="Original corpus sample changed"):
        run_job(job)
    assert not (Path(job["output_dir"]) / "distilled-policy").exists()


def test_cross_split_duplicate_pixels_and_state_fail_in_real_worker(job):
    from conftest import rebind_manifest
    from safetensors.torch import load, save

    root = Path(job["dataset"]["path"])
    first = load((root / "sample-000000.safetensors").read_bytes())
    other = load((root / "sample-000002.safetensors").read_bytes())
    other.update(image=first["image"], state=first["state"])
    raw = save(other)
    (root / "sample-000002.safetensors").write_bytes(raw)
    rebind_manifest(job, lambda d: d["samples"][2].update(sha256=digest(raw), bytes=len(raw)))
    with pytest.raises(ValueError, match="leak across splits"):
        run_job(job)
    assert not (Path(job["output_dir"]) / "distilled-policy").exists()
