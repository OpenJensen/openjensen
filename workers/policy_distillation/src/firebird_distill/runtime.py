"""Pinned CPU ACT training and fresh-process inference. This module is never an API import."""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

from firebird_act.bundle import canonical, inventory, safe_file
from firebird_act.probe import offline_audit, runtime_versions

from .contracts import corpus, digest, load_sample, policy_info, read, teacher_info
from .provenance import action_fps, inherited_files, policy_metadata


def setup(seed):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
    forbidden = tuple(
        Path(p) for p in json.loads(os.environ.get("FIREBIRD_DISTILL_FORBIDDEN", "[]"))
    )
    sys.addaudithook(lambda event, args: offline_audit(event, args, forbidden))
    versions = runtime_versions()
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)
    return torch, versions


def load_policy(root):
    import torch
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.processor import (
        PolicyProcessorPipeline,
        batch_to_transition,
        policy_action_to_transition,
        transition_to_batch,
        transition_to_policy_action,
    )
    from safetensors.torch import load_file

    config = PreTrainedConfig.from_pretrained(root, local_files_only=True)
    config.device, config.pretrained_backbone_weights = "cpu", None
    model = ACTPolicy(config).cpu()
    weights = load_file(root / "model.safetensors", device="cpu")
    if any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in weights.values()):
        raise ValueError("ACT requires complete finite FP32 tensors")
    model.load_state_dict(weights, strict=True)
    model.eval()
    pre = PolicyProcessorPipeline.from_pretrained(
        root,
        config_filename="policy_preprocessor.json",
        local_files_only=True,
        overrides={"device_processor": {"device": "cpu"}},
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    post = PolicyProcessorPipeline.from_pretrained(
        root,
        config_filename="policy_postprocessor.json",
        local_files_only=True,
        overrides={"device_processor": {"device": "cpu"}},
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return model, pre, post


def chunk(policy, batch):
    import torch

    policy.reset()
    result = policy.predict_action_chunk(batch).detach()
    prediction = policy.config.chunk_size
    if tuple(result.shape) != (1, prediction, 6) or not torch.isfinite(result).all():
        raise ValueError("Expected finite full ACT prediction-horizon chunk")
    return result


def batch_for(data, pre, camera):
    pre.reset()
    # Only observations enter normalization. Targets are ALREADY normalized teacher outputs.
    return pre({camera: data["image"].float() / 255, "observation.state": data["state"]})


def processed(raw, post):
    import torch

    post.reset()
    prediction = raw.shape[1]
    if tuple(raw.shape) != (1, prediction, 6) or not 1 <= prediction <= 1024:
        raise ValueError("Invalid full action chunk for postprocessing")
    value = torch.stack([post(raw[:, i, :]) for i in range(prediction)])[:, 0]
    if tuple(value.shape) != (prediction, 6) or not torch.isfinite(value).all():
        raise ValueError("Invalid recorded-coordinate actions")
    return value


def tensor_sha(tensor):
    return digest(tensor.detach().contiguous().numpy().tobytes())


def predictions(policy, pre, post, data_root, doc):
    """Deterministic full corpus receipt; no teacher reads and no loss/optimizer access."""
    import torch

    records = []
    for sample in doc["samples"]:
        data = load_sample(data_root, sample, doc["image_shape"], doc["chunk_size"])
        batch = batch_for(data, pre, doc["camera"])
        with torch.no_grad():
            raw = chunk(policy, batch)
            actual = processed(raw, post)
            policy.reset()
            execution = policy.config.n_action_steps
            queued = torch.stack([policy.select_action(batch) for _ in range(execution)])[:, 0]
            refill = policy.select_action(batch)
            policy.reset()
            first = policy.select_action(batch)
            if (
                not torch.equal(queued, raw[0, :execution])
                or not torch.equal(refill, raw[:, 0])
                or not torch.equal(first, raw[:, 0])
            ):
                raise ValueError("Student execution-prefix queue/refill/reset differs")
        records.append(
            {
                "sample_sha256": sample["sha256"],
                "split": sample["split"],
                "raw_sha256": tensor_sha(raw),
                "postprocessed_sha256": tensor_sha(actual),
                "queue_and_reset_exact": True,
                "queue_refill_exact": True,
                "prediction_horizon": policy.config.chunk_size,
                "execution_horizon": policy.config.n_action_steps,
            }
        )
    return records


def runtime_source_identity():
    import platform

    import lerobot.policies.act.configuration_act as config
    import lerobot.policies.act.modeling_act as model
    import lerobot.policies.act.processor_act as processor

    return {
        "python": platform.python_version(),
        "cpu_threads": 1,
        "files": {
            Path(module.__file__).name: digest(safe_file(Path(module.__file__), 1024**2))
            for module in (model, config, processor)
        },
    }


def train(job, output):
    torch, versions = setup(job["recipe"]["seed"])
    from lerobot.policies.act.modeling_act import ACTPolicy

    teacher_root, data_root = Path(job["teacher"]["path"]), Path(job["dataset"]["path"])
    cfg, camera, processors_sha = teacher_info(teacher_root, job["teacher"]["files"])
    inherited = policy_metadata(teacher_root, cfg)
    doc = corpus(data_root, job["dataset"]["manifest_sha256"], cfg, camera, processors_sha,
                 metadata=inherited, expected_fps=action_fps(teacher_root, inherited))
    # Decode every input before optimization to reject corrupt/misdeclared padding immediately.
    # Prevent identical observation bytes leaking across operator-declared partitions.
    seen = {}
    for sample in doc["samples"]:
        data = load_sample(data_root, sample, doc["image_shape"], doc["chunk_size"])
        key = digest(data["image"].numpy().tobytes() + data["state"].numpy().tobytes())
        if seen.setdefault(key, sample["split"]) != sample["split"]:
            raise ValueError("Identical observations leak across splits")
    teacher, pre, post = load_policy(teacher_root)
    teacher.requires_grad_(False)
    student_config = copy.deepcopy(teacher.config)
    for key, value in {
        "dim_model": 256,
        "dim_feedforward": 1024,
        "n_encoder_layers": 2,
        "n_decoder_layers": 1,
        "n_heads": 4,
        "use_vae": False,
        "dropout": 0.0,
        # Native0.6.1 annotates this bool as int and deserializes False to0.
        "replace_final_stride_with_dilation": False,
    }.items():
        setattr(student_config, key, value)
    student = ACTPolicy(student_config).cpu()
    student.model.backbone.load_state_dict(teacher.model.backbone.state_dict(), strict=True)
    student.model.backbone.requires_grad_(False)
    trainable = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=job["recipe"]["learning_rate"], weight_decay=0.01)
    targets = {}

    def sample_batch(sample):
        data = load_sample(data_root, sample, doc["image_shape"], doc["chunk_size"])
        return data, batch_for(data, pre, camera)

    def target(sample, batch):
        if sample["file"] not in targets:
            with torch.no_grad():
                targets[sample["file"]] = chunk(teacher, batch).clone()
        return targets[sample["file"]]

    def metrics(split):
        absolute, squared, count = 0.0, 0.0, 0
        demo_error = torch.zeros(6, dtype=torch.float64)
        for sample in (s for s in doc["samples"] if s["split"] == split):
            data, batch = sample_batch(sample)
            with torch.no_grad():
                teacher_chunk, student_chunk = target(sample, batch), chunk(student, batch)
                mask = ~data["padding"]
                delta = student_chunk[0, mask] - teacher_chunk[0, mask]
                actual = processed(student_chunk, post)
                absolute += float(delta.abs().sum())
                squared += float(delta.square().sum())
                demo_error += (actual[mask] - data["actions"][mask]).abs().double().sum(dim=0)
                count += int(mask.sum()) * 6
        return {
            "valid_action_coordinates": count,
            "teacher_normalized_l1": absolute / count,
            "teacher_normalized_rmse": (squared / count) ** 0.5,
            "demonstration_per_coordinate_l1": (demo_error / (count / 6)).tolist(),
        }

    baseline = {split: metrics(split) for split in ("train", "validation")}
    training_samples = [s for s in doc["samples"] if s["split"] == "train"]
    generator = torch.Generator().manual_seed(job["recipe"]["seed"])
    order, losses, gradient_norms, cursor = [], [], [], 0
    initial_head = student.model.action_head.weight.detach().clone()
    for step in range(job["recipe"]["steps"]):
        if cursor == len(order):
            order = torch.randperm(len(training_samples), generator=generator).tolist()
            cursor = 0
        sample = training_samples[order[cursor]]
        cursor += 1
        data, batch = sample_batch(sample)
        targets_for_batch = target(sample, batch)
        student.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = student(
            {**batch, "action": targets_for_batch, "action_is_pad": data["padding"].unsqueeze(0)}
        )
        if not torch.isfinite(loss) or not loss.requires_grad:
            raise ValueError("Invalid native distillation loss")
        loss.backward()
        gradients = [p.grad for p in trainable if p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all() for g in gradients):
            raise ValueError("Invalid student gradients")
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in student.parameters()):
            raise ValueError("Non-finite student weights")
        losses.append(float(loss.detach()))
        gradient_norms.append(float(norm))
    if torch.equal(initial_head, student.model.action_head.weight):
        raise ValueError("Student action head did not learn")
    if any(p.grad is not None for p in teacher.parameters()):
        raise ValueError("Teacher unexpectedly received gradients")
    if any(
        not torch.equal(value, teacher.model.backbone.state_dict()[name])
        for name, value in student.model.backbone.state_dict().items()
    ):
        raise ValueError("Frozen teacher backbone changed")
    # The fixed last update is frozen BEFORE the final split is used. No model/candidate selection.
    student.eval()
    policy_dir = output / "policy"
    policy_dir.mkdir(parents=True)
    student.save_pretrained(policy_dir)
    # Preserve exact normalization and sidecar bytes. Distillation changes only
    # architecture/weights, never the teacher's observation or action coordinates.
    for name in inherited_files(teacher_root, cfg):
        raw = safe_file(teacher_root / name)
        expected = job["teacher"]["files"][name]
        if len(raw) != expected["bytes"] or digest(raw) != expected["sha256"]:
            raise ValueError("Teacher inference semantics changed before student save")
        (policy_dir / name).write_bytes(raw)
    frozen_files = inventory(policy_dir)
    student_cfg, _, _ = policy_info(policy_dir, frozen_files)
    if policy_metadata(policy_dir, student_cfg) != inherited:
        raise ValueError("Student lost teacher timing/control semantics")
    after = {split: metrics(split) for split in ("train", "validation", "final")}
    proof = predictions(student, pre, post, data_root, doc)
    if inventory(policy_dir) != frozen_files:
        raise ValueError("Frozen student changed during final assessment")
    teacher_inference_bytes = sum(
        t.numel() * t.element_size()
        for name, t in teacher.state_dict().items()
        if not name.startswith("model.vae_encoder")
    )
    student_bytes = frozen_files["model.safetensors"]["bytes"]
    if student_bytes >= teacher_inference_bytes:
        raise ValueError("Student weights are not smaller than teacher inference tensors")
    report = {
        "schema_version": 1,
        "adapter": "act-act-v1",
        **inherited,
        "versions": versions,
        "device": "cpu",
        "dtype": "float32",
        "dataset_kind": doc["source"]["kind"],
        "steps": len(losses),
        "training_losses": losses,
        "gradient_norms": gradient_norms,
        "objective": "masked L1 to detached normalized teacher actions; no demonstration mixing",
        "untuned_student": baseline,
        "trained_student": after,
        "selection": "fixed last step; final split assessed only after saving immutable student",
        "teacher_weights_bytes": job["teacher"]["files"]["model.safetensors"]["bytes"],
        "teacher_inference_tensor_bytes": teacher_inference_bytes,
        "student_weights_bytes": student_bytes,
        "student_parameters": sum(p.numel() for p in student.parameters()),
        "trainable_parameters": sum(p.numel() for p in trainable),
        "teacher_gradients_absent": True,
        "copied_backbone_unchanged": True,
        "action_head_changed": True,
        "policy_files": frozen_files,
        "teacher_target_sha256": {name: tensor_sha(t) for name, t in sorted(targets.items())},
        "runtime_source_sha256": runtime_source_identity(),
        "predictions": proof,
        "quality_verified": False,
        "calibration_verified": False,
        "speedup_verified": False,
        "task_success": None,
        "teacher_training_overlap": "unknown; student-only held-out split",
        "training_resume_supported": False,
    }
    (output / "training.json").write_bytes(canonical(report))
    return report


def verify(job, output):
    _, versions = setup(job["recipe"]["seed"])
    # No teacher path is accessed in this process (audit hook denies it).
    root, data = output / "policy", Path(job["dataset"]["path"])
    files = inventory(root)
    cfg, camera, processors_sha = policy_info(root, files)
    inherited = policy_metadata(root, cfg)
    doc = corpus(data, job["dataset"]["manifest_sha256"], cfg, camera, processors_sha,
                 metadata=inherited, expected_fps=action_fps(root, inherited))
    policy, pre, post = load_policy(root)
    records = predictions(policy, pre, post, data, doc)
    return {
        "schema_version": 1,
        "versions": versions,
        **inherited,
        "policy_files": inventory(output / "policy"),
        "predictions": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "verify"])
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    value = (train if args.mode == "train" else verify)(read(args.request), args.output)
    with args.result.open("xb") as stream:
        stream.write(canonical(value))


if __name__ == "__main__":
    main()
