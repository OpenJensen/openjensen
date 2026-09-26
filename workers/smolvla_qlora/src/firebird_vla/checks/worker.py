"""One native environment, model and phase per process. Called by firebird-check-qlora."""

import argparse
import hashlib
import json
import random
import subprocess
import time
import traceback
from importlib.metadata import distributions
from pathlib import Path

from firebird_vla.checkpoint import sha256, write_json

from .catalog import CRITICAL_PACKAGES, PROFILES, selected_module


def validate_runtime(entry, runtime):
    from importlib.metadata import distribution, version
    from urllib.parse import unquote, urlparse

    expected = PROFILES[entry["id"]]["code_commit"]
    for package, pin in CRITICAL_PACKAGES[entry["id"]].items():
        if version(package).split("+")[0] != pin:
            raise ValueError(
                f"{entry['id']} requires {package}=={pin}; use its isolated environment"
            )
    if entry["id"] in ("smolvla", "pi0"):
        if version("lerobot") != "0.4.4":
            raise ValueError("LeRobot adapter is pinned to 0.4.4")
    else:
        repo = Path(runtime["native_repo"]).resolve()
        head = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        if head != runtime["native_code_commit"] or (expected and head != expected):
            raise ValueError("Native runtime commit differs from the resolved profile")
        dirty = subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD", "--"], text=True)
        if dirty:
            raise ValueError(
                "Native runtime has tracked modifications; pin the actual implementation"
            )
        # Verify imports resolve to this checkout, not another installed native stack.
        package = {
            "openvla": "prismatic",
            "openvla_oft": "prismatic",
            "pi05": "openpi",
            "gr00t_n17": "gr00t",
        }[entry["id"]]
        import importlib.util

        spec = importlib.util.find_spec(package)
        if spec is None or not Path(spec.origin).resolve().is_relative_to(repo):
            raise ValueError(f"{package} must be installed from {repo}")
    if entry["id"] == "openvla_oft":
        direct = json.loads(distribution("transformers").read_text("direct_url.json") or "{}")
        commit = direct.get("vcs_info", {}).get("commit_id")
        if direct.get("dir_info", {}).get("editable"):
            path = unquote(urlparse(direct["url"]).path)
            commit = subprocess.check_output(
                ["git", "-C", path, "rev-parse", "HEAD"], text=True
            ).strip()
            if subprocess.check_output(["git", "-C", path, "diff", "HEAD", "--"], text=True):
                raise ValueError("OFT Transformers fork has uncommitted changes")
        if commit != PROFILES[entry["id"]]["transformers_code_commit"]:
            raise ValueError("OFT requires its pinned bidirectional-attention Transformers fork")


def pack_and_adapt(model, model_id, rank, adapter_path=None):
    import bitsandbytes as bnb
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model

    model.requires_grad_(False)
    targets = [
        (name, layer)
        for name, layer in model.named_modules()
        if isinstance(layer, torch.nn.Linear) and selected_module(name, model_id)
    ]
    if not targets:
        raise ValueError(
            f"No allowlisted linears in {model_id}; native architecture may have changed"
        )
    names = []
    for name, layer in targets:
        packed = bnb.nn.Linear4bit(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None,
            compute_dtype=torch.bfloat16,
            compress_statistics=True,
            quant_type="nf4",
            quant_storage=torch.bfloat16,
        )
        packed.load_state_dict(layer.state_dict())
        packed.requires_grad_(False)
        parent, _, child = name.rpartition(".")
        # setattr also works in the older, separately pinned OpenVLA torch runtime.
        setattr(model.get_submodule(parent), child, packed.to("cuda:0"))
        names.append(name)
    model.is_loaded_in_4bit = True
    model.to("cuda:0")  # Device move only: casting packed float containers corrupts NF4 bytes.
    if adapter_path:
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=rank,
                lora_alpha=2 * rank,
                lora_dropout=0,
                target_modules=names,
                bias="none",
            ),
        )
    return model, names


def packed_digest(model):
    import bitsandbytes as bnb
    import torch

    digest = hashlib.sha256()
    for name, layer in model.named_modules():
        if isinstance(layer, bnb.nn.Linear4bit):
            if layer.weight.requires_grad or layer.weight.quant_state is None:
                raise ValueError("Expected packed, frozen NF4 base weights")
            digest.update(name.encode())
            digest.update(layer.weight.detach().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def seed_rng(seed):
    """Seed initialization and native loss noise across all supported RNGs."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def check_loss(loss_fn, model, batch, seed, backward=False):
    import copy

    import torch

    seed_rng(seed)

    # Native processors may mutate dictionaries; re-create containers between repeated checks.
    def containers(value):
        if isinstance(value, torch.Tensor):
            return value.clone()
        if isinstance(value, dict):
            return {key: containers(v) for key, v in value.items()}
        if isinstance(value, list):
            return [containers(v) for v in value]
        return copy.deepcopy(value)

    with torch.set_grad_enabled(backward), torch.autocast("cuda", dtype=torch.bfloat16):
        loss = loss_fn(model, containers(batch))
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise FloatingPointError("Native loss must be a finite scalar")
    if backward:
        loss.backward()
    return float(loss.detach())


def run_phase(job, phase):
    import torch

    from .fixtures import load_fixture
    from .native import LOADERS

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA_BF16_UNAVAILABLE: run in the model's native CUDA environment")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    entry, runtime = job["entry"], job["runtime"]
    if sha256(runtime["dependency_lock"]) != job["dependency_lock_sha256"]:
        raise ValueError("Native dependency lock changed after planning")
    validate_runtime(entry, runtime)
    batch, fixture = load_fixture(runtime["fixture"], entry)
    profile = PROFILES[entry["id"]]
    if (
        profile["code_commit"]
        and fixture["provenance"]["native_code_commit"] != profile["code_commit"]
    ):
        raise ValueError("Fixture was prepared by another native code revision")
    # Native heads and PEFT adapter initialization must share the job's reproducible seed.
    # Loss checks reseed separately so repeated comparisons use identical native loss noise.
    seed_rng(job["seed"])
    model, loss_fn, source_path = LOADERS[profile["loader"]](entry, runtime)
    output = Path(job["output"])
    native_linears = {
        name: {"shape": list(layer.weight.shape), "native_dtype": str(layer.weight.dtype)}
        for name, layer in model.named_modules()
        if isinstance(layer, torch.nn.Linear)
    }
    # Bind even auxiliary native heads/configs to each phase; no inference-success promotion.
    source_hashes = {
        str(p.relative_to(source_path)): sha256(p) for p in source_path.rglob("*") if p.is_file()
    }
    evidence = {
        "phase": phase,
        "status": "passed",
        "fixture_kind": fixture["provenance"]["kind"],
        "source_hashes": source_hashes,
        "fixture_sha256": sha256(Path(runtime["fixture"]) / "fixture.json"),
        "device": torch.cuda.get_device_name(),
        "cuda": torch.version.cuda,
        "packages": {d.metadata["Name"]: d.version for d in distributions()},
        "worker_source_hashes": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
    }
    if phase == "float":
        model.to("cuda:0").eval()
        evidence["native_loss"] = check_loss(loss_fn, model, batch, job["seed"])
    else:
        if phase == "reload":
            saved = json.loads((output / "qlora.json").read_text())
            if saved["source_hashes"] != source_hashes:
                raise ValueError("Native checkpoint assets changed between QLoRA and reload")
            if saved["packages"] != evidence["packages"]:
                raise ValueError("Installed native packages changed before reload")
            if saved["worker_source_hashes"] != evidence["worker_source_hashes"]:
                raise ValueError("QLoRA worker implementation changed before reload")
            adapter_files = {
                str(p.relative_to(output / "adapter")): sha256(p)
                for p in (output / "adapter").rglob("*")
                if p.is_file()
            }
            if adapter_files != saved["adapter_hashes"]:
                raise ValueError("Saved adapter is missing or corrupt")
        model, names = pack_and_adapt(
            model,
            entry["id"],
            job["rank"],
            output / "adapter" if phase == "reload" else None,
        )
        evidence["quantized_modules"] = names
        evidence["precision_map"] = {
            name: {
                **details,
                "weight_format": "nf4_double_quant" if name in names else details["native_dtype"],
            }
            for name, details in native_linears.items()
        }
        evidence["protected_linear_modules"] = [
            name for name in native_linears if name not in names
        ]
        evidence["protected_groups"] = [
            "vision",
            "projectors",
            "action_experts",
            "action_heads",
            "embeddings",
            "normalizations",
            "output_token_head",
        ]
        model.eval()
        if phase == "reload":
            from peft import get_peft_model_state_dict
            from safetensors.torch import load_file

            saved_tensors = load_file(str(output / "adapter" / "adapter_model.safetensors"))
            restored_tensors = get_peft_model_state_dict(model, save_embedding_layers=False)
            if set(saved_tensors) != set(restored_tensors) or any(
                not torch.equal(value.float(), restored_tensors[name].detach().cpu().float())
                for name, value in saved_tensors.items()
            ):
                raise ValueError(
                    "Adapter loader did not restore every saved adapter tensor exactly"
                )
            evidence["adapter_tensors_restored_exactly"] = True
            if (
                names != saved["quantized_modules"]
                or evidence["fixture_sha256"] != saved["fixture_sha256"]
            ):
                raise ValueError("Quantization layout or fixture changed at reload")
            actual = check_loss(loss_fn, model, batch, job["seed"])
            expected = saved["native_loss_after"]
            if abs(actual - expected) > 1e-4 + 1e-3 * abs(expected):
                raise ValueError(f"Fresh-process loss changed: expected {expected}, got {actual}")
            evidence.update(native_loss=actual, reload_abs_difference=abs(actual - expected))
        else:
            evidence["native_loss_before"] = check_loss(loss_fn, model, batch, job["seed"])
            trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
            if not trainable or any("lora_" not in n for n, _ in trainable):
                raise ValueError("Only LoRA adapter weights may be trainable in this diagnostic")
            optimizer = torch.optim.AdamW([p for _, p in trainable], lr=job["learning_rate"])
            before = {n: p.detach().cpu().clone() for n, p in trainable}
            frozen_digest = packed_digest(model)
            model.train()
            for step in range(job["steps"]):
                optimizer.zero_grad(set_to_none=True)
                check_loss(loss_fn, model, batch, job["seed"] + step, backward=True)
                if not any(
                    p.grad is not None and torch.count_nonzero(p.grad).item() for _, p in trainable
                ):
                    raise ValueError("No nonzero LoRA gradients from the native action loss")
                torch.nn.utils.clip_grad_norm_(
                    [p for _, p in trainable], 1.0, error_if_nonfinite=True
                )
                optimizer.step()
            changed = [n for n, p in trainable if not torch.equal(before[n], p.detach().cpu())]
            if not changed or packed_digest(model) != frozen_digest:
                raise ValueError("Adapters did not update or packed base weights changed")
            model.eval()
            evidence.update(
                native_loss_after=check_loss(loss_fn, model, batch, job["seed"]),
                adapter_parameters=sum(p.numel() for _, p in trainable),
                changed_adapter_tensors=len(changed),
                frozen_packed_weights_unchanged=True,
                nonzero_adapter_gradients=True,
            )
            model.save_pretrained(
                output / "adapter", safe_serialization=True, save_embedding_layers=False
            )
            evidence["adapter_hashes"] = {
                str(p.relative_to(output / "adapter")): sha256(p)
                for p in (output / "adapter").rglob("*")
                if p.is_file()
            }
    torch.cuda.synchronize()
    evidence["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    evidence["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("phase", choices=["float", "qlora", "reload"])
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    started = time.monotonic()
    code = 0
    try:
        evidence = run_phase(job, args.phase)
    except Exception as error:
        text = str(error)
        status = "oom" if "out of memory" in text.lower() else "failed"
        if isinstance(error, ImportError) or "CUDA_BF16_UNAVAILABLE" in text:
            status = "blocked"
        evidence = {"phase": args.phase, "status": status, "reason": text, "metrics": None}
        traceback.print_exc()
        code = 1
    evidence["elapsed_seconds"] = time.monotonic() - started
    write_json(Path(job["output"]) / f"{args.phase}.json", evidence)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
