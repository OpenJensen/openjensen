"""Native loaders/losses for the six catalog families. No common robotics data conversion."""

import json
import re
from pathlib import Path

from firebird_vla.checkpoint import sha256


def checked_hf_load(cls, path, **kwargs):
    model, info = cls.from_pretrained(path, output_loading_info=True, **kwargs)
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if info.get(key):
            raise ValueError(f"Incomplete pretrained checkpoint ({key}): {info[key][:8]}")
    return model


def snapshot(entry):
    from huggingface_hub import snapshot_download

    cp = entry["checkpoint"]
    prefix = cp.get("subdirectory")
    path = snapshot_download(
        cp["source"], revision=cp["revision"], allow_patterns=[f"{prefix}/*"] if prefix else None
    )
    return Path(path) / prefix if prefix else Path(path)


def load_smolvla(entry, runtime):
    from huggingface_hub import snapshot_download
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    path = snapshot(entry)
    config = SmolVLAConfig.from_pretrained(path)
    config.device = "cpu"
    config.compile_model = False
    config.load_vlm_weights = False
    config.train_expert_only = False
    config.freeze_vision_encoder = True
    config.vlm_model_name = snapshot_download(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        revision="7b375e1b73b11138ff12fe22c8f2822d8fe03467",
        allow_patterns=["*.json", "*.txt", "*.model", "*.jinja"],
    )
    model = SmolVLAPolicy.from_pretrained(path, config=config, strict=True)
    return model, lambda policy, batch: policy(_smol_batch(batch))[0], path


def _smol_batch(batch):
    batch = dict(batch)
    if "action_is_pad" in batch:
        batch["actions_id_pad"] = batch["action_is_pad"]
    return batch


def load_pi0(entry, runtime):
    from lerobot.policies.pi0.configuration_pi0 import PI0Config
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from safetensors.torch import load_file

    path = snapshot(entry)
    config = PI0Config.from_pretrained(path)
    config.device = "cpu"
    config.compile_model = config.gradient_checkpointing = False
    config.freeze_vision_encoder = True
    config.train_expert_only = False
    model = PI0Policy(config)
    # The upstream v0.4.4 from_pretrained catches load errors and returns random
    # weights. Perform its native remapping ourselves and REQUIRE a complete load.
    state = model._fix_pytorch_state_dict_keys(load_file(str(path / "model.safetensors")), config)
    state = {k if k.startswith("model.") else f"model.{k}": v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model, lambda policy, batch: policy(batch)[0], path


def _openvla(path):
    import torch
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction

    return checked_hf_load(
        OpenVLAForActionPrediction,
        str(path),
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )


def load_openvla(entry, runtime):
    path = snapshot(entry)
    model = _openvla(path)
    model.config.use_cache = False

    def loss(policy, batch):
        return policy(
            **{
                k: batch[k]
                for k in (
                    "input_ids",
                    "attention_mask",
                    "pixel_values",
                    "labels",
                )
            }
        ).loss

    return model, loss, path


def load_openvla_oft(entry, runtime):
    import torch
    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.projectors import ProprioProjector
    from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
    from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM

    if (ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM) != (7, 8, 8):
        raise ValueError("OFT runtime must use its native LIBERO constants (7,8,8)")
    path = snapshot(entry)
    vla = _openvla(path)
    vla.config.use_cache = False
    vla.vision_backbone.set_num_images_in_input(2)

    def component(name):
        state = torch.load(path / name, map_location="cpu", weights_only=True)
        return {k.removeprefix("module."): value for k, value in state.items()}

    class OFTTrainingProbe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = {"model_type": "openvla_oft_probe"}
            self.vla = vla
            dim = vla.llm_dim
            self.action_head = L1RegressionActionHead(input_dim=dim, hidden_dim=dim, action_dim=7)
            self.proprio_projector = ProprioProjector(llm_dim=dim, proprio_dim=8)
            self.action_head.load_state_dict(
                component("action_head--150000_checkpoint.pt"), strict=True
            )
            self.proprio_projector.load_state_dict(
                component("proprio_projector--150000_checkpoint.pt"),
                strict=True,
            )
            self.action_head.to(dtype=torch.bfloat16)
            self.proprio_projector.to(dtype=torch.bfloat16)

        def forward(self, batch):
            # Same continuous L1 objective and action-token mask as native finetune.py.
            output = self.vla(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
                proprio=batch["proprio"],
                proprio_projector=self.proprio_projector,
                output_hidden_states=True,
                use_film=False,
            )
            labels = batch["labels"][:, 1:]
            mask = get_current_action_mask(labels) | get_next_actions_mask(labels)
            num_patches = (
                self.vla.vision_backbone.get_num_patches()
                * self.vla.vision_backbone.get_num_images_in_input()
                + 1
            )
            hidden = output.hidden_states[-1][:, num_patches:-1]
            selected = hidden[mask].reshape(batch["input_ids"].shape[0], 8 * 7, -1)
            actions = self.action_head.predict_action(selected.to(torch.bfloat16))
            if actions.shape != batch["actions"].shape:
                raise ValueError("OFT fixture must contain the native [B,8,7] action chunk")
            return torch.nn.functional.l1_loss(actions.float(), batch["actions"].float())

    return OFTTrainingProbe(), lambda policy, batch: policy(batch), path


def validate_conversion(path, manifest, entry):
    from .catalog import PROFILES

    if manifest.get("source") != entry["checkpoint"]["source"]:
        raise ValueError("Converted checkpoint does not refer to the catalog's pi05_libero source")
    if manifest.get("float_action_equivalence_passed") is not True:
        raise ValueError("Verify JAX/PyTorch floating action equivalence before QLoRA")
    if not manifest.get("source_hashes") or not manifest.get("conversion_commit"):
        raise ValueError("Conversion requires source hashes and a native code commit")
    if manifest["conversion_commit"] != PROFILES["pi05"]["code_commit"]:
        raise ValueError("Conversion must use the pinned OpenPI source commit")
    if any(
        not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in manifest["source_hashes"].values()
    ):
        raise ValueError("Source checkpoint files require SHA-256 digests")
    hashes = manifest.get("output_hashes", {})
    actual = {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()}
    if "model.safetensors" not in hashes or set(hashes) != actual:
        raise ValueError("Conversion manifest must hash every file in the converted checkpoint")
    for name, digest in hashes.items():
        asset = (path / name).resolve()
        if not asset.is_relative_to(path.resolve()) or sha256(asset) != digest:
            raise ValueError(f"Converted asset hash mismatch: {name}")


def load_pi05(entry, runtime):
    from openpi.models.model import Observation
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.training.config import get_config
    from safetensors.torch import load_model

    path = Path(runtime["converted_checkpoint"])
    manifest = json.loads(Path(runtime["conversion_manifest"]).read_text())
    validate_conversion(path, manifest, entry)
    config = get_config("pi05_libero")
    model = PI0Pytorch(config.model)
    load_model(model, str(path / "model.safetensors"), strict=True)
    # Preprocessed native observation dict plus padded native action chunk.
    return (
        model,
        lambda policy, batch: policy(
            Observation.from_dict(batch["observation"]),
            batch["actions"],
        ).mean(),
        path,
    )


def load_gr00t(entry, runtime):
    import gr00t.model  # noqa: F401 -- native AutoModel registrations
    import torch
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
    from huggingface_hub import snapshot_download

    path = snapshot(entry)
    config = Gr00tN1d7Config.from_pretrained(path)
    # Native architecture dispatch requires the repository identifier, not a snapshot path.
    # Prefetch the pinned backbone, then bind both its weights and processor to that revision.
    snapshot_download(config.model_name, revision=runtime["backbone_revision"])
    config.tune_llm = True
    config.use_flash_attention = False
    model = checked_hf_load(
        Gr00tN1d7,
        str(path),
        config=config,
        torch_dtype=torch.bfloat16,
        transformers_loading_kwargs={
            "revision": runtime["backbone_revision"],
            "local_files_only": True,
            "trust_remote_code": False,
        },
    )
    return model, lambda policy, batch: policy(batch)["loss"], path


LOADERS = {
    "smolvla": load_smolvla,
    "pi0": load_pi0,
    "openvla": load_openvla,
    "openvla_oft": load_openvla_oft,
    "pi05": load_pi05,
    "gr00t_n17": load_gr00t,
}
