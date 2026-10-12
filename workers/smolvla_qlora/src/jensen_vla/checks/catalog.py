import json
import re
from pathlib import Path

# Each map is architecture-specific. Only language-backbone linears are packed;
# vision, multimodal projectors, action experts/heads and output tokens stay native.
PROFILES = {
    "smolvla": {
        "loader": "smolvla",
        "roots": ["model.vlm_with_expert.vlm.model.text_model.layers."],
        "runtime": "lerobot==0.4.4; requirements-smolvla-linux.txt",
        "code_commit": "8fff0fde7c79f23a93d845d1a50e985de01f8b8a",
    },
    "openvla_oft": {
        "loader": "openvla_oft",
        "roots": ["vla.language_model.model.layers."],
        "runtime": "isolated native OpenVLA-OFT environment (its Transformers fork)",
        "code_commit": "e4287e94541f459edc4feabc4e181f537cd569a8",
        "transformers_code_commit": "bc339d9ad707454c0c115970db43c260067c61ab",
        "auxiliary_files": [
            "action_head--150000_checkpoint.pt",
            "proprio_projector--150000_checkpoint.pt",
        ],
    },
    "openvla": {
        "loader": "openvla",
        "roots": ["language_model.model.layers."],
        "runtime": "isolated native OpenVLA environment (Transformers 4.40.1)",
        "code_commit": "c8f03f48af692657d3060c19588038c7220e9af9",
    },
    "pi0": {
        "loader": "pi0",
        "roots": ["model.paligemma_with_expert.paligemma.model.language_model.layers."],
        "runtime": "lerobot==0.4.4; requirements-smolvla-linux.txt",
        "code_commit": "8fff0fde7c79f23a93d845d1a50e985de01f8b8a",
    },
    "pi05": {
        "loader": "pi05",
        "roots": ["paligemma_with_expert.paligemma.model.language_model.layers."],
        "runtime": "isolated native OpenPI PyTorch environment and converted checkpoint",
        "code_commit": "215abfb217dbac7d5f1273282331b9b1866c0479",
    },
    "gr00t_n17": {
        "loader": "gr00t_n17",
        "roots": ["backbone.model.model.language_model.layers."],
        "runtime": "isolated Isaac-GR00T environment; LIBERO_PANDA processor fixture",
        "code_commit": "51d4c89f72fda44cbf77285c6a8114b52676b8a1",
    },
}
LINEARS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}

CRITICAL_PACKAGES = {
    "smolvla": {
        "lerobot": "0.4.4",
        "torch": "2.7.1",
        "transformers": "4.57.1",
        "peft": "0.18.0",
        "bitsandbytes": "0.48.2",
    },
    "pi0": {
        "lerobot": "0.4.4",
        "torch": "2.7.1",
        "transformers": "4.57.1",
        "peft": "0.18.0",
        "bitsandbytes": "0.48.2",
    },
    "openvla": {
        "torch": "2.2.0",
        "transformers": "4.40.1",
        "peft": "0.11.1",
        "bitsandbytes": "0.43.1",
    },
    # Transformers is checked by VCS commit for OFT, not by an ambiguous version string.
    "openvla_oft": {"torch": "2.2.0", "peft": "0.11.1", "bitsandbytes": "0.43.1"},
    "pi05": {
        "torch": "2.7.1",
        "transformers": "4.53.2",
        "peft": "0.18.0",
        "bitsandbytes": "0.48.2",
    },
    "gr00t_n17": {
        "torch": "2.9.0",
        "transformers": "4.57.3",
        "peft": "0.17.1",
        "bitsandbytes": "0.48.2",
    },
}


def selected_module(name, model_id):
    return (
        name.startswith(tuple(PROFILES[model_id]["roots"])) and name.rsplit(".", 1)[-1] in LINEARS
    )


def load_catalog(path):
    catalog = json.loads(Path(path).read_text())
    if catalog.get("schema_version") != 1 or catalog.get("kind") != "benchmark_catalog":
        raise ValueError("Expected the idea/ multi-model benchmark catalog, not a training recipe")
    if catalog.get("target_suite") != "libero_spatial":
        raise ValueError("These profiles are restricted to the catalog's LIBERO-Spatial check")
    ids = [m["id"] for m in catalog["models"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate catalog model")
    for entry in catalog["models"]:
        if entry["id"] not in PROFILES:
            raise ValueError(f"No native profile for {entry['id']}")
        cp = entry["checkpoint"]
        if not cp["source"].startswith("gs://") and not re.fullmatch(
            r"[0-9a-f]{40}", cp["revision"] or ""
        ):
            raise ValueError(f"Checkpoint must be immutable: {entry['id']}")
        sub = cp.get("subdirectory")
        if sub and (Path(sub).is_absolute() or ".." in Path(sub).parts):
            raise ValueError("Unsafe checkpoint subdirectory")
    return catalog


def result_row(entry):
    return {
        "model_id": entry["id"],
        "checkpoint": entry["checkpoint"],
        "suite": "libero_spatial",
        "status": "pending",
        "reason": None,
        "scope": "native training-batch QLoRA diagnostic; not a closed-loop benchmark",
        "candidate": "Q2_backbone_nf4_lora",
        "runtime_profile": PROFILES[entry["id"]],
        "metrics": None,
        "qlora_verified": False,
        "inference_verified": False,
        "quantization_verified": False,
        "closed_loop_verified": False,
        "export_verified": False,
        "evidence_refs": [],
    }
