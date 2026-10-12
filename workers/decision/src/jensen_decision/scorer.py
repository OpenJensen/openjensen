"""Load only reviewed in-memory bytes; never use the upstream pickle helper."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import sys
import time
import types
from pathlib import Path

from .contracts import (
    CAVEATS,
    LICENSE,
    MAX_TOKENS,
    REPOSITORY,
    REVISION,
    TEMPLATE,
    TEMPLATE_VERSION,
    VERSIONS,
    DecisionError,
    prompt,
    request_hash,
    validate_request,
    weights,
)
from .integrity import FILES, verified_files


def offline_guard():
    """Python audit boundary only, not an OS security sandbox."""
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
    )

    def audit(event, args):
        if event in {"socket.connect", "socket.getaddrinfo", "subprocess.Popen", "os.system"}:
            raise DecisionError(
                "Network and subprocess operations are disabled in the scoring process."
            )

    sys.addaudithook(audit)


def _execute_sources(raw):
    names = ("model", "_firebird_pinned_muose_decision")
    previous = {name: sys.modules.get(name) for name in names}
    modules = []
    try:
        for name, filename in zip(names, ("model.py", "decision_model.py"), strict=True):
            module = types.ModuleType(name)
            sys.modules[name] = module  # dataclasses and the reviewed absolute import need this.
            exec(compile(raw[filename], f"<pinned-muose/{filename}>", "exec"), module.__dict__)
            modules.append(module)
        return modules
    finally:
        for name in names:
            if previous[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous[name]


def _deny_pickle(*args, **kwargs):
    raise DecisionError("Pickle checkpoints are unsupported; use the pinned safetensors file.")


class DecisionScorer:
    """For the supervised worker or isolated offline experiments, not the core process."""

    def __init__(self, model_dir: Path, accepted_license: str):
        started = time.perf_counter()
        raw = verified_files(
            model_dir, accepted_license
        )  # verify ALL bytes before any custom execution
        for name, expected in VERSIONS.items():
            try:
                actual = importlib.metadata.version(name).split("+")[0]
            except importlib.metadata.PackageNotFoundError as exc:
                raise DecisionError(
                    "Install this worker's pinned dependencies in an isolated environment.",
                    "runtime_unavailable",
                ) from exc
            if actual != expected:
                raise DecisionError(
                    f"Unsupported {name} version; expected {expected}.", "runtime_unavailable"
                )
        import torch
        from safetensors.torch import load
        from tokenizers import ByteLevelBPETokenizer

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        torch.load = _deny_pickle
        base_source, decision_source = _execute_sources(raw)
        config = json.loads(raw["config.json"])
        model = decision_source.MuoseDecision(
            base_source.Muose(base_source.MuoseConfig(**config["base_config"]))
        )
        state = load(raw["model.safetensors"])
        # The pinned safetensors header records this ONE tied alias, not arbitrary missing keys.
        state["base.token_embedding.weight"] = state["base.lm_head.weight"]
        expected = model.state_dict()
        if set(state) != set(expected):
            raise DecisionError(
                "Checkpoint tensor inventory differs from the reviewed architecture."
            )
        for name, tensor in state.items():
            if (
                tensor.dtype != torch.float32
                or tensor.shape != expected[name].shape
                or not torch.isfinite(tensor).all()
            ):
                raise DecisionError(
                    "Checkpoint tensor shape, dtype, or finite-value validation failed."
                )
        model.load_state_dict(state, strict=True)
        if sum(parameter.numel() for parameter in model.parameters()) != 50_760_960:
            raise DecisionError("Checkpoint parameter count does not match the pinned model.")
        self.model = model.cpu().eval()
        self.torch = torch
        merges = [
            tuple(line.split())
            for line in raw["merges.txt"].decode("utf-8").splitlines()
            if line and not line.startswith("#version:")
        ]
        self.tokenizer = ByteLevelBPETokenizer(
            json.loads(raw["vocab.json"]), merges, add_prefix_space=False, lowercase=False
        )
        self.load_ms = (time.perf_counter() - started) * 1_000

    def score(self, request):
        validate_request(request)
        started = time.perf_counter()
        sequences = [
            self.tokenizer.encode(prompt(request, criterion)).ids
            for criterion in request["criteria"]
        ]
        if any(not 1 <= len(ids) <= MAX_TOKENS for ids in sequences):
            raise DecisionError(
                "Each complete criterion prompt must fit 512 tokens; input was not truncated.",
                "token_limit",
            )
        logits = []
        with self.torch.inference_mode():
            for ids in sequences:
                tensor = self.torch.tensor([ids], dtype=self.torch.long, device="cpu")
                length = self.torch.tensor([len(ids)], dtype=self.torch.long, device="cpu")
                value = self.model(tensor, length)
                if tuple(value.shape) != (1,):
                    raise DecisionError("Model returned an invalid score shape.")
                logits.append(float(value.item()))
        relative_weights = weights(logits)
        best = max(range(len(logits)), key=logits.__getitem__)
        return {
            "schema_version": 1,
            "model": REPOSITORY,
            "revision": REVISION,
            "model_sha256": FILES["model.safetensors"][1],
            "license": LICENSE,
            "prompt_template": TEMPLATE_VERSION,
            "prompt_template_sha256": hashlib.sha256(TEMPLATE.encode()).hexdigest(),
            "request_sha256": request_hash(request),
            "device": "cpu",
            "threads": 2,
            "runtime_versions": VERSIONS,
            "advisory_only": True,
            "calibrated": False,
            "selected_id": request["criteria"][best]["id"],
            "scores": [
                {
                    "id": criterion["id"],
                    "logit": logits[index],
                    "relative_weight": relative_weights[index],
                    "tokens": len(sequences[index]),
                }
                for index, criterion in enumerate(request["criteria"])
            ],
            "timing_ms": {"load": self.load_ms, "score": (time.perf_counter() - started) * 1_000},
            "caveats": CAVEATS,
        }
