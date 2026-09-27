# QLoRA checks using the quantization benchmark catalog

`firebird-check-qlora` consumes the exact model identities and revisions in
[the six-model catalog](../../configs/benchmarks/open_weight_vlas.json), following
[idea document 18](../idea/18_multimodel-quantization-benchmark.md).
The existing SO-101 trainer remains a separate recipe. Its base checkpoint,
camera mapping and action data are not substituted into a LIBERO comparison.

## Scope and current evidence

These are **native training-batch integration diagnostics**. They check whether
the catalog checkpoint can load, compute its native loss, train LoRA parameters
over actual NF4 backbone weights, and reproduce the saved adapted loss in a new
process. They do not fine-tune on a complete dataset or measure robot success.

All six native loader/loss paths are implemented but **CUDA-unverified**. On the
development Mac, CPU contract tests pass; the native GPU checks cannot run.
The [recorded metadata inspection](qlora-model-inspection-2026-09-26.json) verified
the five Hugging Face revisions and required asset presence. π₀.₅ is blocked on
source-file hashes and a verified PyTorch conversion. Metadata inspection is not
evidence that QLoRA works.

| Catalog model | Native loss/checkpoint handling | Additional prerequisite |
|---|---|---|
| SmolVLA | LeRobot flow matching; `lerobot/smolvla_libero` | Native LIBERO camera/state/language batch, not SO-101 |
| OpenVLA-OFT | Native continuous L1 head, both cameras, proprio projector, eight-action chunk | OFT source and its pinned bidirectional-attention Transformers fork |
| OpenVLA | Native autoregressive action-token loss | Native processed pixels, action-token labels and masks |
| π₀ | LeRobot flow matching; strict checkpoint remapping | π₀ native prepared batch; existing checkpoint normalization |
| π₀.₅ | Native OpenPI PyTorch flow matching | Conversion manifest, all source/output hashes, floating action-equivalence evidence |
| GR00T N1.7 | Native GR00T action-head loss, `libero_spatial` subdirectory | `LIBERO_PANDA` batch and separately pinned Cosmos backbone |

The π₀ loader deliberately avoids the pinned upstream loader's catch-and-return
fallback. A missing or mismatched checkpoint must fail instead of testing a
randomly initialized model. All other loaders also reject incomplete state loads.

## Inspect and plan without a GPU

From `workers/smolvla_qlora`, install the lightweight package:

```bash
uv pip install -e .

# No ML imports, downloads or jobs:
firebird-check-qlora --output-dir outputs/qlora-plan

# Small public metadata/config reads only; no weight downloads:
firebird-check-qlora --inspect --output-dir outputs/qlora-inspection
```

Every report preserves all six rows in the declared priority order. Unselected
models stay pending. Use a fresh output directory for each check.

## Prepare native environments and fixtures

Copy [the runtime example](../../configs/benchmarks/qlora_runtimes.example.json)
and replace its absolute paths. Each model names its own Python interpreter,
native checkout, exact source commit, dependency lock and training-batch fixture.
SmolVLA and π₀ can share the pinned LeRobot environment from the
[existing setup guide](smolvla-qlora.md). Keep OpenVLA, OFT, OpenPI and GR00T
environments separate: their Torch, Transformers and PEFT requirements differ.

Install the selected native repository at the commit recorded in the profile,
following its own installation procedure. In each environment install OPEN JENSEN
with `pip install --no-deps -e /absolute/path/to/this/repository`. Use Python 3.11+
(GR00T's native environment uses Python 3.12). Record the actual resolved packages
with `python -m pip freeze --all > /absolute/path/to/locks/model.txt` and pass that
file as `dependency_lock`. This captures the environment used; it is not a claim
that every possible dependency combination is supported.

The SmolVLA/π₀ lock includes bitsandbytes 0.48.2 and PEFT 0.18.0. For native
OpenVLA/OFT's older Torch stack, bitsandbytes 0.43.1 supports the required packed
BF16 storage; retain the native PEFT version. OpenPI needs PEFT with dataclass
config support (0.18.0) in its own environment. GR00T retains its native PEFT
version. Pin bitsandbytes in each lock and let the actual check decide compatibility;
do not force the SmolVLA dependency lock onto another native stack.

Critical package versions are enforced by the worker:

| Native environment | Torch | Transformers | PEFT | bitsandbytes |
|---|---|---|---|---|
| SmolVLA / π₀ | 2.7.1 | 4.57.1 | 0.18.0 | 0.48.2 |
| OpenVLA | 2.2.0 | 4.40.1 | 0.11.1 | 0.43.1 |
| OpenVLA-OFT | 2.2.0 | pinned native fork | 0.11.1 | 0.43.1 |
| OpenPI π₀.₅ | 2.7.1 | 4.53.2, with native OpenPI setup | 0.18.0 | 0.48.2 |
| GR00T N1.7 | 2.9.0 | 4.57.3 | 0.17.1 | 0.48.2 |

These are source-aligned diagnostic profiles, not GPU-validated compatibility claims.

OFT additionally requires Transformers from
`moojink/transformers-openvla-oft@bc339d9ad707454c0c115970db43c260067c61ab`.
The worker checks the installed VCS commit (or editable checkout), not just the
Transformers version string. Native repository commits are checked against clean
tracked source. Installed package versions are saved and compared at reload.

Export **one batch after the model's native training processor and collator**.
This preserves its actual camera transforms, tokenizer, normalization, embodiment,
action shape and padding. There is no universal conversion from SO-101 to LIBERO.
The helper stores tensors as safetensors and containers/provenance as JSON:

```python
import json
from firebird_vla.checks.catalog import PROFILES
from firebird_vla.checks.fixtures import save_fixture

catalog = json.load(open("configs/benchmarks/open_weight_vlas.json"))
model_id = "smolvla"  # Run this inside that model's native data pipeline.
entry = next(m for m in catalog["models"] if m["id"] == model_id)

# batch = next(iter(native_loader)); batch = native_preprocessor(batch)
save_fixture(
    "/absolute/path/to/fixtures/smolvla",
    entry,
    batch,
    {
        "source": "your pinned LIBERO dataset/revision and training episode IDs",
        "kind": "recorded",
        "split": "train",
        "native_code_commit": PROFILES[model_id]["code_commit"],
        "processor_description": "native processor config/revision and normalization provenance",
    },
)
```

Replace the example source/processor descriptions with actual provenance.
GR00T also requires `"embodiment": "LIBERO_PANDA"`. For OpenPI export
`{"observation": observation.to_dict(), "actions": actions}`. For GR00T export
the native collator's `inputs` dict. For OpenVLA/OFT export the native collated
training batch; OFT requires the combined camera tensor, `proprio`, `actions`,
`input_ids`, `labels` and `attention_mask`. Convert custom containers/NumPy arrays
to ordinary containers/tensors before exporting.

If a synthetic native batch is deliberately used, set `kind` to `synthetic`;
the label propagates into phase evidence. Search/final/test-evaluation fixtures
are rejected for this training check. Never tune on the final benchmark starts.

For π₀.₅, supply an external conversion manifest with:

```json
{
  "source": "gs://openpi-assets/checkpoints/pi05_libero",
  "conversion_commit": "<40-character native conversion code SHA>",
  "source_hashes": {"params/example-file": "<SHA-256>"},
  "output_hashes": {"model.safetensors": "<SHA-256 of converted file>"},
  "float_action_equivalence_passed": true
}
```

Hash **every** converted file, including assets, and supply the actual source
inventory. Keep the conversion manifest outside the converted directory. The
worker verifies the supplied output hashes; the equivalence assertion must come
from a separately completed conversion check, not from setting the flag merely
to pass preflight. This command does not convert or validate JAX source weights.

## Execute on CUDA

```bash
firebird-check-qlora --run \
  --runtimes /absolute/path/to/qlora-runtimes.json \
  --models smolvla openvla_oft \
  --output-dir outputs/qlora-native-checks

# Omit --models to attempt all six, sequentially in catalog priority order.
```

Each selected model executes three fresh processes in its own environment:

1. **float:** load the exact native checkpoint and compute a finite scalar loss
   on the bound batch, with a fixed random seed.
2. **qlora:** load the same checkpoint; freeze it; pack only explicitly allowlisted
   language-backbone attention/MLP linears as double-quantized NF4; attach rank-8
   LoRA; perform two optimizer steps. Require finite nonzero gradients, changed
   adapter weights and byte-identical frozen packed weights. Save the adapter.
3. **reload:** re-create the same quantized base in a new process, verify source/
   adapter/fixture hashes and package versions, load the adapter, and compare
   every restored adapter tensor exactly, then the native loss at the original
   seed (`atol=1e-4`, `rtol=1e-3`).

`--rank`, `--steps`, `--learning-rate`, `--seed` and `--timeout-seconds` control the
diagnostic. The default phase timeout is 30 minutes, including weight downloads.
The run checks CUDA/BF16 before downloading weights. Source weight hashes are
computed per phase, which adds host I/O time. Full-precision construction happens
on CPU, and layers are packed onto CUDA individually. Host RAM is still required.

Vision, multimodal projectors, action experts/heads, embeddings, normalization and
action-token output heads remain frozen in their native precision. Quantization
maps are distinct per architecture and are checked against actual loaded modules.
This differs intentionally from the original SO-101 worker, which quantizes the
SmolVLA expert and trains expert LoRA plus state/action/time projections.

## Reading the report

`report.json` contains all model rows and evidence paths. Each attempted phase has
its own log and JSON result. The QLoRA phase records exact quantized modules,
adapter counts, changed tensors, packed-weight integrity, native loss before/after,
GPU peak allocated/reserved bytes, elapsed time, source hashes and package versions.
The precision map lists every native linear module's shape, native dtype and
whether it was packed; protected linear module names are included explicitly.

OOM, dependency failures, missing fixtures, timeouts and incomplete runs remain
visible. A floating-reference OOM does not prevent attempting the smaller QLoRA
diagnostic, but the row stays `incomplete_float_reference` even if that check passes.
There is no cross-device memory/speedup comparison. A failed QLoRA phase blocks
reload, and any selected incomplete row makes the command exit nonzero.

`qlora_verified` means only the native gradient/update/loss-reload diagnostic
passed. It never promotes `inference_verified`, `quantization_verified`,
`closed_loop_verified` or `export_verified`, and never changes the source catalog's
disabled benchmark-execution flags. Success rate, episode counts and action latency
stay null. No average of incomparable model losses is produced.

To evaluate QLoRA's effect on task quality, next run each completed adapted policy
and its own released baseline through the native LIBERO evaluator using the task/
initial-state splits in documents 14 and 18. C0/C1/C2/C3 inference candidates,
closed-loop admission and complete deployment export remain separate operations.

Native source references:
[LeRobot 0.4.4](https://github.com/huggingface/lerobot/tree/8fff0fde7c79f23a93d845d1a50e985de01f8b8a),
[OpenVLA](https://github.com/openvla/openvla/tree/c8f03f48af692657d3060c19588038c7220e9af9),
[OFT](https://github.com/moojink/openvla-oft/tree/e4287e94541f459edc4feabc4e181f537cd569a8),
[OpenPI](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479),
[GR00T](https://github.com/NVIDIA/Isaac-GR00T/tree/51d4c89f72fda44cbf77285c6a8114b52676b8a1).
