# Run native QLoRA checks

## Install and plan

From `workers/smolvla_qlora`:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e .
. .venv/bin/activate

# No ML imports, downloads or jobs:
firebird-check-qlora --output-dir outputs/qlora-plan

# Small public metadata/config reads only; no weight downloads:
firebird-check-qlora --inspect --output-dir outputs/qlora-inspection
```

Use a new output directory for each command. Model revisions are in the
[catalog](../../configs/benchmarks/open_weight_vlas.json).

## Configure model environments

Copy [qlora_runtimes.example.json](../../configs/benchmarks/qlora_runtimes.example.json)
and set each model's Python path, native checkout/commit, dependency lock and
batch fixture. Follow each pinned native repository's installation procedure.
Use Python 3.11+; GR00T uses Python 3.12. In each environment run
`pip install --no-deps -e /absolute/path/to/this/repository` and save
`python -m pip freeze --all > /absolute/path/to/locks/model.txt` as `dependency_lock`.

Install these profile versions:

| Native environment | Torch | Transformers | PEFT | bitsandbytes |
|---|---|---|---|---|
| SmolVLA / π₀ | 2.7.1 | 4.57.1 | 0.18.0 | 0.48.2 |
| OpenVLA | 2.2.0 | 4.40.1 | 0.11.1 | 0.43.1 |
| OpenVLA-OFT | 2.2.0 | pinned native fork | 0.11.1 | 0.43.1 |
| OpenPI π₀.₅ | 2.7.1 | 4.53.2, with native OpenPI setup | 0.18.0 | 0.48.2 |
| GR00T N1.7 | 2.9.0 | 4.57.3 | 0.17.1 | 0.48.2 |

For OFT, use Transformers from
`moojink/transformers-openvla-oft@bc339d9ad707454c0c115970db43c260067c61ab`.

## Save a processed training batch

Export one batch after the model's native processor and collator:

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

Replace provenance with the actual dataset/revision, training episode IDs and
processor configuration. Set `kind` to `recorded` or `synthetic` as appropriate.
For GR00T, include `embodiment: LIBERO_PANDA` and export its collator's `inputs`
dict. For OpenPI use `{"observation": observation.to_dict(), "actions": actions}`.
For OFT include cameras, `proprio`, `actions`, `input_ids`, `labels` and
`attention_mask`. Use ordinary containers/tensors when saving the fixture.

For π₀.₅, supply an external conversion manifest listing every source and
converted file hash after completing floating action-equivalence checks:

```json
{
  "source": "gs://openpi-assets/checkpoints/pi05_libero",
  "conversion_commit": "<40-character native conversion code SHA>",
  "source_hashes": {"params/example-file": "<SHA-256>"},
  "output_hashes": {"model.safetensors": "<SHA-256 of converted file>"},
  "float_action_equivalence_passed": true
}
```

## Run on CUDA

```bash
firebird-check-qlora --run \
  --runtimes /absolute/path/to/qlora-runtimes.json \
  --models smolvla openvla_oft \
  --output-dir outputs/qlora-native-checks

# Omit --models to attempt all six, sequentially in catalog priority order.
```

Set `--rank`, `--steps`, `--learning-rate`, `--seed` and `--timeout-seconds` to
configure the run. The default phase timeout is 30 minutes. Read `report.json`
and the per-phase files under the selected output directory.
