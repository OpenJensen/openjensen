# Muose scorer setup

Use a POSIX host with Python 3.12.14, uv 0.12.19, Torch 2.11.0,
safetensors 0.8.0 and tokenizers 0.23.2.

## Install

From the repository root:

```sh
uv venv --python 3.12.14 /absolute/path/to/decision-env
# Linux only: preinstall the CPU build, rather than downloading CUDA libraries.
uv pip install --python /absolute/path/to/decision-env/bin/python \
  --index-url https://download.pytorch.org/whl/cpu torch==2.11.0
# macOS: omit the preceding Linux-only command.
uv pip install --python /absolute/path/to/decision-env/bin/python ./workers/decision
```

Place `README.md`, `config.json`, `model.py`, `decision_model.py`,
`model.safetensors`, `vocab.json` and `merges.txt` from
[model revision `5afb8eeff127621fea2d66fc63f56798ada12eda`](https://huggingface.co/muose/Muose-50M-Decision/tree/5afb8eeff127621fea2d66fc63f56798ada12eda)
in one operator-owned external directory. File hashes are in
[integrity.py](src/firebird_decision/integrity.py). Accept the model's
[CC-BY-NC-SA-4.0 license](https://huggingface.co/muose/Muose-50M-Decision/blob/5afb8eeff127621fea2d66fc63f56798ada12eda/README.md)
when running the command.

## Score a request

Save a UTF-8 JSON file using this structure. Supply 2–8 unique criteria; each
complete criterion prompt must fit 512 tokens.

```json
{
  "schema_version": 1,
  "state": "The approved snapshot is ready. Fine-tune an ACT policy on it.",
  "instructions": "Select the criterion that best describes the requested task. This is advisory classification only.",
  "criteria": [
    {"id": "train", "text": "Fine-tune a robot policy using the approved dataset."},
    {"id": "clarify", "text": "Ask the user to clarify the intended task; do not start a job."}
  ]
}
```

Replace the model and request paths:

```sh
/absolute/path/to/decision-env/bin/firebird-decision \
  --model-dir /absolute/operator/path/to/pinned-muose \
  --request /absolute/path/to/request.json \
  --accept-license CC-BY-NC-SA-4.0 \
  --timeout-seconds 30
```

Read the JSON result from stdout. The CLI uses this prompt template:

```text
STATE:
{state}

INSTRUCTIONS:
{instructions}

CRITERION:
{criterion}
```

## Configure the application

Set these variables on the application host and restart the application:

```sh
export FIREBIRD_DECISION_PYTHON=/absolute/path/to/decision-env/bin/python
export FIREBIRD_DECISION_ROOT=/absolute/path/to/firebird/workers/decision
export FIREBIRD_DECISION_MODEL_DIR=/absolute/operator/path/to/pinned-muose
export FIREBIRD_DECISION_ACCEPT_LICENSE=CC-BY-NC-SA-4.0
```

Open **Test → Decision lab**, enter the state and criteria, then choose
**Score criteria**. `GET /api/v1/decision/status` checks configuration;
`POST /api/v1/decision/score` accepts the request structure above.

## Run tests

Choose a new receipt path for the opt-in model test:

```sh
uv pip install --python /absolute/path/to/decision-env/bin/python \
  -r workers/decision/requirements-test.txt
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=workers/decision/src \
  /absolute/path/to/decision-env/bin/python -m pytest workers/decision/tests -q -p no:cacheprovider
FIREBIRD_DECISION_MODEL=/absolute/operator/path/to/pinned-muose \
  FIREBIRD_DECISION_ACCEPT_LICENSE=CC-BY-NC-SA-4.0 \
  FIREBIRD_DECISION_RECEIPT=/absolute/new/path/to/scoring-receipt.json \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=workers/decision/src \
  /absolute/path/to/decision-env/bin/python -m pytest workers/decision/tests -q -p no:cacheprovider
```
