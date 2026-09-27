# Optional Muose decision worker

This standalone CPU worker scores a small list of text criteria against a state and instructions. It has no job, tool, cloud, payment, or robot execution interface. It is **experimental and advisory only**. It is not a VLA. The optional OPEN JENSEN **Dataset → Decision lab** panel exposes manual advisory scoring when the application operator configures this isolated worker. It is disabled by default.

The only supported model is [muose/Muose-50M-Decision](https://huggingface.co/muose/Muose-50M-Decision/tree/5afb8eeff127621fea2d66fc63f56798ada12eda) at commit `5afb8eeff127621fea2d66fc63f56798ada12eda`. Its [model card](https://huggingface.co/muose/Muose-50M-Decision/blob/5afb8eeff127621fea2d66fc63f56798ada12eda/README.md) attributes it to Muose and licenses it under **CC-BY-NC-SA-4.0**. Use requires explicit license acceptance; commercial use needs separate permission from the owner. This optional component does not change the license of the rest of OPEN JENSEN. No model weights or upstream Python files are included here.

## Install separately

The tested tuple is Python 3.12.14, Torch 2.11.0, safetensors 0.8.0, tokenizers 0.23.2, and uv 0.12.19. Keep it out of the application environment. The worker currently requires Linux/macOS POSIX file and process facilities; Windows execution is unsupported. The recorded real-model proof was run on macOS ARM64; Linux model scoring has not yet been verified.

From the repository root, using the pinned uv binary:

```sh
uv venv --python 3.12.14 /absolute/path/to/decision-env
# Linux only: preinstall the CPU build, rather than downloading CUDA libraries.
uv pip install --python /absolute/path/to/decision-env/bin/python \
  --index-url https://download.pytorch.org/whl/cpu torch==2.11.0
# macOS: omit the preceding Linux-only command.
uv pip install --python /absolute/path/to/decision-env/bin/python ./workers/decision
```

Installation may require network access. Scoring never downloads anything. Runtime version admission normalizes a Torch build suffix such as `+cpu`; retain the full installed version in experiment receipts. Dependencies are pinned in this isolated package; they are not added to the application lockfile.

Place the following seven files from the exact pinned revision in one operator-managed external directory: `README.md`, `config.json`, `model.py`, `decision_model.py`, `model.safetensors`, `vocab.json`, and `merges.txt`. The exact sizes and SHA256 values are in [integrity.py](src/firebird_decision/integrity.py). The model weights are 203,051,616 bytes. The worker has no automatic downloader, remote-code approval switch, or support for alternate revisions. Every file must match before any upstream Python executes.

## Run

Save this as a regular UTF-8 JSON file:

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

```sh
/absolute/path/to/decision-env/bin/firebird-decision \
  --model-dir /absolute/operator/path/to/pinned-muose \
  --request /absolute/path/to/request.json \
  --accept-license CC-BY-NC-SA-4.0 \
  --timeout-seconds 30
```

Successful stdout is a single JSON receipt. It contains the selected criterion ID, all logits, **uncalibrated relative softmax weights**, token counts, model/revision/weight identity, prompt-template and request hashes, CPU/runtime information, and load/scoring times. It never claims that the selected answer is correct and never acts on it. Ties choose the first supplied criterion. A failure returns a nonzero exit status and a bounded JSON error on stderr; no partial result is accepted. Unexpected child diagnostics and private paths are not reflected through the public CLI.

Requests are limited to 64 KiB, 2–8 unique criteria, 8,000 state characters, 2,000 instruction characters, and 2,000 characters per criterion. IDs contain only ASCII letters, digits, underscores, and hyphens (1–64 characters). Each complete criterion prompt must fit **512 tokens**; long prompts are rejected, never truncated. Scores are computed separately without padding. User text is data, not an instruction to run an application action.

The experimental `firebird-experimental-sections-v1` template is exactly:

```text
STATE:
{state}

INSTRUCTIONS:
{instructions}

CRITERION:
{criterion}
```

The upstream repository does not publish a complete inference/tokenization helper establishing a canonical prompt format. This template is explicitly OPEN JENSEN's experiment. Do not equate these results with the author's BANKING77 scores.

## Execution and integrity boundaries

- All ancestor directories and final input files use descriptor-based no-follow traversal. Files must be regular, have bounded size, and remain stable during bounded reads. FIFO, symlink, changed-size, and checksum mismatches fail closed.
- The reviewed `model.py` and `decision_model.py` bytes are executed in isolated temporary module namespaces only after all seven files pass verification. The model directory is never added to Python's import path. The original bytes are never modified.
- The source defines an unused `torch.load(..., weights_only=False)` helper. OPEN JENSEN does not call it, and disables `torch.load` in the scoring process. It loads only the pinned safetensors bytes, restores the one declared tied embedding alias, and checks exact tensor inventory, shapes, FP32 dtype, finiteness, and the 50,760,960 parameter count.
- CPU only: two intra-op threads, one inter-op thread, evaluation/inference mode. Python audit hooks reject socket connection/name-resolution and subprocess launch attempts; this is **not an OS network or filesystem sandbox**. The pinned dependency runtime and operator configuration remain trusted.
- Only the public CLI supplies the owned-process deadline: default 30 seconds, configurable above zero through 120 seconds. Output streams are bounded to 64 KiB each. Timeout, interruption, or excessive output kills and reaps the owned scoring child, with up to five seconds for local reaping. The supervisor defers signals during child creation and cleanup. There is no automatic retry.
- The internal `--_child` path and `DecisionScorer` class are implementation/isolated-experiment interfaces, not independent timeout boundaries. An outer application supervisor must give the public CLI its requested deadline plus the cleanup allowance and send graceful termination before forced kill. No software can reap a child after the parent itself is forcibly killed.
- Application integration configures the interpreter and model directory on the operator side. Browser requests cannot select filesystem paths, executables or credentials. The model cannot authorize operations on its own.

## Optional application panel

Configure these variables on the separately running application host, then restart it:

```sh
export FIREBIRD_DECISION_PYTHON=/absolute/path/to/decision-env/bin/python
export FIREBIRD_DECISION_ROOT=/absolute/path/to/firebird/workers/decision
export FIREBIRD_DECISION_MODEL_DIR=/absolute/operator/path/to/pinned-muose
export FIREBIRD_DECISION_ACCEPT_LICENSE=CC-BY-NC-SA-4.0
```

The interpreter must already contain the pinned dependencies above. The worker root is the directory containing `src/firebird_decision`; it is not supplied by a browser. The model directory remains external and unchanged. These settings do not configure GCP or any provider. The application supports this worker on POSIX hosts; Windows reports unavailable.

`GET /api/v1/decision/status` reports configuration and whether a request can be attempted. It does not pre-certify model integrity, runtime compatibility or quality. `POST /api/v1/decision/score` accepts the same bounded request shown above. The application validates the exact model, weights, template, runtime and request identities, every finite score and token count, the selected criterion, and recomputed relative weights. Failed or malformed worker output is rejected without exposing local paths or arbitrary diagnostics.

Only one score runs per application process; concurrent submissions receive HTTP409. There is no automatic write retry, queue, saved job or motion operation. The app sends the fixed public CLI a 30-second deadline within an outer 40-second timeout; cleanup first allows the CLI 6 seconds to reap its child, then forces owned-process-group cleanup if needed. This is a local execution deadline, not a model-quality claim. Child processes receive a minimal environment without inherited provider or cloud credentials. Disconnect, timeout, excessive output and cancellation are covered by local subprocess tests.

Open **Dataset → Decision lab**, enter a state and 2–8 unique criteria, and explicitly choose **Score criteria**. The result is a manual experiment. Changing the input clears the displayed result. Stopping or leaving the panel cancels the browser request; no automatic retry or action follows. The panel displays the existing 3/6 workflow-fixture limitation and uncalibrated labels prominently. Real-model application validation on macOS uses the exact pinned model and unchanged source inventory; it does not establish Linux/Windows model scoring or robotics success.

## Tests and measured limitations

Unit tests need the worker test requirements. The real model test is opt-in and never downloads weights:

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

The receipt destination must not already exist. The real probe has an outer 120-second deadline and tests the public supervised CLI as well. It checks deterministic repetition, criterion-order invariance, token rejection, complete response identity, and all seven source hashes before/after execution. The unit suite exercises malformed/oversized JSON, checksum and symlink/FIFO rejection, mutation during reads, response forgery, real child timeouts/output bounds, creation-time interruption, and repeated signals during cleanup.

On 2026-09-27, the predeclared [12-example fixture](fixtures/scoring-v1.json), SHA256 `0e0053fe2e77d1407714180604acc2f926ca27e51c2c41af1e0661d87b2bee5f`, measured **5/6 banking** and **3/6 workflow examples** correct. Both clarification requests were wrongly classified as dataset inspection; the simulation-evaluation request was also classified as inspection. One lost-card request was classified as a phone-number change. These small, hand-authored cases are neither a representative benchmark nor robotics validation, and were not used for prompt tuning.

That macOS ARM64 run measured 1.677 seconds for checksum validation/import/model loading. Across 12 warmed four-criterion requests, median scoring time was 42.16 ms and nearest-rank p95 was 48.82 ms (only 12 samples; not a latency guarantee). Original source hashes were unchanged, and repetition/order checks passed. The first measurement attempt failed after scoring because platform metadata collection tried a subprocess blocked by the guard; the metadata lookup was corrected without changing the prompt, model, examples, or expected answers. These results support a working optional local scorer; they do **not** support automatic workflow routing or robot action selection.
