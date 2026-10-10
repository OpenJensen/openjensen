# Native application acceptance contract

The application-owned entry point is `policykit.application`. Acceptance checks
validate artifact and runtime contracts. Hardware performance and policy quality
require model-specific hardware and simulator evaluation.

## What is accepted

The native engine checks the GGUF architecture, positive action dimensions,
artifact action-dimension declaration, tensor precision inventory and conversion
audit before running the native probe. Probe output length comes from
`smolvla.chunk_size * smolvla.max_action_dim`. Resident language/vision matrix
counts come from the audited tensors rather than a fixed matrix count. Different
model inventories do not establish checkpoint interchangeability.
The current instrumentation admits at most 50 action steps, at most 32 padded
action channels, and 512px images. Other shapes fail explicitly.

Memory acceptance requires a finite positive peak and positive sample count for
reload, timing, **and every requested rollout**. Missing, zero, negative or
nonfinite telemetry makes the aggregate peak unavailable. A later valid rollout
cannot conceal an earlier telemetry gap. Reports retain coverage and diagnostic
measurements, replacing nonfinite JSON numbers with `null` plus field names.
CUDA figures are sampled whole-device memory, including unrelated processes;
CPU figures are sampled process-tree RSS. Neither is an exact allocation peak.
Engine p95 measures the instrumented prediction call, not the entire robot loop.

Runtime identity version 2 records:

- Native executable hashes, linked ELF library hashes resolved with `ldd`, and
  shared libraries under the build tree, including build-local dynamic plugins.
- Worker source hash, the worker's `pyproject.toml` and `uv.lock` hashes, current
  Python executable hash/version and installed distribution names/versions.
- Prepared vendor adapter and simulator source/configuration hashes when
  configured, plus generated simulator configuration inputs.
- Hardware/driver identity and digests of configured runtime and relevant process
  environment values. Environment credentials are not exported in plaintext.

The fingerprint is checked before and after measurement and simulator evaluation.
Unresolved libraries, missing runtime locks/source and non-Linux hosts are explicit
unsupported states. This first implementation certifies Linux ELF identity only.
Python package versions are recorded, not hashes of every third-party Python file;
arbitrary system libraries loaded later through unlisted `dlopen` calls are outside
this fingerprint. The runtime remains an externally prepared dependency, not a
runtime image embedded in the policy package. Changes invalidate prior runtime
comparisons and require fresh reference/candidate measurements.

## Exact package verification

`policy.run` first copies and verifies the artifact into `package-pending`. A new
Python worker receives that package path and evaluates it with Hugging Face and
Transformers offline modes enabled. It reads model/tokenizer files from the copied
package. The caller checks the child-process identity, package path, input manifest
hash, model hash and fresh-reload flag. It then rechecks that the payload did not
change. Final deployment acceptance additionally requires finite bounded quality,
positive timing/memory, complete memory coverage and the paired final reference.

Only after acceptance are evidence files added and `package-pending` renamed to
`package`. Failure leaves no published `package` and retains staging diagnostics.
`tested-payload.json` records the exact tested file hashes and pre-export manifest
hash. The final manifest additionally inventories `reload-verification.json`,
`runtime-lock.json`, workflow evidence and lineage. Existing payload files with
those reserved names are rejected, so evidence cannot silently replace a tested
input. The report retains the staging path where execution occurred; the final
package has identical tested payload bytes at its returned path.

## Local verification

Run inside `workers/vla_cpp` using an isolated Python 3.11 environment with only
`quantize` and `test` extras (no Torch/GPU dependencies):

```sh
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -rs -p no:cacheprovider
ruff check policykit/acceptance.py policykit/provenance.py policykit/application.py \
  tests/test_application_acceptance.py tests/test_application.py
ruff format --check policykit/acceptance.py policykit/provenance.py policykit/application.py \
  tests/test_application_acceptance.py tests/test_application.py
```

Tests use small GGUF tensors, controlled native log/resource fixtures and a fresh
synthetic worker process. Vendor-source and Linux native integration checks need
their prepared environments. Report unavailable checks as skipped; contract tests
do not establish full-model reload, simulator success or GPU performance.

## Spatial integration boundary (EVAL-003)

The native worker retains the legacy LIBERO Object adapter. The separate
[Spatial application integration](spatial-application.md) adds the newer model
without treating its 112 matrices as sufficient compatibility evidence. It packages the pinned processor, tokenizer,
normalization statistics and derived observation configuration alongside weights;
record the 6-to-8 observation-state correction as explicit lineage; and validate
camera names, state/action layouts, action chunk replay and task instructions.

The Spatial adapter uses an explicit protocol identity: suite/task IDs,
asset revision and task/state hashes, episode horizon, action replay length,
observation preprocessing, seed/noise schedule, and separate development/final
initial states. That protocol is the comparison key between native PyTorch BF16
and CPP float/Q8/experimental Q4. Their runtime identities must remain distinct.
For repeats of the same backend/package, require the full runtime identity to
match. It reuses `artifact_contract`, complete `memory_coverage`, runtime source
fingerprints and `evaluate_package`, with process-tree GPU measurements. Actual L4
packed quality and final package reruns remain required gates.
