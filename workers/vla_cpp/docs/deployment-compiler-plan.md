# VLA deployment compiler plan

This research plan describes a proposed compiler. It is not a release capability
list or evidence that its automation has completed. Use the
[application workflow](../../../docs/policy-workflow.md) and
[native acceptance contract](native-acceptance.md) for current support and limits.

## Product contract

**Input:** a LeRobot-format dataset, a supported VLA checkpoint, target hardware, and deployment constraints.

**Output:** the smallest deployable policy which passes the requested task-success and latency thresholds, plus a reproducible evidence bundle.

```text
dataset + VLA + target hardware + constraints
  → fine-tune a FP16/BF16 master policy
  → generate hardware-aware compression candidates
  → evaluate every viable candidate in cloud simulation
  → select the Pareto-optimal passing artifact
  → deployable model + runtime config + report + rollout videos
```

The selection objective is:

```text
minimize artifact size and/or inference latency
subject to task success ≥ the requested threshold
```

## Required compiler search matrix

The first automatic search must generate and evaluate these candidates for each supported VLA:

| Candidate | Language backbone | Vision tower | Action expert / output head |
|---|---|---|---|
| baseline | BF16 | FP16/BF16 | FP16/BF16 |
| `lm_q8` | Q8_0 | FP16/BF16 | FP16/BF16 |
| `lm_q4` | Q4_0 | FP16/BF16 | FP16/BF16 |
| `lm_q8_vision_q8` | Q8_0 | Q8_0 | FP16/BF16 |
| `lm_q4_vision_q8` | Q4_0 | Q8_0 | FP16/BF16 |

Vision quantization is included because camera encoding can be a meaningful share of edge memory and latency. It is never silently enabled: each vision-packed candidate must have its own closed-loop simulation result. The action expert/output head remains floating-point in the first compiler version because it directly controls robot actions.

## Automation requirements

- Accept target profiles from `configs/targets.yaml`: local CPU, RTX 3070 CUDA, or GCloud CUDA worker.
- Fine-tune only the full-precision master checkpoint; never fine-tune an already quantized artifact.
- Run a five-episode smoke gate before the full fixed-seed simulation suite.
- Persist every candidate, command, source revision, hardware fingerprint, artifact hash, memory usage, latency distribution, and rollout video path.
- Mark failed, slow, or memory-incompatible candidates as rejected evidence—never drop them from the report.
- Render a report with the selected artifact, Pareto frontier, acceptance threshold, and rejected-candidate reasons.

## Delivery phases

1. **Candidate generation now:** BF16, LM-Q8_0, LM-Q4_0, LM-Q8_0 plus vision-Q8_0, and LM-Q4_0 plus vision-Q8_0 through vla.cpp on the local/Docker CPU lane.
2. **Comparable evidence:** each candidate uses the identical simulator matrix and immutable source/checkpoint provenance.
3. **Hardware compiler:** execute the same manifest on RTX 3070 and GCloud CUDA lanes; select per target rather than sharing a universal winner.
4. **Fine-tune → compile service:** expose dataset, VLA, hardware profile, success threshold, and latency budget as one job submission; return the selected artifact and evidence bundle.

QAT, AWQ, GPTQ, and K-Quants remain future candidate generators. They enter the compiler only after they can be measured against this same closed-loop task-success contract.

The concrete experiment order and product admission gate are maintained in the [quantization hypothesis register](quantization-hypotheses.md). No technique is marketed as supported until it passes that gate.
