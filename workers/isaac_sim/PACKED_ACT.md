# Native packed ACT on the existing policy server

The local policy server accepts a complete OPEN JENSEN native ACT packed policy,
with the exact identity produced by the quantization worker. It uses the existing
`/health`, `/reset` and `/predict` protocol and saved pre/postprocessors. Ordinary
FP32 ACT and SmolVLA exports retain their existing loader and identity.

This first consumer is **CPU only**. It does not enable an existing CUDA cloud
profile, qualify packed SmolVLA, or prove simulator success, calibration, GPU
memory reduction or speed. Computation is FP32 eager PyTorch with packed weight
storage and dequantization on access. A smaller saved file is not a latency claim.

## Complete package and identity

Use the published `native-quantized` envelope, or its complete inner `policy`
directory. It contains `config.json`, `encoding.json`, `model.fbq`, both saved
processors, and every referenced statistics file. A floating master must not be
present. The existing checkpoint resolver preserves all envelope files and
verifies its manifest; it never promotes imported quality claims.

Structural inspection has no Torch dependency. Packed metadata uses the existing
`firebird-native-packed-policy-v1` domain and must exactly match the producer's
`model_id`. Unsupported/missing encoding, extra weights, links, corrupt containers,
changed inventories and ambiguous models are rejected. Structural admission is
not an actual model load: the runtime additionally verifies every stored tensor,
its architecture shape/dtype, encoding recipe/precision and saved processors.

The consumer copies the admitted bytes to a private temporary directory, checks
both copies before and after loading, and constructs the trusted ACT architecture
without downloading a backbone or reading `model.safetensors`. It preserves the
saved normalization statistics. The private copy is removed after the model and
processors have loaded. Live requests never reopen the source package.

## Local invocation

Reuse an isolated, already-provisioned ACT CPU environment with LeRobot **0.6.1**,
Torch **2.11.0**, torchvision **0.26.0**, and safetensors **0.8.0**. Local build
suffixes such as `+cpu` are accepted. Unsupported versions fail explicitly.
No dependency installation is part of serving.

From the repository root, expose the three isolated worker source roots:

```sh
export PYTHONPATH="$PWD/workers/isaac_sim:$PWD/workers/firebird_quant/src:$PWD/workers/act_optimizer/src"
# ACT_PYTHON and POLICY_DIRECTORY are operator-owned local paths.
"$ACT_PYTHON" -m sim_worker.rollout.checkpoint_package \
  --checkpoint "$POLICY_DIRECTORY" --inspect-only
"$ACT_PYTHON" -m sim_worker.rollout.server \
  --host 127.0.0.1 --port 8080 --backend lerobot --device cpu \
  --checkpoint "$POLICY_DIRECTORY" --model-id "$VERIFIED_MODEL_ID" \
  --state-dim 6 --action-steps 100 --camera-key observation.images.front
```

For a full outer envelope, first use the existing exclusive-output import CLI;
its receipt names the selected inner policy directory. `VERIFIED_MODEL_ID` must
come from inspection of those exact admitted bytes. Keep the default loopback
binding for local use; this server does not add an authentication layer.

The supported recipe inherits the existing native ACT exporter boundary:
ResNet18, no VAE/PEFT/AMP/temporal ensembling, one RGB camera, six state/action
coordinates and `1 <= execution <= prediction <= 1024`. Existing bounds on image size and architecture still
apply. A different model needs a separately verified adapter.

## Evidence and limits

The regression suite covers wrapped TAR admission, source/model identity,
strict encoding types, runtime pins, no-Torch metadata inspection, corrupt
weights, links, mutation, source snapshot races and early device/shape rejection.
Opt-in pinned-runtime tests convert generated ACT weights, execute real HTTP
full-chunk predictions, reset episodes and reject relabeled packed precision.

Separately, both actual saved user INT8/INT4 packages were loaded in fresh local
processes and served through HTTP on generated seeds171/902. Each100×6
postprocessed chunk exactly matched the frozen packed verification output, and
repeated reset produced the same output. The proof blocked external network and
floating-master reads and checked source hashes afterward. These generated
observations are not independent task-quality data. The prior measured drift
from FP32 remains in each package and is not replaced by a parity claim.

Application selection, local execution ownership and any simulator route must
be integrated and reviewed separately. No cloud image, credentials, room,
provider settings, original weights or physical robot were changed here.


Changed-horizon ACT software proof uses separate full-prediction and execution
prefix HTTP configurations. For an 8/3 checkpoint, `--action-steps 8` returns the
full prediction for inspection; `--action-steps 3` returns its execution prefix.
Observation replay deliberately retains all eight predictions and applies none.
The native policy queue is independently checked to recompute every three
`select_action` calls; HTTP chunk prediction itself does not consume that queue.
The source `temporal-contract.json` remains in the packed model identity. These
CPU checks do not establish a changed-horizon Isaac/GPU rollout.
