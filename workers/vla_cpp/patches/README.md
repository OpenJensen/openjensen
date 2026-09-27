# SmolVLA packed-weight execution

The canonical patch is `../policykit/patches/vla-cpp-smolvla-packed.patch` and ships
in installed wheels. It applies to vla.cpp v0.3.0, commit
`52439f7c6c362d7bee218b400b9080cc32d75cc3`.

The original SmolVLA-specific reader accepts only F32/BF16 and allocates floating
matrices regardless of the GGUF tensor type. The patch allocates LM and vision
matrices in their Q8_0/Q4_0 source type and transfers packed bytes directly to the
backend. `ggml_mul_mat` consumes those packed weights. Action-expert, projector,
embedding, normalization and action/state tensors remain unquantized.

The GGUF token embedding is allocated using its actual F32, F16 or BF16 storage
type and transferred without recasting. The upstream reader hardcoded BF16,
which rejected a trained checkpoint exported to F32 with a raw-byte-count error.
This correction preserves the exporter's lossless floating materialization for
both FP16 and BF16 training runs. Native safetensors retain their existing BF16
loading path. Unsupported packed embedding types are rejected.

Packed input validation rejects other tensor types and protected components.
Loading checks packed tensor shape, type, and byte count before transfer. The
runtime logs packed LM/vision matrix counts and actual weight-buffer allocation.
The benchmark additionally reports model-loading time and rejects nonfinite
predictions. No quantized tensor is expanded to a permanent floating weight buffer.

Both benchmark manifests list this patch. `VlaCpp.ensure_prepared()` applies it
before building; a reverse-apply check makes repeated preparation idempotent.
Conflicting edits fail the apply check rather than being overwritten.

For the existing Docker checkout, rebuild only the required targets:

```bash
docker compose run --rm --entrypoint bash policykit -lc 'cmake --build artifacts/docker/vendor/vla.cpp/build --target vla-bench vla-server vla_predict_check -j 2'
docker compose run --rm --entrypoint artifacts/docker/tooling/bin/python policykit -m pytest -q
docker compose run --rm --entrypoint artifacts/docker/tooling/bin/python policykit -m policykit.packed_bench --reps 20 --run smolvla-packed-repeat
```

`vla_predict_check` requires configuring CMake with `-DVLA_BUILD_TESTS=ON`.
The benchmark consumes the existing audited v1 artifacts and the saved unpatched
fixed-input action log at `artifacts/docker/unpatched-float-actions.log`.
For a fresh checkout, capture that log with the unpatched `vla_predict_check`
before applying the patch, using `VLA_N_THREADS=4 VLA_IMG_SIZE=512` and the same
floating GGUF. The harness refuses changed artifact hashes and mismatched source
patches, records the executable hash, and verifies BF16 action regression.

Historical packed-weight CPU/CUDA measurements are recorded separately in the
worker's quantization reports. The newer embedding reader has compiled regression
coverage for F32/F16/BF16, and the application requires a fresh native forward pass
for each cloud quantization result. See the current
[application verification record](../../../output/verification/README.md) for
completed whole-model checks and their exact scope. Synthetic finite actions do
not establish robot task success.
