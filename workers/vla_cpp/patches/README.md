# Build the SmolVLA packed loader

From `workers/vla_cpp`, use a clean vla.cpp v0.3.0 checkout at commit
`52439f7c6c362d7bee218b400b9080cc32d75cc3` under
`artifacts/docker/vendor/vla.cpp`. Prepare the Docker environment and the frozen
screen inputs at `artifacts/docker/runs/smolvla-screen-v1/`. Use a new
`artifacts/docker/unpatched-float-actions.log` destination for the capture below.

## Capture the floating input log

Configure and build the unpatched predictor first, then capture the floating
control using the same image size and thread settings as the packed benchmark:

```bash
docker compose run --rm --entrypoint cmake policykit -S artifacts/docker/vendor/vla.cpp -B artifacts/docker/vendor/vla.cpp/build -DCMAKE_BUILD_TYPE=Release -DVLA_BUILD_TESTS=ON
docker compose run --rm --entrypoint cmake policykit --build artifacts/docker/vendor/vla.cpp/build --target vla_predict_check -j 2
docker compose run --rm -e VLA_N_THREADS=4 -e OMP_NUM_THREADS=4 -e VLA_IMG_SIZE=512 --entrypoint artifacts/docker/vendor/vla.cpp/build/tests/vla_predict_check policykit artifacts/docker/runs/smolvla-screen-v1/models/smolvla-float.gguf > artifacts/docker/unpatched-float-actions.log 2>&1
```

## Rebuild and run

Apply the patch to that checkout, then rebuild and run against the unchanged
model inputs:

```bash
docker compose run --rm --entrypoint git policykit -C artifacts/docker/vendor/vla.cpp apply /workspace/policykit/patches/vla-cpp-smolvla-packed.patch
docker compose run --rm --entrypoint bash policykit -lc 'cmake --build artifacts/docker/vendor/vla.cpp/build --target vla-bench vla-server vla_predict_check -j 2'
docker compose run --rm --entrypoint artifacts/docker/tooling/bin/python policykit -m pytest -q
docker compose run --rm --entrypoint artifacts/docker/tooling/bin/python policykit -m policykit.packed_bench --reps 20 --run smolvla-packed-repeat
```

Use a new `--run` value for each packed benchmark. Keep the audited input models
and floating action log in the paths required by the benchmark manifest.
