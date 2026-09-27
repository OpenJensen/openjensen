# Native ACT and SmolVLA simulation

**Run → Native Isaac** imports complete native ACT or SmolVLA policies and runs a registered SO-101 cup scene through the existing SkyPilot launcher. **Engine checks** remains the separate optimized-policy inference path. These workflows use the same saved application jobs, cancellation controls and event history.

1. Select a project and the operator-configured cup profile.
2. Upload a complete TAR package containing configuration, safetensors weights, saved processors and their normalization statistics. Wrapper folders are supported. Import is a local job; it does not allocate a GPU. The upload limit is 4 GiB with a five-minute upload deadline.
3. Select the imported policy, or an existing complete native export. A training checkpoint must first be exported. The app checks package integrity and supported metadata; import alone does not prove that arbitrary weights load or solve the task.
4. Acknowledge the experimental calibration and submit the run explicitly. The current profile uses an L4 simulator and an H100 policy worker. The application never retries a paid submission automatically.
5. Follow the saved job in Run or Cloud runs. A verified completed run offers its video and a downloadable record containing the trajectory, completion records, exact model identity and file hashes. Job events contain lifecycle and task-state updates; they are not raw worker log streaming.

**Evaluation** measures a policy against an explicit protocol. The existing engine and LIBERO paths keep their original requirements. Scored Isaac evaluation is unavailable until the cup task outcome and calibration are verified. An experimental Run can establish execution and produce evidence; it does not establish cup pickup or placement success. The original dataset instruction mentions a cube, while the observed object and scene are a cup. Imported policy language is preserved as provenance.

An operator sets `FIREBIRD_SIMULATION_CONFIG` to a private profile file before starting the application. See [the runner configuration and ownership contract](../workers/skypilot/APP_RUNNER.md). Keep credentials, weights, runtime environments and private profiles outside Git. This setup is separate from the single-GPU training target in Settings; it reuses an explicitly configured runner without changing the training connection.

Interrupted jobs are marked interrupted. Recovery requests cancellation only for a saved, identity-matched job; it does not adopt late results or resubmit. A changed profile or missing receipt produces an explicit cleanup warning. Cancellation acknowledgement is not proof that cloud VMs were deleted, and a job deadline is not an exact billing cap.

API clients use `GET /api/v1/simulation-options`, `POST /api/v1/projects/{id}/model-imports?profile_id=...` with the raw TAR body, and the existing policy-job endpoint with `operation: policy.run`, the registered artifact, and `simulation: {profile_id, experimental: true}`. Local upload/job identities remain project-bound; the browser cannot supply runner commands, credential paths or cloud YAML.
