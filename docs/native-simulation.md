# Native ACT and SmolVLA simulation in the web app

Under **Run**, choose the **3D simulation** card. **Replay observations** and **Check inference** open separate workflows; the initial selection follows the project's saved work and available workers. Engine checks test loading and execution; configured LIBERO evaluation can measure benchmark task success. Isaac Run is an experimental rollout in the registered SO-101 **cup** scene. It does not provide a scored pickup benchmark or verified calibration.

The application operator must register an Isaac profile. The browser receives its public ID and supported architectures, never interpreter paths, commands or credentials. Opening the page only reads configuration and recorded jobs; it does not provision workers.

1. Choose a configured scene card, then select a compatible saved local native policy or choose **Import package** to import a complete TAR archive. ACT and SmolVLA packages need safetensors weights, configuration, saved processors and normalization statistics. The server validates the package in a local import job; a weights-only file is insufficient. Empty archives and archives over 4 GiB are refused. Upload has a 300-second server deadline and a 330-second client receipt deadline.
2. Import completion does not start simulation. Select the newly saved artifact explicitly. Full training checkpoints and remote descriptors are excluded; export or materialize them through the supported training flow first.
3. Select a timeout from 30 to 7200 seconds and acknowledge the experimental paid cloud rollout. **Start experimental simulation** sends one request with the registered profile and exact artifact ID. The registered multi-worker target is recorded on the job. This local supervision timeout is not a cloud spending cap.
4. Follow recorded status and the latest 100 events. Task-status events include the runner’s recognized Isaac and VLA states; arbitrary extra fields are not displayed as worker statuses. Cancellation requires confirmation for the selected job and rechecks its identity/status before sending a request. Changing selection dismisses that confirmation. Cancellation status does not prove resource deletion.
5. A successful verified simulation record exposes its download and the fixed application video endpoint when the report includes `artifacts/outputs/video.mp4`. Failed runs do not show completed-output links. Video-load failure leaves the verified record download available. Job completion is displayed separately from cup-pickup success, which remains unmeasured.

A lost, timed-out, malformed or mismatched mutation receipt is ambiguous. The client never automatically retries a POST. Further submissions in the current view pause until the user refreshes and checks recorded jobs, then explicitly allows another request. Leaving a pending upload aborts the browser transfer; check history because a server receipt may already have been created. Leaving a submitted rollout does not cancel it.

**Cloud runs** includes application jobs carrying either a persisted cloud compute target or a persisted native simulation target. Current runtime names and cloud-stored input files are not evidence that a job ran in the cloud. External observer snapshots remain a separate source with their own collection time and stale state.

Browser tests use explicitly generated API fixtures to verify transport, exact request payloads, ambiguity, cancellation, output selection and project isolation. They do not establish GPU execution, robot calibration, pickup success or a working live cloud scene. Server/worker checks and any real execution receipts must be reported separately.

## Operator and API integration

Set `FIREBIRD_SIMULATION_CONFIG` to a private profile file before starting the application. See [runner configuration and ownership](../workers/skypilot/APP_RUNNER.md). Keep credentials, weights, runtime environments and private profiles outside Git. This setup is separate from the single-GPU training target in Settings and reuses the explicitly configured runner.

Interrupted jobs are marked interrupted. Recovery requests cancellation only for a saved, identity-matched job; it does not adopt late results or resubmit. A changed profile or missing receipt produces an explicit cleanup warning. Cancellation acknowledgement is not proof that cloud VMs were deleted.

API clients use `GET /api/v1/simulation-options`, `POST /api/v1/projects/{id}/model-imports?profile_id=...` with the raw TAR body, and the existing policy-job endpoint with `operation: policy.run`, the registered artifact, and `simulation: {profile_id, experimental: true}`. Upload/job identities remain project-bound; the browser cannot supply runner commands, credential paths or cloud YAML.

The original dataset instruction mentions a cube, while the observed object and scene are a cup. Imported policy language is preserved as provenance. Scored Isaac evaluation remains unavailable until the cup task outcome and calibration are verified.
