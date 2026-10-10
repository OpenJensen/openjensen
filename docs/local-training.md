# Train from a verified local dataset

In **Dataset → Sources → Local directory**, select **Prepare immutable training copy** before inspection. The application copies the entire finalized LeRobot v3 dataset into its workspace, checks every declared row and video, and records a content identity. A normal metadata inspection does not grant training eligibility.

Set `FIREBIRD_LOCAL_DATA_ROOT` to the permitted source directory, install the [isolated CPU reader](../workers/_cpu_readers/README.md), and put FFmpeg/ffprobe on the application host's path. Secure snapshots currently support Linux and macOS; Windows metadata inspection remains available but snapshot training is not yet supported. The source must be finalized, contain no symlinks or unfinished files, and fit the current limits: 16 GiB total, 4 GiB per file, 4,096 files, two million rows and 20,000 episodes. Validation is bounded to 120 seconds; exceeding a limit fails instead of granting partial eligibility.

A completed copy enables **Train on this dataset** for native LeRobot adapters. The job receives its own verified copy; cloud staging uploads only that selected snapshot. The native loader rechecks hashes and disables Hub fallback for local data. Training weights still require their configured model source and compute environment. Dataset validation does not establish policy quality.

Optional `meta/firebird-lineage.json` records `schema_version: 1` and a complete `episodes` list with `episode_index`, `origin` (`recorded`, `imported`, `augmented`, or `synthetic`), and `lineage_group`. The train/validation split keeps each lineage group together. At least two groups are required. Missing lineage is accepted with an ancestry-unknown warning; schema validation does not independently prove ancestry or make the validation split suitable for final benchmark claims.

Changing source files requires a new inspection/copy. Existing jobs retain the original content identity. Generated or augmented demonstrations must have valid action labels and related episodes must retain their shared lineage. Repeated copies do not create independent evaluation evidence.

# Teach in simulation

**Teaching**, directly under the sidebar's **Data** group, relays explicit task, recording, pause/reset, named-joint correction, failure annotation and finish commands to an authenticated local executor. Each command binds to the exact executor session and state revision. A correction is acknowledged only after physics advances and the applied-action row is written. A timeout means execution is unverified; inspect the executor before retrying.

Install and run the separate [teaching worker](../workers/teaching/README.md) using the existing Isaac runtime. Set these application-host variables before starting OPEN JENSEN:

```sh
export FIREBIRD_TEACHING_URL=http://127.0.0.1:8768
export FIREBIRD_TEACHING_VOICE_URL=http://127.0.0.1:8769
export FIREBIRD_TEACHING_CONTROL_TOKEN_FILE=/operator/private/teaching-token
```

The token file must be private and shared with the executor/broker; it is never returned to the browser. A remote simulator requires an operator-configured loopback tunnel. Existing rollout monitoring does not provide teaching control. When the executor is absent, the UI reports that state and disables controls.

Voice uses an isolated LiveKit room, OpenRouter for the language model, and configured speech services. Existing LiveKit connection credentials alone are insufficient: the voice worker also requires `OPENROUTER_API_KEY` and an explicitly chosen `FIREBIRD_VOICE_MODEL`. LiveKit Cloud Inference speech requires account entitlement; compatible local speech endpoints are an explicit alternative. Provider secrets stay in operator files outside Git. The browser enables its microphone only after the user clicks **Connect voice**, disconnects on executor-session changes, and limits each connection to five minutes.

Finalized captures are converted by the pinned LeRobot writer, verified by its reader, and imported through the immutable-copy flow above. This produces demonstration data for behavioral cloning. It does not perform model distillation, prove pickup success, or establish physical robot calibration. The Unity/TeleOP prototype rooms are separate because their packet receiver does not enforce the typed control protocol.
