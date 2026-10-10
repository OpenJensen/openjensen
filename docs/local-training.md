# Train from a verified local dataset

1. Install the [CPU reader](../workers/_cpu_readers/README.md) and FFmpeg/ffprobe on the API host.
2. Put a finalized LeRobot v3 dataset under `FIREBIRD_LOCAL_DATA_ROOT`. Use Linux or macOS for snapshot preparation.
3. Open **Dataset → Sources → Local files**, select **Prepare immutable training copy**, and inspect the dataset.
4. Open the completed inspection and select **Train on this dataset**.
5. Choose the native LeRobot model, cameras and runtime, review the recipe and start.

Prepare source files without symlinks or unfinished captures. Keep the dataset within 16 GiB total, 4 GiB per file, 4,096 files, two million rows and 20,000 episodes.

To group related episodes, add `meta/firebird-lineage.json` with `schema_version: 1` and an `episodes` list containing `episode_index`, `origin` (`recorded`, `imported`, `augmented`, or `synthetic`) and `lineage_group`. Supply at least two distinct groups for the training/validation split. Inspect again after changing source files.

# Teach in simulation

Install and start the [Teaching worker](../workers/teaching/README.md) with the Isaac runtime. Set the application-host connection variables:

```sh
export FIREBIRD_TEACHING_URL=http://127.0.0.1:8768
export FIREBIRD_TEACHING_VOICE_URL=http://127.0.0.1:8769
export FIREBIRD_TEACHING_CONTROL_TOKEN_FILE=/operator/private/teaching-token
```

Use a private token file shared with the executor/broker. For a remote executor, configure a loopback tunnel to the API host.

1. Open **Teaching** after starting the configured executor.
2. Wait for its session and camera frame, enter an instruction and start recording.
3. Pause, correct named joints or mark failures as needed.
4. Finish the episode, then [prepare selected captures](recording-preparation.md).
5. Open the saved dataset and choose **Train on this dataset**.

For voice, configure LiveKit, `OPENROUTER_API_KEY`, `FIREBIRD_VOICE_MODEL` and speech services, then select **Connect voice**. Follow [optional service setup](teaching-intelligence.md).
