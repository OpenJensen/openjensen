# Train on a local dataset

Freeze a local LeRobot v3 dataset into a verified, immutable training copy, then fine-tune a native LeRobot model on it. The last section shows how to record such a dataset yourself in simulation.

## What you need

- A Linux or macOS application host. Training copies need one of these.
- The [CPU reader](../workers/_cpu_readers/README.md) and FFmpeg/ffprobe on the application host.
- A finished LeRobot v3 dataset, either:
  - in a folder under `JENSEN_LOCAL_DATA_ROOT`, or
  - uploaded through **Local files** and showing **Ready to inspect** (see [dataset import](dataset-import.md)).

## 1. Prepare a training copy

1. Open **Datasets** and select **Local files** under **Import a dataset**.
2. For a folder, expand **Folder on the app host** and enter its **Dataset directory**, a path inside `JENSEN_LOCAL_DATA_ROOT`.
3. Tick **Prepare immutable training copy** and select **Inspect dataset**.

The app copies and verifies the dataset into the workspace's `dataset-snapshots/` folder, so later edits to the source cannot change saved training runs. Your original files stay unchanged. Inspect again after changing source files.

From the CLI:

```sh
jensen inspect PROJECT_ID --path my-dataset --snapshot-for-training
```

## 2. Train

1. Open the completed inspection and select **Train on this dataset**.
2. On **Fine-tune**, choose a native LeRobot model (ACT included), the cameras and the runtime.
3. Review the recipe and start.

## Dataset rules

The copy accepts only standard LeRobot v3 files:

- `meta/info.json`, `meta/stats.json`, `meta/tasks.parquet`
- `meta/episodes/chunk-NNN/file-NNN.parquet` and `data/chunk-NNN/file-NNN.parquet`
- `videos/<camera>/chunk-NNN/file-NNN.mp4`
- `meta/firebird-lineage.json` and `meta/firebird-demonstrations.json` (optional)
- `README.md` and `.gitattributes`

Any other file is reported as unsupported or unfinished. Symlinks, special files and empty files are rejected, and the dataset must not change while it is copied. Finish recording before you inspect.

| Limit | Maximum |
| --- | --- |
| Total size | 16 GiB |
| Single file | 4 GiB |
| Files | 4,096 |
| Rows | 2,000,000 |
| Episodes | 20,000 |

Training needs at least two complete episodes.

## Group episodes by lineage

The training/validation split keeps related episodes together. To declare which episodes are related, add `meta/firebird-lineage.json`:

```json
{
  "schema_version": 1,
  "episodes": [
    {"episode_index": 0, "origin": "recorded", "lineage_group": "scene-a"},
    {"episode_index": 1, "origin": "recorded", "lineage_group": "scene-b"}
  ]
}
```

- List every episode exactly once, with `episode_index` from 0.
- `origin` is `recorded`, `imported`, `augmented` or `synthetic`.
- `lineage_group` is 1–160 letters, digits, `_`, `.`, `:` or `-`.
- Supply at least two distinct groups for the training/validation split.
- `meta/firebird-demonstrations.json` is accepted only together with this file.

Without the file, each episode counts as its own group, and the inspection warns that the dataset's ancestry is unknown.

## Teach in simulation

To record your own demonstrations in Isaac Sim, install and start the [Teaching worker](../workers/teaching/README.md) with the Isaac runtime. Then set the connection variables on the application host:

```sh
export JENSEN_TEACHING_URL=http://127.0.0.1:8768
export JENSEN_TEACHING_VOICE_URL=http://127.0.0.1:8769
export JENSEN_TEACHING_CONTROL_TOKEN_FILE=/operator/private/teaching-token
```

Use a private token file shared with the executor and broker. For a remote executor, configure a loopback tunnel to the application host. [Connect the Teaching executor](teaching-connections.md) has the full requirements.

1. Open **Teaching** after starting the configured executor.
2. Wait for its session and camera frame, enter an instruction and start recording.
3. Pause, correct named joints or mark failures as needed.
4. Finish the episode, then [prepare selected captures](recording-preparation.md).
5. Open the saved dataset and choose **Train on this dataset**.

For voice, configure LiveKit, `OPENROUTER_API_KEY`, `JENSEN_VOICE_MODEL` and speech services, then select **Connect voice**. Follow [voice and advice setup](teaching-intelligence.md).
