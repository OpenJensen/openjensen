# Dataset import and labeling

## Reader setup

Install the isolated [CPU reader](../workers/_cpu_readers/README.md) and `workers/_cpu_readers/requirements.txt`. Set `FIREBIRD_CPU_READER_PYTHON` to its interpreter and install FFmpeg on the API host.

## Inputs

Prepare a folder, ZIP or HDF5 file using one of these layouts:

| Format | Files to supply |
| --- | --- |
| LeRobot v2/v3 | `meta/info.json` and its declared data/media files. |
| robomimic HDF5 | `/data/demo_*/actions` and `/data/demo_*/obs`. |
| ALOHA HDF5 | `/action`, `/observations/qpos`, `/observations/images`. |
| Images and records | `frames.csv`, `frames.jsonl`, `frames.parquet`, `records.csv` or `records.jsonl`, plus referenced images. |
| RLDS / TFRecord or ROS | Export synchronized observations/actions into one of the layouts above. |

Use contiguous frame indices and finite numeric state/action vectors. For CSV vector cells, use quoted JSON arrays. An image-record row can use:

```json
{"episode_index":0,"frame_index":0,"action":[0.1,0.2],"observation.state":[0.2,0.3],"observation.images.front":"images/front/000.png","task":"Pick up the block"}
```

Prepare uploads within 2 GiB and 4096 files; use up to 100,000 converted frames, 2000 episodes and eight synchronized RGB views.

## Import

1. Open **Dataset → Sources → Local files**.
2. Select the folder, ZIP or HDF5 file from the browser computer.
3. Review detected format, frame rate, task description and field mapping.
4. Start conversion and open the saved dataset after it finishes.

Files are stored on the API host. Converted data is written as LeRobot v3 Parquet, MP4 and metadata.

## Labels and export

Open the dataset, rename camera views and label the displayed samples. Save changes, then export its ZIP. The archive contains LeRobot files, `openjensen/annotations.json` and source/conversion details. Use **Create example** to try the steps with generated data.

References: [LeRobot v3](https://huggingface.co/docs/lerobot/main/lerobot-dataset-v3), [robomimic structure](https://robomimic.github.io/docs/datasets/overview.html#dataset-structure), [ALOHA/ACT](https://github.com/tonyzhaozh/act), [RLDS](https://github.com/google-research/rlds).
