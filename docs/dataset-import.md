# Dataset import and labeling

The browser selects files from the computer running the browser, including when the app runs on a separate application host. A local folder is uploaded to the workspace. ZIP and single HDF5 files are supported. Limits are 2 GiB and 4096 input files, 100,000 converted frames, 2000 episodes and eight synchronized RGB views. Conversion uses one isolated CPU reader at a time. No GPU, cloud job or model download is started.

## Inputs

| Input | Detection | Conversion |
| --- | --- | --- |
| LeRobot v3 | `meta/info.json` and version/schema | Retained; complete validation runs when requesting a training snapshot |
| LeRobot v2 | `meta/info.json` and version/schema | Frame Parquet and declared image/video paths → LeRobot v3 |
| robomimic HDF5 | `/data/demo_*/actions` and `/data/demo_*/obs` | Reviewed numeric observations become state; RGB arrays become camera views |
| ALOHA HDF5 | `/action`, `/observations/qpos`, `/observations/images` | Joint state, actions and raw or JPEG camera arrays → LeRobot v3 |
| Images + records | `frames.csv`, `frames.jsonl`, `frames.parquet`, `records.csv` or `records.jsonl` | Numeric action/state arrays and referenced images → LeRobot v3 |
| RLDS / TFRecord | `dataset_info.json` plus TFRecord files | Recognized; synchronized export required |
| ROS bag / MCAP | `.bag` or `.mcap` files | Recognized; synchronized topic export required |

Ambiguous folders are rejected. Detection never executes Python, pickle, TensorFlow builders or scripts supplied by the dataset. External HDF5 links, symlinks, archive traversal and duplicate archive paths are rejected.

Image record tables use these columns:

```json
{"episode_index":0,"frame_index":0,"action":[0.1,0.2],"observation.state":[0.2,0.3],"observation.images.front":"images/front/000.png","task":"Pick up the block"}
```

CSV vector cells contain quoted JSON arrays. Frame indices must be contiguous within an episode. Actions and state remain finite float32 vectors; units and controller semantics are not inferred. The reviewed frame rate sets synchronized output timestamps. Missing state, actions or camera frames are not invented.

Conversion writes LeRobot v3 Parquet, MP4, episode metadata, task indices and normalization statistics. Original files and their SHA-256 inventory remain in the workspace. Identical uploads in one project reuse the saved dataset. Training snapshots apply the complete validator, including the requirement for two finalized episodes.

## Labels and export

The labeling page exposes beginning, middle and final samples for converted episodes. LeRobot v3 imports expose bounded camera previews where available. Camera display names and text labels use revision checks to prevent overwriting changes from another window. They remain separate from actions and training task instructions.

Save changes before exporting. The ZIP contains LeRobot files, `openjensen/annotations.json` and source/conversion details. It can be used in a LeRobot/Hugging Face dataset repository; publishing to the Hub is a separate action. Use the synthetic playground to try import, labeling and export with generated examples; its data does not establish robot task performance.

## Reader setup

Use the existing isolated CPU reader and install `workers/_cpu_readers/requirements.txt`. Set `FIREBIRD_CPU_READER_PYTHON` to its trusted Python path and install FFmpeg on the app host. The core imports no native ML or dataset libraries. Conversion processes have CPU, file and time limits; Linux caps virtual memory at 3 GiB. A parser subprocess is not a general sandbox for arbitrary code.

References: [LeRobot v3](https://huggingface.co/docs/lerobot/main/lerobot-dataset-v3), [robomimic structure](https://robomimic.github.io/docs/datasets/overview.html#dataset-structure), [ALOHA/ACT](https://github.com/tonyzhaozh/act), [RLDS](https://github.com/google-research/rlds).
