# Import and label a local dataset

Upload a robot dataset from your computer, convert it to LeRobot v3, then name its camera views, label frames and export it as a ZIP. You can also inspect it for training.

## What you need

- The isolated [CPU reader](../workers/_cpu_readers/README.md) on the application host, installed from `workers/_cpu_readers/requirements.txt`. Alternatively, install **Dataset tools** under **Runtime packages** in **Settings → Modules**.
- `JENSEN_CPU_READER_PYTHON` set to the reader's interpreter, unless you use **Dataset tools** or the checkout's `workers/_cpu_readers/.venv`.
- FFmpeg on the application host.

## Supported layouts

Prepare a folder, ZIP or HDF5 file in one of these layouts:

| Format | Files to supply |
| --- | --- |
| LeRobot v2/v3 | `meta/info.json` and its declared data/media files. |
| robomimic HDF5 | `/data/demo_*/actions` and `/data/demo_*/obs`. |
| ALOHA HDF5 | `/action`, `/observations/qpos`, `/observations/images`. |
| Images and records | One of `frames.csv`, `frames.jsonl`, `frames.parquet`, `records.csv` or `records.jsonl`, plus the images it references. |
| RLDS / TFRecord or ROS | Export synchronized observations/actions into one of the layouts above first. |

Each upload must contain exactly one dataset in one format.

**Image-record columns.** Use `action` (or `actions`) for the action, `observation.state` (or `state`, `qpos`) for the state, and `observation.images.<view>` (or `image.<view>`) for each camera. Optional columns are `episode_index`, `frame_index` and `task`. A row can look like this:

```json
{"episode_index":0,"frame_index":0,"action":[0.1,0.2],"observation.state":[0.2,0.3],"observation.images.front":"images/front/000.png","task":"Pick up the block"}
```

- Use contiguous frame indices within each episode.
- State and action vectors hold 1–256 finite numbers. In CSV, write vector cells as quoted JSON arrays.
- Images are RGB, at most 3840×2160.

## Limits

| Limit | Maximum |
| --- | --- |
| Upload size | 2 GiB |
| Upload files | 4096 |
| Converted frames | 100,000 |
| Episodes | 2000 |
| Synchronized RGB camera views | 8 |

An upload is refused if it would leave less than 1 GiB free on the application host.

## Import

1. Open **Datasets** and select **Local files** under **Import a dataset**.
2. Select **Choose folder** or **ZIP / HDF5 file** on the computer running your browser.
3. Review the detected format. Set **Frame rate (FPS)** (1–240) and **Task description**, and adjust **Field mapping & robot** if needed.
4. Select **Convert to LeRobot**. LeRobot v3 uploads are ready without conversion.
5. When the dataset shows **Ready to inspect**, select **Inspect dataset**.

Files are stored on the application host in the workspace's `dataset-library/` folder. Converted data is written as LeRobot v3 Parquet, MP4 and metadata. Uploading identical files to the same project reuses the saved dataset and its labels.

If an upload stops partway, select **Refresh saved upload**, or **Discard unfinished upload** to remove it from the app host. Your original files stay unchanged.

## Label and export

1. Open the dataset. Under **Camera views**, give each camera a name.
2. Step through samples with **Previous image** and **Next image**, and type a note in **Label this image**.
3. Select **Save labels**. Unsaved labels are kept only in the current tab.
4. Select **Export dataset**. The ZIP contains the LeRobot files, `openjensen/annotations.json` and `openjensen/source.json` with the source and conversion details.

To train on it, select **Inspect for training**, which selects the dataset under **Import a dataset**. Tick **Prepare immutable training copy**, select **Inspect dataset**, and continue with [local training](local-training.md#2-train).

References: [LeRobot v3](https://huggingface.co/docs/lerobot/main/lerobot-dataset-v3), [robomimic structure](https://robomimic.github.io/docs/datasets/overview.html#dataset-structure), [ALOHA/ACT](https://github.com/tonyzhaozh/act), [RLDS](https://github.com/google-research/rlds).
