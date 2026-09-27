"""Native LeRobot v3 loading; only training episodes contribute normalization."""

import math

from .config import split_episodes
from .temporal import action_timestamps


def vector_stats(rows, dimensions):
    """Streaming population statistics; rows must already be filtered to training episodes."""
    count = 0
    mean, m2 = [0.0] * dimensions, [0.0] * dimensions
    minimum, maximum = [math.inf] * dimensions, [-math.inf] * dimensions
    for row in rows:
        if len(row) != dimensions or any(not math.isfinite(float(v)) for v in row):
            raise ValueError("Non-finite value or wrong dimension in state/action data")
        count += 1
        for i, value in enumerate(row):
            value = float(value)
            delta = value - mean[i]
            mean[i] += delta / count
            m2[i] += delta * (value - mean[i])
            minimum[i], maximum[i] = min(minimum[i], value), max(maximum[i], value)
    if not count:
        raise ValueError("Empty training split")
    return {
        "mean": mean,
        "std": [max(math.sqrt(max(v, 0) / count), 1e-6) for v in m2],
        "min": minimum,
        "max": maximum,
        "count": [count],
    }


def load_data(cfg, splits=None):
    from huggingface_hub import snapshot_download
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    # SHA-addressed snapshot avoids LeRobot's repo-id-only cache silently reusing another revision.
    root = snapshot_download(cfg.dataset_id, repo_type="dataset", revision=cfg.dataset_revision)
    meta = LeRobotDatasetMetadata(cfg.dataset_id, root=root, revision=cfg.dataset_revision)
    features = meta.features
    for key in ("action", "observation.state"):
        shape = features.get(key, {}).get("shape", [])
        if len(shape) != 1 or not 1 <= shape[0] <= 32:
            raise ValueError(f"{key} must be a vector with 1..32 dimensions")
    for camera_key in cfg.selected_camera_keys:
        if features.get(camera_key, {}).get("dtype") not in ("video", "image"):
            raise ValueError(f"Missing image/video camera {camera_key}")
    timestamps = action_timestamps(cfg.temporal["prediction_horizon"], meta.fps)
    ids = [int(ep["episode_index"]) for ep in meta.episodes]
    expected = split_episodes(ids, cfg.validation_fraction, cfg.seed)
    if splits is not None and splits != expected:
        raise ValueError("Checkpoint split differs from the pinned dataset/recipe")
    splits = expected
    kwargs = dict(
        repo_id=cfg.dataset_id,
        root=root,
        revision=cfg.dataset_revision,
        delta_timestamps={"action": timestamps},
        video_backend="pyav",
    )
    train = LeRobotDataset(**kwargs, episodes=splits["train"])
    validation = LeRobotDataset(**kwargs, episodes=splits["validation"])
    # Access tabular columns directly: no video decode, no temporal padding, no held-out stats.
    stats = {}
    for key in ("action", "observation.state"):
        columns = train.hf_dataset.select_columns([key])
        stats[key] = vector_stats((row[key] for row in columns), features[key]["shape"][0])
    return train, validation, stats, splits


def prepare_batch(batch, preprocessor, camera_keys=None):
    tasks = batch.get("task")
    if not tasks or any(not isinstance(t, str) or not t.strip() for t in tasks):
        raise ValueError("Each frame must have a non-empty dataset task instruction")
    if camera_keys is not None:
        missing = [key for key in camera_keys if key not in batch or batch[key] is None]
        if missing:
            raise ValueError(f"Missing selected cameras in batch: {', '.join(missing)}")
        # Avoid moving unused camera tensors to the GPU. The policy consumes selected views
        # in recipe order through its input_features, independently of dataset column order.
        selected = set(camera_keys) | {f"{key}_padding_mask" for key in camera_keys}
        batch = {
            key: value
            for key, value in batch.items()
            if not key.startswith("observation.images.") or key in selected
        }
    processed = preprocessor(batch)
    if camera_keys is not None:
        missing = [key for key in camera_keys if key not in processed or processed[key] is None]
        if missing:
            raise ValueError(f"Preprocessor dropped selected cameras: {', '.join(missing)}")
    # LeRobot 0.4.4's dataset emits action_is_pad; SmolVLA.forward reads actions_id_pad.
    if "action_is_pad" not in batch:
        raise ValueError("Missing action chunk padding mask")
    processed["actions_id_pad"] = batch["action_is_pad"].to("cuda")
    return processed
