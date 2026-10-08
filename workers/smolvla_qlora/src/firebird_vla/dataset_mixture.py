"""Pinned dataset concatenation with canonical camera names and disjoint splits.

Source indices remain private to each decoder. Public indices, episode ranges
and sampler mappings refer to one combined set; normalization sees training only.
"""

import re
from bisect import bisect_right
from copy import deepcopy

from .config import split_episodes


def bind_sources(job, camera_keys):
    entries = job.get("datasets")
    if not entries:
        return None
    if not 2 <= len(entries) <= 8:
        raise ValueError("Select two to eight datasets")
    sources, identities = [], set()
    primary = job["dataset"]
    for entry in entries:
        data, mapping = entry["profile"], entry["camera_mapping"]
        if data.get("source") != "huggingface" or data.get("format") != "lerobot_v3":
            raise ValueError("Combined training currently requires pinned LeRobot v3 Hub datasets")
        if not re.fullmatch(r"[a-f0-9]{40}", data.get("revision", "")):
            raise ValueError("Every dataset needs an immutable revision")
        identity = (data["repo_id"], data["revision"])
        if identity in identities:
            raise ValueError("Duplicate dataset identity")
        identities.add(identity)
        if set(mapping) != set(camera_keys) or len(set(mapping.values())) != len(mapping):
            raise ValueError("Every source needs a distinct mapping for each selected camera")
        if data["fps"] != primary["fps"]:
            raise ValueError("Combined datasets need matching frame rates")
        for key in ("action", "observation.state"):
            feature, reference = data["features"][key], primary["features"][key]
            if (feature.get("shape"), feature.get("dtype"), feature.get("names")) != (
                reference.get("shape"),
                reference.get("dtype"),
                reference.get("names"),
            ):
                raise ValueError("Combined state/action dimensions and ordered names must match")
        for key, source in mapping.items():
            if data["features"][source]["shape"] != primary["features"][key]["shape"]:
                raise ValueError("Mapped camera dimensions must match")
        sources.append(
            {
                "repo_id": data["repo_id"],
                "revision": data["revision"],
                "camera_mapping": mapping,
                "features": data["features"],
                "fps": data["fps"],
            }
        )
    if (sources[0]["repo_id"], sources[0]["revision"]) != (primary["repo_id"], primary["revision"]):
        raise ValueError("Primary dataset identity differs from the combined recipe")
    return sources


class MixtureMetadata:
    def __init__(self, base, features, episodes, stats, frames):
        self.base = base
        self.features, self.episodes, self.stats = features, episodes, stats
        self.camera_keys = [
            key for key, value in features.items() if value.get("dtype") in {"video", "image"}
        ]
        self.info = {
            **base.info,
            "features": features,
            "total_episodes": len(episodes),
            "total_frames": frames,
        }
        self.total_episodes, self.total_frames = len(episodes), frames

    def __getattr__(self, name):
        return getattr(self.base, name)


class MixtureDataset:
    def __init__(self, components, sources, offsets, metadata):
        import datasets

        self.components, self.sources, self.offsets = components, sources, offsets
        self.meta = metadata
        self.features = metadata.features
        self.ends, self.episodes, tables = [], [], []
        length = 0
        common = set(components[0].hf_dataset.column_names)
        for component in components:
            common &= set(component.hf_dataset.column_names)
        # Decoding remains in the source dataset, never in this concatenated table.
        common = sorted(key for key in common if not key.startswith("observation.images."))
        for component, source, (episode_offset, frame_offset, task_offset) in zip(
            components, sources, offsets, strict=True
        ):
            length += len(component)
            self.ends.append(length)
            self.episodes.extend(int(index) + episode_offset for index in component.episodes)
            table = component.hf_dataset.select_columns(common).with_format(None)
            shifts = {
                "episode_index": episode_offset,
                "index": frame_offset,
                "task_index": task_offset,
            }
            table = table.map(
                lambda row, shifts=shifts: {
                    key: int(row[key]) + shift for key, shift in shifts.items() if key in row
                },
                load_from_cache_file=False,
            )
            tables.append(table)
        self.hf_dataset = datasets.concatenate_datasets(tables)
        self.absolute_to_relative_idx = {
            int(index): row for row, index in enumerate(self.hf_dataset["index"])
        }
        self.delta_indices = components[0].delta_indices
        self.delta_timestamps = components[0].delta_timestamps
        self.num_frames, self.num_episodes = length, len(self.episodes)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        part = bisect_right(self.ends, index)
        local_index = index - (self.ends[part - 1] if part else 0)
        item = dict(self.components[part][local_index])
        mapping = self.sources[part]["camera_mapping"]
        output = {
            key: value for key, value in item.items() if not key.startswith("observation.images.")
        }
        for canonical, source in mapping.items():
            for suffix in ("", "_is_pad", "_padding_mask"):
                if source + suffix in item:
                    output[canonical + suffix] = item[source + suffix]
        for key, offset in zip(
            ("episode_index", "index", "task_index"), self.offsets[part], strict=True
        ):
            if key in output:
                output[key] = output[key] + offset
        return output


def load_mixture(sources, camera_keys, fraction, seed, make_component):
    import datasets
    import numpy as np
    from huggingface_hub import snapshot_download
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    from .data import vector_stats

    train, validation, offsets, episode_rows = [], [], [], []
    episode_offset = frame_offset = task_offset = 0
    first = None
    for index, source in enumerate(sources):
        root = snapshot_download(
            source["repo_id"], repo_type="dataset", revision=source["revision"]
        )
        meta = LeRobotDatasetMetadata(source["repo_id"], root=root, revision=source["revision"])
        if meta.features != source["features"] or meta.fps != source["fps"]:
            raise ValueError("Downloaded source differs from its inspected immutable metadata")
        first = first or meta
        ids = [int(row["episode_index"]) for row in meta.episodes]
        if ids != list(range(meta.total_episodes)):
            raise ValueError("Combined datasets need complete, contiguous episode metadata")
        split = split_episodes(ids, fraction, seed + index)
        offsets.append((episode_offset, frame_offset, task_offset))
        for row in meta.episodes:
            episode_rows.append(
                {
                    "episode_index": int(row["episode_index"]) + episode_offset,
                    "dataset_from_index": int(row["dataset_from_index"]) + frame_offset,
                    "dataset_to_index": int(row["dataset_to_index"]) + frame_offset,
                }
            )
        train.append(make_component(source, root, split["train"], False))
        validation.append(make_component(source, root, split["validation"], True))
        episode_offset += meta.total_episodes
        frame_offset += meta.total_frames
        task_offset += len(meta.tasks)
    features = {
        key: deepcopy(first.features[key]) for key in ("action", "observation.state", *camera_keys)
    }
    metadata = MixtureMetadata(
        first, features, datasets.Dataset.from_list(episode_rows), {}, frame_offset
    )
    training = MixtureDataset(train, sources, offsets, metadata)
    held_out = MixtureDataset(validation, sources, offsets, metadata)
    stats = {}
    for key in ("action", "observation.state"):
        rows = training.hf_dataset[key]
        stats[key] = vector_stats(rows, features[key]["shape"][0])
        # Native quantile-normalized policies also receive training-only quantiles.
        values = np.asarray(list(rows), dtype=np.float32)
        stats[key].update(
            q01=np.quantile(values, 0.01, axis=0).tolist(),
            q99=np.quantile(values, 0.99, axis=0).tolist(),
        )
    for key in camera_keys:
        stats[key] = {
            "mean": [[[value]] for value in (0.485, 0.456, 0.406)],
            "std": [[[value]] for value in (0.229, 0.224, 0.225)],
            "min": [[[0.0]]] * 3,
            "max": [[[1.0]]] * 3,
            "count": [len(training)],
        }
    metadata.stats = stats
    splits = {"train": training.episodes, "validation": held_out.episodes}
    return training, held_out, stats, splits


def native_mixture(cfg, recipe):
    import torch
    from lerobot.datasets.factory import make_dataset

    def component(source, root, episodes, evaluation):
        config = deepcopy(cfg)
        config.dataset.repo_id, config.dataset.revision, config.dataset.root = (
            source["repo_id"],
            source["revision"],
            root,
        )
        config.dataset.episodes = episodes
        if evaluation:
            config.dataset.image_transforms.enable = False
        return make_dataset(config)

    training, validation, stats, _ = load_mixture(
        recipe["dataset_sources"],
        recipe["camera_keys"],
        recipe["validation_fraction"],
        recipe["seed"],
        component,
    )
    training.meta.stats = {
        key: {name: torch.tensor(value) for name, value in values.items()}
        for key, values in stats.items()
    }
    return training, validation
