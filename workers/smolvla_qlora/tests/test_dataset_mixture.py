"""CPU-only combination proof: decoder aliases, sampler indices and held-out stats."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from firebird_vla.dataset_mixture import bind_sources, load_mixture


def test_combination_preserves_decoder_boundaries_and_excludes_validation(monkeypatch):
    import datasets
    import numpy as np

    # The component decoder is injected; exercise real dataset tables without
    # importing LeRobot, PyTorch or a CUDA stack in this CPU contract test.
    lerobot_dataset = ModuleType("lerobot.datasets.lerobot_dataset")
    monkeypatch.setitem(sys.modules, "lerobot", ModuleType("lerobot"))
    monkeypatch.setitem(sys.modules, "lerobot.datasets", ModuleType("lerobot.datasets"))
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", lerobot_dataset)

    features = {
        "action": {"dtype": "float32", "shape": [2]},
        "observation.state": {"dtype": "float32", "shape": [2]},
        "observation.images.front": {"dtype": "video", "shape": [3, 32, 32]},
    }
    sources = [
        {
            "repo_id": f"fixture/{part}",
            "revision": str(part + 1) * 40,
            "features": {**features},
            "fps": 10,
            "camera_mapping": {
                "observation.images.front": "observation.images.front"
                if part == 0
                else "observation.images.wrist"
            },
        }
        for part in range(2)
    ]
    sources[1]["features"] = {
        key.replace("front", "wrist"): value for key, value in features.items()
    }
    calls = []

    def download(repo, *, repo_type, revision):
        calls.append((repo, revision))
        return repo

    def metadata(repo, root, revision):
        part = int(repo.split("/")[-1])
        return SimpleNamespace(
            features=sources[part]["features"],
            fps=10,
            info={"fps": 10},
            total_episodes=4,
            total_frames=8,
            tasks=["Pick"],
            has_language_columns=False,
            episodes=datasets.Dataset.from_list(
                [
                    {
                        "episode_index": episode,
                        "dataset_from_index": episode * 2,
                        "dataset_to_index": episode * 2 + 2,
                    }
                    for episode in range(4)
                ]
            ),
        )

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    monkeypatch.setattr(lerobot_dataset, "LeRobotDatasetMetadata", metadata, raising=False)
    decoded = []

    def component(source, root, episodes, evaluation):
        part = int(source["repo_id"].split("/")[-1])
        rows = [
            {
                "episode_index": episode,
                "index": episode * 2 + frame,
                "task_index": 0,
                "action": [part * 100 + episode, frame],
                "observation.state": [episode, frame],
            }
            for episode in episodes
            for frame in range(2)
        ]

        class Decoder:
            hf_dataset = datasets.Dataset.from_list(rows)
            delta_indices = {"action": [0, 1]}
            delta_timestamps = {"action": [0.0, 0.1]}

            def __len__(self):
                return len(rows)

            def __getitem__(self, index):
                decoded.append((part, index))
                camera = source["camera_mapping"]["observation.images.front"]
                return {
                    **{key: np.asarray(value) for key, value in rows[index].items()},
                    camera: np.zeros((3, 32, 32)) + part,
                    "task": "Pick",
                }

        value = Decoder()
        value.episodes = episodes
        return value

    train, validation, stats, splits = load_mixture(
        sources, ["observation.images.front"], 0.25, 42, component
    )
    assert calls == [(source["repo_id"], source["revision"]) for source in sources]
    assert not set(splits["train"]) & set(splits["validation"])
    assert len(train) == 12 and len(validation) == 4
    assert len(train.absolute_to_relative_idx) == len(train)
    assert train.meta.episodes["dataset_from_index"] == list(range(0, 16, 2))
    boundary = len(train.components[0])
    sample = train[boundary]
    assert "observation.images.front" in sample and "observation.images.wrist" not in sample
    assert sample["episode_index"].item() >= 4 and sample["index"].item() >= 8
    assert decoded == [(1, 0)]  # Indexing the combined set never seeks another source decoder.
    rows = list(train.hf_dataset["action"])
    assert stats["action"]["count"] == [len(rows)]
    assert stats["action"]["mean"][0] == pytest.approx(sum(row[0] for row in rows) / len(rows))
    assert all(
        index not in train.absolute_to_relative_idx for index in validation.hf_dataset["index"]
    )


def test_binding_refuses_unpinned_sources_and_ambiguous_camera_mapping():
    profile = {
        "source": "huggingface",
        "format": "lerobot_v3",
        "repo_id": "a/b",
        "revision": "a" * 40,
        "fps": 30,
        "features": {},
    }
    with pytest.raises(ValueError, match="immutable"):
        bind_sources(
            {
                "dataset": profile,
                "datasets": [
                    {"profile": {**profile, "revision": "main"}, "camera_mapping": {}},
                    {"profile": profile, "camera_mapping": {}},
                ],
            },
            [],
        )
