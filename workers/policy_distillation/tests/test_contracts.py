import copy
from pathlib import Path

import pytest
from conftest import rebind_manifest
from firebird_act.bundle import canonical
from firebird_distill.contracts import corpus, decode, digest, load_sample, request, teacher_info


@pytest.mark.parametrize(
    "payload", [b'{"x":1e309}', b'{"x":NaN}', b'{"x":1,"x":2}', '{"x":1}'.encode("utf-16"), b"[]"]
)
def test_strict_json(payload):
    with pytest.raises((ValueError, UnicodeError)):
        decode(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("steps", True),
        ("seed", 1.0),
        ("steps", 0),
        ("learning_rate", True),
        ("learning_rate", float("inf")),
        ("student", "smolvla"),
        ("adapter", "arbitrary-code"),
    ],
)
def test_request_rejects_unsupported_controls(job, field, value):
    job["recipe"][field] = value
    with pytest.raises(ValueError):
        request(job)


def inspect(job):
    cfg, camera, processors = teacher_info(Path(job["teacher"]["path"]), job["teacher"]["files"])
    return corpus(
        Path(job["dataset"]["path"]), job["dataset"]["manifest_sha256"], cfg, camera, processors
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d["samples"][2].update(lineage_group="scene-0"),
        lambda d: d["samples"][2].update(episode_id=0),
        lambda d: d["samples"][4].update(split="train"),
        lambda d: d["samples"][0].update(frame_index=4),
        lambda d: d["samples"][0].update(bytes=True),
        lambda d: d["samples"][0].update(file="../weights"),
        lambda d: d["semantics"].update(teacher_processors_sha256="f" * 64),
        lambda d: d["semantics"].update(compatibility="simulator_radians"),
        lambda d: d.update(image_shape=[3, 64, 64]),
    ],
)
def test_corpus_rejects_leakage_or_semantic_changes(job, mutation):
    rebind_manifest(job, mutation)
    with pytest.raises(ValueError):
        inspect(job)


def test_corpus_native_three_partitions(job):
    doc = inspect(job)
    assert {s["split"] for s in doc["samples"]} == {"train", "validation", "final"}
    for sample in doc["samples"]:
        data = load_sample(Path(job["dataset"]["path"]), sample, doc["image_shape"])
        assert int((~data["padding"]).sum()) == 4 - sample["frame_index"]


@pytest.mark.parametrize("corrupt", ["padding", "nonfinite", "image", "unexpected"])
def test_rehashed_bad_tensors_rejected(job, corrupt):
    import torch
    from safetensors.torch import load, save

    root = Path(job["dataset"]["path"])
    doc = inspect(job)
    sample = doc["samples"][0]
    tensors = load((root / sample["file"]).read_bytes())
    if corrupt == "padding":
        tensors["padding"][:] = True
    elif corrupt == "nonfinite":
        tensors["state"][0] = float("nan")
    elif corrupt == "image":
        tensors["image"] = tensors["image"].float()
    else:
        tensors["surprise"] = torch.ones(1)
    raw = save(tensors)
    (root / sample["file"]).write_bytes(raw)
    sample.update(sha256=digest(raw), bytes=len(raw))
    with pytest.raises(ValueError):
        load_sample(root, sample, doc["image_shape"])


def test_changed_sample_rejected(job):
    doc = inspect(job)
    sample = doc["samples"][0]
    file = Path(job["dataset"]["path"]) / sample["file"]
    file.write_bytes(file.read_bytes() + b"x")
    with pytest.raises(ValueError, match="changed"):
        load_sample(file.parent, sample, doc["image_shape"])


def test_split_contract_persists_exactly(job):
    assert request(decode(canonical(job))) == request(job)
    changed = copy.deepcopy(job)
    changed["resume"] = True
    with pytest.raises(ValueError):
        request(changed)
