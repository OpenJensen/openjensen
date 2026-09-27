"""Private adapter admission tests; generated captures, no native writer execution."""

import hashlib
import json
import sys

import pytest
from conftest import record_fixture

from firebird_teaching import prepare_dataset as prepare
from firebird_teaching.dataset import _inventory


def request(tmp_path):
    root = tmp_path / "capture"
    episodes = record_fixture(root, 1)
    meta = json.loads((root / "session.json").read_bytes())

    def sha(node):
        return hashlib.sha256(node.read_bytes()).hexdigest()

    return {
        "schema_version": 1,
        "job_id": "job-1",
        "configuration_sha256": "a" * 64,
        "selection_sha256": "b" * 64,
        "output": str(tmp_path / "dataset"),
        "ffmpeg": "/usr/bin/ffmpeg",
        "ffprobe": "/usr/bin/ffprobe",
        "captures": [
            {
                "path": str(root),
                "session_id": meta["session_id"],
                "session_sha256": sha(root / "session.json"),
                "episodes": [
                    {"episode_id": e, "receipt_sha256": sha(root / e / "episode.json")}
                    for e in episodes
                ],
            }
        ],
    }


def test_private_request_preserves_exact_selected_bytes(tmp_path):
    value = request(tmp_path)
    selections, output = prepare.admit(value)
    assert selections[0].root.as_posix() == value["captures"][0]["path"]
    assert selections[0].episodes == tuple(
        e["episode_id"] for e in value["captures"][0]["episodes"]
    )
    assert not output.exists()


@pytest.mark.parametrize(
    "damage", ["schema", "extra", "identity", "metadata", "receipt", "duplicate", "path", "output"]
)
def test_private_admission_rejects_malformed_or_stale_source(tmp_path, damage):
    value = request(tmp_path)
    if damage == "schema":
        value["schema_version"] = True
    elif damage == "extra":
        value["download"] = True
    elif damage == "identity":
        value["captures"][0]["session_id"] = "f" * 32
    elif damage == "metadata":
        value["captures"][0]["session_sha256"] = "f" * 64
    elif damage == "receipt":
        value["captures"][0]["episodes"][0]["receipt_sha256"] = "f" * 64
    elif damage == "duplicate":
        value["captures"] *= 2
    elif damage == "path":
        value["captures"][0]["path"] += "/../capture"
    else:
        value["output"] = value["captures"][0]["path"]
    with pytest.raises(ValueError):
        prepare.admit(value)


@pytest.mark.parametrize("which", ["source", "output_parent"])
def test_adapter_keeps_lexical_no_follow_paths(tmp_path, which):
    value = request(tmp_path)
    link = tmp_path / "linked"
    if which == "source":
        link.symlink_to(tmp_path / "capture", target_is_directory=True)
        value["captures"][0]["path"] = str(link)
    else:
        link.symlink_to(tmp_path, target_is_directory=True)
        value["output"] = str(link / "dataset")
    with pytest.raises((ValueError, OSError)):
        prepare.admit(value)


@pytest.mark.parametrize("destination", ["source", "output", "existing", "request"])
def test_result_refusal_happens_before_any_writer_or_source_change(
    tmp_path, monkeypatch, destination
):
    value = request(tmp_path)
    source = tmp_path / "capture"
    before = _inventory(source)
    input_path = tmp_path / "request.json"
    input_path.write_text(json.dumps(value))
    result = {
        "source": source / "result.json",
        "output": tmp_path / "dataset/result.json",
        "existing": tmp_path / "existing.json",
        "request": input_path,
    }[destination]
    if destination == "existing":
        result.write_text("preserve")
    monkeypatch.setattr(
        sys, "argv", ["prepare", "--request", str(input_path), "--result", str(result)]
    )
    monkeypatch.setattr(prepare, "prepare", lambda _: pytest.fail("Writer must not start"))
    with pytest.raises((ValueError, OSError)):
        prepare.main()
    assert _inventory(source) == before
    assert not (tmp_path / "dataset").exists()


def test_unavailable_native_preflight_never_publishes(tmp_path, monkeypatch):
    value = request(tmp_path)
    before = _inventory(tmp_path / "capture")

    def unavailable(_):
        raise ValueError("Explicit missing native runtime")

    monkeypatch.setattr(prepare, "preflight", unavailable)
    with pytest.raises(ValueError, match="missing native"):
        prepare.prepare(value)
    assert not (tmp_path / "dataset").exists()
    assert _inventory(tmp_path / "capture") == before
