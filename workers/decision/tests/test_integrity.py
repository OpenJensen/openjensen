import hashlib
import os
from pathlib import Path

import pytest

from firebird_decision import integrity
from firebird_decision.contracts import LICENSE, DecisionError


def test_license_checked_before_read(monkeypatch):
    monkeypatch.setattr(integrity, "read_file", lambda *a, **k: pytest.fail("read before opt-in"))
    with pytest.raises(DecisionError, match="accept-license"):
        integrity.verified_files(Path("/does-not-exist"), None)


def test_checked_bytes_only_and_tamper_rejected(tmp_path, monkeypatch):
    path = tmp_path / "model.py"
    raw = b"raise Exception('must never run')"
    path.write_bytes(raw)
    monkeypatch.setattr(
        integrity, "FILES", {"model.py": (len(raw), hashlib.sha256(raw).hexdigest())}
    )
    assert integrity.verified_files(tmp_path, LICENSE) == {"model.py": raw}
    path.write_bytes(b"x" * len(raw))
    with pytest.raises(DecisionError, match="checksum"):
        integrity.verified_files(tmp_path, LICENSE)
    path.write_bytes(b"oversized" * 100)
    with pytest.raises(DecisionError, match="bounded"):
        integrity.verified_files(tmp_path, LICENSE)


def test_final_and_ancestor_symlinks_rejected(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    (actual / "file").write_bytes(b"ok")
    (tmp_path / "link").symlink_to(actual, target_is_directory=True)
    (tmp_path / "leaf").symlink_to(actual / "file")
    for path in (tmp_path / "link/file", tmp_path / "leaf"):
        with pytest.raises(DecisionError, match="symlinks"):
            integrity.read_file(path, 2)


def test_fifo_rejected_without_wait(tmp_path):
    path = tmp_path / "fifo"
    os.mkfifo(path)
    with pytest.raises(DecisionError, match="regular"):
        integrity.read_file(path, 10)


def test_growth_during_read_is_bounded_and_rejected(tmp_path, monkeypatch):
    path = tmp_path / "file"
    path.write_bytes(b"before")
    read = os.read
    calls = []

    def growing_read(fd, count):
        calls.append(count)
        chunk = read(fd, count)
        if len(calls) == 1:
            with path.open("ab") as output:
                output.write(b"x" * 1000)
        return chunk

    monkeypatch.setattr(os, "read", growing_read)
    with pytest.raises(DecisionError, match="changed"):
        integrity.read_file(path, 6)
    assert calls == [6, 1]


def test_parent_traversal_and_directory_rejected(tmp_path):
    with pytest.raises(DecisionError, match="Parent"):
        integrity.read_file(tmp_path / "a/../b", 10)
    with pytest.raises(DecisionError, match="regular"):
        integrity.read_file(tmp_path, 10)


def test_failed_integrity_never_executes_source(monkeypatch):
    from firebird_decision import scorer

    def fail(*args):
        raise DecisionError("checksum failure", "model_integrity")

    monkeypatch.setattr(scorer, "verified_files", fail)
    monkeypatch.setattr(scorer, "_execute_sources", lambda *args: pytest.fail("executed source"))
    with pytest.raises(DecisionError, match="checksum"):
        scorer.DecisionScorer(Path("/not-read"), LICENSE)


def test_wrong_runtime_rejected_before_custom_source(monkeypatch):
    from firebird_decision import scorer

    monkeypatch.setattr(scorer, "verified_files", lambda *args: {})
    monkeypatch.setattr(scorer.importlib.metadata, "version", lambda *args: "0.0.0")
    monkeypatch.setattr(scorer, "_execute_sources", lambda *args: pytest.fail("executed source"))
    with pytest.raises(DecisionError, match="Unsupported torch"):
        scorer.DecisionScorer(Path("/not-read"), LICENSE)
