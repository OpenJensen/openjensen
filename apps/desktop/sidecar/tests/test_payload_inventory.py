"""Complete packaging identity is distinct from production activation."""

import os
import shutil
import struct

import pytest
from test_sidecar import module

payload = module("payload_inventory")


def tree(root):
    (root / "_internal/Python.framework/Versions/A").mkdir(parents=True)
    (root / "firebird-sidecar").write_bytes(b"\xcf\xfa\xed\xfe" + struct.pack("<I", 0x100000C))
    (root / "firebird-sidecar").chmod(0o755)
    (root / "_internal/Python.framework/Versions/A/Python").write_bytes(b"library")
    (root / "_internal/Python.framework/Versions/Current").symlink_to("A", target_is_directory=True)
    (root / "_internal/Python.framework/Python").symlink_to("Versions/Current/Python")


def test_full_identity_survives_relocation_with_relative_links(tmp_path):
    original, relocated = tmp_path / "original", tmp_path / "moved"
    tree(original)
    expected = payload.inventory(original)
    shutil.copytree(original, relocated, symlinks=True)
    payload.verify(relocated, expected)
    assert expected["entries"]["firebird-sidecar"]["macho_cpu_types"] == [0x100000C]
    assert (
        expected["entries"]["_internal/Python.framework/Python"]["target"]
        == "Versions/Current/Python"
    )


@pytest.mark.parametrize("fault", ["bytes", "missing", "extra", "mode", "target", "empty-dir"])
def test_inventory_binds_all_entries(tmp_path, fault):
    tree(tmp_path)
    expected = payload.inventory(tmp_path)
    executable = tmp_path / "firebird-sidecar"
    if fault == "bytes":
        executable.write_bytes(b"changed")
    elif fault == "missing":
        executable.unlink()
    elif fault == "extra":
        (tmp_path / "extra").write_text("injected")
    elif fault == "mode":
        executable.chmod(0o644)
    elif fault == "target":
        link = tmp_path / "_internal/Python.framework/Python"
        link.unlink()
        link.symlink_to("Versions/A/Python")
    else:
        (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="inventory changed"):
        payload.verify(tmp_path, expected)


@pytest.mark.parametrize("fault", ["absolute", "escape", "dangling", "cycle", "fifo"])
def test_unsafe_entries_are_rejected(tmp_path, fault):
    root = tmp_path / "payload"
    root.mkdir()
    (root / "file").write_text("data")
    if fault == "fifo":
        os.mkfifo(root / "pipe")
    else:
        target = {
            "absolute": str(root / "file"),
            "escape": "../outside",
            "dangling": "missing",
            "cycle": "link",
        }[fault]
        (tmp_path / "outside").write_text("outside")
        (root / "link").symlink_to(target)
    with pytest.raises(ValueError):
        payload.inventory(root)


def test_budgets_and_typed_receipt(tmp_path, monkeypatch):
    tree(tmp_path)
    expected = payload.inventory(tmp_path)
    expected["schema_version"] = True
    with pytest.raises(ValueError):
        payload.verify(tmp_path, expected)
    monkeypatch.setattr(payload, "MAX_ENTRIES", 1)
    with pytest.raises(ValueError, match="entry limit"):
        payload.inventory(tmp_path)
    monkeypatch.setattr(payload, "MAX_ENTRIES", 20000)
    monkeypatch.setattr(payload, "MAX_BYTES", 1)
    with pytest.raises(ValueError, match="byte limit"):
        payload.inventory(tmp_path)


def test_native_headers_are_bounded():
    assert payload.native_architectures(b"ordinary text") is None
    for raw in (
        b"\xcf\xfa\xed\xfe",
        b"\xca\xfe\xba\xbe",
        b"\xca\xfe\xba\xbe" + struct.pack(">I", 17),
    ):
        with pytest.raises(ValueError):
            payload.native_architectures(raw)
    fat = b"\xca\xfe\xba\xbe" + struct.pack(">I", 2)
    fat += struct.pack(">IIIII", 0x100000C, 0, 0, 0, 0)
    fat += struct.pack(">IIIII", 0x1000007, 0, 0, 0, 0)
    assert payload.native_architectures(fat) == [0x100000C, 0x1000007]
