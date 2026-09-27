"""Static dependency validation does not execute Mach-O files."""

import pytest
from test_sidecar import module

native = module("native_dependencies")


def test_parse_otool_output():
    assert native.parse_dependencies(
        "file:\n\t@rpath/libPython.dylib (compatibility version 3.14.0, current version 3.14.0)\n"
    ) == ["@rpath/libPython.dylib"]
    assert native.parse_rpaths(
        "Load command 0\n cmd LC_RPATH\n cmdsize 48\n path @executable_path/_internal (offset 12)\n"
    ) == ["@executable_path/_internal"]
    with pytest.raises(ValueError):
        native.parse_dependencies("file:\n unexplained text\n")
    with pytest.raises(ValueError):
        native.parse_rpaths("cmd LC_RPATH\ncmd LC_LOAD_DYLIB\n")


@pytest.mark.parametrize(
    "reference", ["/private/build/lib.dylib", "@loader_path/../../escape", "unqualified.dylib"]
)
def test_private_or_escaping_paths_rejected(tmp_path, reference):
    root = tmp_path / "payload"
    root.mkdir()
    binary = root / "firebird-sidecar"
    binary.write_bytes(b"macho")
    with pytest.raises((ValueError, FileNotFoundError)):
        native._location(reference, binary, binary, root)


def test_resolve_relocated_dependency_and_reject_unknown_rpath(tmp_path, monkeypatch):
    main = tmp_path / "firebird-sidecar"
    main.write_bytes(b"main")
    internal = tmp_path / "_internal"
    internal.mkdir()
    (internal / "lib.dylib").write_bytes(b"library")
    recorded = {"entries": {"firebird-sidecar": {"kind": "file", "macho_cpu_types": [0x100000C]}}}

    def fake_tool(path, option, limit):
        if option == "-L":
            return (
                "file:\n\t@rpath/lib.dylib (compatibility version 0.0.0, current version 0.0.0)\n"
                "\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, "
                "current version 1.0.0)\n"
            )
        return "cmd LC_RPATH\npath @executable_path/_internal (offset 12)\n"

    monkeypatch.setattr(native, "_tool", fake_tool)
    result = native.inspect(tmp_path, recorded)
    assert result["native_files"]["firebird-sidecar"]["resolved_dependencies"][0]["resolved"] == [
        "_internal/lib.dylib"
    ]
    (internal / "lib.dylib").unlink()
    with pytest.raises(ValueError, match="Unresolved"):
        native.inspect(tmp_path, recorded)
