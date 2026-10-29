"""Portable source guard without naming an installation or contributor."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/check_portability.py"
spec = importlib.util.spec_from_file_location("portability", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_guard_reports_custom_home_paths_without_echoing_contents(tmp_path):
    source = tmp_path / "settings.py"
    source.write_text("root = '/home/" + "developer-name/runtime'\n")
    result = module.check(tmp_path, [source.name])
    assert result == [("settings.py", 1, "developer-home-path")]
    assert "developer-name" not in repr(result)


def test_generic_paths_and_dynamic_discovery_are_portable():
    assert list(module.violations("root = Path.home() / 'runtime'")) == []
    assert list(module.violations("root = '/home/user/runtime'")) == []
    assert list(module.violations(r"root = 'C:\Users\username\runtime'")) == []


def test_guard_handles_mac_and_windows_custom_homes():
    assert list(module.violations("path = '/Users/" + "developer-name/runtime'")) == [
        (1, "developer-home-path")
    ]
    assert list(module.violations(r"path = 'C:\Users" + r"\developer-name\runtime'")) == [
        (1, "developer-home-path")
    ]


def test_private_endpoints_are_configured_instead_of_shipped():
    assert list(module.violations("url = 'http://" + "10.24.5.9:8000'")) == [
        (1, "private-default-endpoint")
    ]
    assert list(module.violations("ssh user@" + "192.168.12.8 command")) == [
        (1, "private-default-endpoint")
    ]
    assert list(module.violations("http://127.0.0.1:8000 http://192.0.2.3:8000")) == []
    assert list(module.violations("https://compute.example.invalid")) == []


def test_license_and_vendor_attribution_are_outside_guard(tmp_path):
    license_file = tmp_path / "LICENSE"
    license_file.write_text("Contributor originally used /home/" + "contributor-name/source")
    assert module.check(tmp_path, ["LICENSE"]) == []
    assert not module.first_party("workers/vendor/component/source.py")
    assert module.first_party("workers/policy/source.py")


def test_synthetic_test_topology_is_not_a_shipped_default(tmp_path):
    folder = tmp_path / "tests"
    folder.mkdir()
    fixture = folder / "test_network.py"
    fixture.write_text("endpoint = 'http://" + "10.24.5.9:8000'")
    assert module.check(tmp_path, ["tests/test_network.py"]) == []
