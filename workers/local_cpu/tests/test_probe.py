from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("local_cpu_probe", ROOT / "probe.py")
p = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = p
spec.loader.exec_module(p)


class Distribution:
    def __init__(self, name, version, root):
        self.name, self.version, self.root = name, version, root
        self.reads = 0

    @property
    def metadata(self):
        self.reads += 1
        return {"Name": self.name, "Version": self.version}

    def locate_file(self, name):
        assert name == ""
        return self.root


@pytest.mark.parametrize("pin_count", [1, 10])
def test_each_distribution_metadata_is_read_once(tmp_path, monkeypatch, pin_count):
    pins = {f"package-{i}": "1.2.3" for i in range(pin_count)}
    distributions = [
        Distribution(f"PACKAGE_{i}", "1.2.3", tmp_path / "site-packages") for i in range(63)
    ]
    monkeypatch.setattr(p.importlib.metadata, "distributions", lambda: iter(distributions))
    assert p.installed_versions(pins, tmp_path) == pins
    assert [d.reads for d in distributions] == [1] * 63


@pytest.mark.parametrize(
    "case, error",
    [
        ("duplicate", "missing or duplicate"),
        ("missing", "missing or duplicate"),
        ("wrong_version", "requires"),
        ("no_version", "requires"),
        ("borrowed", "borrowed"),
        ("no_name", "package name"),
    ],
)
def test_pin_admission_checks_remain_strict(tmp_path, monkeypatch, case, error):
    prefix = tmp_path / "runtime"
    distributions = [Distribution("Torch", "2.11.0+cpu", prefix / "site-packages")]
    if case == "duplicate":
        distributions.append(Distribution("TORCH", "2.11.0+cpu", prefix))
    elif case == "missing":
        distributions.clear()
    elif case == "wrong_version":
        distributions[0].version = "2.10.0"
    elif case == "no_version":
        distributions[0].version = None
    elif case == "borrowed":
        distributions[0].root = tmp_path / "different-runtime"
    elif case == "no_name":
        distributions[0].name = None
    monkeypatch.setattr(p.importlib.metadata, "distributions", lambda: iter(distributions))
    with pytest.raises(ValueError, match=error):
        p.installed_versions({"torch": "2.11.0"}, prefix)


def test_cpu_version_suffix_is_preserved(tmp_path, monkeypatch):
    distribution = Distribution("torch", "2.11.0+cpu", tmp_path)
    monkeypatch.setattr(p.importlib.metadata, "distributions", lambda: [distribution])
    assert p.installed_versions({"torch": "2.11.0"}, tmp_path) == {"torch": "2.11.0+cpu"}
    assert distribution.reads == 1
