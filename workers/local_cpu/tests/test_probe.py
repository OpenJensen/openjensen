from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

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


@pytest.fixture
def worker_sources(tmp_path, monkeypatch):
    """Only tiny source markers and fake module metadata; never import an ML worker."""
    files, modules = {}, {}
    groups = (
        (
            "distillation",
            "firebird_distill",
            "policy_distillation/src/firebird_distill",
            ("__init__", "application", "contracts", "prepare", "runtime", "provenance"),
        ),
        (
            "act",
            "firebird_act",
            "act_optimizer/src/firebird_act",
            ("bundle", "probe", "control_schema"),
        ),
        (
            "data",
            "firebird_vla",
            "smolvla_qlora/src/firebird_vla",
            ("control_contract", "control_schema"),
        ),
    )
    for prefix, package, folder, names in groups:
        for name in names:
            path = tmp_path / "workers" / folder / f"{name}.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# generated {prefix}/{name} source marker\n")
            files[f"{prefix}/{name}.py"] = path
            imported = package if name == "__init__" else f"{package}.{name}"
            modules[imported] = SimpleNamespace(__file__=str(path))

    def identity():
        return {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in files.items()}

    modules["firebird_distill.application"].implementation_identity = identity

    def selected_import(name):
        if name not in modules:
            raise AssertionError(f"Unexpected import, including native ML: {name}")
        return modules[name]

    monkeypatch.setattr(p.importlib, "import_module", selected_import)
    return tmp_path, files, modules, identity


def test_distillation_receipt_changes_when_provenance_changes(worker_sources):
    repo, files, _, identity = worker_sources
    before = p.distillation_source_identity(repo)
    assert before == identity() and len(before) == 11
    files["distillation/provenance.py"].write_text("# changed coordinate derivation\n")
    after = p.distillation_source_identity(repo)
    assert [key for key in before if before[key] != after[key]] == ["distillation/provenance.py"]


@pytest.mark.parametrize("fault", ["missing", "wrong_hash", "extra"])
def test_declared_worker_identity_must_match_exact_sources(worker_sources, fault):
    repo, _, modules, identity = worker_sources
    reported = identity()
    if fault == "missing":
        reported.pop("distillation/provenance.py")
    elif fault == "wrong_hash":
        reported["data/control_contract.py"] = "0" * 64
    else:
        reported["distillation/undeclared.py"] = "0" * 64
    modules["firebird_distill.application"].implementation_identity = lambda: reported
    with pytest.raises(ValueError, match="implementation identity"):
        p.distillation_source_identity(repo)


@pytest.mark.parametrize("fault", ["missing_file", "missing_module", "borrowed", "no_source"])
def test_dependency_source_origin_and_presence_are_required(worker_sources, fault, monkeypatch):
    repo, files, modules, _ = worker_sources
    key = "firebird_distill.provenance"
    if fault == "missing_file":
        files["distillation/provenance.py"].unlink()
    elif fault == "missing_module":

        def missing(name):
            if name == key:
                raise ModuleNotFoundError(name)
            return modules[name]

        monkeypatch.setattr(p.importlib, "import_module", missing)
    elif fault == "borrowed":
        # Even a module elsewhere in this same checkout must not satisfy the fixed path.
        modules[key].__file__ = str(files["distillation/contracts.py"])
    else:
        modules[key].__file__ = None
    with pytest.raises((ValueError, FileNotFoundError, ModuleNotFoundError)):
        p.distillation_source_identity(repo)
