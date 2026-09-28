"""Generated native ACT/robotics-shaped fixtures; never claim recorded policy quality."""

from pathlib import Path

import pytest
from firebird_act.bundle import canonical
from firebird_distill.contracts import digest
from native_fixture import make_job, make_teacher


@pytest.fixture(scope="session")
def teacher(tmp_path_factory):
    return make_teacher(tmp_path_factory.mktemp("distill-teacher"))


@pytest.fixture
def job(teacher, tmp_path):
    return make_job(teacher, tmp_path)


def rebind_manifest(job, mutate):
    import json

    root = Path(job["dataset"]["path"])
    doc = json.loads((root / "manifest.json").read_bytes())
    mutate(doc)
    raw = canonical(doc)
    (root / "manifest.json").write_bytes(raw)
    job["dataset"]["manifest_sha256"] = digest(raw)
    return doc
