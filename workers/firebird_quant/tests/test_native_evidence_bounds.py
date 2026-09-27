"""Bounded JSON evidence regression; no Torch/model imports or inference."""

from types import SimpleNamespace

import pytest

from firebird_quant import native_application as app
from firebird_quant import native_package as package


def evidence():
    row = {
        "raw": [[-0.12345679104328156] * 6 for _ in range(1024)],
        "postprocessed": [[0.12345679104328156] * 6 for _ in range(1024)],
    }
    return {"baseline": [row, row], "packed": [row, row]}


@pytest.mark.parametrize("mode", ["convert", "reload"])
def test_max_horizon_probe_reads_large_evidence(tmp_path, mode):
    value = evidence()
    raw = package.canonical(value)
    assert package.JSON_LIMIT < len(raw) < 8 * 1024**2
    result = tmp_path / "probe.json"
    owner = SimpleNamespace(run=lambda *_: result.write_bytes(raw))
    assert app._probe(owner, mode, tmp_path, result) == value


@pytest.mark.parametrize("mode", ["convert", "reload"])
def test_probe_still_rejects_above_evidence_cap(tmp_path, mode):
    result = tmp_path / "probe.json"

    def write(*_):
        with result.open("wb") as stream:
            stream.truncate(8 * 1024**2 + 1)

    with pytest.raises(ValueError, match="bounded"):
        app._probe(SimpleNamespace(run=write), mode, tmp_path, result)


@pytest.mark.parametrize("name", ["config.json", "manifest.json"])
def test_ordinary_metadata_keeps_original_limit(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(package.canonical(evidence()))
    with pytest.raises(ValueError, match="bounded"):
        package.read_json(path)
