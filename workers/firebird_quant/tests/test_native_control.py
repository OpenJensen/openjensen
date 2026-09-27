"""Pure contract/publication checks. Stubbed probes are not native execution evidence."""

# ruff: noqa: E402 -- optional sibling validators and stdlib fixture builders only.
import copy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

WORKERS = Path(__file__).resolve().parents[2]
for path in ("isaac_sim", "isaac_sim/tests", "act_optimizer/src"):
    sys.path.insert(0, str(WORKERS / path))
from test_packed_checkpoint import add_control_contract, packed_fixture

from firebird_quant import native_application as app
from firebird_quant import native_package as package
from firebird_quant.native_consumer import load_packed_act


@pytest.fixture
def source(tmp_path):
    root = packed_fixture(tmp_path / "source")
    (root / "model.fbq").rename(root / "model.safetensors")
    (root / "encoding.json").unlink()
    add_control_contract(root)
    return root


def job(source, output):
    return {
        "schema_version": 1,
        "job_id": "pure-control",
        "operation": "policy.quantize",
        "source": {
            "path": str(source),
            "files": package.inventory(source),
            "artifact_id": "generated-structural-fixture",
            "artifact_manifest_sha256": "a" * 64,
            "manifest_sha256": None,
        },
        "output_dir": str(output),
        "native_quantization": {"format": "firebird_quant", "bits": 8, "group_size": 64},
        "timeout_seconds": 30,
    }


@pytest.fixture
def probes(monkeypatch):
    calls = []

    def probe(owner, mode, source, result, *, destination=None, bits=None):
        calls.append(mode)
        policy = source
        if mode == "convert":
            destination.mkdir()
            for path in source.iterdir():
                name = "model.fbq" if path.name == "model.safetensors" else path.name
                (destination / name).write_bytes(path.read_bytes())
            (destination / "encoding.json").write_bytes(package.canonical(package.encoding(bits)))
            policy = destination
        info = package.inspect_policy(policy)
        rows = [
            {
                "seed": seed,
                "input_sha256": f"{seed:064x}",
                "image_shape": [3, 32, 32],
                "raw": [[0.25] * 6 for _ in range(info["prediction_horizon"])],
                "postprocessed": [[0.5] * 6 for _ in range(info["prediction_horizon"])],
                "queue_and_reset_exact": True,
            }
            for seed in (171, 902)
        ]
        report = {
            "schema_version": 1,
            "versions": package.RUNTIME,
            "model_id": info["model_id"],
            "policy_files": info["files"],
            **{key: info[key] for key in package.CONTROL_FIELDS if key in info},
            "packed": rows,
            "network_disabled": True,
            "floating_master_reads_blocked": True,
        }
        if mode == "convert":
            report |= {"baseline": copy.deepcopy(rows), "source_files": package.inventory(source)}
        return report

    monkeypatch.setattr(app, "_probe", probe)
    return calls


def test_control_temporal_processors_manifest_and_reports_are_bound(source, tmp_path, probes):
    before = package.inventory(source)
    result = app.run_job(job(source, tmp_path / "out"))
    root = Path(result["artifact"]["path"])
    info = package.inspect_policy(root / "policy")
    expected = {key: info[key] for key in package.CONTROL_FIELDS}
    for record in (
        result["report"],
        package.read_json(root / "manifest.json")["metadata"],
        package.read_json(root / "verification.json"),
    ):
        assert {key: record[key] for key in package.CONTROL_FIELDS} == expected
        assert (record["prediction_horizon"], record["execution_horizon"]) == (8, 3)
        assert record["quality_verified"] is record["calibration_verified"] is False
    for name in before.keys() - {"model.safetensors"}:
        assert (root / "policy" / name).read_bytes() == (source / name).read_bytes()
    manifest = package.read_json(root / "manifest.json")
    for name, digest in manifest["files"].items():
        assert package.sha((root / name).read_bytes()) == digest
    assert info["model_id"] == package.model_identity(info["files"])
    assert package.inventory(source) == before
    assert probes == ["convert", "reload"]


@pytest.mark.parametrize("mode", ["convert", "reload"])
@pytest.mark.parametrize("change", ["omitted", "null", "digest", "joint_order"])
def test_probe_control_claims_cannot_disagree(source, tmp_path, probes, monkeypatch, mode, change):
    original = app._probe

    def wrong(*args, **kwargs):
        report = original(*args, **kwargs)
        if args[1] == mode:
            if change == "omitted":
                report.pop("control_contract")
                report.pop("control_contract_sha256")
            elif change == "null":
                report["control_contract"] = report["control_contract_sha256"] = None
            elif change == "digest":
                report["control_contract_sha256"] = "0" * 64
            else:
                report["control_contract"]["joint_order"].reverse()
        return report

    monkeypatch.setattr(app, "_probe", wrong)
    with pytest.raises(ValueError, match="Probe runtime/package identity"):
        app.run_job(job(source, tmp_path / "out"))
    assert not (tmp_path / "out/native-quantized").exists()


@pytest.mark.parametrize(
    "change", ["drop_control", "drop_temporal", "joint_order", "processors", "introduced"]
)
def test_conversion_cannot_drop_or_replace_bound_files(
    source, tmp_path, probes, monkeypatch, change
):
    if change == "introduced":
        (source / "control-contract.json").unlink()
    original = app._probe

    def wrong(*args, **kwargs):
        report = original(*args, **kwargs)
        if args[1] == "convert":
            policy = kwargs["destination"]
            if change.startswith("drop_"):
                (policy / (change[5:] + "-contract.json")).unlink()
            elif change == "joint_order":
                record = package.read_json(policy / "control-contract.json")
                record["joint_order"].reverse()
                (policy / "control-contract.json").write_bytes(package.canonical(record))
            elif change == "introduced":
                add_control_contract(policy)
            else:
                # Valid equivalent JSON still changes exact saved processor bytes.
                path = policy / "policy_preprocessor.json"
                path.write_bytes(package.canonical(package.read_json(path)))
        return report

    monkeypatch.setattr(app, "_probe", wrong)
    with pytest.raises(ValueError, match="Packing changed source"):
        app.run_job(job(source, tmp_path / "out"))
    assert probes == ["convert"]
    assert not (tmp_path / "out/native-quantized").exists()


@pytest.mark.parametrize("null_claims", [False, True])
def test_legacy_absent_or_null_probe_claims_keep_identity(
    source, tmp_path, probes, monkeypatch, null_claims
):
    (source / "control-contract.json").unlink()
    original = app._probe

    def legacy(*args, **kwargs):
        report = original(*args, **kwargs)
        if null_claims:
            report.update(dict.fromkeys(package.CONTROL_FIELDS))
        return report

    monkeypatch.setattr(app, "_probe", legacy)
    result = app.run_job(job(source, tmp_path / "out"))
    assert not set(package.CONTROL_FIELDS).intersection(result["report"])
    info = package.inspect_policy(Path(result["artifact"]["path"]) / "policy")
    assert result["report"]["model_id"] == package.model_identity(info["files"])


@pytest.mark.parametrize("change", ["units", "fps", "camera", "joints", "null"])
def test_bad_control_rejected_before_probe(source, tmp_path, probes, change):
    path = source / "control-contract.json"
    record = package.read_json(path)
    if change == "units":
        record["action_units"] = "degrees"
    elif change == "fps":
        record["action_fps"] = 30
    elif change == "camera":
        record["camera"]["key"] = "observation.images.other"
    elif change == "joints":
        record["joint_order"].pop()
    else:
        record = None
    path.write_bytes(package.canonical(record))
    with pytest.raises(ValueError):
        app.run_job(job(source, tmp_path / "out"))
    assert not probes


def test_packed_control_is_bound_to_inventory_not_later_read(tmp_path, monkeypatch):
    policy = packed_fixture(tmp_path / "policy")
    record = add_control_contract(policy)
    original = package.inventory

    def changed(root):
        result = original(root)
        record["joint_order"].reverse()
        (root / "control-contract.json").write_bytes(package.canonical(record))
        return result

    monkeypatch.setattr(package, "inventory", changed)
    with pytest.raises(ValueError, match="control contract changed"):
        package.inspect_policy(policy)


def test_consumer_cannot_drop_contract_under_previous_identity(tmp_path):
    policy = packed_fixture(tmp_path / "policy")
    add_control_contract(policy)
    identity = package.inspect_policy(policy)["model_id"]
    (policy / "control-contract.json").unlink()
    with patch("firebird_act.probe.runtime_versions") as runtime:
        with pytest.raises(ValueError, match="identity"):
            load_packed_act(policy, device="cpu", expected_model_id=identity)
        runtime.assert_not_called()


def test_control_does_not_enable_packed_cuda(tmp_path):
    from sim_worker.rollout.backend import LeRobotPolicy

    policy = packed_fixture(tmp_path / "policy")
    add_control_contract(policy)
    with patch("sim_worker.rollout.backend.version") as version:
        with pytest.raises(ValueError, match="CPU only"):
            LeRobotPolicy(policy, "cuda", 8, 6, "observation.images.front")
        version.assert_not_called()
