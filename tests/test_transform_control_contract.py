"""Protocol-only compression lineage regressions; no Torch or simulator execution."""

import copy
import json
from types import SimpleNamespace

import pytest
import test_native_distillation as baseline
from test_native_distillation import check, digest, manifest, output, prepared, write
from test_simulator_control_provenance import control_record
from vla_platform.contracts import DatasetSnapshot
from vla_platform.datasets import snapshots
from vla_platform.lifecycle import native_distillation as nd
from vla_platform.lifecycle import native_quantization as nq
from vla_platform.lifecycle.temporal import resolved_temporal

fixture = baseline.fixture


def bind_teacher(f):
    """Attach internally consistent metadata to deliberately non-native fixture bytes."""
    snap = copy.deepcopy(f.data["manifest"])
    raw = b'{"protocol_only":true}\n'
    snap["files"].append(
        {"path": "meta/firebird-demonstrations.json", "size": len(raw), "sha256": digest(raw)}
    )
    for name in ("observation.state", "action"):
        snap["features"][name]["dtype"] = "float32"
    snapshot_raw = snapshots.canonical(snap)
    snapshot_sha = digest(snapshot_raw)
    root = f.data_dir / "dataset-snapshots" / snapshot_sha
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_bytes(b"{}\n")
    (root / "meta/firebird-demonstrations.json").write_bytes(raw)
    (root / snapshots.MANIFEST).write_bytes(snapshot_raw)
    f.profile.snapshot = DatasetSnapshot(**snapshots.descriptor(snap, snapshot_sha))
    policy = f.admitted["path"]
    config = json.loads((policy / "config.json").read_text())
    config.update(chunk_size=8, n_action_steps=3, n_obs_steps=1)
    config["input_features"]["observation.state"]["type"] = "STATE"
    config["input_features"]["observation.images.front"]["type"] = "VISUAL"
    config["output_features"]["action"]["type"] = "ACTION"
    write(policy / "config.json", config)
    record = control_record()
    record["joint_order"] = [f"j{i}" for i in range(6)]
    record["camera"].update(width=32, height=32)
    record["source"].update(
        dataset_snapshot_id=f.profile.snapshot.id,
        dataset_manifest_sha256=snapshot_sha,
        demonstrations_sha256=digest(raw),
        origins=["synthetic"],
    )
    write(policy / "control-contract.json", record)
    temporal = resolved_temporal(
        SimpleNamespace(
            chunk_size=8,
            n_action_steps=3,
            n_obs_steps=1,
            action_delta_indices=list(range(8)),
            observation_delta_indices=[0],
        ),
        30,
        "act",
    )
    write(policy / "temporal-contract.json", temporal)
    claims = {
        "control_contract": record,
        "control_contract_sha256": digest((policy / "control-contract.json").read_bytes()),
    }
    f.source.metadata.update(claims)
    f.source.manifest_sha256 = manifest(policy.parent, f.source.metadata)
    f.admitted = nq.source_info(f.source, f.data_dir, require_inference=False)
    f.job.request.native_distillation.units = ["radians"] * 6
    implementation = f.data["implementation"]
    f.data = nd.dataset_info(
        f.profile, f.data_dir / "dataset-snapshots", f.admitted, f.job.request.native_distillation
    )
    f.data.update(episode_lengths={0: 2, 1: 2, 2: 2}, implementation=implementation)
    return f


def corpus(f, path):
    doc, response = prepared(f, path)
    doc["chunk_size"] = f.admitted["prediction_horizon"]
    for key in (
        "execution_horizon",
        "action_fps",
        "temporal_contract_sha256",
        "control_contract",
        "control_contract_sha256",
    ):
        doc[key] = f.data[key]
    write(path / "manifest.json", doc)
    response["manifest_sha256"] = digest((path / "manifest.json").read_bytes())
    return doc, response


def annotated_output(f, directory, doc, corpus_sha):
    response = output(f, directory, doc, corpus_sha)
    claims = {
        key: f.admitted[key]
        for key in (
            "prediction_horizon",
            "execution_horizon",
            "temporal_contract_sha256",
            "control_contract",
            "control_contract_sha256",
        )
    }
    metadata = json.loads((directory / "manifest.json").read_text())["metadata"] | claims
    for row in response["report"]["predictions"]:
        row.update(queue_refill_exact=True, prediction_horizon=8, execution_horizon=3)
    response["report"].update(claims)
    for name in ("training.json", "verification.json"):
        value = json.loads((directory / name).read_text()) | claims
        value["predictions"] = response["report"]["predictions"]
        write(directory / name, value)
    manifest(directory, metadata)
    return response


def test_control_bound_nondefault_horizons_survive_student_admission(fixture):
    f = bind_teacher(fixture)
    path = f.root / "corpus"
    doc, response = corpus(f, path)
    nd.check_corpus(path, response, f.data, f.job.request.native_distillation)
    destination = f.root / "student"
    response = annotated_output(f, destination, doc, response["manifest_sha256"])
    result, _ = check(f, response, doc, digest(nd.canonical(doc)), destination)
    assert result["metadata"]["execution_horizon"] == 3
    assert result["metadata"]["control_contract"] == f.source.metadata["control_contract"]


@pytest.mark.parametrize(
    "field",
    ["execution_horizon", "action_fps", "control_contract_sha256", "temporal_contract_sha256"],
)
def test_corpus_cannot_drop_new_semantics(fixture, field):
    f = bind_teacher(fixture)
    path = f.root / "corpus"
    doc, response = corpus(f, path)
    del doc[field]
    write(path / "manifest.json", doc)
    response["manifest_sha256"] = digest(nd.canonical(doc))
    with pytest.raises(ValueError, match="corpus fields"):
        nd.check_corpus(path, response, f.data, f.job.request.native_distillation)


@pytest.mark.parametrize("fault", ["snapshot", "fps", "units", "joint_order", "demonstrations"])
def test_control_admission_rejects_incompatible_dataset(fixture, fault):
    f = bind_teacher(fixture)
    data = copy.deepcopy(f.data["manifest"])
    descriptor = copy.deepcopy(f.data["descriptor"])
    names = list(f.data["semantics"]["state_names"])
    recipe = f.job.request.native_distillation.model_copy(deep=True)
    if fault == "snapshot":
        descriptor["id"] = "sha256:" + "e" * 64
    elif fault == "fps":
        data["fps"] = 15
    elif fault == "units":
        recipe.units = ["degrees"] * 6
    elif fault == "joint_order":
        names.reverse()
    else:
        data["files"][-1]["sha256"] = "e" * 64
    with pytest.raises(ValueError, match="snapshot|coordinates"):
        nd.dataset_provenance(data, descriptor, f.admitted, recipe, f.data["camera"], names)


@pytest.mark.parametrize("target", ["manifest", "report", "verification", "policy"])
def test_student_cannot_lose_control_identity(fixture, target):
    f = bind_teacher(fixture)
    doc, receipt = corpus(f, f.root / "corpus")
    destination = f.root / "student"
    response = annotated_output(f, destination, doc, receipt["manifest_sha256"])
    metadata = json.loads((destination / "manifest.json").read_text())["metadata"]
    if target == "manifest":
        metadata.pop("control_contract_sha256")
    elif target == "report":
        response["report"].pop("control_contract_sha256")
    elif target == "verification":
        path = destination / "verification.json"
        value = json.loads(path.read_text())
        value.pop("control_contract_sha256")
        write(path, value)
    else:
        (destination / "policy/control-contract.json").unlink()
    manifest(destination, metadata)
    with pytest.raises(ValueError):
        check(f, response, doc, receipt["manifest_sha256"], destination)


def test_fp32_identity_excludes_temporal_but_includes_control_bytes(fixture):
    f = bind_teacher(fixture)
    policy = f.admitted["path"]
    before = nd.model_id(policy, nq.inventory(policy))
    (policy / "temporal-contract.json").write_text('{"changed":true}\n')
    assert nd.model_id(policy, nq.inventory(policy)) == before
    (policy / "control-contract.json").write_text('{"changed":true}\n')
    assert nd.model_id(policy, nq.inventory(policy)) != before


def test_admission_rejects_orphan_control_copy(fixture):
    f = bind_teacher(fixture)
    policy = f.admitted["path"]
    (policy / "control-contract.json").rename(policy.parent / "control-contract.json")
    f.source.metadata = {"architecture": "act"}
    f.source.manifest_sha256 = manifest(policy.parent, f.source.metadata)
    with pytest.raises(ValueError, match="lost or changed"):
        nq.source_info(f.source, f.data_dir, require_inference=False)


@pytest.mark.parametrize(
    "key,value", [("control_contract_sha256", "f" * 64), ("prediction_horizon", 8)]
)
def test_legacy_student_report_cannot_introduce_unbound_provenance(fixture, key, value):
    f = fixture
    doc, receipt = prepared(f, f.root / "corpus")
    destination = f.root / "student"
    response = output(f, destination, doc, receipt["manifest_sha256"])
    response["report"][key] = value
    training = json.loads((destination / "training.json").read_text())
    training[key] = value
    write(destination / "training.json", training)
    metadata = json.loads((destination / "manifest.json").read_text())["metadata"]
    manifest(destination, metadata)
    with pytest.raises(ValueError, match="report provenance"):
        check(f, response, doc, receipt["manifest_sha256"], destination)


def packed_fixture(f):
    import runpy
    from pathlib import Path

    from test_native_quantization import request

    config_path = f.admitted["path"] / "config.json"
    config = json.loads(config_path.read_text())
    config["use_vae"] = False
    write(config_path, config)
    f.source.manifest_sha256 = manifest(config_path.parent.parent, f.source.metadata)
    f.admitted = nq.source_info(f.source, f.data_dir)
    root = f.root / "packed-output"
    payload = {
        "job_id": "packing",
        "source": {
            "path": str(f.admitted["path"]),
            "files": f.admitted["files"],
            "manifest_sha256": f.admitted["manifest_sha256"],
            "artifact_id": f.source.id,
            "artifact_manifest_sha256": f.source.manifest_sha256,
        },
        "output_dir": str(root),
        "native_quantization": {"bits": 8},
    }
    worker = (
        Path(__file__).parent
        / "fixtures/native_quant_worker/src/firebird_quant/native_application.py"
    )
    response = runpy.run_path(str(worker))["package"](payload)
    job = SimpleNamespace(id="packing", request=request(artifact_id=f.source.id))
    return root / "native-quantized", response, job


def test_packed_result_preserves_control_and_independent_horizons(fixture):
    f = bind_teacher(fixture)
    path, response, job = packed_fixture(f)
    result, _ = nq.check_result(response, job, f.source, f.admitted, path)
    assert result["metadata"]["control_contract"] == f.admitted["control_contract"]
    assert result["metadata"]["execution_horizon"] == 3


@pytest.mark.parametrize(
    "fault", ["lost-file", "changed-file", "missing-proof", "coerced-proof", "coerced-report"]
)
def test_packed_result_refuses_lost_or_forged_control(fixture, fault):
    f = bind_teacher(fixture)
    path, response, job = packed_fixture(f)
    metadata = json.loads((path / "manifest.json").read_text())["metadata"]
    if fault == "lost-file":
        (path / "policy/control-contract.json").unlink()
    elif fault == "changed-file":
        (path / "policy/control-contract.json").write_text("{}")
    elif fault == "coerced-report":
        response["report"]["control_contract"]["action_fps"] = 30.0
    else:
        proof = json.loads((path / "verification.json").read_text())
        if fault == "missing-proof":
            proof.pop("control_contract_sha256")
        else:
            proof["control_contract"]["action_fps"] = 30.0
        write(path / "verification.json", proof)
    manifest(path, metadata)
    with pytest.raises(ValueError):
        nq.check_result(response, job, f.source, f.admitted, path)


@pytest.mark.parametrize("location", ["top", "nested"])
@pytest.mark.parametrize(
    "fault", ["prediction", "execution", "digest", "partial", "missing-file", "coerced"]
)
def test_source_temporal_metadata_must_match_actual_files(fixture, location, fault):
    f = bind_teacher(fixture)
    claims = {
        key: f.admitted[key]
        for key in ("prediction_horizon", "execution_horizon", "temporal_contract_sha256")
    }
    if fault == "prediction":
        claims["prediction_horizon"] = 100
    elif fault == "execution":
        claims["execution_horizon"] = 8
    elif fault == "digest":
        claims["temporal_contract_sha256"] = "f" * 64
    elif fault == "partial":
        claims.pop("execution_horizon")
    elif fault == "coerced":
        claims["prediction_horizon"] = 8.0
    else:
        (f.admitted["path"] / "temporal-contract.json").unlink()
    if location == "top":
        f.source.metadata.update(claims)
    else:
        f.source.metadata["checkpoint"] = claims
    f.source.manifest_sha256 = manifest(f.admitted["path"].parent, f.source.metadata)
    with pytest.raises(ValueError, match="temporal"):
        nq.source_info(f.source, f.data_dir, require_inference=False)


def test_source_cannot_drop_policy_temporal_file_while_retaining_outer_copy(fixture):
    f = bind_teacher(fixture)
    policy = f.admitted["path"]
    (policy / "temporal-contract.json").rename(policy.parent / "temporal-contract.json")
    f.source.manifest_sha256 = manifest(policy.parent, f.source.metadata)
    with pytest.raises(ValueError, match="lost or changed its temporal"):
        nq.source_info(f.source, f.data_dir, require_inference=False)
