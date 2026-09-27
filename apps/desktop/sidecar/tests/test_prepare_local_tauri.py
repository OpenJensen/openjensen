"""Preparation never executes a payload or activates a native build."""

import json
import struct
from pathlib import Path

import pytest
from test_sidecar import module, resource_tree

prepare = module("prepare_local_tauri")


def fixture(tmp_path):
    payload = tmp_path / "payload"
    payload.mkdir()
    executable = payload / "firebird-sidecar"
    executable.write_bytes(b"\xcf\xfa\xed\xfe" + struct.pack("<I", 0x100000C) + b"fixture")
    executable.chmod(0o755)
    (payload / "safe-link").symlink_to("firebird-sidecar")
    resource_tree(payload / "_internal/sidecar-resources")
    value = module("payload_inventory").inventory(payload)
    manifest = tmp_path / "inventory.json"
    manifest.write_text(json.dumps(value))
    resources = prepare.load_resources(payload / "_internal/sidecar-resources")
    report = {
        "source_build_id": resources.build_id,
        "resources_sha256": resources.manifest_sha256,
        "payload_identity_sha256": value["identity_sha256"],
        "native_files": {"firebird-sidecar": [0x100000C]},
        **{
            key: True
            for key in [
                "source_and_relocated_payloads_unchanged",
                "workspace_owner_exclusion",
                "shutdown_and_parent_eof_exit_zero",
                "restart_preserved_project_and_job",
                "parent_eof_before_start_created_no_workspace",
                "tampered_static_rejected_before_workspace",
                "wrong_control_nonce_shut_down_owned_listener",
            ]
        },
    }
    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text(json.dumps(report))
    return payload, manifest, acceptance


def test_prepare_copies_complete_payload_and_generates_default_off_fixed_inputs(tmp_path):
    payload, manifest, acceptance = fixture(tmp_path)
    output = tmp_path / "candidate"
    result = prepare.prepare(payload, manifest, acceptance, output)
    assert (output / "resources/firebird-sidecar/safe-link").is_symlink()
    assert (output / "resources/firebird-sidecar/safe-link").readlink() == Path("firebird-sidecar")
    assert result["production_activation"] is False
    assert result["native_candidate_built"] is False
    assert result["identifier"].startswith("dev.firebird.workbench.experiment.")
    config = json.loads((output / "tauri.local.json").read_text())
    assert config["identifier"] == result["identifier"]
    assert set(config["bundle"]["resources"].values()) == {
        "sidecar/firebird-sidecar",
        "sidecar/payload-manifest.json",
    }
    assert Path(config["build"]["frontendDist"]).is_absolute()
    assert "LOCAL_EXPERIMENT_PAYLOAD" in (output / "local-payload-pin.rs").read_text()
    assert "PRODUCTION_PAYLOAD" not in (output / "local-payload-pin.rs").read_text()
    prepare.verify(payload, json.loads(manifest.read_text()))
    prepare.verify(output / "resources/firebird-sidecar", json.loads(manifest.read_text()))
    with pytest.raises(ValueError, match="must not exist"):
        prepare.prepare(payload, manifest, acceptance, output)


@pytest.mark.parametrize(
    "fault",
    [
        "modified",
        "missing",
        "extra",
        "wrong-identity",
        "wrong-build",
        "bool-proof",
        "missing-proof",
        "wrong-arch",
        "symlink",
        "overlap",
        "relative",
    ],
)
def test_bad_inputs_fail_before_creating_candidate(tmp_path, fault):
    payload, manifest, acceptance = fixture(tmp_path)
    output = tmp_path / "candidate"
    if fault == "modified":
        (payload / "firebird-sidecar").write_bytes(b"changed")
    elif fault == "missing":
        (payload / "firebird-sidecar").unlink()
    elif fault == "extra":
        (payload / "extra").write_text("extra")
    elif fault == "symlink":
        link = tmp_path / "link"
        link.symlink_to(payload, target_is_directory=True)
        payload = link
    elif fault == "overlap":
        output = payload / "nested"
    elif fault == "relative":
        output = Path("relative")
    else:
        value = json.loads(acceptance.read_text())
        if fault == "wrong-identity":
            value["payload_identity_sha256"] = "c" * 64
        elif fault == "wrong-build":
            value["source_build_id"] = "c" * 40
        elif fault == "bool-proof":
            value["workspace_owner_exclusion"] = 1
        elif fault == "missing-proof":
            del value["workspace_owner_exclusion"]
        elif fault == "wrong-arch":
            value["native_files"] = {"firebird-sidecar": [7]}
        acceptance.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        prepare.prepare(payload, manifest, acceptance, output)
    assert not output.exists()
