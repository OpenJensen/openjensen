"""Small actual formats exercise static checks; no real model execution is claimed."""

import importlib.abc
import io
import json
import shutil
import socket
import subprocess
import sys
import zipfile

import gguf
import numpy as np
import pytest
from policykit import spatial_preflight as preflight
from policykit import spatial_protocol as protocol
from policykit import spatial_validation as validation
from policykit.spatial_fixture import load_arrays
from policykit.worker import atomic_json, canonical, sha256
from safetensors.numpy import save_file
from test_spatial_application import asset_bundle, raw_arrays


def make_case(tmp_path, monkeypatch, *, dimension=None, stats=None, packed=False, vectors=8):
    bundle, record = asset_bundle(tmp_path, monkeypatch, reference=True)
    model = tmp_path / "source.gguf"
    writer = gguf.GGUFWriter(model, "smolvla")
    for key, value in {**validation.DIMENSIONS, **(dimension or {})}.items():
        writer.add_uint32("smolvla." + key, value)
    for filename, prefix, feature, size in validation.NORMALIZERS:
        tensors = {}
        for suffix in ("mean", "std"):
            value = np.linspace(0.25, 1, size, dtype=np.float32)
            if stats in {"nan", "infinity"}:
                value[-1] = np.nan if stats == "nan" else np.inf
            elif stats == "negative-std" and suffix == "std":
                value[-1] = -1
            elif stats == "zero-std" and suffix == "std":
                value[0] = 0
            tensors[feature + "." + suffix] = value.copy()
            if stats == "mismatch":
                value[-1] += 1
            if prefix == "state" and vectors != 8:
                value = np.ones(vectors, dtype=np.float32)
            writer.add_tensor(prefix + "_" + suffix, value)
        # Additional feature statistics must not be rejected or loaded eagerly.
        tensors[feature + ".min"] = np.zeros(size, dtype=np.float32)
        if stats == "dtype":
            tensors[feature + ".mean"] = tensors[feature + ".mean"].astype(np.float64)
        save_file(tensors, bundle / "policy" / filename)
    if packed:
        values = gguf.quants.quantize(
            np.ones((32, 32), dtype=np.float32), gguf.GGMLQuantizationType.Q8_0
        )
        writer.add_tensor("vlm.blk.0.test.weight", values, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    writer.add_tensor("extra111", np.ones(1, dtype=np.float32))
    writer.add_tensor("extra222", np.ones(1, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    refresh(bundle, record, model)
    return bundle, record, model


def refresh(bundle, record, model):
    record["files"] = {name: sha256(bundle / name) for name in record["files"]}
    record["floating_gguf_sha256"] = sha256(model)
    atomic_json(bundle / "spatial-assets.json", record)


def check(bundle, model):
    return preflight.preflight(bundle, sha256(bundle / "spatial-assets.json"), model, sha256(model))


def snapshot(root):
    return {p.relative_to(root).as_posix(): sha256(p) for p in root.rglob("*") if p.is_file()}


def local_sources(tmp_path, monkeypatch, **kwargs):
    bundle, record, model = make_case(tmp_path, monkeypatch, **kwargs)
    sources = tmp_path / "snapshots"
    sources.mkdir()
    policy, backbone, fixtures = (
        sources / protocol.REVISION,
        sources / protocol.BACKBONE_REVISION,
        sources / "fixtures",
    )
    shutil.copytree(bundle / "policy", policy)
    shutil.copytree(bundle / "backbone", backbone)
    shutil.copytree(bundle / "fixtures", fixtures)
    return policy, backbone, fixtures, model


def test_valid_receipt_is_deterministic_static_only_and_leaves_inputs_unchanged(
    tmp_path, monkeypatch
):
    bundle, record, model = make_case(tmp_path, monkeypatch)
    before = snapshot(tmp_path)
    first = check(bundle, model)
    assert first == check(bundle, model)
    assert snapshot(tmp_path) == before
    assert first["schema_version"] == 1 and first["status"] == "static_assets_verified"
    assert first["inputs"]["files"] == record["files"]
    assert first["inputs"]["gguf_sha256"] == sha256(model)
    assert first["fixtures"][0]["sha256"] == record["files"]["fixtures/a.npz"]
    assert first["layout"]["real_state_dim"] == 8
    assert set(first["not_checked"]) >= {
        "gpu",
        "runtime",
        "episodes",
        "authorization",
        "action_parity",
        "model_load",
    }
    assert "not complete model weights" in first["scope"]
    assert first["validator"]["worker_lock_sha256"]
    assert "deployment_verified" not in canonical(first)


@pytest.mark.parametrize(
    "failure", ["model-pin", "bundle-pin", "binding", "weight-pin", "extra", "missing"]
)
def test_exact_identity_failures_are_rejected(tmp_path, monkeypatch, failure):
    bundle, record, model = make_case(tmp_path, monkeypatch)
    bundle_sha, model_sha = sha256(bundle / "spatial-assets.json"), sha256(model)
    if failure == "model-pin":
        model_sha = "0" * 64
    elif failure == "bundle-pin":
        bundle_sha = "0" * 64
    elif failure in {"binding", "weight-pin"}:
        if failure == "binding":
            record["floating_gguf_sha256"] = "0" * 64
        else:
            record["checkpoint"]["weights_sha256"] = "0" * 64
        atomic_json(bundle / "spatial-assets.json", record)
        bundle_sha = sha256(bundle / "spatial-assets.json")
    elif failure == "extra":
        (bundle / "fixtures/unlisted.npz").write_bytes(b"extra")
    else:
        (bundle / "backbone/tokenizer.json").unlink()
    with pytest.raises(ValueError):
        preflight.preflight(bundle, bundle_sha, model, model_sha)


@pytest.mark.parametrize(
    "dimension",
    [{"real_state_dim": 6}, {"max_action_dim": 7}, {"chunk_size": 49}, {"num_steps": 9}],
)
def test_incompatible_model_layout_is_rejected(tmp_path, monkeypatch, dimension):
    bundle, _, model = make_case(tmp_path, monkeypatch, dimension=dimension)
    with pytest.raises(ValueError, match="state/action contract"):
        check(bundle, model)


@pytest.mark.parametrize("stats", ["mismatch", "nan", "infinity", "negative-std", "dtype"])
def test_bad_statistics_are_rejected(tmp_path, monkeypatch, stats):
    bundle, _, model = make_case(tmp_path, monkeypatch, stats=stats)
    with pytest.raises(ValueError, match="normalization"):
        check(bundle, model)


def test_zero_std_is_preserved_for_processor_epsilon_semantics(tmp_path, monkeypatch):
    bundle, _, model = make_case(tmp_path, monkeypatch, stats="zero-std")
    assert check(bundle, model)["status"] == "static_assets_verified"


def test_packed_candidate_is_not_a_floating_source(tmp_path, monkeypatch):
    bundle, _, model = make_case(tmp_path, monkeypatch, packed=True)
    with pytest.raises(ValueError, match="master tensors"):
        check(bundle, model)


def test_duplicate_gguf_tensor_is_rejected(tmp_path, monkeypatch):
    bundle, record, model = make_case(tmp_path, monkeypatch)
    model.write_bytes(model.read_bytes().replace(b"extra222", b"extra111"))
    refresh(bundle, record, model)
    with pytest.raises(ValueError, match="duplicated tensor|Duplicate GGUF"):
        check(bundle, model)


@pytest.mark.parametrize("kind", ["shape", "file-size", "header-size", "missing", "duplicate-key"])
def test_normalizer_bounds_checked_before_native_tensor_reads(tmp_path, monkeypatch, kind):
    import safetensors.numpy

    bundle, record, model = make_case(tmp_path, monkeypatch, vectors=9 if kind == "shape" else 8)
    path = bundle / "policy" / validation.NORMALIZERS[0][0]
    if kind == "file-size":
        with path.open("wb") as stream:
            stream.truncate(validation.MAX_NORMALIZER_BYTES + 1)
    elif kind == "header-size":
        path.write_bytes((validation.MAX_NORMALIZER_HEADER_BYTES + 1).to_bytes(8, "little") + b"{}")
    elif kind == "missing":
        save_file({"unrelated": np.ones(1, dtype=np.float32)}, path)
    elif kind == "duplicate-key":
        header = b'{"observation.state.mean":{},"observation.state.mean":{}}'
        path.write_bytes(len(header).to_bytes(8, "little") + header)
    refresh(bundle, record, model)
    monkeypatch.setattr(
        safetensors.numpy,
        "load",
        lambda *a, **k: pytest.fail("Native tensor read reached before validation"),
    )
    with pytest.raises(ValueError):
        check(bundle, model)


@pytest.mark.parametrize(
    "kind", ["corrupt", "nan", "dtype", "shape", "pickle", "oversized", "duplicate"]
)
def test_fixture_failures_reject_static_preflight(tmp_path, monkeypatch, kind):
    bundle, record, model = make_case(tmp_path, monkeypatch)
    path = bundle / "fixtures/a.npz"
    values = raw_arrays()
    if kind == "corrupt":
        path.write_bytes(b"not an archive")
    elif kind == "duplicate":
        with (
            zipfile.ZipFile(path, "a") as archive,
            pytest.warns(UserWarning, match="Duplicate name"),
        ):
            archive.writestr("noise.npy", b"duplicate")
    else:
        if kind == "nan":
            values["noise"][0, 0, 0] = np.nan
        elif kind == "dtype":
            values["noise"] = values["noise"].astype(np.float64)
        elif kind == "shape":
            values["noise"] = values["noise"][:, :49]
        elif kind == "pickle":
            values["task"] = np.array([{"executable": "rejected"}], dtype=object)
        else:
            values["noise"] = np.zeros(17 * 1024 * 1024, dtype=np.uint8)
        np.savez_compressed(path, **values)
    refresh(bundle, record, model)
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        check(bundle, model)


def test_tiny_npy_claiming_huge_shape_fails_before_allocation(tmp_path, monkeypatch):
    bundle, record, model = make_case(tmp_path, monkeypatch)
    path = bundle / "fixtures/a.npz"
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    fake = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        fake, {"descr": "<f4", "fortran_order": False, "shape": (2**50,)}
    )
    members["noise.npy"] = fake.getvalue()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    refresh(bundle, record, model)
    monkeypatch.setattr(
        np, "load", lambda *a, **k: pytest.fail("Array allocation must not be reached")
    )
    with pytest.raises(ValueError, match="header shape"):
        check(bundle, model)


def test_no_network_subprocess_torch_cuda_or_model_import_is_needed(tmp_path, monkeypatch):
    bundle, _, model = make_case(tmp_path, monkeypatch)

    class BlockML(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in {
                "torch",
                "lerobot",
                "transformers",
                "huggingface_hub",
                "cuda",
            }:
                pytest.fail("Model import attempted: " + fullname)

    finder = BlockML()
    sys.meta_path.insert(0, finder)

    def forbidden(*args, **kwargs):
        pytest.fail("Network or subprocess execution attempted")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    try:
        assert check(bundle, model)["status"] == "static_assets_verified"
    finally:
        sys.meta_path.remove(finder)


def test_cli_emits_scoped_json_on_success_and_failure(tmp_path, monkeypatch, capsys):
    bundle, _, model = make_case(tmp_path, monkeypatch)
    args = [
        "--bundle",
        str(bundle),
        "--bundle-sha256",
        sha256(bundle / "spatial-assets.json"),
        "--gguf",
        str(model),
        "--gguf-sha256",
        sha256(model),
    ]
    assert preflight.main(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "static_assets_verified"
    args[-1] = "0" * 64
    assert preflight.main(args) == 1
    failure = json.loads(capsys.readouterr().out)
    assert failure["status"] == "failed" and failure["not_checked"] == preflight.NOT_CHECKED
    assert "static_assets_verified" not in canonical(failure)


def test_preparation_publishes_validated_bundle_without_changing_inputs(tmp_path, monkeypatch):
    inputs = local_sources(tmp_path, monkeypatch)
    before = snapshot(tmp_path / "snapshots")
    output = tmp_path / "published"
    record = protocol.prepare(*inputs, output)
    assert record == protocol.verify_assets(output, require_reference=True)
    assert check(output, inputs[-1])["status"] == "static_assets_verified"
    assert snapshot(tmp_path / "snapshots") == before
    assert not list(tmp_path.glob(".published-*.staging"))


@pytest.mark.parametrize("failure", ["copy", "validation", "write", "publish"])
def test_failed_preparation_cleans_only_owned_staging(tmp_path, monkeypatch, failure):
    inputs = local_sources(tmp_path, monkeypatch)
    before = snapshot(tmp_path / "snapshots")
    output = tmp_path / "published"
    other = tmp_path / ".published-other.staging"
    other.mkdir()
    (other / "sentinel").write_text("keep")
    if failure == "validation":
        (inputs[2] / "a.npz").write_bytes(b"bad fixture")
        before = snapshot(tmp_path / "snapshots")
    else:

        def fail(*args, **kwargs):
            raise OSError("injected " + failure)

        if failure == "copy":
            monkeypatch.setattr(protocol.shutil, "copyfile", fail)
        elif failure == "write":
            monkeypatch.setattr(protocol, "atomic_json", fail)
        else:
            monkeypatch.setattr(protocol, "publish_new_directory", fail)
    with pytest.raises((OSError, ValueError, zipfile.BadZipFile)):
        protocol.prepare(*inputs, output)
    assert not output.exists()
    assert list(tmp_path.glob(".published-*.staging")) == [other]
    assert (other / "sentinel").read_text() == "keep"
    assert snapshot(tmp_path / "snapshots") == before


@pytest.mark.parametrize("raced,empty", [(False, False), (True, False), (True, True)])
def test_existing_or_raced_destination_is_never_overwritten(tmp_path, monkeypatch, raced, empty):
    inputs = local_sources(tmp_path, monkeypatch)
    output = tmp_path / "published"
    identity = []

    def occupy():
        output.mkdir()
        identity.append(output.stat().st_ino)
        if not empty:
            (output / "sentinel").write_text("not ours")

    if raced:
        real = preflight.preflight

        def create_after_validation(*args, **kwargs):
            result = real(*args, **kwargs)
            occupy()
            return result

        monkeypatch.setattr(preflight, "preflight", create_after_validation)
    else:
        occupy()
    with pytest.raises(FileExistsError):
        protocol.prepare(*inputs, output)
    assert output.stat().st_ino == identity[0]
    assert sorted(x.name for x in output.iterdir()) == ([] if empty else ["sentinel"])
    assert not list(tmp_path.glob(".published-*.staging"))


def test_output_inside_source_is_rejected_without_mutation(tmp_path, monkeypatch):
    inputs = local_sources(tmp_path, monkeypatch)
    before = snapshot(tmp_path / "snapshots")
    with pytest.raises(ValueError, match="outside"):
        protocol.prepare(*inputs, inputs[0] / "output")
    assert snapshot(tmp_path / "snapshots") == before


def test_fixture_decoder_uses_checked_snapshot_after_path_replacement(tmp_path, monkeypatch):
    from policykit import spatial_fixture

    bundle, _, _ = make_case(tmp_path, monkeypatch)
    path = bundle / "fixtures/a.npz"
    original = sha256(path)
    real = spatial_fixture.validate_header

    def replace_after_header(*args, **kwargs):
        result = real(*args, **kwargs)
        path.write_bytes(b"replacement must never be decoded")
        return result

    monkeypatch.setattr(spatial_fixture, "validate_header", replace_after_header)
    assert load_arrays(path, expected_sha256=original)["noise"].shape == (1, 50, 32)
    with pytest.raises(ValueError, match="snapshot SHA256"):
        load_arrays(path, expected_sha256=original)


def test_normalizer_decoder_uses_checked_snapshot_after_path_replacement(tmp_path, monkeypatch):
    import safetensors.numpy

    bundle, record, model = make_case(tmp_path, monkeypatch)
    real = safetensors.numpy.load

    def replace_before_decode(payload):
        for filename, *_ in validation.NORMALIZERS:
            path = bundle / "policy" / filename
            if path.read_bytes() == payload:
                path.write_bytes(b"replacement must never be decoded")
        return real(payload)

    monkeypatch.setattr(safetensors.numpy, "load", replace_before_decode)
    assert (
        validation.validate_spatial_layout(gguf.GGUFReader(model), bundle, files=record["files"])[
            "real_state_dim"
        ]
        == 8
    )
    with pytest.raises(ValueError, match="snapshot SHA256"):
        validation.validate_spatial_layout(gguf.GGUFReader(model), bundle, files=record["files"])


def test_preflight_rejects_changes_during_fixture_decoding(tmp_path, monkeypatch):
    from policykit import spatial_fixture

    bundle, _, model = make_case(tmp_path, monkeypatch)
    real = spatial_fixture.load_arrays

    def mutate_after_decode(path, **kwargs):
        result = real(path, **kwargs)
        path.write_bytes(b"replaced after decoding")
        return result

    monkeypatch.setattr(spatial_fixture, "load_arrays", mutate_after_decode)
    with pytest.raises(ValueError):
        check(bundle, model)


def test_corrupt_gguf_fails_even_with_updated_hashes(tmp_path, monkeypatch):
    bundle, record, model = make_case(tmp_path, monkeypatch)
    with model.open("r+b") as stream:
        stream.write(b"BAD!")
    refresh(bundle, record, model)
    with pytest.raises(ValueError, match="magic"):
        check(bundle, model)


def test_compressed_fixture_bound_precedes_archive_decode(tmp_path, monkeypatch):
    path = tmp_path / "large.npz"
    with path.open("wb") as stream:
        stream.truncate(16 * 1024 * 1024 + 1)
    monkeypatch.setattr(
        zipfile, "ZipFile", lambda *a, **k: pytest.fail("Archive must not be opened")
    )
    with pytest.raises(ValueError, match="compressed file"):
        load_arrays(path)
