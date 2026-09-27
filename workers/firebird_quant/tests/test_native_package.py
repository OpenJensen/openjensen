"""Admission/publication regressions; stubbed probes are never runtime evidence."""

import copy
import hashlib
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

# Optional adapter reuses repository ACT validators; standalone quant needs no LeRobot install.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "act_optimizer" / "src"))
from firebird_quant import native_application as app
from firebird_quant import native_package as package


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    features = {
        "observation.state": {"type": "STATE", "shape": [6]},
        "observation.images.front": {"type": "VISUAL", "shape": [3, 32, 32]},
    }
    outputs = {"action": {"type": "ACTION", "shape": [6]}}
    config = {
        "type": "act",
        "n_obs_steps": 1,
        "n_action_steps": 100,
        "chunk_size": 100,
        "use_vae": False,
        "use_peft": False,
        "use_amp": False,
        "temporal_ensemble_coeff": None,
        "vision_backbone": "resnet18",
        "pre_norm": False,
        "replace_final_stride_with_dilation": False,
        "feedforward_activation": "relu",
        "input_features": features,
        "output_features": outputs,
        "normalization_mapping": {"VISUAL": "MEAN_STD", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
        "dim_model": 32,
        "n_heads": 4,
        "dim_feedforward": 64,
        "n_encoder_layers": 1,
        "n_decoder_layers": 1,
        "n_vae_encoder_layers": 1,
        "latent_dim": 8,
    }
    (root / "config.json").write_bytes(package.canonical(config))
    stats = {}
    for key, item in (features | outputs).items():
        shape = [3, 1, 1] if item["type"] == "VISUAL" else [6]
        stats[key + ".mean"] = torch.zeros(shape)
        stats[key + ".std"] = torch.ones(shape)
    save_file(stats, root / "pre.safetensors")
    save_file(
        {k: v for k, v in stats.items() if k.startswith("action.")}, root / "post.safetensors"
    )
    device = {"registry_name": "device_processor", "config": {"device": "cpu", "float_dtype": None}}
    pre = [
        {"registry_name": "rename_observations_processor", "config": {"rename_map": {}}},
        {"registry_name": "to_batch_processor", "config": {}},
        device,
        {
            "registry_name": "normalizer_processor",
            "config": {
                "eps": 1e-8,
                "features": features | outputs,
                "norm_map": config["normalization_mapping"],
            },
            "state_file": "pre.safetensors",
        },
    ]
    post = [
        {
            "registry_name": "unnormalizer_processor",
            "config": {
                "eps": 1e-8,
                "features": outputs,
                "norm_map": config["normalization_mapping"],
            },
            "state_file": "post.safetensors",
        },
        device,
    ]
    for name, steps in [("policy_preprocessor", pre), ("policy_postprocessor", post)]:
        (root / (name + ".json")).write_bytes(package.canonical({"name": name, "steps": steps}))
    save_file({"weight": torch.ones(2, 128)}, root / "model.safetensors")
    return root


def request(source, output):
    files = package.inventory(source)
    return {
        "schema_version": 1,
        "job_id": "test-job",
        "operation": "policy.quantize",
        "source": {
            "path": str(source),
            "files": files,
            "artifact_id": "test-artifact",
            "artifact_manifest_sha256": "a" * 64,
            "manifest_sha256": files.get("manifest.json", {}).get("sha256"),
        },
        "output_dir": str(output),
        "native_quantization": {"format": "firebird_quant", "bits": 8, "group_size": 64},
        "timeout_seconds": 30,
    }


@pytest.fixture
def stub_probes(monkeypatch):
    calls = []

    def probe(owner, mode, source, result, *, destination=None, bits=None):
        calls.append(mode)
        if mode == "convert":
            from safetensors.torch import load_file

            from firebird_quant import Recipe, quantize_state_dict

            destination.mkdir()
            for name in package.BASE | {"pre.safetensors", "post.safetensors"}:
                (destination / name).write_bytes((source / name).read_bytes())
            (destination / "encoding.json").write_bytes(package.canonical(package.encoding(bits)))
            quantize_state_dict(load_file(source / "model.safetensors"), Recipe(bits=bits)).save(
                destination / "model.fbq"
            )
            policy = destination
        else:
            policy = source
        info = package.inspect_policy(policy)
        rows = [
            {
                "seed": seed,
                "input_sha256": f"{seed:064x}",
                "image_shape": [3, 32, 32],
                "raw": [[0.25] * 6 for _ in range(100)],
                "postprocessed": [[0.5] * 6 for _ in range(100)],
                "queue_and_reset_exact": True,
            }
            for seed in (171, 902)
        ]
        report = {
            "schema_version": 1,
            "versions": package.RUNTIME,
            "model_id": info["model_id"],
            "policy_files": info["files"],
            "packed": rows,
            "network_disabled": True,
            "floating_master_reads_blocked": True,
        }
        if mode == "convert":
            report |= {"baseline": copy.deepcopy(rows), "source_files": package.inventory(source)}
            report["baseline"][0]["raw"][0][0] = 0.0
        return report

    monkeypatch.setattr(app, "_probe", probe)
    return calls


def test_complete_package_identity_lineage_and_no_float_master(source, tmp_path, stub_probes):
    before = package.inventory(source)
    result = app.run_job(request(source, tmp_path / "output"))
    root = Path(result["artifact"]["path"])
    doc = package.read_json(root / "manifest.json")
    assert doc["files"] == {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file() and p != root / "manifest.json"
    }
    assert not (root / "policy/model.safetensors").exists()
    assert result["report"]["model_id"] == package.inspect_policy(root / "policy")["model_id"]
    assert result["report"]["runtime_verified"] is False
    assert result["report"]["fresh_reload_verified"] is True
    assert result["report"]["quality_verified"] is False
    assert result["report"]["source_artifact_manifest_sha256"] == "a" * 64
    assert package.inventory(source) == before
    assert stub_probes == ["convert", "reload"]
    assert result["report"]["drift_from_fp32"][0]["raw"]["maximum_absolute_difference"] == 0.25


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 1.0),
        ("operation", "policy.run"),
        ("timeout_seconds", True),
        ("timeout_seconds", 0),
        ("timeout_seconds", 601),
        ("extra", 1),
    ],
)
def test_strict_request(source, tmp_path, stub_probes, field, value):
    job = request(source, tmp_path / "out")
    job[field] = value
    with pytest.raises(ValueError):
        app.run_job(job)
    assert not stub_probes


@pytest.mark.parametrize(
    "field,value",
    [
        ("bits", True),
        ("bits", 4.0),
        ("bits", 3),
        ("group_size", True),
        ("group_size", 32),
        ("format", "gguf"),
        ("extra", 1),
    ],
)
def test_strict_recipe(source, tmp_path, stub_probes, field, value):
    job = request(source, tmp_path / "out")
    job["native_quantization"][field] = value
    with pytest.raises(ValueError):
        app.run_job(job)
    assert not stub_probes


@pytest.mark.parametrize(
    "change", ["hash", "missing", "extra", "boolsize", "manifest", "family", "vae", "chunk", "stat"]
)
def test_source_rejection(source, tmp_path, stub_probes, change):
    job = request(source, tmp_path / "out")
    if change == "hash":
        job["source"]["files"]["config.json"]["sha256"] = "0" * 64
    elif change == "missing":
        del job["source"]["files"]["config.json"]
    elif change == "extra":
        (source / "unexpected.json").write_text("{}")
    elif change == "boolsize":
        job["source"]["files"]["config.json"]["bytes"] = True
    elif change == "manifest":
        job["source"]["manifest_sha256"] = "a" * 64
    elif change == "stat":
        save_file({"action.mean": torch.zeros(6)}, source / "post.safetensors")
        job["source"]["files"] = package.inventory(source)
    else:
        doc = package.read_json(source / "config.json")
        doc[{"family": "type", "vae": "use_vae", "chunk": "chunk_size"}[change]] = {
            "family": "smolvla",
            "vae": True,
            "chunk": 20,
        }[change]
        (source / "config.json").write_bytes(package.canonical(doc))
        job["source"]["files"] = package.inventory(source)
    with pytest.raises(ValueError):
        app.run_job(job)
    assert not stub_probes


def test_generic_manifest_source_is_supported(source, tmp_path, stub_probes):
    files = package.inventory(source)
    (source / "manifest.json").write_bytes(
        package.canonical(
            {"schema_version": 1, "files": {k: v["sha256"] for k, v in files.items()}}
        )
    )
    result = app.run_job(request(source, tmp_path / "out"))
    assert (Path(result["artifact"]["path"]) / "source-manifest.json").read_bytes() == (
        source / "manifest.json"
    ).read_bytes()


@pytest.mark.parametrize("where", ["root", "file", "ancestor", "output"])
def test_path_links_rejected(source, tmp_path, stub_probes, where):
    job = request(source, tmp_path / "out")
    if where == "root":
        link = tmp_path / "alias"
        link.symlink_to(source, target_is_directory=True)
        job["source"]["path"] = str(link)
    elif where == "file":
        (source / "model.safetensors").rename(tmp_path / "weights")
        (source / "model.safetensors").symlink_to(tmp_path / "weights")
    elif where == "ancestor":
        link = tmp_path / "alias"
        link.symlink_to(tmp_path, target_is_directory=True)
        job["source"]["path"] = str(link / "source")
    else:
        link = tmp_path / "alias"
        link.symlink_to(tmp_path, target_is_directory=True)
        job["output_dir"] = str(link / "out")
    with pytest.raises(ValueError):
        app.run_job(job)
    assert not stub_probes


def test_refuse_existing_and_nested_output(source, tmp_path, stub_probes):
    with pytest.raises(ValueError):
        app.run_job(request(source, source / "out"))
    (tmp_path / "out/native-quantized").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        app.run_job(request(source, tmp_path / "out"))
    assert not stub_probes


@pytest.mark.parametrize(
    "change",
    [
        "source",
        "packed",
        "replay",
        "version",
        "finite",
        "short",
        "queue",
        "input",
        "model",
        "shape",
    ],
)
def test_verification_failure_never_publishes(source, tmp_path, stub_probes, monkeypatch, change):
    original = app._probe

    def probe(*args, **kwargs):
        result = original(*args, **kwargs)
        if args[1] == "reload":
            if change == "source":
                (source / "config.json").write_text("{}")
            elif change == "packed":
                (args[2] / "model.fbq").write_bytes(b"changed")
            elif change == "version":
                result["versions"] = {**package.RUNTIME, "lerobot": "0.6.2"}
            elif change == "model":
                result["model_id"] = "sha256:" + "0" * 64
            elif change == "shape":
                result["packed"][0]["image_shape"] = [3, 10, 10]
            elif change == "finite":
                result["packed"][0]["raw"][0][0] = float("inf")
            elif change == "short":
                result["packed"][0]["raw"].pop()
            elif change == "queue":
                result["packed"][0]["queue_and_reset_exact"] = False
            elif change == "input":
                result["packed"][0]["input_sha256"] = "0" * 64
            else:
                result["packed"][0]["raw"][0][0] += 1
        return result

    monkeypatch.setattr(app, "_probe", probe)
    with pytest.raises(ValueError):
        app.run_job(request(source, tmp_path / "out"))
    assert not (tmp_path / "out/native-quantized").exists()
    assert not list((tmp_path / "out").glob(".native-quant-*"))


def test_raced_destination_not_replaced(source, tmp_path, stub_probes, monkeypatch):
    from firebird_act import bundle

    original = bundle.publish_new_directory

    def race(staging, destination):
        destination.mkdir()
        (destination / "sentinel").write_text("keep")
        original(staging, destination)

    monkeypatch.setattr(bundle, "publish_new_directory", race)
    with pytest.raises(OSError):
        app.run_job(request(source, tmp_path / "out"))
    assert (tmp_path / "out/native-quantized/sentinel").read_text() == "keep"


def test_strict_utf8_finite_duplicate_json():
    for value in [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e309}', b"{}\xff"]:
        with pytest.raises((ValueError, UnicodeError)):
            package.decode(value)


def test_model_identity_binds_each_inference_file():
    files = {"a": {"sha256": "a" * 64, "bytes": 1}, "b": {"sha256": "b" * 64, "bytes": 2}}
    initial = package.model_identity(files)
    assert initial == package.model_identity(dict(reversed(list(files.items()))))
    for field, value in [("sha256", "c" * 64), ("bytes", 3)]:
        changed = copy.deepcopy(files)
        changed["a"][field] = value
        assert initial != package.model_identity(changed)


def test_fifo_does_not_block(tmp_path):
    import os
    import time

    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    start = time.monotonic()
    with pytest.raises(ValueError):
        package.read(fifo)
    assert time.monotonic() - start < 1


@pytest.mark.parametrize("key", ["schema_version", "format_version"])
def test_encoding_bool_versions_rejected(source, tmp_path, stub_probes, key):
    result = app.run_job(request(source, tmp_path / "out"))
    policy = Path(result["artifact"]["path"]) / "policy"
    doc = package.read_json(policy / "encoding.json")
    doc[key] = True
    (policy / "encoding.json").write_bytes(package.canonical(doc))
    with pytest.raises(ValueError):
        package.inspect_policy(policy)


def test_no_symlink_race_through_ancestor(tmp_path, monkeypatch):
    import os

    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "data.json").write_text("{}")
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (replacement / "data.json").write_text('{"wrong":true}')
    original = os.open

    def raced(path, *args, **kwargs):
        if path == "parent":
            parent.rename(tmp_path / "old")
            parent.symlink_to(replacement, target_is_directory=True)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", raced)
    with pytest.raises(OSError):
        package.read(parent / "data.json")


def test_signal_during_publication_never_returns_success(
    source, tmp_path, stub_probes, monkeypatch
):
    import os
    import signal

    from firebird_act import bundle

    original = bundle.publish_new_directory

    def interrupted(staging, destination):
        original(staging, destination)
        os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(bundle, "publish_new_directory", interrupted)
    with pytest.raises(InterruptedError):
        app.run_job(request(source, tmp_path / "out"))
    # An intact artifact after the commit boundary remains unregistered, not a job success.
    assert (tmp_path / "out/native-quantized/manifest.json").exists()
