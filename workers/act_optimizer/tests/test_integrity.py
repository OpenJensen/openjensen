"""Invalid inputs, interrupted verification and concurrent publication fail closed."""

import json
import os
import shutil
import struct
import time
from pathlib import Path

import pytest

from firebird_act import bundle
from firebird_act.bundle import (
    canonical,
    decode,
    inventory,
    publish_new_directory,
    read_json,
    safe_file,
    strip_vae,
    tensor_header,
    validate_config,
    validate_processors,
)
from firebird_act.export import compare_probes, export_policy, run_probe


def test_duplicate_and_nonfinite_json_rejected() -> None:
    for payload in (b'{"x":1,"x":2}', b'{"x":NaN}', b"[]"):
        with pytest.raises(ValueError):
            decode(payload)


def small_tensor(header: dict, payload: bytes = b"\0\0\0\0") -> bytes:
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    return struct.pack("<Q", len(raw)) + raw + payload


@pytest.mark.parametrize(
    "change", ["dtype", "shape", "offset", "gap", "overlap", "extra", "metadata", "empty"]
)
def test_malformed_tensor_records_rejected(change: str) -> None:
    header = {"tensor": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    if change == "dtype":
        header["tensor"]["dtype"] = "F16"
    if change == "shape":
        header["tensor"]["shape"] = [True]
    if change == "offset":
        header["tensor"]["data_offsets"] = [0, 3]
    if change == "gap":
        header["tensor"]["data_offsets"] = [1, 5]
    if change == "overlap":
        header["extra"] = dict(header["tensor"])
    if change == "extra":
        header["tensor"]["unexpected"] = 1
    if change == "metadata":
        header["__metadata__"] = {"foo": 1}
    if change == "empty":
        header = {}
    with pytest.raises(ValueError):
        tensor_header(small_tensor(header))


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"1234",
        struct.pack("<Q", 10_000_000),
        struct.pack("<Q", 1) + b"{}",
        small_tensor({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}, b""),
    ],
)
def test_truncated_or_oversize_tensor_data_rejected(data: bytes) -> None:
    with pytest.raises(ValueError):
        tensor_header(data)


def test_nonsymlink_regular_bounded_files_only(tmp_path: Path) -> None:
    regular = tmp_path / "data"
    regular.write_bytes(b"hello")
    link = tmp_path / "link"
    link.symlink_to(regular)
    with pytest.raises(ValueError, match="Symlink"):
        safe_file(link)
    with pytest.raises(ValueError, match="size"):
        safe_file(regular, 4)
    with pytest.raises(ValueError):
        safe_file(tmp_path)
    with pytest.raises(ValueError):
        inventory(link)
    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "fifo"
        os.mkfifo(fifo)
        with pytest.raises(ValueError):
            safe_file(fifo)


@pytest.mark.parametrize(
    "key,value",
    [
        ("type", "smolvla"),
        ("use_vae", False),
        ("use_peft", True),
        ("chunk_size", 50),
        ("n_obs_steps", True),
        ("pre_norm", True),
        ("dim_model", 31),
        ("n_heads", 3),
        ("temporal_ensemble_coeff", 0.1),
    ],
)
def test_unsupported_act_configs_fail(act_source: Path, key: str, value: object) -> None:
    cfg = read_json(act_source / "config.json")
    cfg[key] = value
    with pytest.raises(ValueError):
        validate_config(cfg, source=True)


@pytest.mark.parametrize("mutation", ["traversal", "custom", "stats", "features", "extra"])
def test_processor_contract_rejects_untrusted_or_incompatible_steps(
    act_source: Path, tmp_path: Path, mutation: str
) -> None:
    source = tmp_path / "source"
    shutil.copytree(act_source, source)
    path = source / "policy_preprocessor.json"
    cfg = read_json(path)
    if mutation == "traversal":
        cfg["steps"][-1]["state_file"] = "../state.safetensors"
    if mutation == "custom":
        cfg["steps"][0]["registry_name"] = "external.Custom"
    if mutation == "stats":
        (source / cfg["steps"][-1]["state_file"]).write_bytes(b"invalid")
    if mutation == "features":
        cfg["steps"][-1]["config"]["norm_map"]["STATE"] = "IDENTITY"
    if mutation == "extra":
        cfg["steps"][0]["state_file"] = "ignored.safetensors"
    path.write_bytes(canonical(cfg))
    with pytest.raises(ValueError):
        validate_processors(source, read_json(source / "config.json"))


def test_extra_source_file_and_nested_output_rejected(act_source: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside"):
        export_policy(act_source, act_source / "nested")
    source = tmp_path / "source"
    shutil.copytree(act_source, source)
    (source / "unknown.bin").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="Unexpected"):
        export_policy(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("timeout", [0, 601, float("nan"), True])
def test_invalid_deadline_rejected(tmp_path: Path, timeout: float) -> None:
    with pytest.raises(ValueError, match="deadline"):
        export_policy(tmp_path / "source", tmp_path / "output", timeout=timeout)


def test_no_replace_directory_publication_wins_race(tmp_path: Path) -> None:
    candidate, output = tmp_path / "candidate", tmp_path / "output"
    candidate.mkdir()
    (candidate / "payload").write_text("new")
    output.mkdir()
    with pytest.raises(OSError):
        publish_new_directory(candidate, output)
    assert list(output.iterdir()) == []
    assert (candidate / "payload").read_text() == "new"
    output.rmdir()
    publish_new_directory(candidate, output)
    assert (output / "payload").read_text() == "new"


def test_unsupported_publication_os_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bundle.platform, "system", lambda: "UnsupportedOS")
    with pytest.raises(OSError, match="unavailable"):
        publish_new_directory(tmp_path / "candidate", tmp_path / "out")


def test_real_process_deadline_kills_verification_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    executable = tmp_path / "slow-python"
    marker = tmp_path / "finished"
    executable.write_text(
        f"#!{sys.executable}\nimport time\nfrom pathlib import Path\ntime.sleep(2)\n"
        f'Path({str(marker)!r}).write_text("should never happen")\n'
    )
    executable.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(executable))
    start = time.monotonic()
    with pytest.raises(ValueError, match="exceeded"):
        run_probe(tmp_path, tmp_path / "report.json", 0.2)
    assert 0.1 <= time.monotonic() - start < 1.5
    assert not marker.exists()


def valid_probe() -> dict:
    from firebird_act.probe import FIXTURE_SEEDS, VERSIONS

    return {
        "schema_version": 1,
        "device": "cpu",
        "dtype": "float32",
        "cpu_threads": 1,
        "network_disabled": True,
        "python": "3.12.14",
        "versions": dict(VERSIONS),
        "fixtures": [
            {
                "seed": seed,
                "input_sha256": "a" * 64,
                "queue_and_reset_exact": True,
                "chunk": [[0.0] * 6 for _ in range(100)],
                "postprocessed": [[0.0] * 6 for _ in range(100)],
            }
            for seed in FIXTURE_SEEDS
        ],
    }


def test_parity_mismatch_never_accepted() -> None:
    original, candidate = valid_probe(), valid_probe()
    candidate["fixtures"][0]["chunk"][0][0] = 1.0
    with pytest.raises(ValueError, match="parity"):
        compare_probes(original, candidate)
    candidate = valid_probe()
    candidate["python"] = "3.12.13"
    with pytest.raises(ValueError, match="runtimes"):
        compare_probes(original, candidate)


@pytest.mark.parametrize(
    "mutation", ["empty", "partial", "nan", "bool", "queue", "schema", "versions"]
)
def test_equal_but_invalid_probe_reports_fail_closed(mutation: str) -> None:
    report = valid_probe()
    if mutation == "empty":
        report["fixtures"] = []
    if mutation == "partial":
        report["fixtures"][0]["chunk"].pop()
    if mutation == "nan":
        report["fixtures"][0]["postprocessed"][0][0] = float("nan")
    if mutation == "bool":
        report["fixtures"][0]["chunk"][0][0] = True
    if mutation == "queue":
        report["fixtures"][0]["queue_and_reset_exact"] = False
    if mutation == "schema":
        report["schema_version"] = True
    if mutation == "versions":
        report["versions"] = {}
    with pytest.raises(ValueError):
        compare_probes(report, report)


def test_failed_probe_cleans_owned_staging_and_never_publishes(
    act_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from firebird_act import export

    before = inventory(act_source)

    def fail(*args: object) -> dict:
        raise ValueError("forced parity failure")

    monkeypatch.setattr(export, "run_probe", fail)
    with pytest.raises(ValueError, match="forced parity"):
        export_policy(act_source, tmp_path / "output")
    assert list(tmp_path.iterdir()) == []
    assert inventory(act_source) == before


def test_missing_vae_key_not_silently_dropped(act_source: Path, tmp_path: Path) -> None:
    from safetensors.torch import load_file, save_file

    tensors = load_file(act_source / "model.safetensors")
    tensors.pop("model.vae_encoder_pos_enc")
    path = tmp_path / "source.safetensors"
    save_file(tensors, path)
    with pytest.raises(ValueError, match="inventory"):
        strip_vae(path, tmp_path / "out.safetensors", read_json(act_source / "config.json"))


@pytest.mark.runtime
@pytest.mark.parametrize("mutation", ["missing", "extra", "shape", "nonfinite"])
def test_real_strict_loader_rejects_invalid_retained_weights(
    act_source: Path, tmp_path: Path, mutation: str
) -> None:
    import torch
    from safetensors.torch import load_file, save_file

    source = tmp_path / "source"
    shutil.copytree(act_source, source)
    tensors = load_file(source / "model.safetensors")
    name = "model.action_head.bias"
    if mutation == "missing":
        tensors.pop(name)
    if mutation == "extra":
        tensors["unexpected.weight"] = torch.ones(2)
    if mutation == "shape":
        tensors[name] = torch.ones(7)
    if mutation == "nonfinite":
        tensors[name][0] = float("nan")
    save_file(tensors, source / "model.safetensors")
    with pytest.raises(ValueError, match="verification failed"):
        export_policy(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".act-export-*"))


def test_network_attempt_is_rejected() -> None:
    from firebird_act.probe import offline_audit

    with pytest.raises(RuntimeError, match="Network"):
        offline_audit("socket.connect", ())
    offline_audit("open", ())


def test_wrong_runtime_version_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from firebird_act import probe

    monkeypatch.setattr(probe.importlib.metadata, "version", lambda _: "0.0.0")
    with pytest.raises(ValueError, match="requires"):
        probe.runtime_versions()


def test_source_mutation_during_verification_rejects_publication(
    act_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from firebird_act import export

    source = tmp_path / "source"
    shutil.copytree(act_source, source)
    original_probe = export.run_probe

    def mutate(checkpoint: Path, report: Path, timeout: float, **kwargs: object) -> dict:
        result = original_probe(checkpoint, report, timeout)
        (source / "train_config.json").write_text('{"changed":true}')
        return result

    monkeypatch.setattr(export, "run_probe", mutate)
    with pytest.raises(ValueError, match="changed during verification"):
        export_policy(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".act-export-*"))


@pytest.mark.runtime
def test_nonfinite_processor_statistics_fail_even_if_unused(
    act_source: Path, tmp_path: Path
) -> None:
    from safetensors.torch import load_file, save_file

    source = tmp_path / "source"
    shutil.copytree(act_source, source)
    state_path = source / read_json(source / "policy_preprocessor.json")["steps"][-1]["state_file"]
    state = load_file(state_path)
    next(iter(state.values())).flatten()[0] = float("inf")
    save_file(state, state_path)
    with pytest.raises(ValueError, match="Non-finite processor"):
        export_policy(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.runtime
def test_published_package_rejects_tampering(act_source: Path, tmp_path: Path) -> None:
    from firebird_act.bundle import verify_export

    output = tmp_path / "out"
    export_policy(act_source, output)
    manifest = read_json(output / "manifest.json")
    verify_export(output)
    manifest["schema_version"] = True
    (output / "manifest.json").write_bytes(canonical(manifest))
    with pytest.raises(ValueError, match="manifest"):
        verify_export(output)
    manifest["schema_version"] = 1
    (output / "manifest.json").write_bytes(canonical(manifest))
    path = output / "policy_postprocessor.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash verification"):
        verify_export(output)
    with pytest.raises(ValueError, match="hash verification"):
        run_probe(output, tmp_path / "bad-reload.json", 120)


def test_empty_directory_and_symlink_source_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "empty"
    directory.mkdir()
    with pytest.raises(ValueError, match="at most"):
        inventory(directory)
    link = tmp_path / "linked"
    link.symlink_to(directory, target_is_directory=True)
    with pytest.raises(ValueError, match="nonsymlink"):
        inventory(link)


def test_changed_file_length_detected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "data"
    path.write_bytes(b"abc")
    original = bundle.os.fstat

    def changed(fd: int) -> object:
        from types import SimpleNamespace

        info = original(fd)
        return SimpleNamespace(st_mode=info.st_mode, st_size=info.st_size + 1)

    monkeypatch.setattr(bundle.os, "fstat", changed)
    with pytest.raises(ValueError, match="changed while reading"):
        safe_file(path)


def test_final_verifier_cannot_open_original_source(tmp_path: Path) -> None:
    from firebird_act.probe import offline_audit

    with pytest.raises(RuntimeError, match="Source reads"):
        offline_audit("open", (str(tmp_path / "model.safetensors"), "r", 0), (tmp_path,))
    offline_audit("open", (str(tmp_path.parent / "other"), "r", 0), (tmp_path,))


@pytest.mark.parametrize(
    "processor,feature",
    [("policy_preprocessor.json", "observation.state"), ("policy_postprocessor.json", "action")],
)
def test_missing_feature_statistics_never_become_identity_normalization(
    act_source: Path, tmp_path: Path, processor: str, feature: str
) -> None:
    from safetensors.torch import load_file, save_file

    source = tmp_path / "source"
    shutil.copytree(act_source, source)
    document = read_json(source / processor)
    step = next(s for s in document["steps"] if "state_file" in s)
    path = source / step["state_file"]
    tensors = {
        name: tensor
        for name, tensor in load_file(path).items()
        if not name.startswith(feature + ".")
    }
    save_file(tensors, path)
    with pytest.raises(ValueError, match="statistics"):
        validate_processors(source, read_json(source / "config.json"))


@pytest.mark.parametrize(
    "mutation", ["shape", "negative_std", "count", "unknown", "range", "count_shape"]
)
def test_processor_statistics_shapes_and_semantics_are_checked(
    act_source: Path, tmp_path: Path, mutation: str
) -> None:
    import torch
    from safetensors.torch import load_file, save_file

    source = tmp_path / "source"
    shutil.copytree(act_source, source)
    path = source / read_json(source / "policy_preprocessor.json")["steps"][-1]["state_file"]
    tensors = load_file(path)
    if mutation == "shape":
        tensors["observation.state.mean"] = torch.ones(1)
    if mutation == "negative_std":
        tensors["action.std"][0] = -1
    if mutation == "count":
        tensors["action.count"] = torch.tensor([0.5])
    if mutation == "unknown":
        tensors["mystery.mean"] = torch.ones(1)
    if mutation == "range":
        tensors["action.min"] = torch.ones(6) * 2
        tensors["action.max"] = torch.ones(6)
    if mutation == "count_shape":
        tensors["action.count"] = torch.ones(6)
    save_file(tensors, path)
    with pytest.raises(ValueError, match="statistics"):
        validate_processors(source, read_json(source / "config.json"))


@pytest.mark.parametrize("existing", [False, True])
def test_probe_receipt_cannot_mutate_package_or_replace_file(
    tmp_path: Path, existing: bool
) -> None:
    import subprocess
    import sys

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    report = tmp_path / "receipt.json" if existing else checkpoint / "receipt.json"
    if existing:
        report.write_text("preserve me")
    result = subprocess.run(
        [sys.executable, "-m", "firebird_act.probe", str(checkpoint), str(report)],
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert result.returncode != 0
    assert "must be" in result.stderr
    assert list(checkpoint.iterdir()) == []
    if existing:
        assert report.read_text() == "preserve me"


def test_auxiliary_quantile_roundoff_does_not_reject_valid_mean_std(
    act_source: Path, tmp_path: Path
) -> None:
    import torch
    from safetensors.torch import load_file, save_file

    source = tmp_path / "source"
    shutil.copytree(act_source, source)
    path = source / read_json(source / "policy_preprocessor.json")["steps"][-1]["state_file"]
    tensors = load_file(path)
    # Observed in the supplied checkpoint's unused task-index histogram statistic.
    tensors["task_index.max"] = torch.tensor([0.0])
    tensors["task_index.q99"] = torch.tensor([3.96e-14])
    save_file(tensors, path)
    validate_processors(source, read_json(source / "config.json"))


@pytest.mark.parametrize("shape", [[3, 1080, 1920], [3, 1920, 1080], [3, 1440, 1440]])
def test_full_hd_image_admission_preserves_exact_shape(act_source: Path, shape: list[int]) -> None:
    cfg = read_json(act_source / "config.json")
    cfg["input_features"]["observation.images.front"]["shape"] = shape.copy()
    validate_config(cfg, source=True)
    assert cfg["input_features"]["observation.images.front"]["shape"] == shape


@pytest.mark.parametrize(
    "shape",
    [
        [3, 1921, 1080],
        [3, 2048, 1024],
        [3, 2049, 32],
        [3, 32, 2049],
        [3, 31, 1920],
        [3, 1080.0, 1920],
        [3, True, 1920],
        [3.0, 1080, 1920],
    ],
)
def test_image_admission_rejects_excess_area_dimensions_and_nonintegers(
    act_source: Path, shape: list[object]
) -> None:
    cfg = read_json(act_source / "config.json")
    cfg["input_features"]["observation.images.front"]["shape"] = shape
    with pytest.raises(ValueError, match="RGB image shape"):
        validate_config(cfg, source=True)


@pytest.mark.parametrize("prediction,execution", [(8, 3), (1, 1), (1024, 7)])
def test_independent_horizons_are_bounded(prediction, execution):
    assert bundle.temporal_dimensions({"chunk_size": prediction, "n_action_steps": execution}) == {
        "prediction_horizon": prediction,
        "execution_horizon": execution,
    }


@pytest.mark.parametrize(
    "prediction,execution", [(8, 9), (True, 1), (8, True), (8.0, 3), (1025, 1), (8, 0)]
)
def test_invalid_horizons_fail_before_runtime(prediction, execution):
    with pytest.raises(ValueError):
        bundle.temporal_dimensions({"chunk_size": prediction, "n_action_steps": execution})
