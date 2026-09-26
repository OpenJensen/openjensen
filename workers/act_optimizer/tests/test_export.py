"""Behavioral export, integrity and no-replacement regressions."""

from pathlib import Path

import pytest

from firebird_act.export import export_policy


def test_existing_output_is_never_replaced(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(FileExistsError):
        export_policy(tmp_path / "missing-source", output)
    assert list(output.iterdir()) == []


def test_missing_source_does_not_publish(tmp_path: Path) -> None:
    output = tmp_path / "output"
    with pytest.raises((FileNotFoundError, ValueError)):
        export_policy(tmp_path / "missing-source", output)
    assert not output.exists()


@pytest.mark.runtime
def test_real_act_export_fresh_process_parity_and_source_preserved(
    act_source: Path, tmp_path: Path
) -> None:
    from firebird_act.bundle import inventory, read_json, tensor_header
    from firebird_act.export import run_probe

    before = inventory(act_source)
    output = tmp_path / "export"
    result = export_policy(act_source, output)
    assert inventory(act_source) == before
    original, original_start = tensor_header((act_source / "model.safetensors").read_bytes())
    exported, export_start = tensor_header((output / "model.safetensors").read_bytes())
    assert len(original) - len(exported) == 20  # 12 attention/FFN + 8 VAE-only tensors.
    original_bytes = (act_source / "model.safetensors").read_bytes()
    export_bytes = (output / "model.safetensors").read_bytes()
    for name in exported:
        if name == "__metadata__":
            continue
        start, stop = original[name]["data_offsets"]
        out_start, out_stop = exported[name]["data_offsets"]
        assert (
            original_bytes[original_start + start : original_start + stop]
            == export_bytes[export_start + out_start : export_start + out_stop]
        )
    assert read_json(output / "config.json") == read_json(act_source / "config.json") | {
        "use_vae": False
    }
    for name in before:
        if name not in {"config.json", "model.safetensors"}:
            assert (act_source / name).read_bytes() == (output / name).read_bytes()
    parity = read_json(output / "parity.json")
    assert parity["exact_equal"] is True
    assert parity["maximum_absolute_error"] == 0
    assert parity["fixture_count"] == 2
    assert all(f["queue_and_reset_exact"] for f in parity["fixtures"])
    assert parity["task_success"] is None
    assert result["export_model_bytes"] < result["source_model_bytes"]
    assert result["package_bytes"] == sum(p.stat().st_size for p in output.iterdir())
    # A published package reloads independently; the source is not a loader argument.
    final = run_probe(output, tmp_path / "final.json", 120)
    assert final["fixtures"][0]["chunk"]
