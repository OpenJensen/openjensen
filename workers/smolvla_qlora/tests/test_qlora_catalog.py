import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from firebird_vla.checkpoint import sha256, write_json
from firebird_vla.checks.catalog import PROFILES, load_catalog, result_row, selected_module
from firebird_vla.checks.cli import run_entry, runtime_blockers, summarize_phases
from firebird_vla.checks.fixtures import validate_fixture
from firebird_vla.checks.native import checked_hf_load, validate_conversion

CATALOG = Path(__file__).resolve().parents[1] / "configs/benchmarks/open_weight_vlas.json"


def entry(model_id):
    return next(m for m in load_catalog(CATALOG)["models"] if m["id"] == model_id)


def fixture(model_id="smolvla"):
    return {
        "schema_version": 1,
        "kind": "native_training_batch",
        "model_id": model_id,
        "checkpoint": entry(model_id)["checkpoint"],
        "suite": "libero_spatial",
        "provenance": {
            "source": "test-fixture",
            "kind": "synthetic",
            "split": "diagnostic",
            "native_code_commit": PROFILES[model_id]["code_commit"],
            "processor_description": "test only",
            "embodiment": "LIBERO_PANDA",
        },
    }


def test_catalog_uses_all_six_libero_checkpoints_not_so101_base():
    catalog = load_catalog(CATALOG)
    assert {e["id"] for e in catalog["models"]} == set(PROFILES)
    assert entry("smolvla")["checkpoint"]["source"] == "lerobot/smolvla_libero"
    assert entry("gr00t_n17")["checkpoint"]["subdirectory"] == "libero_spatial"
    assert all(not result_row(e)["qlora_verified"] for e in catalog["models"])


@pytest.mark.parametrize("model_id", list(PROFILES))
def test_each_profile_targets_its_own_backbone_only(model_id):
    root = PROFILES[model_id]["roots"][0]
    assert selected_module(root + "0.self_attn.q_proj", model_id)
    assert selected_module(root + "0.mlp.down_proj", model_id)
    for protected in (
        "vision_model.encoder.layers.0.self_attn.q_proj",
        "action_head.q_proj",
        "model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj",
        "model.paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj",
        root + "0.input_layernorm",
        root + "0.lm_head",
    ):
        assert not selected_module(protected, model_id)


@pytest.mark.parametrize(
    "field,value",
    [
        ("suite", "so101"),
        ("model_id", "openvla"),
        ("kind", "raw_observation"),
        ("checkpoint", {"source": "lerobot/smolvla_base", "revision": "main"}),
    ],
)
def test_cross_model_or_embodiment_substitution_rejected(field, value):
    manifest = fixture()
    manifest[field] = value
    with pytest.raises(ValueError):
        validate_fixture(manifest, entry("smolvla"))


@pytest.mark.parametrize("split", ["search", "final", "test", None])
def test_no_training_on_search_or_final_fixture(split):
    manifest = fixture()
    manifest["provenance"]["split"] = split
    with pytest.raises(ValueError, match="must not train"):
        validate_fixture(manifest, entry("smolvla"))


def test_gr00t_requires_native_embodiment():
    manifest = fixture("gr00t_n17")
    validate_fixture(manifest, entry("gr00t_n17"))
    manifest["provenance"]["embodiment"] = "SO101"
    with pytest.raises(ValueError, match="LIBERO_PANDA"):
        validate_fixture(manifest, entry("gr00t_n17"))


@pytest.mark.parametrize(
    "bad_key", ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]
)
def test_native_load_never_silently_accepts_random_or_missing_weights(bad_key):
    class NativeClass:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert kwargs["output_loading_info"]
            return object(), {bad_key: ["unloaded action head"]}

    with pytest.raises(ValueError, match="Incomplete pretrained"):
        checked_hf_load(NativeClass, "/fixture")


def test_conversion_needs_matching_source_equivalence_and_unchanged_assets(tmp_path):
    model = entry("pi05")
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"test-not-weights")
    manifest = {
        "source": model["checkpoint"]["source"],
        "float_action_equivalence_passed": True,
        "source_hashes": {"params/file": "0" * 64},
        "conversion_commit": PROFILES["pi05"]["code_commit"],
        "output_hashes": {"model.safetensors": sha256(weights)},
    }
    validate_conversion(tmp_path, manifest, model)
    for field, value in [
        ("float_action_equivalence_passed", False),
        ("source", "gs://openpi-assets/checkpoints/pi05_base"),
        ("source_hashes", {}),
    ]:
        changed = copy.deepcopy(manifest)
        changed[field] = value
        with pytest.raises(ValueError):
            validate_conversion(tmp_path, changed, model)
    weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_conversion(tmp_path, manifest, model)


@pytest.mark.parametrize("failure", ["oom", "failed", "timeout", "blocked"])
def test_failure_and_unmeasured_metrics_never_become_success(failure):
    row = summarize_phases(
        result_row(entry("openvla")),
        {
            "float": {"status": "passed", "native_loss": 1.0},
            "qlora": {"status": failure, "reason": "fixture failure"},
            "reload": {"status": "blocked"},
        },
    )
    assert row["status"] == failure
    assert row["metrics"]["task_success"] is None
    assert row["metrics"]["nf4_loss_after"] is None
    for state in (
        "qlora_verified",
        "quantization_verified",
        "closed_loop_verified",
        "export_verified",
    ):
        assert row[state] is False


def test_successful_gradient_reload_is_not_a_closed_loop_or_export_score():
    row = summarize_phases(
        result_row(entry("smolvla")),
        {phase: {"status": "passed"} for phase in ("float", "qlora", "reload")},
    )
    assert row["qlora_verified"] is True
    assert row["closed_loop_verified"] is False
    assert row["export_verified"] is False
    assert row["metrics"]["task_success"] is None


def test_reference_oom_remains_visible_even_if_qlora_passes():
    row = summarize_phases(
        result_row(entry("openvla")),
        {
            "float": {"status": "oom"},
            "qlora": {"status": "passed"},
            "reload": {"status": "passed"},
        },
    )
    assert row["status"] == "incomplete_float_reference"
    assert row["metrics"]["native_float_loss"] is None


def test_cli_keeps_unselected_models_pending_without_ml_imports(tmp_path):
    output = tmp_path / "plan"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "firebird_vla.checks.cli",
            "--catalog",
            str(CATALOG),
            "--models",
            "smolvla",
            "--output-dir",
            str(output),
        ],
        check=True,
    )
    rows = json.loads((output / "report.json").read_text())["rows"]
    assert len(rows) == 6
    assert all(r["status"] == "pending" and not r["qlora_verified"] for r in rows)


def test_missing_runtimes_produce_six_blocked_rows_and_failure_exit(tmp_path):
    output = tmp_path / "checks"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "firebird_vla.checks.cli",
            "--catalog",
            str(CATALOG),
            "--run",
            "--output-dir",
            str(output),
        ],
        check=False,
    )
    assert result.returncode == 1
    rows = json.loads((output / "report.json").read_text())["rows"]
    assert len(rows) == 6 and all(r["status"] == "blocked" for r in rows)


def test_runtime_preflight_rejects_fixture_for_another_checkpoint(tmp_path):
    write_json(tmp_path / "fixture.json", fixture("openvla"))
    lock = tmp_path / "lock.txt"
    lock.write_text("torch==2.7.1")
    reasons = runtime_blockers(
        entry("smolvla"),
        {
            "python": sys.executable,
            "fixture": str(tmp_path),
            "dependency_lock": str(lock),
        },
    )
    assert reasons and "another model" in reasons[0]


@pytest.mark.parametrize("mode", ["nonzero", "timeout"])
def test_real_subprocess_failures_are_retained_and_reload_is_not_launched(tmp_path, mode):
    # A real tiny executable exercises supervision; its output is explicitly a test fixture.
    executable = tmp_path / "fake-native-python"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys, time\n"
        "job=json.loads(pathlib.Path(sys.argv[-2]).read_text())\n"
        "phase=sys.argv[-1]\n"
        + ("time.sleep(2)\n" if mode == "timeout" else "")
        + "pathlib.Path(job['output'], phase+'.json').write_text(json.dumps({'status':'passed'}))\n"
        "sys.exit(3 if phase == 'qlora' else 0)\n"
    )
    executable.chmod(0o755)
    lock = tmp_path / "lock.txt"
    lock.write_text("test fixture")
    args = SimpleNamespace(
        steps=2,
        rank=8,
        learning_rate=1e-4,
        seed=42,
        timeout_seconds=0.05 if mode == "timeout" else 5,
    )
    row = run_entry(
        entry("smolvla"),
        {"python": str(executable), "dependency_lock": str(lock)},
        tmp_path / "run",
        args,
    )
    assert row["status"] == ("timeout" if mode == "timeout" else "failed")
    assert row["phases"]["reload"]["status"] == "blocked"
    assert row["qlora_verified"] is False
    assert (tmp_path / "run" / "qlora.log").exists()
    assert not (tmp_path / "run" / "reload.log").exists()
    assert json.loads((tmp_path / "run" / "reload.json").read_text())["status"] == "blocked"
