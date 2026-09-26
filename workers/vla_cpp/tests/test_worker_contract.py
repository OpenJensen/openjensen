"""Pack real small GGUFs through the standalone worker, without the app core."""

import hashlib
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import gguf
import numpy as np
import pytest
from policykit.quantization import quantize, tensor_quantization
from policykit.worker import code_hash


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@pytest.fixture
def job(tmp_path):
    source = tmp_path / "source.gguf"
    writer = gguf.GGUFWriter(source, "smolvla")
    for name in ("vlm.blk.0.attn_q.weight", "vit.blk.0.attn_q.weight",
                 "aex.blk.0.attn_q.weight", "mm.fc.weight", "token_embd.weight"):
        writer.add_tensor(name, np.arange(1024, dtype=np.float32).reshape(32, 32) / 1024)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    vendor = tmp_path / "vendor"
    (vendor / "scripts").mkdir(parents=True)
    script = vendor / "scripts/quantize_gguf.py"
    script.write_text("import numpy as np\nimport gguf\n"
                      "def to_f32(t):\n"
                      "    if t.tensor_type == gguf.GGMLQuantizationType.BF16:\n"
                      "        return gguf.quants.dequantize(t.data, t.tensor_type).reshape(t.shape[::-1])\n"
                      "    return np.asarray(t.data, dtype=np.float32).reshape(t.shape[::-1])\n")
    for command in (["git", "init", "-q"], ["git", "add", "scripts"],
                    ["git", "-c", "user.name=Test", "-c", "user.email=test@example.org",
                     "commit", "-qm", "synthetic F32 decoder"]):
        subprocess.run(command, cwd=vendor, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=vendor, text=True).strip()
    output = tmp_path / "run"
    output.mkdir()
    return {
        "schema_version": 1, "run_id": "synthetic", "attempt_id": "attempt-1",
        "output_dir": str(output),
        "recipe": {
            "schema_version": 1, "operation": "policy.quantize", "backend": "vla_cpp_smolvla",
            "policy": {"architecture": "smolvla", "task": "synthetic-test"},
            "source": {"path": str(source), "sha256": sha(source)},
            "runtime": {
                "vendor_path": str(vendor), "commit": commit,
                "patch_sha256": hashlib.sha256(b"").hexdigest(),
                "quantizer_sha256": sha(script), "worker_sha256": code_hash(),
                "dependencies": {name: version(name) for name in ("gguf", "numpy")},
            },
            "target": "synthetic-cpu-test", "language": "Q4_0", "vision": None,
        },
    }


def launch(job):
    job["recipe_sha256"] = hashlib.sha256(canonical(job["recipe"]).encode()).hexdigest()
    path = Path(job["output_dir"]) / "job.json"
    path.write_text(canonical(job))
    result = subprocess.run([sys.executable, "-m", "policykit.worker", "--job", str(path)],
                            capture_output=True, text=True, timeout=30)
    payload = json.loads(path.with_name("result.json").read_text())
    return result, payload, path.parent


def replace_master(job, tensor_types):
    source = Path(job["recipe"]["source"]["path"]).with_name("typed-master.gguf")
    writer = gguf.GGUFWriter(source, "smolvla")
    for name, dtype in tensor_types.items():
        data = np.arange(1024, dtype=np.float32).reshape(32, 32) / 1024
        if dtype == "BF16":
            writer.add_tensor(name, gguf.quants.quantize(data, gguf.GGMLQuantizationType.BF16),
                              raw_dtype=gguf.GGMLQuantizationType.BF16)
        else:
            writer.add_tensor(name, data.astype(np.float16 if dtype == "F16" else np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    job["recipe"]["source"] = {"path": str(source), "sha256": sha(source)}


@pytest.mark.parametrize("master_type", ["F32", "BF16"])
@pytest.mark.parametrize("language,vision", [("Q8_0", None), ("Q4_0", None),
                                            ("Q8_0", "Q8_0"), ("Q4_0", "Q8_0")])
def test_worker_preserves_protected_values_and_packs_requested_groups(job, language, vision,
                                                                   master_type):
    names = [t.name for t in gguf.GGUFReader(job["recipe"]["source"]["path"]).tensors]
    replace_master(job, dict.fromkeys(names, master_type))
    job["recipe"].update(language=language, vision=vision)
    result, payload, output = launch(job)
    assert result.returncode == 0, payload
    manifest = json.loads((output / "bundle/manifest.json").read_text())
    source = {t.name: t for t in gguf.GGUFReader(job["recipe"]["source"]["path"]).tensors}
    packed = {t.name: t for t in gguf.GGUFReader(output / "bundle/model.gguf").tensors}
    assert packed["vlm.blk.0.attn_q.weight"].tensor_type.name == language
    assert packed["vit.blk.0.attn_q.weight"].tensor_type.name == (vision or master_type)
    for name in manifest["protected_tensors"]:
        np.testing.assert_array_equal(packed[name].data, source[name].data)
    assert manifest["evidence_scope"] == "conversion_only"
    assert manifest["task_success"] is None and manifest["deployment_verified"] is False
    assert manifest["deployed_weight_bytes"] < sum(t.data.nbytes for t in source.values())
    assert not list(output.glob(".conversion-*"))


@pytest.mark.parametrize("f16_group", ["all", "language", "vision", "action", "projector"])
def test_f16_and_mixed_f16_masters_fail_before_conversion(job, f16_group):
    names = {"language": "vlm.blk.0.attn_q.weight", "vision": "vit.blk.0.attn_q.weight",
             "action": "aex.blk.0.attn_q.weight", "projector": "mm.fc.weight"}
    tensor_types = {name: "F16" if f16_group in {"all", group} else "F32"
                    for group, name in names.items()}
    replace_master(job, tensor_types)
    result, payload, output = launch(job)
    assert result.returncode != 0 and payload["status"] == "failed"
    assert "F32/BF16" in payload["error"] and "F16" in payload["error"]
    assert not (output / "bundle").exists()
    assert not list(output.glob(".conversion-*"))
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert [event["state"] for event in events] == ["preflight", "failed"]


@pytest.mark.parametrize("language_type", ["F32", "F16"])
def test_direct_quantizer_rejects_f16_before_creating_output(job, language_type):
    replace_master(job, {"vlm.blk.0.attn_q.weight": language_type,
                         "aex.blk.0.attn_q.weight": "F16"})
    destination = Path(job["output_dir"]) / "direct.gguf"
    recipe = job["recipe"]
    vendor_script = Path(recipe["runtime"]["vendor_path"]) / "scripts/quantize_gguf.py"
    with pytest.raises(ValueError, match="F32/BF16"):
        quantize(Path(recipe["source"]["path"]), destination, vendor_script, "Q4_0", None)
    assert not destination.exists()
    assert not destination.with_suffix(".audit.json").exists()


@pytest.mark.parametrize("change", ["source", "patch", "worker", "dependencies"])
def test_identity_mismatch_never_publishes_a_bundle(job, change):
    recipe = job["recipe"]
    if change == "source":
        Path(recipe["source"]["path"]).write_bytes(b"changed")
    elif change == "patch":
        (Path(recipe["runtime"]["vendor_path"]) / "scripts/quantize_gguf.py").write_text("# changed")
    elif change == "worker":
        recipe["runtime"]["worker_sha256"] = "0" * 64
    else:
        recipe["runtime"]["dependencies"]["numpy"] = "wrong"
    result, payload, output = launch(job)
    assert result.returncode != 0 and payload["status"] == "failed"
    assert not (output / "bundle").exists()
    assert not list(output.glob(".conversion-*"))


def test_action_and_projector_are_never_in_the_quantization_allowlist():
    for name in ("aex.blk.0.attn_q.weight", "action_out_proj.weight", "mm.fc.weight",
                 "token_embd.weight", "vlm.blk.0.attn_norm.weight"):
        assert tensor_quantization(name, [960, 960], "Q4_0", "Q8_0") is None
