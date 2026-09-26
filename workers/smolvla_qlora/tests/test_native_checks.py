"""CPU contract checks for native loaders and initialization; no model downloads or CUDA."""

import random
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest

from firebird_vla.checkpoint import sha256
from firebird_vla.checks import fixtures, native, worker


def test_gr00t_keeps_dispatch_identifier_and_pins_backbone_and_processor(monkeypatch, tmp_path):
    backbone = "nvidia/Cosmos-Reason2-2B"
    revision = "a" * 40
    snapshot_path = str(tmp_path / "models--nvidia--Cosmos-Reason2-2B/snapshots" / revision)
    events = []
    config = SimpleNamespace(model_name=backbone)

    def prefetch(repo, **kwargs):
        assert (repo, kwargs) == (backbone, {"revision": revision})
        events.append("prefetch")
        return snapshot_path

    def load_config(path):
        assert path == tmp_path
        return config

    def load_backbone_or_processor(repo, **kwargs):
        assert events[0] == "prefetch"
        assert repo == backbone
        assert kwargs == {
            "revision": revision,
            "local_files_only": True,
            "trust_remote_code": False,
        }
        events.append("nested_load")

    class Gr00tConstructor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert path == str(tmp_path)
            assert kwargs["output_loading_info"] is True
            cfg = kwargs["config"]
            # Native get_backbone_cls at GR00T commit 51d4c89 dispatches on these names.
            if (
                "nvidia/Cosmos-Reason2" not in cfg.model_name
                and "Qwen/Qwen3-VL" not in cfg.model_name
            ):
                raise ValueError(f"Unsupported model name: {cfg.model_name}")
            assert cfg.tune_llm is True and cfg.use_flash_attention is False
            # Its constructor forwards the same kwargs to Qwen3Backbone and DataCollator.
            load_backbone_or_processor(cfg.model_name, **kwargs["transformers_loading_kwargs"])
            load_backbone_or_processor(cfg.model_name, **kwargs["transformers_loading_kwargs"])
            return (lambda batch: {"loss": batch["loss"]}), {}

    modules = {
        "gr00t": {},
        "gr00t.model": {},
        "gr00t.configs": {},
        "gr00t.configs.model": {},
        "gr00t.configs.model.gr00t_n1d7": {
            "Gr00tN1d7Config": SimpleNamespace(from_pretrained=load_config),
        },
        "gr00t.model.gr00t_n1d7": {},
        "gr00t.model.gr00t_n1d7.gr00t_n1d7": {"Gr00tN1d7": Gr00tConstructor},
        "huggingface_hub": {"snapshot_download": prefetch},
        "torch": {"bfloat16": "bfloat16"},
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(native, "snapshot", lambda entry: tmp_path)

    model, loss, path = native.load_gr00t({}, {"backbone_revision": revision})
    assert path == tmp_path
    assert config.model_name == backbone
    assert loss(model, {"loss": 7}) == 7
    assert events == ["prefetch", "nested_load", "nested_load"]


@pytest.fixture
def rng_stubs(monkeypatch):
    # Independent generators stand in for optional NumPy/Torch; exercise ordering on CPU.
    numpy_rng, torch_rng = random.Random(), random.Random()
    state = random.getstate()
    torch = SimpleNamespace(
        manual_seed=torch_rng.seed,
        cuda=SimpleNamespace(
            is_available=lambda: True,
            is_bf16_supported=lambda: True,
            set_device=lambda _: None,
            reset_peak_memory_stats=lambda: None,
            get_device_name=lambda: "CPU test stub, not CUDA evidence",
        ),
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace()), cudnn=SimpleNamespace()
        ),
        version=SimpleNamespace(cuda=None),
        nn=SimpleNamespace(Linear=type("Linear", (), {})),
        Tensor=type("Tensor", (), {}),
        bfloat16="bfloat16",
        set_grad_enabled=lambda _: nullcontext(),
        autocast=lambda *args, **kwargs: nullcontext(),
        isfinite=lambda _: True,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace(random=numpy_rng))
    yield lambda: (random.random(), numpy_rng.random(), torch_rng.random())
    random.setstate(state)


def test_phase_seeds_model_and_adapter_initialization(monkeypatch, tmp_path, rng_stubs):
    lock = tmp_path / "lock.txt"
    lock.write_text("CPU test fixture")
    (tmp_path / "fixture.json").write_text("{}")
    job = {
        "entry": {"id": "smolvla"},
        "runtime": {"dependency_lock": str(lock), "fixture": str(tmp_path)},
        "dependency_lock_sha256": sha256(lock),
        "output": str(tmp_path),
        "seed": 42,
        "rank": 8,
    }
    initialization = []

    class AdapterReached(Exception):
        pass

    def load_model(entry, runtime):
        initialization.append(("model", rng_stubs()))
        return SimpleNamespace(named_modules=lambda: []), None, tmp_path

    def initialize_adapter(*args):
        initialization.append(("adapter", rng_stubs()))
        raise AdapterReached

    monkeypatch.setattr(worker, "validate_runtime", lambda *args: None)
    monkeypatch.setattr(worker, "distributions", lambda: [])
    monkeypatch.setattr(
        fixtures,
        "load_fixture",
        lambda *args: (
            {},
            {"provenance": {"kind": "synthetic", "native_code_commit": "test"}},
        ),
    )
    monkeypatch.setitem(worker.PROFILES, "smolvla", {"loader": "smolvla", "code_commit": None})
    monkeypatch.setitem(native.LOADERS, "smolvla", load_model)
    monkeypatch.setattr(worker, "pack_and_adapt", initialize_adapter)

    def run(seed):
        initialization.clear()
        job["seed"] = seed
        with pytest.raises(AdapterReached):
            worker.run_phase(job, "qlora")
        assert [stage for stage, _ in initialization] == ["model", "adapter"]
        return list(initialization)

    first = run(42)
    for _ in range(13):
        rng_stubs()  # Simulate unrelated RNG use between separate phases/processes.
    assert run(42) == first
    different_seed = run(43)
    for (_, before), (_, after) in zip(first, different_seed, strict=True):
        assert all(a != b for a, b in zip(before, after, strict=True))


def test_native_loss_noise_is_reseeded_for_comparisons(rng_stubs):
    samples = []

    class Loss(float):
        ndim = 0

        def detach(self):
            return self

    def loss_fn(model, batch):
        noise = rng_stubs()
        samples.append(noise)
        return Loss(sum(noise))

    first = worker.check_loss(loss_fn, None, {}, seed=42)
    rng_stubs()
    assert worker.check_loss(loss_fn, None, {}, seed=42) == first
    assert samples[0] == samples[1]
    worker.check_loss(loss_fn, None, {}, seed=43)
    assert all(a != b for a, b in zip(samples[0], samples[2], strict=True))
