"""Feature installation plans stay isolated and never hide missing external runtimes."""

import json
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner
from vla_platform.install_cli import install
from vla_platform.install_profiles import (
    GIB,
    check_profile,
    environment_path,
    execute_install,
    install_plan,
    interpreter,
    profile,
    public_profiles,
    source_root,
)


@pytest.fixture
def checkout(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test-source'\n")
    (tmp_path / "workers/robot_lab").mkdir(parents=True)
    (tmp_path / "workers/_cpu_readers").mkdir(parents=True)
    (tmp_path / "workers/robot_lab/requirements-hardware.txt").write_text(
        "pyserial==3.5\nfeetech-servo-sdk==1.0.0\nopencv-python-headless==5.0.0.93\n"
        "numpy==2.5.4\nPillow==13.0.0\npyarrow==25.0.1\n"
    )
    (tmp_path / "workers/_cpu_readers/requirements.txt").write_text(
        "pyarrow==25.0.1\nnumpy==2.5.3\nh5py==3.16.0\nPillow==12.3.0\n"
    )
    return tmp_path


def test_catalog_explains_feature_and_python_boundaries():
    values = {item["id"]: item for item in public_profiles()}
    assert set(values) == {"core", "robot", "datasets", "training", "decision", "simulation"}
    assert values["robot"]["python"] == values["core"]["python"] == "3.14.7"
    assert values["training"]["python"] == "3.11"
    assert "torch" not in values["robot"]["distributions"]
    assert values["simulation"]["manual"]
    assert "NVIDIA" in values["simulation"]["description"]


def test_optional_plans_use_distinct_environments_and_pinned_sources(checkout):
    robot = install_plan(profile("robot"), checkout)
    datasets = install_plan(profile("datasets"), checkout)
    training = install_plan(profile("training"), checkout)
    assert robot[0]["argv"][:4] == ["uv", "venv", "--python", "3.14.7"]
    assert training[0]["argv"][:4] == ["uv", "venv", "--python", "3.11"]
    assert robot[0]["argv"][-1] != datasets[0]["argv"][-1] != training[0]["argv"][-1]
    assert robot[1]["argv"][-1].endswith("workers/robot_lab/requirements-hardware.txt")
    assert "--no-deps" in robot[1]["argv"] and "--no-deps" in datasets[1]["argv"]
    assert training[1]["argv"][-1].endswith("workers/smolvla_qlora[smolvla]")
    assert all("shell" not in command for command in robot + datasets + training)
    assert install_plan(profile("simulation"), checkout) == []


def test_dry_run_and_list_never_create_environments_or_run_commands(checkout, monkeypatch):
    app = typer.Typer()
    app.command()(install)
    before = sorted(path.relative_to(checkout) for path in checkout.rglob("*"))
    monkeypatch.setattr(
        "subprocess.run", lambda *args, **kwargs: pytest.fail("dry-run ran a process")
    )
    listing = CliRunner().invoke(app, ["list"])
    assert listing.exit_code == 0 and len(json.loads(listing.stdout)["profiles"]) == 6
    result = CliRunner().invoke(app, ["robot", "--dry-run", "--source", str(checkout)])
    assert result.exit_code == 0, result.output
    value = json.loads(result.stdout)
    assert value["profile"] == "robot" and len(value["commands"]) == 2
    assert sorted(path.relative_to(checkout) for path in checkout.rglob("*")) == before
    simulation = CliRunner().invoke(app, ["simulation", "--dry-run", "--source", str(checkout)])
    assert simulation.exit_code == 0
    assert (
        json.loads(simulation.stdout)["manual"] and json.loads(simulation.stdout)["commands"] == []
    )


def test_checks_read_only_metadata_and_report_pin_mismatch(checkout, monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    value = profile("robot")
    python = interpreter(environment_path(value, checkout))
    python.parent.mkdir(parents=True)
    python.touch()
    wanted = {
        "pyserial": "3.5",
        "feetech-servo-sdk": "1.0.0",
        "opencv-python-headless": "5.0.0.93",
        "numpy": "2.5.4",
        "Pillow": "13.0.0",
        "pyarrow": "25.0.1",
    }
    calls = []

    def metadata(args, **kwargs):
        calls.append((args, kwargs))
        assert args[1:3] == ["-I", "-c"] and "import torch" not in args[3]
        assert kwargs["timeout"] == 15 and kwargs["check"] is False
        return SimpleNamespace(
            returncode=0, stdout=json.dumps({"python": [3, 14, 7], "packages": wanted})
        )

    state = check_profile(value, checkout, runner=metadata, which=lambda _: "/bin/tool")
    assert state["ready"] and len(calls) == 1
    wanted["numpy"] = "old-version"
    assert not check_profile(value, checkout, runner=metadata, which=lambda _: "/bin/tool")["ready"]
    sim = check_profile(
        profile("simulation"),
        checkout,
        runner=lambda *a, **k: pytest.fail("Isaac check launched process"),
        which=lambda _: "/bin/tool",
    )
    assert not sim["ready"] and sim["manual"]


def test_incomplete_existing_environment_and_original_files_are_preserved(checkout, monkeypatch):
    value = profile("robot")
    target = environment_path(value, checkout)
    target.mkdir()
    original = target / "original-config"
    original.write_bytes(b"keep this environment")
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("shutil.which", lambda _: "/bin/tool")
    monkeypatch.setattr("shutil.disk_usage", lambda _: SimpleNamespace(free=100 * GIB))
    with pytest.raises(ValueError, match="existing environment was preserved"):
        execute_install(
            value,
            checkout,
            runner=lambda *a, **k: pytest.fail("Existing environment was overwritten"),
        )
    assert original.read_bytes() == b"keep this environment"


def test_low_disk_and_manual_simulation_start_no_install_commands(checkout, monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("shutil.which", lambda _: "/bin/tool")
    monkeypatch.setattr("shutil.disk_usage", lambda _: SimpleNamespace(free=10 * GIB))

    def no_process(*args, **kwargs):
        pytest.fail("A prohibited installation started")

    with pytest.raises(ValueError, match="10 GiB"):
        execute_install(profile("robot"), checkout, runner=no_process)
    with pytest.raises(ValueError, match="external Isaac"):
        execute_install(profile("simulation"), checkout, runner=no_process)
    assert not (checkout / "workers/robot_lab/.venv").exists()


def test_unknown_profiles_and_source_fail_without_side_effects(tmp_path):
    with pytest.raises(ValueError, match="Unknown profile"):
        profile("everything")
    with pytest.raises(ValueError, match="source checkout"):
        source_root(tmp_path)
    assert not list(tmp_path.iterdir())


def test_training_check_rejects_different_lerobot_and_torch_versions(checkout, monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    value = profile("training")
    directory = checkout / value.directory
    directory.mkdir()
    (directory / "pyproject.toml").write_text(
        '[project]\nname="jensen-vla"\nversion="0.1.0"\n'
        "[project.optional-dependencies]\nsmolvla=["
        '"lerobot[smolvla,peft]==0.4.4", "torch==2.7.1", '
        '"torchvision==0.22.1", "torchcodec==0.5.0; sys_platform == \'linux\'", '
        '"transformers==4.57.1", "peft==0.18.0", "bitsandbytes==0.48.2", '
        '"accelerate==1.10.1", "huggingface-hub==0.35.3", "safetensors==0.6.2"]\n'
    )
    python = interpreter(environment_path(value, checkout))
    python.parent.mkdir(parents=True)
    python.touch()
    packages = {
        "jensen-vla": "0.1.0",
        "lerobot": "0.4.4",
        "torch": "2.7.1",
        "torchvision": "0.22.1",
        "torchcodec": "0.5.0",
        "transformers": "4.57.1",
        "peft": "0.18.0",
        "bitsandbytes": "0.48.2",
        "accelerate": "1.10.1",
        "huggingface-hub": "0.35.3",
        "safetensors": "0.6.2",
    }

    def metadata(args, **kwargs):
        return SimpleNamespace(
            returncode=0, stdout=json.dumps({"python": [3, 11, 14], "packages": packages})
        )

    assert check_profile(value, checkout, runner=metadata, which=lambda _: "/bin/tool")["ready"]
    for package in ("lerobot", "torch"):
        saved = packages[package]
        packages[package] = "wrong-version"
        assert not check_profile(value, checkout, runner=metadata, which=lambda _: "/bin/tool")[
            "ready"
        ]
        packages[package] = saved
