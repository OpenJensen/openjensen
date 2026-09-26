from pathlib import Path

import pytest
from vla_platform.lifecycle.psi_flash_install import wheel_url
from vla_platform.lifecycle.sky_psi import UPSTREAM_REVISION, WORKER_MODULE, stage_psi


def test_psi_setup_uses_its_pinned_environment_without_staging_weights(tmp_path):
    training_root = Path(__file__).resolve().parents[1] / "workers/smolvla_qlora"
    bundle = tmp_path / "bundle"
    (bundle / "worker").mkdir(parents=True)
    setup = "\n".join(stage_psi(bundle, training_root))
    assert WORKER_MODULE == "firebird_vla.psi_application"
    assert UPSTREAM_REVISION in setup
    assert "--project psi --frozen --group psi --inexact" in setup
    assert 'UV_PROJECT_ENVIRONMENT="$PWD/.venv"' in setup
    assert ".venv/bin/python psi_flash_install.py" in setup
    assert "command -v nvcc" not in setup
    assert "huggingface-cli download" not in setup
    assert {path.name for path in bundle.iterdir()} == {"worker", "psi_flash_install.py"}


@pytest.mark.parametrize("abi", [False, True])
def test_official_flash_wheel_selects_actual_torch_abi_and_verifies_hash(abi):
    url = wheel_url(
        python_version=(3, 11),
        system="Linux",
        machine="x86_64",
        torch_version="2.7.0+cu126",
        cuda_version="12.6",
        cxx11_abi=abi,
    )
    assert ("cxx11abiTRUE" if abi else "cxx11abiFALSE") in url
    assert "#sha256=" in url
    assert len(url.rsplit("#sha256=", 1)[1]) == 64


def test_flash_wheel_cannot_silently_install_on_incompatible_torch():
    assert (
        wheel_url(
            python_version=(3, 11),
            system="Linux",
            machine="x86_64",
            torch_version="2.8.0",
            cuda_version="12.8",
            cxx11_abi=True,
        )
        is None
    )


def test_psi_setup_refuses_missing_application_source(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        stage_psi(tmp_path, tmp_path / "missing")
