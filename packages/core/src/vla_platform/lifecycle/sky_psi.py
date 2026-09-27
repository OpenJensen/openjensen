"""Fixed, isolated Psi-Zero dependency setup for cloud workers."""

import shutil
from pathlib import Path

WORKER_MODULE = "firebird_vla.psi_application"
UPSTREAM_REVISION = "4f3720d45e102b36d7c3e9465ab8062274170518"


def stage_psi(bundle: Path, training_root: Path) -> list[str]:
    """Append after creating Python 3.11 .venv and installing the OPEN JENSEN worker.

    The pinned upstream lock owns the ML environment. ``--inexact`` preserves
    the already installed OPEN JENSEN bridge and Google Cloud storage client.
    No upstream model weights are downloaded or staged on the application host.
    """
    for name in ("psi_application.py", "psi_train.py", "psi_verify.py", "psi_profile.py"):
        if not (training_root / "src" / "firebird_vla" / name).is_file():
            raise ValueError("The bundled Psi-Zero worker is missing on the application server")
    if not (bundle / "worker").is_dir():
        raise ValueError("Stage the shared OPEN JENSEN worker before Psi-Zero setup")
    shutil.copyfile(
        Path(__file__).with_name("psi_flash_install.py"), bundle / "psi_flash_install.py"
    )
    return [
        "sudo apt-get install -y build-essential ninja-build",
        "git init psi",
        "git -C psi fetch --depth 1 https://github.com/physical-superintelligence-lab/Psi0.git "
        + UPSTREAM_REVISION,
        "git -C psi checkout --detach FETCH_HEAD",
        'UV_PROJECT_ENVIRONMENT="$PWD/.venv" GIT_LFS_SKIP_SMUDGE=1 '
        "python3 -m uv sync --project psi --frozen --group psi --inexact "
        "--index-strategy unsafe-best-match",
        "python3 -m uv pip install --python .venv/bin/python ninja==1.11.1.4 wheel packaging",
        'export PATH="/usr/local/cuda/bin:$PATH"',
        ".venv/bin/python psi_flash_install.py",
        '.venv/bin/python -c "import psi, flash_attn; from psi.data.lerobot.compat import '
        "LEROBOT_LAYOUT; print('Psi-Zero environment ready:', LEROBOT_LAYOUT)\"",
    ]
