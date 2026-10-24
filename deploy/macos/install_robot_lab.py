#!/usr/bin/env python3
"""Install the SO-101 companion using the shared macOS/Linux installer."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "install_robot_lab.py"), run_name="__main__"
    )
