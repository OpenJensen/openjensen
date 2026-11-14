"""Which code this service runs, file by file.

An application that cannot inspect or replace this service (another platform, a
tunnel, an installation made by hand) can still learn whether it runs the same code
as the application's own copy: it compares these fingerprints with its files. The
application computes them the same way; keep the two in step.
"""

import hashlib
from pathlib import Path


def source_files(root=None):
    """The SHA-256 of every code file, by its path below the service's folder."""
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    files = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if (
            path.is_symlink()
            or not path.is_file()
            or "tests" in relative.parts
            or "__pycache__" in relative.parts
            # Code and its pinned requirements; notes and saved copies do not change behaviour.
            or (path.suffix != ".py" and relative.as_posix() != "requirements-hardware.txt")
        ):
            continue
        files[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(files.items()))
