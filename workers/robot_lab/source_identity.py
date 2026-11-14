"""One fingerprint of this service's own code.

An application that cannot inspect or replace this service (another platform, a
tunnel, an installation made by hand) can still learn whether it runs the same code
as the application's own copy. The application computes the same value over its
copy; keep the two in step.
"""

import hashlib
from pathlib import Path


def source_identity(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    digest = hashlib.sha256()
    names = []
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
        names.append(relative.as_posix())
    for name in sorted(names):
        digest.update(name.encode() + b"\0")
        digest.update(hashlib.sha256((root / name).read_bytes()).digest())
    return digest.hexdigest()
