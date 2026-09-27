"""Read a configured control token without following links or opening special files."""

import os
import stat
from pathlib import Path


def control_token():
    filename = os.environ.get("FIREBIRD_TEACHING_CONTROL_TOKEN_FILE")
    direct = os.environ.get("FIREBIRD_TEACHING_CONTROL_TOKEN")
    if filename and direct:
        raise ValueError("Configure either control token file or token, not both")
    if filename:
        path = Path(filename)
        if path.is_symlink():
            raise ValueError("Control token file must not be a symlink")
        fd = os.open(
            path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > 4096:
                raise ValueError("Control token must be a bounded regular file")
            if os.name == "posix" and info.st_mode & 0o077:
                raise ValueError("Control token file must be private to its owner")
            value = os.read(fd, 4097).decode().strip()
        finally:
            os.close(fd)
    else:
        value = (direct or "").strip()
    if not 32 <= len(value) <= 4096 or not value.isascii() or any(c.isspace() for c in value):
        raise ValueError("Control token must contain32..4096 nonspace ASCII characters")
    return value
