"""Full onedir identity for local packaging evidence, not a production activation pin."""

import argparse
import hashlib
import json
import os
import stat
import struct
from pathlib import Path
from typing import Any

MAX_ENTRIES = 20000
MAX_BYTES = 2 * 1024**3
MAX_MANIFEST = 8 * 1024**2


def native_architectures(header: bytes) -> list[int] | None:
    """Read Mach-O CPU identifiers without loading or executing a library."""
    thin = {
        b"\xce\xfa\xed\xfe": "<",
        b"\xcf\xfa\xed\xfe": "<",
        b"\xfe\xed\xfa\xce": ">",
        b"\xfe\xed\xfa\xcf": ">",
    }
    fat = {
        b"\xca\xfe\xba\xbe": (">", 20),
        b"\xbe\xba\xfe\xca": ("<", 20),
        b"\xca\xfe\xba\xbf": (">", 32),
        b"\xbf\xba\xfe\xca": ("<", 32),
    }
    magic = header[:4]
    if magic in thin:
        if len(header) < 8:
            raise ValueError("Truncated Mach-O header")
        return [struct.unpack(thin[magic] + "I", header[4:8])[0]]
    if magic in fat:
        if len(header) < 8:
            raise ValueError("Truncated fat Mach-O header")
        endian, stride = fat[magic]
        count = struct.unpack(endian + "I", header[4:8])[0]
        if not 1 <= count <= 16 or len(header) < 8 + count * stride:
            raise ValueError("Invalid or oversized fat Mach-O header")
        return [struct.unpack_from(endian + "I", header, 8 + n * stride)[0] for n in range(count)]
    return None


def inventory(root: Path) -> dict[str, Any]:
    """Hash every entry, preserving safe relative symlinks without traversing them.

    This checks a trusted same-user build directory; it is not a hostile-writer sandbox.
    """
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("Payload must be an absolute real directory")
    root = root.resolve()
    entries: dict[str, dict[str, Any]] = {}
    total = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in sorted([*dirs, *files]):
            path = Path(directory) / name
            key = path.relative_to(root).as_posix()
            if len(entries) >= MAX_ENTRIES:
                raise ValueError("Payload entry limit exceeded")
            info = path.lstat()
            mode = stat.S_IMODE(info.st_mode)
            if stat.S_ISLNK(info.st_mode):
                target = os.readlink(path)
                if not target or Path(target).is_absolute() or len(os.fsencode(target)) > 4096:
                    raise ValueError("Payload symlink must have a bounded relative target")
                try:
                    resolved = path.resolve(strict=True)
                except (OSError, RuntimeError) as exc:
                    raise ValueError("Payload symlink is dangling or cyclic") from exc
                if not resolved.is_relative_to(root) or not (
                    resolved.is_file() or resolved.is_dir()
                ):
                    raise ValueError("Payload symlink escapes or targets a special file")
                entries[key] = {"kind": "symlink", "target": target, "mode": mode}
            elif stat.S_ISDIR(info.st_mode):
                entries[key] = {"kind": "directory", "mode": mode}
            elif stat.S_ISREG(info.st_mode):
                total += info.st_size
                if total > MAX_BYTES:
                    raise ValueError("Payload byte limit exceeded")
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(fd, "rb") as stream:
                    opened = os.fstat(stream.fileno())
                    if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mode) != (
                        info.st_dev,
                        info.st_ino,
                        info.st_size,
                        info.st_mode,
                    ):
                        raise ValueError("Payload file changed before hashing")
                    digest, seen, header = hashlib.sha256(), 0, b""
                    while block := stream.read(min(65536, info.st_size + 1 - seen)):
                        if not header:
                            header = block[:520]
                        seen += len(block)
                        if seen > info.st_size:
                            raise ValueError("Payload file grew while hashing")
                        digest.update(block)
                    after = os.fstat(stream.fileno())
                    if seen != info.st_size or (after.st_mtime_ns, after.st_ctime_ns) != (
                        opened.st_mtime_ns,
                        opened.st_ctime_ns,
                    ):
                        raise ValueError("Payload file changed while hashing")
                entries[key] = {
                    "kind": "file",
                    "bytes": seen,
                    "sha256": digest.hexdigest(),
                    "mode": mode,
                    "macho_cpu_types": native_architectures(header),
                }
            else:
                raise ValueError("Payload contains a special file")
    if not entries:
        raise ValueError("Payload is empty")
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": 1,
        "entries": entries,
        "entry_count": len(entries),
        "file_bytes": total,
        "identity_sha256": hashlib.sha256(
            b"firebird-onedir-inventory-v1\0" + canonical
        ).hexdigest(),
    }


def verify(root: Path, expected: dict[str, Any]) -> None:
    """Reject any extra, missing, altered or differently typed recorded entry."""
    actual = inventory(root)
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(
        expected, sort_keys=True, allow_nan=False
    ):
        raise ValueError("Payload inventory changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.resolve().is_relative_to(
        args.payload.resolve()
    ):
        raise ValueError("Inventory output must be outside the payload")
    value = inventory(args.payload)
    raw = json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"
    if len(raw) > MAX_MANIFEST:
        raise ValueError("Inventory manifest is too large")
    with args.output.open("xb") as stream:
        stream.write(raw)


if __name__ == "__main__":
    main()
