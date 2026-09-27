"""CPU-only, bounded ACT/SmolVLA package admission into a private snapshot.

This verifies storage integrity and declared compatibility, not a complete model
load, task quality, calibration, or trust in imported training/parity claims.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import struct
import sys
import tarfile
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from .checkpoint import Checkpoint, inspect_checkpoint

MAX_FILES = 256
MAX_EXPANDED = 4 * 1024**3
MAX_FILE = 2 * 1024**3
MAX_JSON = 2 * 1024**2
MAX_HEADER = 16 * 1024**2
MAX_PATH = 512
MAX_DEPTH = 8
BLOCK = 1024**2
_SHA = re.compile(r"[0-9a-f]{64}")
_PROCESSORS = {
    "rename_observations_processor",
    "to_batch_processor",
    "device_processor",
    "normalizer_processor",
    "unnormalizer_processor",
}
_SMOL_PROCESSORS = {"tokenizer_processor", "smolvla_new_line_processor"}
_DTYPES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}
_STATS = {
    "IDENTITY": (),
    "MEAN_STD": ("mean", "std"),
    "MIN_MAX": ("min", "max"),
    "QUANTILES": ("q01", "q99"),
    "QUANTILE10": ("q10", "q90"),
}


@dataclass(frozen=True)
class ResolvedCheckpoint:
    directory: Path
    checkpoint: Checkpoint
    source_sha256: str | None
    manifests: tuple[dict, ...]
    files: dict[str, dict]
    package_root: Path

    def metadata(self):
        return {
            "schema_version": 1,
            "checkpoint": asdict(self.checkpoint),
            "source_sha256": self.source_sha256,
            "manifests": list(self.manifests),
            "files": self.files,
            "runtime_verified": False,
            "calibration_verified": False,
            "task_success": None,
        }


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _finite(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite JSON number")
    return result


def _constant(_):
    raise ValueError("Nonfinite JSON number")


def _decode(raw):
    try:
        result = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_float=_finite,
            parse_constant=_constant,
        )
    except (UnicodeError, RecursionError, ValueError) as error:
        raise ValueError("Invalid bounded UTF-8 JSON") from error
    if not isinstance(result, dict):
        raise ValueError("JSON must be an object")
    return result


def _json(path):
    with path.open("rb") as stream:
        raw = stream.read(MAX_JSON + 1)
    if len(raw) > MAX_JSON:
        raise ValueError("JSON size limit exceeded")
    return _decode(raw)


def _relative(name):
    if (
        not isinstance(name, str)
        or not name
        or len(name) > MAX_PATH
        or "\\" in name
        or ":" in name
        or any(ord(c) < 32 for c in name)
    ):
        raise ValueError("Unsafe package path")
    parts = name.split("/")
    if any(p in {"", ".", ".."} for p in parts) or len(parts) > MAX_DEPTH:
        raise ValueError("Unsafe package path")
    return PurePosixPath(name)


def _stamp(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _open_regular(path):
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_EXPANDED:
        os.close(fd)
        raise ValueError("Package input must be a bounded regular file, not a symlink")
    return os.fdopen(fd, "rb")


def _copy(stream, destination, size):
    if not 0 <= size <= MAX_FILE:
        raise ValueError("Package member size limit exceeded")
    digest = hashlib.sha256()
    remaining = size
    with destination.open("xb") as output:
        while remaining:
            block = stream.read(min(BLOCK, remaining))
            if not block:
                raise ValueError("Truncated package file")
            output.write(block)
            digest.update(block)
            remaining -= len(block)
    return {"sha256": digest.hexdigest(), "bytes": size}


def _snapshot_directory(source, destination):
    """Open every component relative to owned directory FDs; never follow links."""
    if os.open not in os.supports_dir_fd or os.scandir not in os.supports_fd:
        raise ValueError("Safe directory snapshots require directory-FD support on this platform")
    inventory, total, count = {}, 0, 0
    seen = set()
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)

    def walk(fd, relative):
        nonlocal total, count
        before = _stamp(os.fstat(fd))
        with os.scandir(fd) as entries:
            for entry in entries:
                count += 1
                if count > MAX_FILES:
                    raise ValueError("Package member count limit exceeded")
                name = str(relative / entry.name)
                _relative(name)
                if name.casefold() in seen:
                    raise ValueError("Package duplicate member path")
                seen.add(name.casefold())
                info = entry.stat(follow_symlinks=False)
                target = destination / name
                if stat.S_ISDIR(info.st_mode):
                    child = os.open(entry.name, flags, dir_fd=fd)
                    try:
                        target.mkdir()
                        walk(child, PurePosixPath(name))
                    finally:
                        os.close(child)
                elif stat.S_ISREG(info.st_mode):
                    total += info.st_size
                    if total > MAX_EXPANDED:
                        raise ValueError("Package expanded size limit exceeded")
                    child = os.open(
                        entry.name,
                        os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=fd,
                    )
                    with os.fdopen(child, "rb") as stream:
                        opened = os.fstat(stream.fileno())
                        if not stat.S_ISREG(opened.st_mode) or _stamp(opened) != _stamp(info):
                            raise ValueError("Package file changed while opening")
                        inventory[name] = _copy(stream, target, info.st_size)
                        if stream.read(1) or _stamp(os.fstat(stream.fileno())) != _stamp(info):
                            raise ValueError("Package file changed while copying")
                else:
                    raise ValueError(
                        "Package members must be regular files/directories, not symlinks"
                    )
        if _stamp(os.fstat(fd)) != before:
            raise ValueError("Package directory changed while copying")

    fd = os.open(source, flags)
    try:
        walk(fd, PurePosixPath())
    finally:
        os.close(fd)
    return inventory


class _BoundedTarInfo(tarfile.TarInfo):
    def _proc_pax(self, tarfile):
        tarfile._package_headers = getattr(tarfile, "_package_headers", 0) + 1
        if tarfile._package_headers > MAX_FILES:
            raise ValueError("TAR metadata count limit exceeded")
        if self.size > 64 * 1024:
            raise ValueError("TAR metadata size limit exceeded")
        return super()._proc_pax(tarfile)

    def _proc_gnulong(self, tarfile):
        raise ValueError("GNU long-name TAR extensions are unsupported")

    def _proc_sparse(self, tarfile):
        raise ValueError("Sparse TAR files are unsupported")

    def _proc_gnusparse_00(self, *args):
        raise ValueError("Sparse TAR files are unsupported")

    def _proc_gnusparse_01(self, *args):
        raise ValueError("Sparse TAR files are unsupported")

    def _proc_gnusparse_10(self, *args):
        raise ValueError("Sparse TAR files are unsupported")


class _HashReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.count = 0

    def read(self, size=BLOCK):
        if size < 0 or size > BLOCK:
            raise ValueError("Archive read limit exceeded")
        raw = self.stream.read(size)
        self.count += len(raw)
        if self.count > MAX_EXPANDED:
            raise ValueError("Archive size limit exceeded")
        self.digest.update(raw)
        return raw


def _snapshot_archive(source, destination):
    inventory, seen, total = {}, set(), 0
    with _open_regular(source) as stream:
        before = _stamp(os.fstat(stream.fileno()))
        reader = _HashReader(stream)
        try:
            with tarfile.open(fileobj=reader, mode="r|*", tarinfo=_BoundedTarInfo) as archive:
                for item in archive:
                    name = item.name.rstrip("/") if item.isdir() else item.name
                    _relative(name)
                    if name.casefold() in seen:
                        raise ValueError("TAR duplicate member path")
                    seen.add(name.casefold())
                    if len(seen) > MAX_FILES:
                        raise ValueError("TAR member count limit exceeded")
                    if item.pax_headers and any(
                        k.startswith("GNU.sparse") for k in item.pax_headers
                    ):
                        raise ValueError("Sparse TAR files are unsupported")
                    target = destination / name
                    if item.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    elif item.isreg() and not item.issparse():
                        total += item.size
                        if total > MAX_EXPANDED:
                            raise ValueError("TAR expanded size limit exceeded")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.extractfile(item) as payload:
                            inventory[name] = _copy(payload, target, item.size)
                    else:
                        raise ValueError(
                            "TAR requires regular files/directories; links are forbidden"
                        )
            while reader.read():
                pass
        except (tarfile.TarError, EOFError, OSError) as error:
            raise ValueError("Invalid or incomplete checkpoint TAR") from error
        if _stamp(os.fstat(stream.fileno())) != before:
            raise ValueError("Archive changed during admission")
        return inventory, reader.digest.hexdigest()


def _verify_manifests(root, inventory):
    verified = []
    for name in sorted(inventory):
        if PurePosixPath(name).name != "manifest.json":
            continue
        path = root / name
        data = _json(path)
        if (
            type(data.get("schema_version")) is not int
            or data["schema_version"] != 1
            or not isinstance(data.get("files"), dict)
        ):
            raise ValueError("Unsupported package manifest")
        prefix = path.parent.relative_to(root).as_posix()
        prefix = "" if prefix == "." else prefix + "/"
        actual = {
            p[len(prefix) :]: v for p, v in inventory.items() if p.startswith(prefix) and p != name
        }
        expected = {}
        for key, value in data["files"].items():
            _relative(key)
            sha = (
                value
                if isinstance(value, str)
                else value.get("sha256")
                if isinstance(value, dict)
                else None
            )
            if not isinstance(sha, str) or not _SHA.fullmatch(sha) or key not in actual:
                raise ValueError("Invalid package manifest entry")
            if isinstance(value, dict) and (
                "bytes" in value
                and (type(value["bytes"]) is not int or value["bytes"] != actual[key]["bytes"])
            ):
                raise ValueError("Package manifest size mismatch")
            expected[key] = sha
        if expected != {p: v["sha256"] for p, v in actual.items()}:
            raise ValueError("Package manifest inventory/hash mismatch")
        verified.append({"path": name, "sha256": inventory[name]["sha256"]})
    return tuple(verified)


def _tensor_header(path, *, statistics=False):
    size = path.stat().st_size
    if not 8 < size <= (MAX_JSON if statistics else MAX_FILE):
        raise ValueError("Invalid safetensors file size")
    with path.open("rb") as stream:
        raw = stream.read(8)
        length = struct.unpack("<Q", raw)[0]
        if not 1 <= length <= MAX_HEADER or 8 + length > size:
            raise ValueError("Invalid safetensors header size")
        header = _decode(stream.read(length))
    intervals, tensors = [], {}
    for key, item in header.items():
        if key == "__metadata__":
            if not isinstance(item, dict) or any(not isinstance(v, str) for v in item.values()):
                raise ValueError("Invalid safetensors metadata")
            continue
        if not isinstance(item, dict) or set(item) != {"dtype", "shape", "data_offsets"}:
            raise ValueError("Invalid safetensors tensor metadata")
        shape, offsets = item["shape"], item["data_offsets"]
        if (
            not isinstance(item["dtype"], str)
            or item["dtype"] not in _DTYPES
            or not isinstance(shape, list)
            or len(shape) > 8
            or any(type(n) is not int or not 0 <= n <= 2**31 for n in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(n) is not int or n < 0 for n in offsets)
            or offsets[1] - offsets[0] != math.prod(shape) * _DTYPES[item["dtype"]]
            or offsets[1] > size - 8 - length
        ):
            raise ValueError("Invalid safetensors layout")
        intervals.append(tuple(offsets))
        tensors[key] = item
    end = 0
    for start, stop in sorted(intervals):
        if start != end:
            raise ValueError("Overlapping or incomplete safetensors tensor inventory")
        end = stop
    if not tensors or end != size - 8 - length:
        raise ValueError("Empty or incomplete safetensors tensor inventory")
    return tensors, 8 + length


def _statistics(path, features, norm_map):
    header, offset = _tensor_header(path, statistics=True)
    raw = path.read_bytes()  # Already bounded to MAX_JSON in the private snapshot.
    for name, feature in features.items():
        mode = norm_map.get(feature["type"], "IDENTITY")
        if mode not in _STATS:
            raise ValueError("Unsupported normalization mode")
        for key in _STATS[mode]:
            item = header.get(name + "." + key)
            shapes = [feature["shape"]] if feature["type"] != "VISUAL" else [[3], [3, 1, 1]]
            if item is None or item["shape"] not in shapes or item["dtype"] not in {"F32", "F64"}:
                raise ValueError("Missing or incompatible required normalization statistics")
            start, stop = item["data_offsets"]
            code = "f" if item["dtype"] == "F32" else "d"
            values = [
                x[0] for x in struct.iter_unpack("<" + code, raw[offset + start : offset + stop])
            ]
            if any(not math.isfinite(x) or (key == "std" and x < 0) for x in values):
                raise ValueError("Invalid normalization statistics values")


def _validate_policy(path):
    config = _json(path / "config.json")
    # Bound both saved recipes before the existing inspector reads them.
    for recipe in ("policy_preprocessor.json", "policy_postprocessor.json"):
        _json(path / recipe)
    # Existing model family, one-camera, one-observation and joint-dimension rules.
    try:
        info = inspect_checkpoint(path)
    except (AttributeError, TypeError, IndexError, KeyError) as error:
        raise ValueError("Invalid saved checkpoint metadata") from error
    if (
        info.width * info.height > 4096 * 2160
        or max(info.state_dim, info.action_dim, info.chunk_size) > 4096
    ):
        raise ValueError("Checkpoint dimensions exceed serving bounds")
    _tensor_header(path / "model.safetensors")
    allowed = _PROCESSORS | (_SMOL_PROCESSORS if info.policy_type == "smolvla" else set())
    for name, normalizer, required in (
        ("policy_preprocessor.json", "normalizer_processor", config["input_features"]),
        ("policy_postprocessor.json", "unnormalizer_processor", config["output_features"]),
    ):
        data = _json(path / name)
        steps = data.get("steps")
        if not isinstance(steps, list) or not 1 <= len(steps) <= 16:
            raise ValueError("Invalid saved processor steps")
        found = False
        for step in steps:
            if (
                not isinstance(step, dict)
                or set(step) - {"registry_name", "config", "state_file"}
                or step.get("registry_name") not in allowed
                or not isinstance(step.get("config"), dict)
            ):
                raise ValueError("Unsupported saved processor registry/configuration")
            cfg = step["config"]
            filename = step.get("state_file")
            if filename is not None:
                if (
                    not isinstance(filename, str)
                    or not re.fullmatch(r"[A-Za-z0-9_.-]+\.safetensors", filename)
                    or filename == "model.safetensors"
                ):
                    raise ValueError("Processor statistics must use a local safetensors file")
                _tensor_header(path / filename, statistics=True)
            if step["registry_name"] == normalizer:
                found = True
                norm_map = cfg.get("norm_map")
                if (
                    not isinstance(norm_map, dict)
                    or any(not isinstance(v, str) or v not in _STATS for v in norm_map.values())
                    or any(cfg.get("features", {}).get(k) != v for k, v in required.items())
                ):
                    raise ValueError("Processor statistics features do not match the model")
                if any(
                    norm_map.get(v["type"], "IDENTITY") != "IDENTITY" for v in required.values()
                ):
                    if filename is None:
                        raise ValueError("Missing saved normalization statistics")
                    _statistics(path / filename, required, norm_map)
        if not found:
            raise ValueError("Required built-in normalization processor is missing")
    return info


@contextmanager
def resolve_checkpoint(source: Path, *, archive: bool = False):
    """Yield an owned immutable copy; keep this context open until sync/use completes."""
    source = Path(source).expanduser().absolute()
    with tempfile.TemporaryDirectory(prefix="firebird-policy-") as temporary:
        root = Path(temporary)
        if archive:
            inventory, source_sha = _snapshot_archive(source, root)
        else:
            if source.is_symlink() or not source.is_dir():
                raise ValueError(
                    "Provide a complete policy directory, not bare weights; "
                    "use --checkpoint-archive for TAR"
                )
            inventory, source_sha = _snapshot_directory(source, root), None
        manifests = _verify_manifests(root, inventory)
        candidates = [
            root / str(PurePosixPath(name).parent)
            for name in inventory
            if PurePosixPath(name).name == "model.safetensors"
            and (root / str(PurePosixPath(name).parent) / "config.json").is_file()
        ]
        if len(candidates) != 1:
            raise ValueError("Package must contain exactly one complete ACT or SmolVLA checkpoint")
        selected = candidates[0]
        info = _validate_policy(selected)
        yield ResolvedCheckpoint(selected, info, source_sha, manifests, inventory, root)


def import_checkpoint(source: Path, output: Path, receipt_path: Path, *, archive: bool = False):
    """Publish a job-owned copy; receipt existence signals completed admission.

    Caller supplies private job paths. Exclusive creation prevents replacing an
    earlier import. A failed import removes only the output directory it created.
    """
    output = output.absolute()
    receipt_path = receipt_path.absolute()
    original = source.expanduser().absolute()
    if output == original or (not archive and output.is_relative_to(original)):
        raise ValueError("Output cannot replace or be inside the original checkpoint")
    if output.exists() or output.is_symlink() or receipt_path.exists() or receipt_path.is_symlink():
        raise ValueError("Import output and receipt must not already exist")
    if receipt_path.is_relative_to(output):
        raise ValueError("Receipt must be adjacent to, not inside, the copied payload")
    owned = False
    owned_receipt = False
    try:
        with resolve_checkpoint(source, archive=archive) as resolved:
            snapshot = resolved.package_root
            output.mkdir(mode=0o700)
            owned = True
            shutil.copytree(snapshot, output / "contents")
            result = resolved.metadata()
            result["directory"] = "contents/" + resolved.directory.relative_to(snapshot).as_posix()
            if result["directory"].endswith("/."):
                result["directory"] = "contents"
            result["files"] = {"contents/" + name: value for name, value in result["files"].items()}
            result["manifests"] = [
                dict(item, path="contents/" + item["path"]) for item in result["manifests"]
            ]
            encoded = (json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode()
            with receipt_path.open("xb") as stream:
                owned_receipt = True
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            return result
    except BaseException:
        if owned_receipt:
            receipt_path.unlink(missing_ok=True)
        if owned:
            shutil.rmtree(output)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--checkpoint", type=Path)
    inputs.add_argument("--checkpoint-archive", type=Path)
    inputs.add_argument("--source", type=Path, help="Fixed job import source")
    parser.add_argument("--archive", action="store_true", help="Interpret --source as TAR")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Inspect a private copy and remove it; no model imports or cloud actions",
    )
    args = parser.parse_args()
    try:
        if args.source is not None:
            if args.inspect_only or args.output_dir is None or args.json_output is None:
                raise ValueError("--source requires --output-dir and --json-output")
            result = import_checkpoint(
                args.source, args.output_dir, args.json_output, archive=args.archive
            )
        else:
            if not args.inspect_only or args.archive or args.output_dir or args.json_output:
                raise ValueError("Inspection requires --inspect-only and no import output options")
            with resolve_checkpoint(
                args.checkpoint or args.checkpoint_archive,
                archive=args.checkpoint_archive is not None,
            ) as resolved:
                result = resolved.metadata()
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"Checkpoint refused: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
