"""Bounded NumPy fixture decoding; no model import or executable pickle."""

import hashlib
import io
import math
import zipfile

import numpy as np

FIELDS = {
    "observation.images.image",
    "observation.images.image2",
    "observation.state",
    "task",
    "noise",
}


def validate_arrays(values):
    if set(values) != FIELDS:
        raise ValueError("Unexpected raw observation/noise fixture fields")
    for key, shape in (
        ("observation.images.image", (1, 3, 360, 360)),
        ("observation.images.image2", (1, 3, 360, 360)),
        ("observation.state", (1, 8)),
        ("noise", (1, 50, 32)),
    ):
        value = values[key]
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("Unsupported fixture shape, dtype or nonfinite values: " + key)
        if key.startswith("observation.images") and (value.min() < 0 or value.max() > 1):
            raise ValueError("Raw fixture image must be RGB float32 in [0,1]")
    task = values["task"]
    if task.shape != (1,) or task.dtype.kind != "U" or not 1 <= len(task[0]) <= 4096:
        raise ValueError("Fixture needs one bounded task string")
    return values


def validate_header(stream, name, file_size):
    # NPY can claim enormous shapes in a tiny archive. Inspect its bounded header
    # before np.load has an opportunity to allocate the declared array.
    version = np.lib.format.read_magic(stream)
    readers = {
        (1, 0): np.lib.format.read_array_header_1_0,
        (2, 0): np.lib.format.read_array_header_2_0,
    }
    if version not in readers:
        raise ValueError("Unsupported fixture NPY version")
    shape, _, dtype = readers[version](stream, max_header_size=4096)
    if name == "task":
        valid = shape == (1,) and dtype.kind == "U" and 4 <= dtype.itemsize <= 4096 * 4
    else:
        shapes = {
            "observation.images.image": (1, 3, 360, 360),
            "observation.images.image2": (1, 3, 360, 360),
            "observation.state": (1, 8),
            "noise": (1, 50, 32),
        }
        valid = shape == shapes[name] and dtype == np.dtype("float32")
    if not valid or stream.tell() + math.prod(shape) * dtype.itemsize != file_size:
        raise ValueError("Fixture header shape, dtype or payload length differs from contract")


def load_arrays(path, *, expected_sha256=None):
    # Both header validation and decoding use the same immutable bounded bytes.
    # Reopening the pathname would let a concurrent replacement bypass the bounds.
    with open(path, "rb") as stream:
        payload = stream.read(16 * 1024 * 1024 + 1)
    if len(payload) > 16 * 1024 * 1024:
        raise ValueError("Fixture compressed file exceeds the 16 MiB bound")
    if expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Fixture snapshot SHA256 mismatch")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        entries = archive.infolist()
        if (
            len(entries) != len(FIELDS)
            or {x.filename for x in entries} != {name + ".npy" for name in FIELDS}
            or sum(x.file_size for x in entries) > 16 * 1024 * 1024
            or any(x.flag_bits & 1 for x in entries)
        ):
            raise ValueError("Fixture archive exceeds the fixed observation inventory")
        for entry in entries:
            with archive.open(entry) as stream:
                validate_header(stream, entry.filename[:-4], entry.file_size)
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        return validate_arrays({name: data[name] for name in data.files})
