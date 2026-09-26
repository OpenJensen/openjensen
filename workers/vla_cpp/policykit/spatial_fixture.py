"""Bounded NumPy fixture decoding; no model import or executable pickle."""

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


def load_arrays(path):
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if (
            len(entries) != len(FIELDS)
            or {x.filename for x in entries} != {name + ".npy" for name in FIELDS}
            or sum(x.file_size for x in entries) > 16 * 1024 * 1024
            or any(x.flag_bits & 1 for x in entries)
        ):
            raise ValueError("Fixture archive exceeds the fixed observation inventory")
    with np.load(path, allow_pickle=False) as data:
        return validate_arrays({name: data[name] for name in data.files})
