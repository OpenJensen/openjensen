# Isolated CPU Parquet preview reader

Hugging Face episode/frame previews and the provisional local preview helper use
this separate CPU environment. From the application repository root on POSIX:

```sh
uv venv --python 3.14.7 workers/_cpu_readers/.venv
uv pip install --python workers/_cpu_readers/.venv/bin/python --no-deps pyarrow==25.0.1
uv run --frozen pytest -q tests/test_explore.py tests/test_hub_parquet.py tests/test_local_preview.py
```

On Windows (PowerShell):

```powershell
uv venv --python 3.14.7 workers/_cpu_readers/.venv
uv pip install --python workers/_cpu_readers/.venv/Scripts/python.exe --no-deps pyarrow==25.0.1
uv run --frozen pytest -q tests/test_explore.py tests/test_hub_parquet.py
```

Dataset import extends the reader with the exact NumPy, HDF5 and Pillow pins in `requirements.txt`; FFmpeg is installed on the app host. The core application and its
tests do not import PyArrow; even synthetic Parquet fixtures are generated in the
reader environment. The app does not install packages at runtime or import these native dependencies. Torch and pandas are not required by this reader. Both fixed reader scripts run with `-I -B`.

## Hugging Face previews

The application retains its pinned Hub revision, metadata hash, host/redirect
allowlist, download budgets and camera URLs. Downloaded Parquet bytes cross stdin
to a fixed decoder with only an operation and selected episode number. Dataset
paths never become local opens, commands, imports or executable code.

Each decoder has a ten-second deadline, a 32 MiB input ceiling (8 MiB for episode
indexes), an 8 MiB JSON output ceiling and 16 KiB diagnostic ceiling. Existing row,
projected-value, vector and materialization limits still apply. Timeout, crash,
output overflow and request cancellation kill and reap the child before releasing
an explorer slot. Camera previews may still be returned with an explicit warning
if sample rows cannot be read; an unavailable index cannot report successful rows.

The source-checkout interpreter defaults to this venv. A packaged application can
set `FIREBIRD_CPU_READER_PYTHON` to a trusted interpreter with `pyarrow==25.0.1`, or
pass `reader_python` when constructing `DatasetExplorer`. This is operator
configuration, never a dataset field. Missing or wrong readers fail explicitly.
No setup script or cloud operation runs automatically.

Hub preview decoding supports Windows because its input is already bounded bytes;
it does not need the local helper's POSIX filesystem security primitives. Tests
cover native imports blocked in core, real hung/crashed processes, blocked stdin,
output/diagnostic overflow, cancellation, and v2/v3 functional previews. CI must
install this reader on both Linux and Windows. Native decode memory is not an
OS-enforced RSS limit, and subprocess separation is not an adversarial-code sandbox.

## Provisional local preview

Local callers use `preview_local(..., python=trusted_interpreter)` to override the
same source-checkout default. No dataset files are copied by this helper.

```python
from vla_platform.datasets.local_preview import PreviewLimits, preview_local

preview = preview_local(
    "lerobot_v3_preview",
    "/absolute/canonical/path/to/tests/fixtures",
    limits=PreviewLimits(max_rows=2),
)
```

`kind="episodes"` selects `meta/episodes/chunk-000/file-000.parquet`; the default
selects `data/chunk-000/file-000.parquet`. An explicit relative `parquet_path` may
select another file in the same area. Metadata path templates are never executed
or followed. The helper previews one file's first rows; it does not discover or
validate all episodes. Unsupported columns are omitted, including images/video.

Limits cover returned rows, each source file's stored bytes, aggregate source
read bytes (including hashing and repeated Parquet reads), declared uncompressed
column bytes, emitted JSON bytes, and subprocess wall time including imports.
Oversize inputs fail without a partial success. Native decode memory is not an
OS-enforced RSS budget: declared uncompressed sizes and actual batch sizes are
checked, with a hard subprocess deadline. This is not an adversarial-code sandbox.

Paths reject traversal, absolute file selectors, symlinks in **every** component,
and nonregular files. Configure a canonical absolute root (e.g. `/private/tmp`,
not macOS's `/tmp` symlink). Descriptor-relative no-follow opens protect against
path replacement races on supported POSIX systems. Windows currently returns
`unsupported_platform`; it needs a separately reviewed native safe-open adapter
and real Windows tests. Do not advertise cross-platform local row previews yet.

The returned dict validates against `LocalDatasetPreview`. `DatasetProfile` may
carry it as `preview` only with `inspection_scope="bounded_parquet_rows"`, local
v3 format, and matching metadata identity. Existing metadata-only profiles omit
that optional field and retain their original serialization. The helper is not
yet a registered job/API/UI operation; existing intake stays metadata-only.

Local row-preview tests run on supported POSIX hosts. Windows runs local contract
and fail-closed boundary checks; local row-preview tests remain explicitly skipped
because secure opens are unsupported. Hub preview tests do not use this exemption.
Missing reader setup on a supported host is a failure, never an automatic skip.
Neither green CI nor synthetic fixtures establish robotics task success.
