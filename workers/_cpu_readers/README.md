# CPU Parquet reader setup

Run from the repository root with Python 3.14.7. Install FFmpeg on the application
host for video import.

## Install on POSIX

```sh
uv venv --python 3.14.7 workers/_cpu_readers/.venv
uv pip install --python workers/_cpu_readers/.venv/bin/python --no-deps -r workers/_cpu_readers/requirements.txt
uv run --frozen pytest -q tests/test_explore.py tests/test_hub_parquet.py tests/test_local_preview.py
```

## Install on Windows

Use PowerShell:

```powershell
uv venv --python 3.14.7 workers/_cpu_readers/.venv
uv pip install --python workers/_cpu_readers/.venv/Scripts/python.exe --no-deps -r workers/_cpu_readers/requirements.txt
uv run --frozen pytest -q tests/test_explore.py tests/test_hub_parquet.py
```

## Configure the interpreter

The source checkout uses `workers/_cpu_readers/.venv` by default. To use another
installed interpreter, set `FIREBIRD_CPU_READER_PYTHON` to its absolute path;
install the exact requirements above, including `pyarrow==25.0.1`.
`DatasetExplorer` also accepts the operator-side `reader_python` argument.

## Preview a local file

Use POSIX and a canonical absolute root containing the dataset directory. Select
`kind="episodes"` for `meta/episodes/chunk-000/file-000.parquet`; the default reads
`data/chunk-000/file-000.parquet`. Set a relative `parquet_path` to select another
regular file in that area.

```python
from vla_platform.datasets.local_preview import PreviewLimits, preview_local

preview = preview_local(
    "lerobot_v3_preview",
    "/absolute/canonical/path/to/tests/fixtures",
    limits=PreviewLimits(max_rows=2),
)
```

Pass `python=trusted_interpreter` to use the configured reader interpreter explicitly.
