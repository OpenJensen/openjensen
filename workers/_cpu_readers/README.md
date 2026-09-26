# Isolated local preview reader

From the application repository root:

```sh
uv venv --python 3.14.7 workers/_cpu_readers/.venv
uv pip install --python workers/_cpu_readers/.venv/bin/python --no-deps pyarrow==25.0.1
uv run --frozen pytest -q tests/test_local_preview.py
```

The ignored `.venv` contains only PyArrow; neither core manifests nor `uv.lock`
change. There is no runtime install, network request, dataset copy, video decoder,
Torch, NumPy or pandas dependency. The fixed reader script runs with `-I -B`.
The source-checkout default is this venv; packaged callers must pass the trusted
interpreter path explicitly. Missing/wrong readers produce explicit errors.

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
and real Windows tests. Do not advertise cross-platform preview support yet.

The returned dict validates against `LocalDatasetPreview`. `DatasetProfile` may
carry it as `preview` only with `inspection_scope="bounded_parquet_rows"`, local
v3 format, and matching metadata identity. Existing metadata-only profiles omit
that optional field and retain their original serialization. The helper is not
yet a registered job/API/UI operation; existing intake stays metadata-only.

CI installs this isolated reader on POSIX hosts. Windows runs contract and
fail-closed boundary checks, while actual row-preview tests are explicitly
skipped because secure opens are unsupported. Missing reader setup on a
supported host is a failure, never an automatic skip. Neither green CI nor the
synthetic fixture establishes cross-platform robotics support.
