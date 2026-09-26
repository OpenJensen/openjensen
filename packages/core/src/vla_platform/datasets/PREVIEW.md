# Local Parquet sample previews

Application intake still returns a `metadata_only` DatasetProfile. The explicit
CPU helper below reads actual Parquet values; it is not wired into the web UI or
the application's registered intake job. It neither installs nor imports LeRobot,
Torch or another ML runtime. `uv sync --frozen` installs pinned PyArrow 25.0.1;
the normal CPU test suite includes actual Parquet fixtures with no optional skips.

From the repository, on Windows or Linux:

```text
uv run --frozen python -m vla_platform.datasets.preview my-dataset --root /absolute/local/datasets --parquet data/chunk-000/file-000.parquet --rows 3
```

For native Windows, `--root D:/datasets` is a valid local root. A relative dataset
argument is rooted there. `FIREBIRD_LOCAL_DATA_ROOT` may supply the root instead.
Select a file explicitly from your local LeRobot v2/v3 dataset: this helper does
not glob or traverse the dataset to choose a shard. Use the file layout actually
present in that dataset. Optional repeated `--column action --column episode_index`
arguments select columns; default columns are episode/frame indices, timestamp,
action, observation.state, index and task_index, when present.

The function is also callable as
`preview_local(path, allowed_root, parquet_path, max_rows=8, columns=None)` from
`vla_platform.datasets.preview`. Its versioned helper JSON is separate from shared
application contracts. Success includes metadata SHA-256, selected-file SHA-256,
source path, reader version, first row-group location, actual sample values,
sample episode indices when available, total file rows and a truncation flag.
Hashes describe exactly the metadata bytes and bounded file snapshot read for
that invocation, not a complete dataset revision or a promise the source remains
unchanged. The selected file is copied only into bounded memory; the helper writes
no dataset copies or application records.

`previewed` means sample rows were decoded, all declared non-video feature columns
exist, and the bounded declared-file probes did not find missing files. It does
not establish complete episode coverage, action units, controller semantics,
reference frames, calibration, train/eval split provenance or policy compatibility.
Missing selected files yield `missing`; empty data or missing feature/declared files
yield `incomplete`. Missing data/video templates yield explicit `unverified` presence
records. Schema dtype/shape compatibility beyond safe preview types is not established.
CLI success exits 0; incomplete/missing results exit 2 with JSON on stdout; refusal
exits 2 with a bounded JSON error on stderr. The helper never invents episode identity.

Video decoding is **unsupported**, regardless of whether a video file exists.
Binary, image, struct, extension and other unsupported columns cannot be selected.
Omitted columns are listed. No remote source is read by this helper. Local metadata
intake only probes the first declared shard (known template substitutions set to
zero) and at most eight declared camera video paths; these are bounded presence
checks, not a full missing-file inventory or validation of all source-declared counts.

## Read and output limits

| Boundary | Fixed limit |
|---|---|
| `meta/info.json` | 2 MiB; 24 JSON levels; 16,384 nodes; 16,384 characters per string |
| Feature declarations / local paths | 128 features; 1,024 characters per path |
| Selected Parquet file snapshot | 32 MiB; one file, read at most limit + 1 bytes |
| Parquet footer before native parsing | 256 KiB; bounded Thrift strings/containers |
| Parquet schema / row-group metadata | 128 leaf columns; 1,024 row groups |
| First row group before decoding | 4 MiB declared uncompressed bytes; 50,000 rows |
| Each column chunk before decoding | 100,000 values; 4 MiB declared uncompressed bytes |
| Returned sample | 1–16 rows from the first row group; at most 8 columns |
| Values and final JSON | 128 numeric vector elements; 512 string characters; 96 KiB JSON |

The reader disables parallel decoding, prebuffering, memory mapping and Arrow
extension interpretation. Whole row-group and column-chunk budgets are checked
before requesting a small batch, because a small batch alone does not limit native
page/dictionary allocations. These are application limits on declared Parquet
metadata, not a process-level sandbox for hostile native-parser exploits. Keep the
configured local root under trusted local control; concurrent hostile filesystem
replacement is outside this local application's security boundary. The helper
detects ordinary size/mtime changes while capturing the file and decodes the same
immutable snapshot it hashes, so later source mutation cannot change returned rows.

Containment is checked lexically before access, including nonexistent paths.
UNC/device paths, drive-relative paths, ADS, `..`, reserved names, symlinks,
Windows junctions and other reparse points are refused. Local metadata is parsed
as strict JSON, with duplicate keys/nonfinite numbers refused. Known LeRobot path
placeholders accept only small integer formatting; attribute access, indexing,
arbitrary conversions, absolute paths and unknown substitutions are refused.

## Reproducible verification

```text
uv sync --frozen
uv run --frozen pytest -q tests/test_local_preview.py tests/test_intake.py
uv run --frozen pytest -q
uv run --frozen ruff check packages/core scripts tests
uv run --frozen ruff format --check packages/core scripts tests
```

`tests/test_local_preview.py` creates real, small CPU Parquet fixtures in a test
directory. Assertions cover decoded values, provenance, no source modifications,
CLI exit states, v2/v3 metadata, file growth and snapshot mutation, compressed
allocation budgets, missing files/columns/semantics, malformed metadata, unsafe
paths and a real Windows junction (a symlink on Linux). These are fixture results,
not live-source robotics, decoded camera, GPU or policy evidence.
