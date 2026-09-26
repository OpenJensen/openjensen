# Synthetic LeRobot v3 style fixture

`generate.py` writes actual Apache Parquet files using isolated PyArrow 25.0.1.
There are six frame rows (two episodes of three frames) and two episode rows,
with v3 `data/chunk-000/file-000.parquet` and
`meta/episodes/chunk-000/file-000.parquet` paths. The accompanying `meta/info.json`
declares the same counts and features. `manifest.json` records exact bytes, row
counts and SHA-256 for every input file.
The metadata uses UTF-8 with LF line endings; `.gitattributes` preserves its
recorded bytes even when Git's Windows checkout conversion is enabled.

Frame `i` has action `[i, -i]` and state `[i + 0.25, i + 0.5]`; timestamps restart
at zero for each episode. Episode ranges are `[0, 3)` and `[3, 6)`. This is a
minimal synthetic format fixture, not a recorded robot dataset, loader-certified
LeRobot export, semantics validation or GPU evidence. No upstream dataset was
downloaded; no video, images, credentials, weights or licensed third-party data
are included. Task strings are synthetic and carried in the episode table.

Regenerate from the repository root:

```sh
workers/_cpu_readers/.venv/bin/python tests/fixtures/lerobot_v3_preview/generate.py
```

Tests check these committed bytes and known values; they do not regenerate the
fixture during testing or download/install dependencies. The generator is only
a reproducibility aid. There is no `tasks.parquet`, global statistics or media;
full LeRobot training/loading and all dataset-file completeness are out of scope.
