# Operation capabilities

`registry(local_root=...)` is the sole source for `/api/v1/capabilities`, used by
both CLI and web. Existing `available`/`planned` status remains compatible with
the clients. `available` and `runnable` describe a registered, configured adapter;
they do not promise successful input validation or claim a run on this host.

Each operation names its backend and implementation state, plus explicit OS and
device targets. `supported` requires a referenced run, with `fixture` or
`live_source` evidence. `untested` is a registered target without recorded runs.
`unsupported` means this implementation does not offer that operation/target;
it is not a claim about upstream backend feasibility. Registration alone never
promotes evidence. Planned policy operations have no backend and cannot run.

The metadata adapter uses CPU only. Historical macOS live public HF metadata
intake and synthetic local fixture checks have separate target evidence.
Windows local fixture evidence is separate from pinned public-HF live metadata
intake at `f7f9a41`, verified through the web transport, CLI and a fresh process.
The latter does not establish interactive-browser rendering or media validation.
Linux local CPU metadata has fixture CI evidence at `6319c09`; later Parquet
preview and local-hardening revisions have no Linux verification. Linux public-HF
intake remains untested. None of these records establish native policy/GPU support.
No hardware probing or ML imports occur when retrieving capabilities.

Local intake is disabled unless a root is configured. Enabling it does not prove
the directory or dataset is valid; the intake worker enforces the path boundary.
An operating system outside Linux, Windows and macOS disables metadata operations.
