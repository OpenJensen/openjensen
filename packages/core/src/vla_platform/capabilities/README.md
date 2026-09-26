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
Windows local fixture evidence does not establish Windows live HF intake,
Linux behavior, media validation, native policy execution or GPU support.
Linux CPU metadata remains untested until native evidence is recorded.
No hardware probing or ML imports occur when retrieving capabilities.

Local intake is disabled unless a root is configured. Enabling it does not prove
the directory or dataset is valid; the intake worker enforces the path boundary.
An operating system outside Linux, Windows and macOS disables metadata operations.
