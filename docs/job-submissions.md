# Durable JSON job submissions

An optional `Idempotency-Key` header is supported on the existing project `intakes`,
`augmentations`, and `policy-jobs` POST endpoints. Existing clients without a key retain
their behavior. A key is a strict, case-sensitive ASCII string of 1–128 characters:
the first character is a letter or digit; later characters may also be `.`, `_`, `:`,
or `-`. Duplicate headers and malformed keys are rejected before admission.

Save a random key with the exact intended recipe **before** sending it. Its scope is
the project and operation (for example, `policy.finetune`). Reusing that key with the
same typed, normalized request returns the original job's **current saved state**.
Reusing it with another normalized request returns HTTP 409. The successful response
is the existing `Job` shape and echoes `Idempotency-Key`; no worker dispatch occurs
when returning an already accepted identity.

Use `GET /api/v1/projects/{project_id}/submissions/{key}?operation=policy.finetune`
to reconcile a lost response without submitting work. A missing binding returns 404;
an unverifiable saved binding returns 503 and cannot create replacement work. A 404
during an outstanding admission is not proof that the earlier request will never
commit. Clients must retain their uncertainty controls, and never manufacture a new
key or automatically repeat POST after a transport failure. There is no automatic
POST retry in the API, CLI or this implementation.

```sh
firebird policy submit PROJECT recipe.json --idempotency-key UNIQUE-KEY
firebird jobs submission PROJECT policy.finetune UNIQUE-KEY
firebird inspect PROJECT --path dataset --idempotency-key ANOTHER-KEY
firebird augmentation submit PROJECT augmentation.json --idempotency-key ANOTHER-KEY
```

The key is not an authorization token. Existing project admission and local API
access boundaries still apply. Binary `model-imports`, project creation and
cancellation do not use this contract. Binary uploads explicitly reject the header
rather than imply upload deduplication. Existing upload/cancellation recovery remains
unchanged. Browser journals have not yet been wired to these keys in this backend
slice; that is a separate integration gate for JOB-003.

## Persistence and recovery

Migration `0002` adds `job_submissions` without modifying existing job records or
artifacts. Its composite primary key is `(project_id, operation, idempotency_key)`.
Each row stores fingerprint version 1, the original typed request JSON and SHA256,
the accepted job ID and immutable accepted response, and the binding timestamp.
For version 1 the digest is SHA256 of UTF-8 JSON with sorted keys, compact separators,
ASCII escaping and nonfinite values forbidden. Canonical input is bounded to 1 MiB.
Model defaults participate; object-key order does not. Ordered arrays and typed
numeric distinctions not normalized by the request model remain significant.

Fingerprinting precedes live source resolution, resume enrichment, catalog changes
and other admission. The accepted job's request may legitimately differ from the
original request. An identical retry does not resolve a moving Hub branch again or
revalidate a now-unavailable runtime. Fresh metadata keys still reuse the existing
verified inspection cache; several keys may therefore bind one cached job. The
accepted response may already be running or terminal, and its creation time need
not equal the later key-binding time.

The new job and binding commit in one SQLite transaction. Only its winner schedules
the worker. Commit and scheduling are owned through request cancellation; shutdown
drains acceptance before stopping workers. If the process dies after commit but
before dispatch, the existing restart reconciliation marks the saved job interrupted.
A retry returns that identity instead of dispatching it. An in-process failure after
commit also cannot cause a retry to dispatch the saved job. Cloud cleanup uncertainty,
terminal failures and interruptions remain visible; idempotency is not proof of
provider-side exactly-once execution, safe cloud cleanup or task success.

Bindings are retained for the lifetime of the workspace; there is no automatic expiry,
eviction or downgrade that drops them. SQLite uniqueness also fences competing
insertions, while the existing workspace-owner lock remains the supported process
ownership boundary. This does not add distributed scheduling or cross-workspace keys.

Desktop/offline tools that only understand revision `0001` must reject `0002` until
their own strict compatibility checks are updated. Do not copy a new workspace into
an older payload or remove its submission table to make it open. No live workspace,
operator configuration or provider was migrated or activated by the disposable tests.
