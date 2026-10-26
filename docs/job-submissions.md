# Submit jobs with a saved request key

A request key (the `Idempotency-Key` header) lets you check or resend a submission without creating a duplicate job when the response is lost. Save the key with the exact recipe before you send it.

## Requests that take a key

| Request | Operation |
| --- | --- |
| `POST /api/v1/projects/{project_id}/intakes` | `dataset.inspect` (dataset inspections and recording preparation) |
| `POST /api/v1/projects/{project_id}/augmentations` | `dataset.augment` |
| `POST /api/v1/projects/{project_id}/policy-jobs` | The recipe's `operation`: `policy.finetune`, `policy.distill`, `policy.quantize`, `policy.evaluate`, `policy.run`, `policy.export`, `policy.workflow` or `policy.import` |
| `POST /api/v1/projects/{project_id}/teaching/sessions` | `teaching.capture` |

Binary uploads and cancellation take no key. For those, use the selected job's ordinary upload or cancellation procedure.

## Choose a key

- Generate a random key and save it with the exact JSON recipe before sending.
- Use 1–128 case-sensitive ASCII characters. Start with a letter or digit, and continue with letters, digits, `.`, `_`, `:` or `-`.
- Send one header value per request; repeated headers are rejected.
- Use a separate key for each intended project and operation request. Keep the key with its original recipe while you follow the job.

Resending the identical request with the same key returns the original job's current state, even if that job has finished or was interrupted. A different request with the same key is rejected with 409. Nothing is retried automatically.

## Submit and follow

```sh
jensen policy submit PROJECT recipe.json --idempotency-key UNIQUE-KEY
jensen jobs submission PROJECT policy.finetune UNIQUE-KEY
jensen inspect PROJECT --path dataset --idempotency-key ANOTHER-KEY
jensen augmentation submit PROJECT augmentation.json --idempotency-key ANOTHER-KEY
```

Replace `PROJECT`, the keys and the recipe files with your saved values. Use the returned job ID to follow status, events and output, for example with `jensen jobs show JOB_ID` and `jensen jobs events JOB_ID`.

## Recover a lost response

Look up the key with its original project and operation:

```text
GET /api/v1/projects/{project_id}/submissions/{key}?operation=policy.finetune
```

`jensen jobs submission PROJECT OPERATION KEY` makes the same lookup. It only reads; it never submits or retries.

| Result | What to do |
| --- | --- |
| A job is returned | Open that saved job. |
| 404 | No accepted job is bound to the key yet. Inspect fresh project history and keep the original key and recipe before explicitly retrying with them. |
| 409 when resending | The key belongs to a different request. Restore the original recipe, or use a new key for a separate intended request. |
| 503 | The saved acceptance cannot be verified. Inspect the saved job and workspace binding before making another request. |

## Recover in the browser

The browser forms keep each unresolved attempt in the tab's session storage. If a response is lost:

1. Select **Check saved request**. It looks up the saved key and never resubmits.
2. If no job is found, **Retry same request** resends the saved request with the same key, even if you have edited the form since.
3. An older attempt without a saved key offers **Refresh job history**. Review the fresh history, then select **I reviewed the jobs; allow a new request**.

Managed Teaching sessions use **Check saved teaching request**, **Retry same teaching request** and **I reviewed teaching jobs; clear legacy recovery**. If browser storage is unavailable, restore it and reload before submitting again.
