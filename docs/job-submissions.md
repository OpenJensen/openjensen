# Submit jobs with a saved request key

## Prepare the request

Save a random `Idempotency-Key` with the exact recipe before sending it. Use 1–128 case-sensitive ASCII characters: start with a letter/digit and continue with letters, digits, `.`, `_`, `:` or `-`. Send one header value per request.

Use a separate key for each intended project/operation request. Keep that key with its original JSON recipe when following the job.

## Submit and follow

```sh
firebird policy submit PROJECT recipe.json --idempotency-key UNIQUE-KEY
firebird jobs submission PROJECT policy.finetune UNIQUE-KEY
firebird inspect PROJECT --path dataset --idempotency-key ANOTHER-KEY
firebird augmentation submit PROJECT augmentation.json --idempotency-key ANOTHER-KEY
```

Replace `PROJECT`, keys and recipe files with your saved values. Send the key on project `intakes`, `augmentations` or `policy-jobs` POST requests. Read the returned job ID to follow status, events and output.

## Recover a lost response

Read `GET /api/v1/projects/{project_id}/submissions/{key}?operation=policy.finetune` with the original project, operation and key.

- If a job is returned, open that saved job.
- For 409, restore the original recipe or use a new key for a separate intended request.
- For 404 after a lost response, inspect fresh project history and retain the original key/recipe before explicitly retrying.
- For 503, inspect the saved job and workspace binding before making another request.

In a browser form, use its reconciliation action to recover the saved attempt. Review legacy uncertain submissions in fresh job history before clearing them. For binary imports and cancellation, use the selected job's ordinary upload/cancellation procedure.
