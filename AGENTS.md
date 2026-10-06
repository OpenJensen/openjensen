# Working safely in this repository

Read the current task card and source before editing. Preserve working configuration, user data and previously verified behavior. Planning and task records belong to the separate `firebird-hackathon-prep` repository; application code belongs here. The coordinator owns shared interfaces and integration.

## Local verification and publication

GitHub Actions is unavailable for this project. Run the applicable checks locally and retain source-bound results; hosted CI is not an acceptance gate. Keep each independent change in its own focused commit, push a feature branch and open a pull request with the local verification results. Do not merge or push directly to `main` unless the user explicitly requests it. Leave previously published runs and commit history as they are. Do not report resource-stopped or unexecuted checks as passing, and do not remove workflow definitions merely because hosted execution is unavailable.

## Local resource budget

- One resource-heavy verification lane at a time across all agents: browser/build, native model proof, or Docker. Coordinate ownership before starting. Lightweight source work can run in parallel.
- Measure free disk and current memory pressure before a heavy run. Keep at least 10 GiB disk headroom; below that, stop new builds/downloads and retire reviewed disposable output first. Plan additional headroom for the expected output size. Do not interpret low resident memory as low usage when swap/compression is high.
- Reuse an existing compatible environment, browser cache or Docker image. Do not create another environment/image merely for a new branch or test attempt. Do not download model weights or rebuild large dependencies without a measured need and capacity check.
- Local Docker verification defaults to one named, task-labeled container, one CPU, 1536 MiB memory and 256 processes, with automatic removal. Use the existing cached image; use offline execution where the test supports it. Larger limits require coordinator review of actual available resources. These defaults do not modify the user's deployed GPU workers.
- Record the exact owned process/container ID and its output directory. After normal completion, failure or cancellation, verify that its process group/container and temporary listener are gone. An observation timeout does not authorize a replacement run; inspect the original handle first.
- Retain concise source-bound receipts, failure logs and required deliverables. Once a run is terminal and its evidence is saved, remove its generated fixture copies and build intermediates that have no runtime dependency. Do not accumulate a fresh model/environment copy per retry.
- Cleanup requires an exact allowlist, ownership and active-use checks. Preserve credentials, original models/datasets, project databases, saved videos, application bundles and supporting runtimes. A temporary directory can still be a live runtime dependency.
- Never run blanket Docker pruning, delete `Docker.raw`, or clear all package/model caches. Do not prune shared uv caches while other tools execute from them. When Docker is unresponsive, stop issuing unbounded requests; do not reset it without determining the effect on existing workloads.
- Retire managed worktrees through the workspace tools after checking for uncommitted/unpublished work and live dependencies; do not remove their checkout directories in the shell.

These rules apply to the coordinator and every subagent. User instructions can refine the budget, but a task is not verified merely because testing was interrupted to protect resources.
