# Working safely in this repository

Read the current issue/task record and source before editing. Preserve working configuration, user data and previously verified behavior. Keep application code and concise in-app help here. Extended setup guides, standalone experiment/benchmark suites and unused configuration examples belong outside this repository. Local verification results, experiment outputs, benchmarks and run reports stay local: never commit them or put them in commit messages or pull requests. Upstream files never name the maintainers' machines, hardware, hostnames, usernames or local paths. Use the designated planning location for larger coordination records. The coordinator owns shared interfaces and integration.

Run `python3 .github/scripts/check_doc_links.py` before committing documentation.

## Versions and builds

A version number is not an identity. Release automation writes the next patch number into every open pull request, so several unmerged branches carry the same number at the same time, and only one commit ever becomes that release. Mixing them up is how two machines both "on 0.1.31" ended up running different code.

- Identify code by its **build label**, never by the bare number. `uv run jensen build` prints it; the app shows it in the header and at `/api/v1/health`. `0.1.30` is the published release. `0.1.31-dev+62227c5` is a branch build that is not 0.1.31. A `.dirty` suffix means uncommitted source changes. Two installations run the same code only when their labels are equal.
- Put the build label of every machine involved in each bug report, verification note and handoff. "Works on 0.1.31" is not evidence.
- Never edit a version by hand. Ten files carry it and `.github/scripts/version_files.py` owns all of them; `python3 .github/scripts/version_files.py check` confirms they agree.
- Start every task from current main: `git fetch origin`, then branch from `origin/main`. Do not branch from another unmerged branch. If the work needs an open pull request, get that one merged first: a stack of unmerged branches leaves `main` behind and puts every device on a different build.
- Robots and shared machines run `main`, installed through **Settings → Updates** or a checkout of `main`. If a branch has to be tried on a device, record its build label in the task record and return the device to `main` when the trial ends.
- One checkout per task, created and retired through the workspace tools. Do not copy the repository into sibling folders or link one checkout's `.venv` or `node_modules` into another: the linked checkout silently runs the other one's dependencies.

## Checks before committing

Run `uv run python scripts/check.py`. It compares the work with `origin/main`, runs only the checks that change can break, and prints why each test was chosen. `--list` shows the selection without running it, `--all` runs every application check, and `--record` runs the whole Python suite once and refreshes the map of which tests execute which source files. Selection fails open: a changed file nothing accounts for runs everything. CI still runs the complete lanes on the pull request.

`tests/conftest.py` gives every Python test three guarantees. Keep them.

- **No spending.** Sockets that leave the machine are refused, cloud, Hugging Face and model-provider credentials are hidden, and the cloud command-line tools are replaced by stubs that fail. A test that tries anyway fails with the address it reached for. A deliberate live check carries `@pytest.mark.allow_network` and never runs in the default suite.
- **No hanging.** Each test has a 60 second deadline and fails with the stack of where it was stuck. Raise it with `@pytest.mark.deadline(N)` only for a measured, legitimately slow test.
- **Same machine everywhere.** Free disk space and installed cloud tools are normalised so a full laptop gives the same result as CI.

How to make a test worth its runtime:

- A bug fix starts with a test that fails for the user's reason. Send the request the page really sends, use real files, watch it fail, then fix. The dataset-inspection regression passed 4,400 tests because every dataset fixture had two or more episodes and one browser test asserted the faulty wording.
- Test what a person does, not what the code does. One journey through real files (record, save, open, train) catches more than ten tests of a mocked boundary.
- Vary the data, not only the code path: one item, none, the largest allowed, a name with spaces.
- Prefer a real file or a real subprocess to a mock. Mock only what cannot exist on a test machine: a robot, a GPU, a paid service.

Cost: this is a public repository and CI runs on free hosted runners. Do not route workflows to paid or self-hosted runners, and do not add a test that needs cloud credentials, without the owner's decision. Start cloud or GPU jobs only on an explicit request, tear them down when they finish, and report anything left running.

## Local verification and publication

GitHub Actions runs the selected application, browser and worker checks. Inspect the current-head jobs and their actual results; a green contract suite does not establish GPU training, native desktop installation or simulator acceptance. Run applicable checks locally when practical. If hosted execution is unavailable, report that limitation and use explicit local evidence instead of assuming failure or success. Keep each independent change in its own focused commit, push a feature branch and open a pull request that describes the change. Do not merge or push directly to `main` unless the user explicitly requests it. Leave previously published runs and commit history as they are. Do not report resource-stopped or unexecuted checks as passing, and do not remove workflow definitions merely because hosted execution is unavailable.

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
