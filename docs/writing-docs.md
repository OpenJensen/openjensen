# Write documentation

These are the rules for writing or editing any documentation in this repository: the README, guides in `docs/`, worker and app READMEs, API references, and the in-app guide. They apply to AI agents and human contributors alike. `AGENTS.md` points agents here.

The readers are robotics researchers and operators. They skim, then act. Every document should let them find the next step in seconds and trust that it is correct.

## Before you write

1. **Read the code the document describes.** Commands, paths, settings, limits and UI labels come from the source, not from older docs. Search `packages/core/src`, `apps/web/src` and `workers/`.
2. **Find the existing document.** Edit it rather than adding a new one. One topic lives in one place; other documents link to it.
3. **Find who links to it.** Run `grep -rn "<file>.md#" --include=*.md --include=*.ts --include=*.tsx --include=*.py .` before renaming a heading.
4. **Check what reads it.** `docs/workspace-guide.md` is built into the app at `/guide/`, and some tests read documentation files. Run `grep -rln "<file>" tests apps/web/src workers` before moving or deleting one.

## Choose the document type

| Type | Purpose | Shape | Example |
| --- | --- | --- | --- |
| README | The front page: what this is, what it does, how to start | Summary, screenshot, capability table, quick start, where to go next | `README.md` |
| Guide | Help someone finish one task | Steps in the order the reader works | `docs/getting-started.md`, `docs/so101-robot-lab.md` |
| Reference | Exact contracts | Endpoint, setting and option tables, then details | `workers/robot_lab/API.md`, `docs/compute-settings.md` |
| Worker or app README | Run and understand one component | Run, dependencies, behavior under headings, tests | `workers/robot_lab/README.md` |
| In-app guide | One short section per page, in sidebar order | Two to five sentences per page and links | `docs/workspace-guide.md` |
| Release notes | What changed for users in a release | Bullets grouped by area, then upgrade notes | `docs/releases/` |

## Structure a guide

```markdown
# <The task, as a verb phrase>

<One or two sentences: what this is for and what the reader ends up with.>

## What you need
## 1. <First stage>
## 2. <Next stage>
## Use it
## Reference
## Troubleshooting
## Technical notes
```

- Order sections the way the reader works: requirements, setup, use, then reference, troubleshooting and technical notes.
- Use numbered steps for a sequence, bullets for options and rules, and tables for paths, settings, environment variables, commands and limits.
- Put safety warnings, billing consent and data-loss risks at the step where they apply, and repeat a critical one in a short **Safety** section when the task involves hardware.
- Move implementation detail (protocol internals, algorithms, history) to **Technical notes** at the end, or to the component's README.

## Write

- **Short.** One idea per paragraph and no paragraph over 80 words. Sentences under about 25 words.
- **Direct.** Second person, active voice, present tense: "Open **Datasets**", "the server stops the job".
- **Current.** Describe what the software does today. Never write changelog phrasing such as "now", "no longer", "former", "previously", "was removed" or "new". History belongs in release notes and commit messages.
- **Concrete.** Name the exact control, command, file or value. Avoid "simply", "just", "easily", "powerful", "seamless" and other filler or marketing words.
- **Honest.** Mark experimental features as experimental. Say what a check does not prove only where a reader could otherwise be misled, for example that simulated tests do not prove a real arm works.
- **Formatted.** Bold exact UI labels (**Inspect dataset**). Use code formatting for commands, paths, settings, keys, endpoints and file names. Use fenced code blocks with a language for anything the reader runs or copies.

## Use the product's names

| Thing | Name |
| --- | --- |
| Product | OpenJensen (never "OPEN JENSEN" or "Open Jensen") |
| Command line and settings | `jensen`, `JENSEN_*`. Mention `firebird` and `FIREBIRD_*` only in compatibility notes |
| Sidebar group | ROBOT LAB, containing Initial setup and AI teleoperation |
| Pages | Dashboard, Initial setup, AI teleoperation, Datasets, Augmentation, Fine-tune, Distill, Quantize, Evaluate, Run, Decision Lab, Models, Cloud runs, Settings |
| Settings tabs | Compute, Modules, Workflow settings, Diagnostics (write **Settings → Modules**) |
| Initial setup steps | Connect, Motor IDs, Calibrate, Teleoperate, Record |
| Datasets form | **Import a dataset**, with **Hugging Face** and **Local files** |

Confirm any other label by searching `apps/web/src`. Never use internal codenames for features.

## Keep it accurate

- Keep every command, path, setting, config key, version pin, limit and warning when you restructure. If you delete a sentence, its facts must survive elsewhere.
- When a document disagrees with the code, fix the document to match the code and say what you corrected in the commit message.
- Pin exact versions where the project pins them (Python 3.14.7, Node 24.21.0, pnpm 12.6.0, uv 0.12.19).
- Run the commands you document when it is practical and safe. Never run hardware, GPU, paid-cloud or long jobs only to document them.

## Keep it public-safe

This repository is public. Never put any of these in documentation, screenshots or commit messages:

- Local test results, benchmark numbers or run logs from someone's machine.
- Maintainers' or contributors' machine names, hardware, hostnames, usernames, private project names, local paths or tokens.

Use placeholders instead: `application-server` for an SSH host, `/absolute/path/to/...`, `PROJECT_ID`, `JOB_ID`, `COPY_THE_PRIVATE_TOKEN`.

## Screenshots

- Capture the current interface at 1440×900 in the light theme, from the browser-test fixtures or a clean workspace.
- Check every pixel for machine names, hardware, usernames and private projects before committing.
- Show only public datasets, with a caption naming the dataset and its license.
- Store images in `docs/images/`, list them in `docs/images/README.md`, and delete images no document uses.
- Write alt text that says what the image shows, not "screenshot".

## Links

- Use relative links between repository files, and link the first mention of another guide.
- Renaming a heading changes its anchor (`## 2. Open a dataset` becomes `#2-open-a-dataset`). Update every link to it, including links in `apps/web/src`.
- CI checks every relative link and anchor on each change. Run the same check locally:

  ```sh
  python3 .github/scripts/check_doc_links.py
  ```

## Before you commit

- [ ] The title says the task and the opening says what the reader gets.
- [ ] Sections follow the reader's order; steps are numbered; settings and paths are in tables.
- [ ] No paragraph is over 80 words, and nothing describes past behavior.
- [ ] Commands, paths, settings, limits and UI labels match the code.
- [ ] Safety, billing and data-loss warnings sit at the step they apply to.
- [ ] No local results, machine details or private names in the text, images or commit message.
- [ ] `python3 .github/scripts/check_doc_links.py` reports no broken links.
- [ ] Each independent documentation change is its own commit, and the message names any factual corrections.
