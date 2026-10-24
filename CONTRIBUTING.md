# Contributing to OPEN JENSEN

Use [GitHub issues](https://github.com/OpenJensen/openjensen/issues) for reproducible bugs and bounded proposals. Include the source revision, OS, relevant tool versions, steps, expected behavior and redacted error output. Follow [security reporting](SECURITY.md) for sensitive concerns.

Read [AGENTS.md](AGENTS.md) and the affected source before editing. Use the [pinned source setup](README.md#run-locally), keep ML workers in their separate environments, and run the checks relevant to your change from [docs/ci.md](docs/ci.md). Regenerate the OpenAPI schema/client when the API changes; do not hand-edit generated types. Format web and browser-test changes with `pnpm format` and check them with `pnpm lint:web`; Python uses `ruff format`.

Keep changes focused and submit a feature-branch pull request describing the problem, resulting behavior and tests you ran. Include any checks you could not complete. Preserve user workspaces, credentials and saved artifacts.

Project code is licensed under [Apache License 2.0](LICENSE); [NOTICE](NOTICE) records project notices. Third-party code, datasets, model weights and assets retain their own terms and attribution. Adding an adapter or fixture does not grant redistribution rights for its upstream model or data.
