# Develop workspace pages

1. Start the API and development frontend using [the setup guide](getting-started.md#develop-the-frontend).
2. Edit the shared shell in `WorkspaceShell`, `Icon` and `ThemeToggle`. Update shared styling in `apps/web/src/app/globals.css`.
3. Edit workspace routes under `apps/web/src/app/(workspace)/` and section components under `apps/web/src/components/workspace-sections/`.
4. Update [the workspace guide](workspace-guide.md) when changing navigation or run steps. Open `/docs/` to check the API reference.

From the repository root, run:

```sh
pnpm check:web
pnpm build:web
pnpm exec playwright install chromium
pnpm test:web
```

Check the edited pages in light and dark themes, at desktop and narrow mobile widths, and with keyboard navigation. See [development checks](ci.md) for the full commands.

## Operational interface

OpenJensen is a scientific workbench. Use the shared page header, controls, spacing
and theme tokens. Feature pages should not add promotional banners, slogans,
decorative badges, or a second introduction below the page title.

Use short names for operations and measured state. Put the next required instruction
beside its control; keep technical background in documentation or a disclosure.
Do not repeat field labels, navigation labels, and instructions as subtitles. Keep
consequential action details (torque, persistent writes, incomplete recovery) visible
at the action that needs them.

Robot illustrations must distinguish the model from live telemetry. Use the actual
mechanism and joint axes; motion previews should be explicit and bounded, with a
static alternative for reduced motion. Verify them in light and dark themes and at
mobile widths, alongside keyboard controls.
