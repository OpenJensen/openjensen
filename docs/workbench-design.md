# Develop workspace pages

1. Start the API and development frontend using [the setup guide](getting-started.md#work-on-the-web-interface).
2. Edit the shared shell in `WorkspaceShell`, `Icon` and `ThemeToggle`. Update shared styling in `apps/web/src/app/globals.css`.
3. Edit workspace routes under `apps/web/src/app/(workspace)/` and section components under `apps/web/src/components/workspace-sections/`.
4. Update [the workspace guide](workspace-guide.md) when changing navigation or run steps. Open `/docs/` to check the API reference.

From the repository root, run:

```sh
pnpm format
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

Write status and error messages as one short sentence: what happened, then the next
step. Do not list what an action does not do or what has not been measured; state
measured results and leave validation scope to the documentation. Consent text names
the recipient and any charge once, at the checkbox.

## Names

Call the product OpenJensen. Use one name per page everywhere it appears: the
sidebar, page title, browser title, guide heading and any button that opens it
(Dashboard, Robot Lab, Teaching, Datasets, Augmentation, Fine-tune, Distill,
Quantize, Evaluate, Run, Decision Lab, Models, Cloud runs, Settings). Buttons that
open a page say "Open <page>"; links back to a list say "← Back to <list>". Do not
end link text with an arrow, and do not put internal model codenames in labels;
name what the action does.

## Shared components

Use the components in `apps/web/src/components/ui/` and the classes in
`apps/web/src/app/design-system.css` instead of feature-specific copies:

- `Notice` for every message box: a short `title`, the next step as children, raw
  driver or server output in `details` (collapsed), and `live="alert"` for failures.
- `StatusBadge` for state labels such as Connected, Not installed or Failed.
- Buttons: `primary-button` for the one main action in a view, `secondary-button`,
  `danger-button` for stopping hardware or deleting, and `text-button` for inline
  link-styled actions.

Older classes such as `error-notice`, `warning-box`, `metadata-badge` and
`text-link` render with the same styles until their components move to the shared
markup. Do not add new feature-specific notice, badge or button styles.

Robot illustrations must distinguish the model from live telemetry. Use the actual
mechanism and joint axes; motion previews should be explicit and bounded, with a
static alternative for reduced motion. Verify them in light and dark themes and at
mobile widths, alongside keyboard controls.
