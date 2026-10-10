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

Check the edited pages in light and dark themes, at desktop and narrow mobile widths, and with keyboard navigation. See [local verification](ci.md) for the full commands.
