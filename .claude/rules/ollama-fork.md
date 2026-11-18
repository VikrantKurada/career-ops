# This checkout is the ollama-local fork

It adds a local Ollama runner (`docs/OLLAMA.md`) and takes upstream changes through git, never through the updater:

```
git fetch upstream
git merge upstream/main
```

What to do before and after that merge is in `docs/OLLAMA.md`, "Keeping the fork in sync with upstream". Follow it when you run the merge.

This rule takes precedence over the "Update Check" section of AGENTS.md and over `modes/update.md`, the mode behind `/career-ops update` and "update career-ops". Asked to update, do not run that mode: the release is taken with the two git commands above. A release check does not show whether there is something to take: the merge takes upstream's `main`, and the check looks at releases only. Three things of that section stay in force: the session check; `node update-system.mjs check --force` when the user asks for a check; and, in an unattended run, say nothing about any update, apply none and dismiss none.

- Never run or offer `node update-system.mjs apply`, `node update-system.mjs rollback`, `npm run update` or `npm run rollback` here. The updater half-updates a fork and then blocks the git merge, and its rollback removes the fork's uncommitted edits. One exception: when the updater has already run in this checkout, follow "If the updater ran anyway" in `docs/OLLAMA.md`.
- The session update check in AGENTS.md still runs. Except in an unattended run, when it, or a check the user asked for, reports `update-available`, do not ask "Want me to update?". Say first, in your reply, that a new upstream release is out (v{local} → v{remote}) and that this fork takes it with the two git commands above. In the same turn, without waiting for an answer, run `node update-system.mjs dismiss --version {remote}`, so that the same release is not announced again. A newer release is announced again.
