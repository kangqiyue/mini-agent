# Terminal UI and approval improvements

- `README.md`: artifact map and validation notes.
- `cache/`: task-owned test and visual-QA scratch, ignored by Git.
- `*.log`: validation output, ignored by Git.
- `pytest-verified.log`: final full regression result, ignored by Git.
- `cache/dist/`: the tested local wheel, ignored by Git.
- `cache/uv/`: package build/install cache, ignored by Git.

Source changes live in `src/mini_agent`; regression tests live in `tests`.

The UI now renders committed tool activity, supports `/output brief|detailed`
and `/history`, fits status blocks to narrow terminals, and pauses Rich live
rendering during explicit approval input. `/permissions` describes existing
rules. Explicit `--auto-approve` now works for chat and resume as well as
headless runs; it uses the shared allow-once decision, does not create durable
command grants, and displays the active mode. No LLM approval is involved.

Validation:

```text
pytest tests scripts/local_gateway_startup/test_startup.py:
  1048 passed, 1 opt-in local integration test skipped
ruff check src tests scripts: passed
pyright: 0 errors, 0 warnings
wheel build and archive-content checks: passed
installed package: all 51 Python sources match the working tree
```

The manual gateway script and its tests are in
[`scripts/local_gateway_startup`](../../scripts/local_gateway_startup/README.md).
Independent terminal acceptance is recorded in
[`reports/mini_agent_acceptance_20260906`](../mini_agent_acceptance_20260906/results.md).
The tests include durable-before-display ordering, redaction, display failure
recovery without replay, output bounds, nonzero command exits, safe default
approval, and mode retention while switching sessions. Disposable pytest
workspaces were removed after validation; the test log and wheel remain in this
task directory.

The first regression run after adding interactive auto approval exposed two
existing timing assumptions. The one-token request-floor test now uses the same
typed runtime snapshot for estimation and validation, instead of repeated live
Git probes. The detached-pipe deadline test now waits for the child to confirm
`setsid()` before timing capture, and explicitly terminates that test-owned child.
Both targeted checks passed after the changes.
