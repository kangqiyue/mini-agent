# Interactive CLI acceptance

This directory contains repeatable PTY-based acceptance checks for the interactive
CLI changes requested on 2026-09-06.

- `acceptance_pty.py`: runs the source CLI with a deterministic in-process fake
  provider, real local tools, and real session persistence inside `cache/`.
- `installed_cli_acceptance.py`: runs the user-installed entry point in an
  isolated workspace without contacting a provider.
- `gateway_smoke.py`: one bounded read-only request through the installed entry
  point and local LiteLLM gateway; it uses only a synthetic fixture in `cache/`.
- `baseline.md`: findings captured while the earlier UI was still active.
- `results.md`: integrated acceptance and targeted regression outcomes.
- `cache/`: generated PTY transcripts and isolated workspaces. The harness
  produces brief-mode width captures, allow/deny/empty/invalid approval
  captures, auto-approve chat/resume captures, a spinner-pause capture, a
  detailed `/history` capture, and a `/permissions` capture. It contains no
  credentials and is ignored by Git.

Run from the repository root:

```bash
.venv/bin/python reports/mini_agent_acceptance_20260906/acceptance_pty.py
```
