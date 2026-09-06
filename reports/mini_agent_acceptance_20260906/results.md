# Acceptance results

The deterministic PTY harness passed against the integrated CLI changes.

- Brief output at 40, 80, and 120 columns keeps status fields intact and prints
  a tool start/completion summary without leaking the full result.
- Detailed output displays bounded arguments and the persisted tool result;
  `/history` displays that saved result again.
- Command approval displays its command, directory, scope, and plain `o`/`d`
  choices. Allow-once executes; deny, empty, and invalid input do not execute.
- The delayed-provider capture shows no work indicator inside the approval
  interval, and the prompt's pause/resume context contract passes.
- `/permissions` reports the current policy before a session exists.

Relevant source test runs passed:

```text
tests/test_terminal_ui.py tests/test_cli.py: 67 passed
tests/test_session.py tests/test_agent_approval.py tests/test_permissions.py: 93 passed
```

The startup script passed `bash -n` and its help path. Read-only review found
that repeat runs only start existing Colima/Compose services and wait for the
local health endpoint; it does not create services or read credentials.

Final installed-entry validation passed with `~/.local/bin/mini-agent --help`
and an isolated PTY chat that ran `/output detailed`, `/permissions`, and
`/exit` without creating a session or contacting a provider. The same installed
environment passed the full deterministic PTY harness.

The local gateway health endpoint responded successfully, and its registry
contained the exact requested `cfuse/DeepSeek-V4-Flash-0731` identifier. One
read-only smoke task then passed through the installed entry point with a
synthetic fixture: no auto-approval, retries disabled, at most three model
calls, a 512-token output limit, and a 60-second process timeout. Credentials
were read only from `LITELLM_API_KEY` and were neither printed nor persisted.

After installing the auto-approve build, isolated installed-entry PTY checks
passed for both `mini-agent chat --auto-approve` and
`mini-agent --auto-approve`: each ran a harmless synthetic `pwd` command
without displaying an approval prompt. The default entry without the flag, with
an empty approval response, displayed the command review and denied execution.
The installed `resume SESSION_ID --auto-approve` path also passed against a
persisted synthetic session. The initial resume diagnostic established that a
bare interactive `resume` first asks the user to select a session; the harness
now supplies the persisted session identifier before testing command execution.
