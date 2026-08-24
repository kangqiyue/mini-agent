# Mini Agent

English | [简体中文](README.zh-CN.md)

Mini Agent is a small, observable local coding agent for experimenting with
long-horizon runtime design. The implementation favors explicit state,
append-only events, and simple typed modules.

The documentation ownership map is in [docs/INDEX.md](docs/INDEX.md). As-built
module boundaries and invariants are in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); product goals and acceptance
criteria are in [docs/PRD.md](docs/PRD.md). See
[SECURITY.md](SECURITY.md) for the trust boundary and vulnerability reporting,
[CONTRIBUTING.md](CONTRIBUTING.md) for development rules, and
[docs/RELEASING.md](docs/RELEASING.md) for the publication checklist. This
project is licensed under [Apache License 2.0](LICENSE).

## Current milestone

The current v0.1 release candidate implements the M4 goal and evaluation scope
on top of the bounded, approval-gated coding runtime. It is not yet a stable
v0.1 release:

- canonical typed messages;
- canonical assistant tool calls and `role="tool"` results;
- an OpenAI-compatible provider boundary;
- append-only JSONL events with truncated-tail recovery;
- single-writer session create, resume, inspect, and stop;
- interrupted/unknown tool-call recovery without automatic replay;
- bounded workspace reads and search;
- one-file, expected-content `apply_patch`;
- explicit-argv `exec_command` with no shell expansion;
- approval events persisted before any write or command starts;
- redacted artifacts plus bounded session-history retrieval;
- conservative full-request budgeting with provider capability clamps;
- incremental, transcript-committed checkpoints with a single writer;
- automatic and manual context rebuild across logical cycles;
- deterministic overflow recovery and optional model-backed checkpoint extraction;
- one durable, editable goal whose acceptance criteria remain in model context;
- a mechanical completion gate over persisted assistant/tool/artifact evidence;
- versioned information-atom fixtures and deterministic session evaluation;
- a three-task read/patch/exec benchmark with explicit unavailable usage/cost/human fields.

## How it works

The durable append-only event stream is the source of truth. For each turn, the
agent persists the user message, builds a bounded model request from validated
session state, persists the assistant response, and then executes any requested
tools through preflight checks and approval gates. Every side effect is enclosed
by durable `tool_started` and terminal events.

```text
user input
  → append event
  → build bounded context
  → call model
  → append assistant/tool-call events
  → preflight and approval
  → append tool_started
  → execute tool
  → append completed/failed/unknown result
  → continue the model loop
```

Context compression never rewrites or summarizes away the transcript itself.
Checkpoints capture incremental working state, while request rebuilding selects
the system instructions, authoritative goal, latest checkpoint, and recent
atomic tool rounds that fit the budget. Omitted exact details remain available
through bounded history and artifact retrieval.

On resume, the session is reconstructed from events. An incomplete read is
recorded as interrupted; an incomplete write or command is recorded as unknown
and is never replayed automatically. Goal completion is accepted only when each
criterion references matching durable evidence.

## Install

Mini Agent supports macOS and Linux with Python 3.12 through 3.14. Windows is
not currently supported.

The built-in `search` tool requires [ripgrep](https://github.com/BurntSushi/ripgrep)
as the `rg` executable on `PATH`; a Python wheel cannot install this system
binary. Install ripgrep with your operating-system package manager and verify
it before starting:

```bash
rg --version
```

Without it, the agent remains installable but `search` fails explicitly with a
`missing_dependency` tool error.

Install the command from a source checkout:

```bash
git clone https://github.com/kangqiyue/mini-agent.git
cd mini-agent
uv tool install .
mini-agent --help
```

This installation path uses [uv](https://docs.astral.sh/uv/). Alternatively,
install the checkout into an activated Python 3.12-3.14 virtual environment with
`python -m pip install .`. For development, use the locked, two-stage setup in
[Development](#development) so the pinned build backend is installed before the
project is built without isolation. That toolchain pins
`hatchling==1.32.0` and its required editable-install helper, `editables==0.6`.
Repository development and release checks require `uv==0.11.28` so lock and
build behavior do not drift between runs.

The project has not published the v0.1 package yet. To test a built release
candidate without relying on a source checkout, install its wheel directly:

```bash
python -m pip install /path/to/mini_agent-0.1.0rc1-py3-none-any.whl
mini-agent --help
```

After an official package-index release exists, the equivalent package-name
installation will be documented with that release. Do not assume the name is
already available from a public index.

## Run

```bash
mini-agent                                      # chat in the current directory
mini-agent init-config --workspace /path/to/project
mini-agent chat --workspace /path/to/project
mini-agent sessions --workspace /path/to/project
mini-agent resume                               # latest active session here
mini-agent resume SESSION_ID --workspace /path/to/project
mini-agent inspect SESSION_ID --workspace /path/to/project
mini-agent evaluate SESSION_ID --fixture fixture.json --output report.json
mini-agent benchmark --output benchmark.json
```

Running `mini-agent` without a subcommand is equivalent to
`mini-agent chat --workspace .`. The explicit `chat` command remains available
for scripts and for choosing another workspace.

Chat startup is lazy: no durable session is created until the first ordinary
request. Exiting, interrupting, or closing an untouched prompt therefore leaves
no empty session. Local `/system`, `/context`, `/goal`, and `/resume` commands
remain available before persistence begins. Lifecycle-only sessions created by
older versions are hidden from `sessions` and implicit resume selection, but
are not deleted.

`mini-agent resume` shows matching session history in an interactive terminal,
ordered by last activity with the latest user-message preview. Press Enter for
the latest session, or choose a number or session id. In a non-interactive
script it resumes the latest match automatically. Passing `SESSION_ID` keeps
the direct exact-selection behavior. After resume, the terminal replays the
latest 30 visible user/assistant messages before the next input; tool and
lifecycle events do not consume this display limit.

Inside chat, `/system` shows the exact assembled system prompt for the next
request, `/context` shows the derived request view, `/checkpoint` commits
incremental working state, and `/compact [focus]` checkpoints then starts a new
bounded context cycle. `/goal` shows the durable objective and criteria.
`/resume` opens the same history selector inside chat, while `/resume
SESSION_ID` switches directly to another session with the same workspace and
model after releasing the current writer. `/exit` ends the interaction and
persists the session's stopped state. There is no separate `/model` command:
the startup panel shows the model name, and `/system` shows it under `Runtime
context`'s `Model` field. The
append-only transcript is never rewritten; exact older details remain available
through history and artifact retrieval tools.

Goal commands are deliberately explicit:

```text
/goal set Ship and verify the change
/goal criteria [{"criterion_id":"tests","description":"Tests pass","evidence_kind":"command_succeeded"}]
/goal complete [{"criterion_id":"tests","event_ids":[42],"note":"pytest exited zero"}]
/goal block Waiting for an external service
/goal resume
/goal cancel
```

The first ordinary user request creates an editable objective if no goal exists.
A goal without criteria cannot complete. `completed` requires one evidence item
per criterion, references existing durable events, and checks the requested
mechanical kind (`assistant_response`, `tool_completed`, `command_succeeded`,
`file_modified`, or `artifact_created`). The runtime injects this authoritative
goal state into every model request; model prose cannot bypass the gate.

`mini-agent evaluate` is read-only: it loads a versioned information-atom
fixture and calculates retention, state precision/recall/F1, stale facts,
retrieval, false completion, recovery, cycle counts, and estimated request
tokens from validated durable state. `mini-agent benchmark` runs three isolated
synthetic coding workspaces through the actual runtime read → patch → exec →
completion gate → three-rebuild path. It does not call an external model. Provider usage,
cost, and human evaluation remain explicitly `unavailable` unless real data is
supplied; the tool never fabricates these measurements.

Reads are automatic. Every command and every eligible file replacement without
an existing exact-path grant asks for an explicit terminal decision before
`tool_started` is persisted. A valid, non-sensitive `apply_patch` can be
granted for one exact relative file path for the rest of the session; each
later patch's complete redacted parameters remain persisted for audit.
Credential-shaped or already-redacted replacement content is rejected before
approval and is never written.
`exec_command` is always approved once per call, accepts an `argv` array, and
never a shell string. Credential-shaped or already-redacted command arguments
are rejected before approval.

`exec_command` is not an operating-system sandbox. It resolves command names
from the current user `PATH`, then runs the resolved executable with a minimal
environment and terminates the command's original process group on timeout,
but a program that deliberately detaches into a new session may outlive that
cleanup. Review each command before approving it.

Workspace path checks reject absolute paths, parent traversal that escapes the
workspace, excluded runtime data, and symlink escapes. File tools and command
working directories traverse from a verified workspace root through held,
no-follow directory descriptors; mutations also revalidate the bound parent
when its identity matters. This closes ordinary pathname replacement races, but
it is not an operating-system sandbox: another process running as the same user
can still modify workspace contents, and approved commands retain that user's OS
access. Use Mini Agent only in a workspace whose filesystem namespace you trust
while the session is active.

## Development

```bash
uv sync --extra dev --locked --no-install-project
uv sync --extra dev --locked --no-build-isolation
uv run pytest
uv run ruff check .
uv run pyright
uv build --no-build-isolation
uv run mini-agent benchmark --output reports/v0.1-benchmark.json
```

These commands are also the ordinary local release checks. The maintainer
checklist adds clean-root history scanning, a private denylist pass, and a fresh
wheel installation; see [docs/RELEASING.md](docs/RELEASING.md).

## Configuration

Install the complete packaged example into the workspace, then replace the
provider values:

```bash
mini-agent init-config --workspace /path/to/project
# edit /path/to/project/.mini-agent/config.toml
export MINI_AGENT_API_KEY="<provider-key>"
mini-agent chat --workspace /path/to/project
```

`init-config` works from an installed wheel or a source checkout, creates a
private file, and refuses to overwrite an existing configuration. The real
`.mini-agent/config.toml` is ignored by Git. The example documents every
supported field, including each actual default or an explicitly disabled
optional example, and is validated by the test suite. Source-checkout users can
also review or copy
[.mini-agent/config.example.toml](.mini-agent/config.example.toml) directly.
Configuration lookup order is an explicit `--config` path, then
`<workspace>/.mini-agent/config.toml`, then the platform-specific user config
directory. Relative `runtime.data_dir` values are resolved from the directory
containing the selected config file. The documented optional `data_dir = "data"`
therefore resolves to `.mini-agent/data/`, which this repository ignores. If you
choose another repository-local data directory, add it to that project's Git
ignore rules before using it.

For a local LiteLLM gateway, first discover its registered model ids and use the
exact id returned by that gateway. Set `MINI_AGENT_LITELLM_MODEL` explicitly
before enabling the optional local integration test; it deliberately has no
default model:

```toml
[model]
provider = "openai-compatible"
model = "<gateway-model-id>"
base_url = "http://127.0.0.1:4000/v1"
api_key_env = "LITELLM_API_KEY"
```

Loopback HTTP is allowed for local gateways; remote providers must use HTTPS.
The configuration stores only the credential environment-variable name, never
the credential value. See [.mini-agent/config.example.toml](.mini-agent/config.example.toml)
for all supported fields, actual defaults, and disabled optional examples.

The `SystemPromptAssembler` builds one authoritative leading system message for
every model request. Cache-stable sections come first: Identity, Operating
rules, optional Project instructions, Available tools, and static runtime facts.
Before a main-model or checkpoint-model request leaves the process, known
workspace and home-directory prefixes are normalized to `<workspace-root>` and
`<home>`. This applies to the system prompt, conversation/tool views, history,
checkpoint input, and custom tool schemas. Automatically collected Git branch
telemetry is reduced to a category before it enters the system prompt. The
runtime probes only repository/ref metadata and deliberately does not collect
working-tree status, so it cannot invoke repository clean/smudge filters or
status-related hooks. This is not branch-name DLP: branch text present in
conversation, project content, or approved tool output can still reach the
provider. Git metadata, Goal, and the latest Checkpoint remain at the end so their
changes preserve as much reusable provider prefix as possible. The current
OpenAI-compatible adapter relies on provider-side automatic prefix caching; it
does not yet send provider-specific `cache_control` fields or report cached
tokens. Use `system_prompt.instructions_file` for workspace-specific
instructions; the file must pass the same sensitive-path boundary as workspace
read tools (including configured runtime stores) and is classified as complete
bounded bytes before use, rejecting PEM private keys while allowing public
certificates. Its contents, conversation
messages, selected file contents, and tool results are sent to the configured
provider as needed, so do not place
confidential organization information there unless that provider is approved
to receive it. This normalization is not general DLP: other absolute paths,
organization names, domains, and project contents can still be disclosed.
Rich also prints a compact model/Git/context/checkpoint/goal status line before
every input, for example:

```text
provider/model-name │ git main │ ctx ~31.2k/83.6k 37% │ checkpoint v2+ │ goal active
```

Because automatic Git telemetry does not collect working-tree status, the line
never uses `*` for uncommitted changes. `~` marks a conservative local token
estimate, and `+` after the Checkpoint version means newer Transcript events
are not yet included in that Checkpoint. `/context` remains the detailed view.

## License

Copyright 2026 kangqiyue. Mini Agent is licensed under the
[Apache License 2.0](LICENSE). See [LICENSE](LICENSE) for the complete terms.
