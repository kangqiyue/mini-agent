# Architecture

This document describes the v0.1 release-candidate implementation as it exists.
Product goals and future work belong in [PRD.md](PRD.md); installation and CLI
usage belong in the [README](../README.md); release operations belong in
[RELEASING.md](RELEASING.md).

## System shape

Mini Agent is one local foreground agent with a restricted checkpoint writer.
It is deliberately not a plugin framework, distributed workflow system, or
operating-system sandbox.

```text
CLI / terminal UI
        │
        ▼
Agent loop ───────────────► Provider adapter
   │  │                         │
   │  ├──► Context projection  canonical response
   │  ├──► Goal gate
   │  └──► Tool runtime ──────► Workspace side effects
   │             │
   ▼             ▼
Session ───► append-only events ───► artifacts / checkpoints
   │
   └──► recovery validates lifecycles before resume
```

Dependencies flow inward from CLI orchestration to typed domain boundaries.
Provider adapters translate external JSON into canonical models; the agent loop
does not parse provider-specific payloads. Persistence records facts and
lifecycle boundaries; it does not decide agent actions.

## Module ownership

| Capability | Primary modules | Responsibility |
|---|---|---|
| Entry points | `cli.py`, `headless.py`, `terminal_ui.py`, `storage_paths.py` | Commands, one-turn headless execution and reporting, local rendering, config/session selection |
| Orchestration | `agent.py` | Turn loop, provider-call budget, tool ordering, context transitions |
| Request context | `context.py`, `system_prompt.py` | Conservative request sizing, projections, provider-facing prompt |
| Provider boundary | `provider.py`, `providers/openai_compatible.py`, `messages.py` | Canonical requests/responses and bounded external protocol parsing |
| Durable sessions | `session.py`, `events.py`, `event_store.py`, `session_recovery.py` | Append-only state, publication, validation, crash recovery |
| Checkpoints | `checkpoint.py`, `checkpoint_writer.py`, `checkpoint_model.py` | Immutable checkpoint storage, serialized extraction, deterministic fallback |
| Tools and approval | `tools/`, `workspace.py`, `workspace_directory_fd.py`, `workspace_subprocess.py`, `permissions.py`, `tool_facts.py` | Workspace validation, directory-descriptor anchoring, approval scopes, execution, structured completion facts |

Interactive CLI rendering observes typed session events only after the event
store has redacted and durably appended them. The scoped observer does not decide
permissions or modify the transcript. A renderer failure propagates and stops
the turn; recovery reads the durable event to distinguish completed operations
from unknown side effects. Terminal verbosity affects display only, and tool
detail views are bounded. Headless execution does not install this observer.
| Large results and retrieval | `artifacts.py`, `history.py` | Redacted externalization and bounded exact-data recovery |
| Task state and evidence | `goal.py` | Durable goal lifecycle and mechanical completion gate |
| Evaluation | `evaluation.py`, `benchmark.py` | Deterministic information-retention and runtime regression evidence |
| Configuration | `config.py`, `config_template.py` | Strict Pydantic boundary and packaged complete example |

A module should not absorb a second row's responsibility merely to avoid a
small typed interface. Split a module when its parsing, decision, persistence,
or side-effect rules can no longer be understood locally.

## One turn

1. Validate the user input and active Goal state.
2. Append the user message and create a default Goal when needed.
3. Build or extend a bounded context projection.
4. Coordinate a due checkpoint or rebuild before the next provider request.
5. Consume one shared provider-call budget unit and append
   `model_request_started`.
6. Append either `assistant_message` or `model_request_failed`.
7. For each tool call, persist request, approval, start, and exactly one terminal
   result around the real side effect.
8. Continue until a text response, an observable failure, or the hard per-turn
   provider-call cap returns control.

Main responses, checkpoint-model calls, provider retries, and overflow recovery
share `max_model_calls_per_turn`. An automatic checkpoint reserves one call for
the main/recovery response and uses the deterministic extractor if no surplus
capacity remains. Manual checkpoint and compact commands receive their own
operation budget.

## Durable lifecycle invariants

`events.jsonl` is the source of truth and is never rewritten. Derived context,
checkpoint indexes, and session summaries are rebuildable views.

```text
model_request_started ──► assistant_message | model_request_failed

assistant tool call ──► tool_requested ──► approval? ──► tool_started
                                                └─────► tool_completed
                                                └─────► tool_failed
                                                └─────► tool_interrupted

checkpoint_started ──► checkpoint_committed | checkpoint_failed

rebuild_started ─────► rebuild_completed | rebuild_failed
```

- A model start has one matching terminal event. Resume records an interrupted
  request as a non-retryable model failure; it never invents a response.
- Assistant events optionally retain validated provider token usage. Headless
  totals cover only the current turn's main-model responses; missing counts
  stay unknown rather than contributing zero to a partial total.
- A tool terminal is unique. A started non-read-only tool without a terminal is
  recovered as `unknown` and is never automatically replayed.
- File creation publishes complete content with an exclusive link in an
  existing workspace directory. Once publication starts, an ambiguous failure
  is `unknown`; a target collision never overwrites the concurrent file.
- A checkpoint commit must match its immutable stored content, version, and
  watermark. Failure does not advance the visible checkpoint.
- On reopen, durable checkpoint registrations are validated before any cleanup.
  A staged candidate without a matching transcript commit is ignored by
  read-only consumers and reconciled by a writable resume, so a crash cannot
  reserve the next checkpoint version or discard committed state.
- Only `rebuild_completed` may advance `cycle_id`, exactly once and only after
  its source/checkpoint facts match the start event. Failure stays in the source
  cycle.
- In-memory cycle and projection state changes only after the durable terminal
  append succeeds.
- New event records use schema v2 and must validate strictly. Published v1
  records are migrated only in memory at the read boundary; compatibility
  flags are injected only while decoding that recognized legacy shape, and the
  append-only file is not rewritten.
- A successful session creation has durably published its final marker. Any
  final directory-fsync failure is surfaced as an ambiguous failure, never a
  reported success.
- Provider, session-stop, or writer-close cleanup failures produce a nonzero
  CLI result. A writer-close failure also prevents an in-process session switch;
  an append/fsync failure is reported as an unknown outcome because the terminal
  event may already be complete on disk.

## Context and checkpoint policy

The request budget uses a tokenizer-independent, conservative UTF-8 estimate.
It accounts for system text, tools, messages, protocol overhead, and output
reserve. Provider capabilities and configured overrides can only reduce the
physical limit.

The transcript remains complete. A projection may omit old rounds only as
atomic units, keeping an assistant tool call with every tool result. The latest
user request is required and may be visibly truncated only when necessary to
fit; exact content remains in history.

Checkpoint extraction is incremental over a fixed event watermark. The
optional model extractor has no tools and must return a validated typed draft;
expected extraction failures fall back to deterministic state. Provider output
tokens and durable checkpoint bytes are separate limits. Rebuild injects a
bounded checkpoint view, while the complete validated checkpoint remains in
local durable storage for audit and later retrieval.

## Trust and privacy boundaries

- Credentials enter through the configured environment-variable name. Config,
  events, artifacts, errors, and examples must never contain credential values.
- External provider responses are bounded and validated before entering core
  state. Remote endpoints require HTTPS; plain HTTP is restricted to loopback.
- Provider requests necessarily contain the conversation, selected project
  instructions/files, tool results, Goal, and checkpoint state required for the
  task. Users must trust the configured provider for that content.
- Before main-model and checkpoint-model calls, known workspace and home
  prefixes are normalized to `<workspace-root>` and `<home>` across messages,
  tool/history views, checkpoint input, and custom tool schemas. Exact workspace
  metadata remains local for resume. Automatically collected Git branch
  telemetry is reduced to a class before system-prompt assembly. It uses only
  repository/ref probes and deliberately does not collect working-tree status,
  avoiding repository clean/smudge filters and status-related hooks. Branch
  text in task content or approved tool output can still reach the provider.
- Configured project instructions pass the workspace lexical and resolved
  sensitive-path policy, including configured runtime-store exclusions, before
  they are opened. Their complete bounded bytes are classified for PEM
  private-key boundaries; public certificates remain eligible.
- Path normalization is not general data-loss prevention. Other absolute paths,
  organization names, domains, project instructions, and selected file content
  can still reach the configured provider.
- Workspace tools accept relative paths and reject static escapes and excluded
  runtime data. File reads, search, patch replacement, project-instruction
  loading, and subprocess working-directory setup use held directory
  descriptors and fail closed when the workspace root or a traversed directory
  is exchanged during an operation. This narrows filesystem race exposure; it
  does not turn an approved command or the host filesystem into a sandbox.
- `exec_command` uses an argument vector and minimal environment but is not an
  OS sandbox. Approved programs retain the current user's operating-system
  authority.
- Redaction is defense in depth. It cannot justify placing secrets or private
  organization data into prompts, repositories, or command arguments.

## Configuration boundary

`MiniAgentConfig` and nested Pydantic models reject unknown fields. The complete
executable source of fields and defaults is
`.mini-agent/config.example.toml`, installed by `mini-agent init-config`.
Documentation should link to that example instead of duplicating a second full
parameter table.

Configuration selection is explicit `--config`, then workspace config, then
the platform user config. Relative runtime data paths resolve from the selected
config file. Runtime session data is excluded from workspace tools and source
search.

## Test ownership

Tests mirror capability ownership rather than implementation call depth:

- `tests/providers/` validates raw provider payload boundaries.
- `tests/tools/` validates individual tool preflight, execution, and failures.
- context/checkpoint/rebuild modules test projection and cycle invariants.
- event/session/recovery modules test durable schemas, publication, corruption,
  and interrupted lifecycles.
- agent integration modules test turn ordering, budget sharing, approval, and
  end-to-end tool rounds using typed canonical responses.
- release-security modules test packaging metadata, workflows, staged/history
  privacy gates, and synthetic secret fixtures.

Large test modules should be split by these observable capabilities. Shared
fixtures belong in a narrowly named support module only after the same setup is
used by multiple test modules; raw dictionaries remain limited to boundary
tests.

## Release boundary

An open-source candidate is created from a clean public root, not the legacy
private Git history. Before publication, verify the built wheel, run Gitleaks
and the local privacy checker, and require both `Security / secrets` and the
aggregate `CI / release-gate` on the protected branch.

The deterministic benchmark is an RC regression baseline. It does not replace
the anonymized real-model tasks, provider token/latency data, and human rubric
required by the PRD for a stable v0.1 quality claim.

## Change discipline

Update this document when a dependency direction, durable lifecycle, trust
boundary, or module owner changes. Ordinary local implementation details should
stay in code and tests. Add a new abstraction only when a second observed use
case needs it and the dependency direction remains explicit.
