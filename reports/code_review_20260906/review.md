# Repository code review

The review covers the current working tree, including the pre-existing headless,
file-creation, directory-listing, and token-usage changes. Those related changes
are retained with the fixes so the resulting commit can build independently.
The review prioritizes workspace access, approval, durable side effects,
provider usage, and CLI outcomes; it is not a claim that every defect is absent.

## Confirmed findings and fixes

| Priority | Finding | Fix and regression evidence |
| --- | --- | --- |
| P1 | Creation could publish a file and then report `write_failed` if linking raised after its side effect. The agent could treat the operation as safely failed. | Mark publication ambiguous before linking; preserve a definite `target_exists` only for collisions. The post-link failure test observes complete file content and `write_status_unknown`; the concurrent-create test verifies no clobber. |
| P1 | Headless provider, session-stop, or writer-close failures still returned successful JSON and exit status. | Attempt every cleanup step and fail the command if any cleanup fails. Tests inject each failure, require nonzero exit, and reopen the session. |
| P1 | Opening a validated file could block indefinitely if it became a FIFO before the descriptor open. | Open nonblocking and reject non-regular descriptors with `fstat`. A subprocess regression with a timeout reproduces the old hang and verifies prompt rejection. |
| P2 | Creation preflight skipped workspace policy; an excluded root that was itself the absent leaf could also be created. | Validate lexical policy, symlinks, the parent directory, and the final excluded path before descriptor access. Tests cover sensitive paths, symlinks, and excluded leaves in both preflight and execution. |
| P2 | Patch size rejection occurred outside descriptor cleanup, leaking an opened target on failure. | Move bounded-content checks inside the cleanup scope. A file-growth regression verifies the target descriptor is closed after rejection. |
| P2 | Headless CLI opened an artifact store but never passed it to the agent, losing large outputs after preview truncation. | Share the artifact store with the agent. The CLI regression reopens durable events and checks the completed tool's artifact registration. |
| P2 | Headless usage converted missing fields to zero, summed earlier turns, and missed newly created active goals. | Sum only new turn events; leave incomplete fields unknown; read the authoritative session goal. Tests cover complete, absent, partial, zero, multi-response, and repeated-turn accounting. |
| P2 | Directory listing materialized and inspected the whole directory despite a bounded result limit. | Stream at most the entry limit plus one lookahead; inspect only displayed entries and sort that bounded selection. Tests verify metadata and output bounds at empty, exact-limit, and overflow sizes. |

Supporting fixes resolve the initial Ruff import-order error, strict type
narrowing errors, a creation fixture that omitted its required parent, and a
host-path test that incorrectly assumed test scratch could not be under the
home directory. README and architecture documentation describe the resulting
CLI, creation, approval, usage, and recovery behavior.

## Verification

- Initial baseline: 963 passed, 4 failed, 1 skipped; Ruff failed; Pyright reported
  79 errors, mostly cascading from missing event-union narrowing.
- Regression reproduction: 18 failing cases before runtime fixes, including
  the FIFO timeout, ambiguous creation, cleanup failures, lost artifacts, and
  unbounded metadata reads.
- Final full offline suite, including usage-boundary regressions: 998 passed,
  1 skipped.
- Ruff: passed. Strict Pyright: zero errors and warnings. `git diff --check`: passed.
- Gitleaks staged scan and the public-tree privacy hook: passed. The initial
  Gitleaks invocation exited with signal 9; direct scanning and the complete
  hook rerun passed with `GOMAXPROCS=2`. No hook was skipped or disabled.
- Offline locked-toolchain build: wheel and source distribution built successfully.
- Installed-wheel smoke: package imports from the installed wheel, command help,
  `run --help`, config initialization, and the deterministic three-task benchmark pass.
- Both distribution inventories exclude this task's caches and logs.

The real-provider integration test remains opt-in and was skipped. Checks were
performed on local macOS with Python 3.13; the Linux and other supported Python
matrix remains a CI responsibility. No external model calls were made.

Local raw evidence is retained in this directory's ignored `*.log` files and
`cache/`; regression tests live beside their corresponding source capabilities.
