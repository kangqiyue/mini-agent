# Security policy

## Supported versions

Security fixes are provided for the latest release on the default branch.
Mini Agent currently supports Python 3.12 through 3.14 on macOS and Linux. Windows is not
supported.

## Reporting a vulnerability

Do not open a public issue for a suspected credential disclosure or an
exploitable vulnerability. Use GitHub's
[private vulnerability report](https://github.com/kangqiyue/mini-agent/security/advisories/new).
Include a minimal reproduction, affected version, impact, and any suggested
mitigation. Do not include real credentials; use synthetic fixtures.

## Security model

Mini Agent is a local coding agent, not an operating-system sandbox.

- Built-in read, search, and apply-patch path resolution is constrained to a
  configured workspace using canonical path and symlink checks. File access and
  subprocess working-directory setup additionally use held directory
  descriptors and fail closed if the workspace root or a traversed directory is
  exchanged during the operation. An approved command is not subject to the
  file-tool boundary, and these checks are not an operating-system sandbox.
- Reads are automatic. File modifications and commands require explicit
  approval before `tool_started` is persisted.
- Commands use an argument vector with `shell=False` and a minimal environment,
  but approved executables can still perform arbitrary actions available to the
  current user. Command lookup and optional `git`/`rg` integrations trust the
  user's process environment and `PATH`.
- Timeout cleanup targets the original process group. A program that
  deliberately detaches into a new session may survive that cleanup.
- Provider credentials are read from the configured environment-variable name.
  Remote providers must use HTTPS; plain HTTP is accepted only for loopback
  gateways.
- Model requests contain the conversation, selected project instructions and
  file content, tool results, goal state, and checkpoints needed for the task.
  Known workspace/home prefixes are normalized at both main-model and
  checkpoint-model boundaries. Automatically collected Git branch telemetry is
  reduced to a category before system-prompt assembly. The runtime probes only
  repository/ref metadata; it does not collect working-tree status and cannot
  invoke repository clean/smudge filters or status-related hooks. Exact workspace
  metadata remains local for safe resume. This is not general DLP: branch text in task
  content, other paths, organization names, domains, and project content can
  still reach the provider. Do not include private organization information
  unless the provider is approved to receive it.
- Project instructions pass the same lexical and resolved sensitive-path policy
  and runtime-store exclusions as workspace read tools before opening. Complete
  bounded bytes are classified for PEM private-key boundaries before instructions
  enter a provider message.
- Credential redaction is defense in depth, not a substitute for keeping secrets
  out of prompts, files, command arguments, and logs.

Run Mini Agent only in workspaces, terminals, provider endpoints, and user
environments that you trust.

## Repository secret scanning

Install the repository hooks with `uv run pre-commit install`. They run
Gitleaks and reject common private filenames, machine-specific paths, and
private denylist matches in staged content and commit messages. Local hooks can
be bypassed with `--no-verify`, so the required `Security / secrets` workflow
repeats generic scanning on pushes and pull requests.

Organization-specific denylist terms are used only by the maintainer's local
tree check immediately before publication. The denylist is never passed to the
agent, provider calls, tests, hooks, or a GitHub workflow. See
[docs/RELEASING.md](docs/RELEASING.md). Secret-shaped test values are assembled
only at test runtime; `.gitleaks.toml` has no allowlist, test directories are
not excluded wholesale, and `.gitleaksignore` has no active exceptions.

Repository maintainers must also configure the default-branch ruleset to:

1. require pull requests plus the stable `Security / secrets` and
   `CI / release-gate` status checks;
2. require CODEOWNER approval for changes to scanning controls;
3. disallow direct-push, force-push, deletion, and ruleset bypass; and
4. enable GitHub secret scanning and push protection when available.

Local hooks and Actions are deterministic gates; an AI-agent review can add a
second opinion but must not replace them. Actions run after an unprotected push
has reached GitHub, so branch rules and host-side push protection are required.
