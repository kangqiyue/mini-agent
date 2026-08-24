# Contributing

Contributions should be small, typed, and backed by observable behavior.
Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) before changing module
ownership, durable lifecycles, or trust boundaries.

## Development principles

- Validate configuration, provider payloads, persisted records, and other
  dynamic input at their boundaries. Prefer Pydantic models that reject unknown
  fields, then pass trusted typed values internally.
- Keep dependencies one-way: CLI → agent loop → domain models, with provider
  and persistence adapters below the loop.
- Keep `events.jsonl` append-only and every durable schema explicitly
  versioned. Interrupted non-idempotent side effects are `unknown` and must
  never be replayed automatically.
- Make side effects and failure states explicit. Do not silently swallow
  exceptions or add abstractions for hypothetical future implementations.
- Use pytest for normal behavior, boundaries, and failure semantics. Preserve
  the invariants that acknowledged events are not lost and credentials are not
  printed or persisted.

## Development setup

```bash
uv sync --extra dev --locked --no-install-project
uv sync --extra dev --locked --no-build-isolation
uv run pre-commit install
uv run pytest
uv run ruff check .
uv run pyright
uv build --no-build-isolation
uv run mini-agent --help
```

These commands are the standard development and local release checks. The
maintainer checklist in [docs/RELEASING.md](docs/RELEASING.md) adds clean-root
history scans, the private denylist, and wheel installation smoke tests.

The first sync installs the locked development toolchain, including
`hatchling==1.32.0` and `editables==0.6`, which Hatchling needs for editable
installs. The second installs this project without build isolation. Keep that
order for local release-equivalent checks, using the required `uv==0.11.28`.

Add pytest coverage for normal behavior, boundaries, and failure semantics.
Changes to durable events must remain backward compatible or include an
explicit migration and incompatibility behavior. Never put real credentials in
tests, issue reports, commits, configuration examples, or benchmark artifacts.

The default suite is offline and skips the real-provider test. Maintainers may
explicitly exercise a local LiteLLM-compatible gateway after the ordinary suite
passes:

```bash
MINI_AGENT_RUN_LOCAL_INTEGRATION=1 \
MINI_AGENT_LITELLM_API_KEY_ENV=NAME_OF_EXISTING_KEY_ENV \
MINI_AGENT_LITELLM_MODEL=deepseek-flash \
uv run pytest -m local_integration
```

The optional test stores only the credential environment-variable name, uses a
temporary session directory, and checks that the credential value was not
persisted. Do not put the value itself in the command line or configuration.

Do not commit generated session data or `.mini-agent/config.toml`. The default
optional `.mini-agent/data/` session directory is ignored; add any other
repository-local `runtime.data_dir` you choose to Git ignore rules yourself. The tracked
`.mini-agent/config.example.toml` must continue to document every supported
configuration field. Do not add symbolic links or local credential-store files
to the public tree.

The pre-commit hooks run Gitleaks and the repository privacy check on staged
content and commit messages. They provide fast local feedback; `--no-verify`
can bypass them, so configure `Security / secrets`, `CI / release-gate`, and
CODEOWNER approval as required GitHub checks.

Organization-specific denylist terms are a local release-only input. Supply
them only to the process-scoped check documented in the release guide; never
commit them or pass them to an agent, provider, test, or workflow. Agent review
is supplemental and does not replace deterministic scanning.

See [docs/RELEASING.md](docs/RELEASING.md) for the maintainer preflight and
first-publication sequence.

The GitHub security workflow scans pushes and pull requests without receiving
the private organization denylist. Branch protection and GitHub push
protection remain necessary because a workflow runs after content reaches the
remote.
