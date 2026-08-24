# Release checklist

This is the practical checklist for the first public release. The legacy Git
history remains private; the public repository starts from one reviewed root
commit.

## 1. Verify the project

```bash
uv sync --extra dev --locked --no-install-project
uv sync --extra dev --locked --no-build-isolation
uv run pytest
uv run ruff check .
uv run pyright
uv run mini-agent benchmark --output /tmp/mini-agent-benchmark.json
uv build --no-build-isolation
```

CI repeats the tests on Python 3.12–3.14 on macOS and Linux and installs the
built wheel in a fresh virtual environment.

## 2. Enable commit checks

```bash
uv run pre-commit install
```

Every commit then runs Gitleaks on staged content and checks filenames,
machine-specific paths, and the commit message. GitHub Actions repeats secret
scanning on pushes and pull requests, so `--no-verify` does not bypass the
remote check.

The optional private denylist is newline-delimited and is read only from the
process environment. Use it locally immediately before publication:

```bash
MINI_AGENT_PUBLICATION_DENYLIST='<load privately>' \
  python scripts/check_public_tree.py --tracked --require-denylist
```

Do not store the real denylist in the repository, shell history, agent prompts,
tests, or GitHub Actions. A failed scan blocks publication.

## 3. Create the clean public root

Create a separate, freshly initialized Git directory and copy only reviewed
public files into it. Do not copy the legacy `.git` directory, refs, hooks,
objects, remotes, `AGENTS.md`, `HANDOFF.md`, `.mini-agent/config.toml`, runtime
data, caches, or build output.

In the new directory:

```bash
git init
git add .
pre-commit run --all-files
python scripts/check_public_tree.py --staged
git commit -m "Initial public release"
gitleaks git . --redact --config .gitleaks.toml
```

Verify that the repository has exactly one root commit and inspect the public
identity, tracked paths, and remotes:

```bash
git rev-list --max-parents=0 HEAD
git log --format=fuller --decorate --stat
git ls-files
git remote -v
```

Do not push while preparing the repository. When ready, push only the reviewed
commit; never use `--mirror`, `--all`, or `--tags` for the first publication.

```bash
EXPECTED_COMMIT=$(git rev-parse HEAD)
git push --dry-run <remote> "$EXPECTED_COMMIT:refs/heads/main"
git push <remote> "$EXPECTED_COMMIT:refs/heads/main"
```

## 4. Configure GitHub

- Enable secret scanning and push protection.
- Require pull requests and CODEOWNER review.
- Require `Security / secrets` and `CI / release-gate`.
- Disallow force pushes, branch deletion, and ruleset bypass.
- Keep the renamed legacy repository private.

## 5. Tag a release

After the protected checks pass, create an annotated tag and scan once more:

```bash
git tag -a <release-tag> "$EXPECTED_COMMIT"
gitleaks git . --redact --config .gitleaks.toml
git push --dry-run <remote> "refs/tags/<release-tag>:refs/tags/<release-tag>"
git push <remote> "refs/tags/<release-tag>:refs/tags/<release-tag>"
```

Upload only the wheel and source distribution built from that commit.
