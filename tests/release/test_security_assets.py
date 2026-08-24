from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_private_local_files_are_ignored_and_untracked() -> None:
    ignore_text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")

    assert "AGENTS.md" in ignore_text
    assert "HANDOFF.md" in ignore_text
    assert ".mini-agent/config.toml" in ignore_text
    assert b"AGENTS.md" not in tracked
    assert b"HANDOFF.md" not in tracked
    assert b".mini-agent/config.toml" not in tracked


def test_commit_and_remote_secret_checks_are_configured() -> None:
    hooks = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/security.yml").read_text(encoding="utf-8")

    assert "default_install_hook_types: [pre-commit, commit-msg]" in hooks
    assert "id: gitleaks" in hooks
    assert "scripts/check_public_tree.py --staged" in hooks
    assert "scripts/check_public_tree.py --commit-message" in hooks
    assert "gitleaks/gitleaks-action@v2" in workflow
    assert "fetch-depth: 0" in workflow
    assert "scripts/check_public_tree.py --tracked" in workflow


def test_release_guide_requires_clean_root_and_private_denylist() -> None:
    guide = (ROOT / "docs/RELEASING.md").read_text(encoding="utf-8")

    assert "freshly initialized Git directory" in guide
    assert "MINI_AGENT_PUBLICATION_DENYLIST" in guide
    assert "--require-denylist" in guide
    assert "--mirror" in guide
    assert "Do not push while preparing" in guide
