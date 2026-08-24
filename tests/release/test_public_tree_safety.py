from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "check_public_tree.py"


def _run(
    *arguments: str,
    cwd: Path,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _stage_file(repository: Path, path: str, content: str) -> None:
    target = repository / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "--", path], cwd=repository, check=True)


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    return tmp_path


def test_staged_scan_accepts_public_source(repository: Path) -> None:
    _stage_file(repository, "src/example.py", "answer = 42\n")

    result = _run("--staged", cwd=repository)

    assert result.returncode == 0


@pytest.mark.parametrize(
    ("path", "content", "expected_reason"),
    [
        (".mini-agent/config.toml", "model = 'example'\n", "credential-related filename"),
        (
            "notes.txt",
            "/" + "Users/example/private/project\n",
            "machine-specific absolute path",
        ),
    ],
)
def test_staged_scan_rejects_private_material(
    repository: Path,
    path: str,
    content: str,
    expected_reason: str,
) -> None:
    _stage_file(repository, path, content)

    result = _run("--staged", cwd=repository)

    assert result.returncode == 1
    assert expected_reason in result.stderr


def test_private_denylist_blocks_without_printing_term(repository: Path) -> None:
    private_term = "private-company-marker"
    _stage_file(repository, "notes.txt", f"mentions {private_term}\n")
    environment = os.environ.copy()
    environment["MINI_AGENT_PUBLICATION_DENYLIST"] = private_term

    result = _run("--staged", cwd=repository, environment=environment)

    assert result.returncode == 1
    assert "private publication denylist" in result.stderr
    assert private_term not in result.stderr


def test_required_denylist_fails_closed(repository: Path) -> None:
    environment = os.environ.copy()
    environment.pop("MINI_AGENT_PUBLICATION_DENYLIST", None)

    result = _run("--tracked", "--require-denylist", cwd=repository, environment=environment)

    assert result.returncode == 2
    assert "is required" in result.stderr
